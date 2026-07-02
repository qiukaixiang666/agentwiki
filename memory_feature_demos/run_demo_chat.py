from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = DEMO_ROOT / "demo_manifest.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_EMBEDDING_URL = "http://127.0.0.1:18083/v1"
DEFAULT_TOP_K = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one memory feature demo through /chat. "
            "Each turn uses a fresh session_id, so answers do not include previous QA pairs."
        )
    )
    parser.add_argument(
        "--demo",
        default=None,
        help="Demo id. Omit this to choose interactively.",
    )
    parser.add_argument("--api-host", default=DEFAULT_HOST)
    parser.add_argument("--api-port", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--embedding-url", default=DEFAULT_EMBEDDING_URL)
    parser.add_argument("--embedding-provider", default="vllm")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--debug-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete this demo's data/runtime outputs before starting.",
    )
    parser.add_argument(
        "--keep-api-running",
        action="store_true",
        help="Leave the Memory API child process running after the demo.",
    )
    parser.add_argument(
        "--refresh-user-card-after-build",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Refresh user card between build and probe turns. Defaults to true only for user_card demo.",
    )
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
        help="Print selected demo and prompts without starting services or calling models.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def choose_demo(manifest: dict[str, Any], requested: str | None) -> dict[str, Any]:
    demos = list(manifest.get("demos") or [])
    by_id = {str(item.get("id")): item for item in demos}
    if requested:
        if requested not in by_id:
            raise ValueError(f"Unknown demo id {requested!r}. Available: {', '.join(by_id)}")
        return by_id[requested]

    print("Available demos:")
    for index, demo in enumerate(demos, start=1):
        print(f"  {index}. {demo['id']} - {demo.get('title', '')}")
    choice = input("Select demo id or number: ").strip()
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(demos):
            return demos[index - 1]
    if choice in by_id:
        return by_id[choice]
    raise ValueError(f"Invalid demo selection: {choice!r}")


def load_project_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def prepare_memory_root(memory_root: Path, reset: bool) -> None:
    if reset and memory_root.exists():
        data_dir = memory_root / "data"
        runtime_dir = memory_root / "runtime"
        for child in [data_dir, runtime_dir]:
            if child.exists():
                shutil.rmtree(child)
    (memory_root / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (memory_root / "runtime").mkdir(parents=True, exist_ok=True)


def api_env(args: argparse.Namespace, demo: dict[str, Any], memory_root: Path, api_port: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["MEMORY_ROOT"] = str(memory_root)
    env["MEMORY_API_HOST"] = args.api_host
    env["MEMORY_API_PORT"] = str(api_port)
    env["MEMORY_TOP_K"] = str(args.top_k)
    env["EMBEDDING_PROVIDER"] = args.embedding_provider
    env["EMBEDDING_SERVICE_URL"] = args.embedding_url
    env["EMBEDDING_MODEL"] = args.embedding_model
    env.setdefault("EMBEDDING_API_KEY", "EMPTY")
    if args.llm_provider:
        env["LLM_PROVIDER"] = args.llm_provider
    if args.llm_model:
        env["LLM_MODEL"] = args.llm_model
    if args.llm_base_url:
        env["LLM_BASE_URL"] = args.llm_base_url
    return env


def health(base_url: str, timeout: float = 10.0) -> dict[str, Any] | None:
    import httpx

    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("service") != "memory_api":
            return None
        return payload
    except Exception:
        return None


def start_api(args: argparse.Namespace, demo: dict[str, Any], memory_root: Path, api_port: int) -> subprocess.Popen | None:
    base_url = f"http://{args.api_host}:{api_port}"
    existing = health(base_url)
    if existing is not None:
        existing_root = str((existing.get("details") or {}).get("memory_root") or "")
        if Path(existing_root).resolve() == memory_root.resolve():
            print(f"Using existing Memory API at {base_url} for {memory_root}")
            return None
        raise RuntimeError(
            f"{base_url} is already serving a different MEMORY_ROOT: {existing_root}. "
            "Use --api-port to choose another port."
        )

    log_path = memory_root / "runtime" / "memory_api.log"
    env = api_env(args, demo, memory_root, api_port)
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "memory_system.api"],
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Memory API exited early. Check log: {log_path}")
        if health(base_url) is not None:
            print(f"Started Memory API at {base_url}, pid={proc.pid}, log={log_path}")
            return proc
        time.sleep(1.0)
    proc.terminate()
    raise TimeoutError(f"Timed out waiting for Memory API at {base_url}. Check log: {log_path}")


def post_json(client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object from {url}")
    return data


def run_demo(args: argparse.Namespace, demo: dict[str, Any]) -> None:
    demo_id = str(demo["id"])
    memory_root = resolve_demo_path(str(demo["memory_root"]))
    prompt_file = resolve_demo_path(str(demo["prompt_file"]))
    api_port = int(args.api_port or demo.get("default_api_port") or 18240)
    base_url = f"http://{args.api_host}:{api_port}"

    prompts = read_jsonl(prompt_file)
    print(f"Demo: {demo_id}")
    print(f"Memory root: {memory_root}")
    print(f"Prompt file: {prompt_file}")
    print(f"Turns: {len(prompts)}, top_k={args.top_k}")
    print("Each turn uses a fresh session_id; no previous QA pairs are passed as recent_messages.")

    if args.dry_run_plan:
        for row in prompts:
            print(f"{row['turn_index']:02d} [{row['phase']}] {row['user_prompt'][:120]}")
        return

    prepare_memory_root(memory_root, reset=args.reset)
    results_path = memory_root / "runtime" / "demo_chat_results.jsonl"
    run_id = uuid.uuid4().hex[:8]
    refresh_after_build = (
        args.refresh_user_card_after_build
        if args.refresh_user_card_after_build is not None
        else demo_id == "user_card"
    )

    proc: subprocess.Popen | None = None
    try:
        proc = start_api(args, demo, memory_root, api_port)
        import httpx

        with httpx.Client(timeout=args.timeout) as client:
            build_phase_done = False
            for row in prompts:
                if (
                    refresh_after_build
                    and not build_phase_done
                    and row.get("phase") == "probe"
                ):
                    card = post_json(client, f"{base_url}/user-card/refresh?limit=30", {})
                    append_jsonl(
                        results_path,
                        {
                            "kind": "user_card_refresh",
                            "demo_id": demo_id,
                            "run_id": run_id,
                            "response": card,
                        },
                    )
                    build_phase_done = True

                turn_index = int(row["turn_index"])
                payload = {
                    "session_id": f"demo_{demo_id}_{run_id}_turn_{turn_index:02d}",
                    "user_input": row["user_prompt"],
                    "recent_messages": [],
                    "top_k": args.top_k,
                    "async_memory_write": False,
                    "debug_prompt": args.debug_prompt,
                }
                started = time.time()
                error = None
                response_payload: dict[str, Any] | None = None
                try:
                    response_payload = post_json(client, f"{base_url}/chat", payload)
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"

                append_jsonl(
                    results_path,
                    {
                        "kind": "chat_turn",
                        "demo_id": demo_id,
                        "run_id": run_id,
                        "turn_index": turn_index,
                        "phase": row.get("phase"),
                        "purpose": row.get("purpose"),
                        "expected_observation": row.get("expected_observation"),
                        "request": payload,
                        "response": response_payload,
                        "error": error,
                        "elapsed_seconds": round(time.time() - started, 3),
                    },
                )
                reply = (response_payload or {}).get("reply", "") if response_payload else ""
                print(
                    f"turn {turn_index:02d} [{row.get('phase')}] "
                    f"error={error} reply={reply[:80]!r}",
                    flush=True,
                )
                if error:
                    break
    finally:
        if proc is not None and not args.keep_api_running:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    print(f"Results path: {results_path}")


def resolve_demo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        repo_candidate = (REPO_ROOT / path).resolve()
        if repo_candidate.exists():
            return repo_candidate
        demo_candidate = (DEMO_ROOT / path).resolve()
        if demo_candidate.exists():
            return demo_candidate
        return repo_candidate
    if path.exists():
        return path.resolve()
    marker = "memory_feature_demos"
    parts = path.parts
    if marker in parts:
        index = parts.index(marker)
        candidate = DEMO_ROOT.joinpath(*parts[index + 1 :])
        return candidate.resolve()
    return path.resolve()


def main() -> None:
    load_project_env()
    args = parse_args()
    manifest = load_json(MANIFEST_PATH)
    demo = choose_demo(manifest, args.demo)
    run_demo(args, demo)


if __name__ == "__main__":
    main()
