from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = REPO_ROOT / "LongMemEval" / "longmemeval_s_sampled_100.json"
DEFAULT_OUTPUT = REPO_ROOT / "LongMemEval" / "results" / "memory_framework_predictions.jsonl"
DEFAULT_RUN_ROOT = REPO_ROOT / "runtime" / "longmemeval_memory_framework"
DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.5-27b"


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    temperature: float
    top_p: float | None
    extra_body: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the memory framework on LongMemEval sampled data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "qwen"],
        default="qwen",
        help="LLM used by the memory framework for extraction, retrieval rewrite, user card, and answering.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--session-concurrency", type=int, default=1)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=18182)
    parser.add_argument("--embedding-url", default=None)
    parser.add_argument(
        "--embedding-provider",
        choices=["legacy", "vllm", "openai", "openai_compatible"],
        default=None,
        help="Embedding backend used by the Memory API child process.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name sent to OpenAI-compatible embedding backends.",
    )
    parser.add_argument(
        "--memory-write-mode",
        choices=["sync", "async"],
        default="sync",
        help="Deprecated for historical ingest; /sessions/ingest always writes synchronously.",
    )
    parser.add_argument(
        "--ingest-turns",
        choices=["user_with_context", "user_only", "all"],
        default="user_with_context",
        help=(
            "user_with_context ingests true user-assistant QA pairs; "
            "user_only is conservative; all also replays assistant turns."
        ),
    )
    parser.add_argument(
        "--enable-session-summary",
        action="store_true",
        help="After each session, add a summary-style extraction turn for assistant-origin evidence.",
    )
    parser.add_argument("--recent-context-turns", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--refresh-user-card-every", type=int, default=25)
    parser.add_argument("--user-card-limit", type=int, default=30)
    parser.add_argument("--reuse-item-memory", action="store_true")
    parser.add_argument("--keep-item-memory", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run-plan", action="store_true")
    return parser.parse_args()


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


def provider_config(args: argparse.Namespace) -> ProviderConfig:
    load_project_env()
    if args.provider == "qwen":
        return ProviderConfig(
            provider="qwen",
            base_url=os.environ.get("ALIYUN_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or DEFAULT_ALIYUN_BASE_URL,
            api_key_env="DASHSCOPE_API_KEY",
            model=args.model
            or os.environ.get("ALIYUN_MODEL")
            or os.environ.get("DASHSCOPE_MODEL")
            or DEFAULT_QWEN_MODEL,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body={"result_format": "message", "enable_thinking": False},
        )
    return ProviderConfig(
        provider="deepseek",
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key_env="LLM_API_KEY",
        model=args.model or os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        temperature=args.temperature,
        top_p=None,
        extra_body=None,
    )


def api_key_for(config: ProviderConfig) -> str:
    if config.provider == "qwen":
        return (
            os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("ALIYUN_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or ""
        )
    return os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question_id = row.get("question_id")
            if question_id:
                completed.add(str(question_id))
    return completed


def selected_items(data: list[dict[str, Any]], start: int, limit: int | None) -> list[tuple[int, dict[str, Any]]]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    items = data[start:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        items = items[:limit]
    return [(start + idx, item) for idx, item in enumerate(items)]


def item_memory_root(run_root: Path, question_id: str, reuse: bool) -> Path:
    root = run_root / "items" / question_id
    if root.exists() and not reuse:
        shutil.rmtree(root)
    (root / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "runtime").mkdir(parents=True, exist_ok=True)
    return root


def build_api_env(
    args: argparse.Namespace,
    config: ProviderConfig,
    memory_root: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MEMORY_ROOT"] = str(memory_root)
    env["MEMORY_API_HOST"] = args.api_host
    env["MEMORY_API_PORT"] = str(args.api_port)
    if args.embedding_url:
        env["EMBEDDING_SERVICE_URL"] = args.embedding_url
    embedding_url = env.get("EMBEDDING_SERVICE_URL", "")
    embedding_provider = args.embedding_provider or os.environ.get("EMBEDDING_PROVIDER")
    if not embedding_provider and embedding_url.rstrip("/").endswith("/v1"):
        embedding_provider = "vllm"
    if embedding_provider:
        env["EMBEDDING_PROVIDER"] = embedding_provider

    embedding_model = args.embedding_model or os.environ.get("EMBEDDING_MODEL")
    if not embedding_model and (
        env.get("EMBEDDING_PROVIDER") in {"vllm", "openai", "openai_compatible"}
        or embedding_url.rstrip("/").endswith("/v1")
    ):
        embedding_model = "bge-m3"
    if embedding_model:
        env["EMBEDDING_MODEL"] = embedding_model

    for key in ["EMBEDDING_API_KEY", "EMBEDDING_TIMEOUT_SECONDS"]:
        if os.environ.get(key):
            env[key] = os.environ[key]
    env["LLM_PROVIDER"] = "deepseek"
    env["LLM_BASE_URL"] = config.base_url
    env["LLM_MODEL"] = config.model
    env["LLM_TEMPERATURE"] = str(config.temperature)
    if config.top_p is not None:
        env["LLM_TOP_P"] = str(config.top_p)
    else:
        env.pop("LLM_TOP_P", None)
    if config.extra_body is not None:
        env["LLM_EXTRA_BODY_JSON"] = json.dumps(config.extra_body, ensure_ascii=False)
    else:
        env.pop("LLM_EXTRA_BODY_JSON", None)

    api_key = api_key_for(config)
    if not api_key:
        raise RuntimeError(f"Missing API key for provider {config.provider}.")
    env["LLM_API_KEY"] = api_key
    return env


def start_memory_api(args: argparse.Namespace, config: ProviderConfig, memory_root: Path) -> subprocess.Popen[str]:
    env = build_api_env(args, config, memory_root)
    log_path = memory_root / "runtime" / "memory_api.log"
    log_handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "memory_system.api"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


async def wait_for_api(base_url: str, proc: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=5.0) as client:
        last_error = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Memory API exited early with code {proc.returncode}.")
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("ok"):
                        return
                    last_error = json.dumps(payload, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Memory API did not become healthy: {last_error}")


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def message_to_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def build_ingest_text(session_id: str, date: str, message: dict[str, Any]) -> str:
    role = str(message.get("role", "unknown"))
    content = message_to_text(message)
    return (
        f"[LongMemEval session_id={session_id}; date={date}; speaker={role}]\n"
        f"{content}"
    )


def with_ingest_header(session_id: str, date: str, message: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(message.get("role", "user")),
        "content": build_ingest_text(session_id, date, message),
    }


def to_chat_message(message: dict[str, Any]) -> dict[str, str]:
    role = message.get("role")
    if role not in {"user", "assistant"}:
        role = "user"
    return {"role": role, "content": message_to_text(message)}


def session_summary_text(session_id: str, date: str, session: list[dict[str, Any]]) -> str:
    lines = [
        f"[LongMemEval session_id={session_id}; date={date}; speaker=session_summary]",
        "Extract LongMemEval evidence memories from this complete historical conversation.",
        "Store user facts, preferences, activities, plans, purchases, counts, dated events, user questions, user attitudes, and concrete assistant recommendations or conclusions that may help answer future questions.",
        "Preserve the date and session_id in each memory when possible.",
        "Do not store generic assistant advice, boilerplate, jokes, private credentials, or unsupported claims as user facts.",
        "",
        "Conversation:",
    ]
    for message in session:
        role = message.get("role", "unknown")
        if role in {"user", "assistant"}:
            lines.append(f"{role}: {message_to_text(message)}")
    return "\n".join(lines)


def iter_ingest_turns(
    item: dict[str, Any],
    ingest_turns: str,
    enable_session_summary: bool,
    recent_context_turns: int,
) -> list[dict[str, Any]]:
    dates = item.get("haystack_dates") or []
    session_ids = item.get("haystack_session_ids") or []
    sessions = item.get("haystack_sessions") or []
    rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        session_id = str(session_ids[session_index] if session_index < len(session_ids) else f"session_{session_index}")
        date = str(dates[session_index] if session_index < len(dates) else "")
        if not isinstance(session, list):
            continue
        valid_messages = [message for message in session if message.get("role") in {"user", "assistant"}]
        if ingest_turns == "user_with_context":
            message_index = 0
            while message_index < len(valid_messages):
                message = valid_messages[message_index]
                if message.get("role") != "user":
                    message_index += 1
                    continue
                qa_messages = [with_ingest_header(session_id, date, message)]
                if (
                    message_index + 1 < len(valid_messages)
                    and valid_messages[message_index + 1].get("role") == "assistant"
                ):
                    qa_messages.append(with_ingest_header(session_id, date, valid_messages[message_index + 1]))
                    message_index += 2
                else:
                    message_index += 1
                rows.append(
                    {
                        "kind": "qa_pair",
                        "session_id": session_id,
                        "date": date,
                        "messages": qa_messages,
                    }
                )
            if enable_session_summary and valid_messages:
                rows.append(
                    {
                        "kind": "session_summary",
                        "session_id": session_id,
                        "date": date,
                        "messages": [
                            {
                                "role": "user",
                                "content": session_summary_text(session_id, date, valid_messages),
                            }
                        ],
                    }
                )
            continue

        for message_index, message in enumerate(valid_messages):
            role = message.get("role")
            if ingest_turns == "user_only" and role != "user":
                continue
            if role not in {"user", "assistant"}:
                continue
            rows.append(
                {
                    "kind": "turn",
                    "session_id": session_id,
                    "date": date,
                    "messages": [with_ingest_header(session_id, date, message)],
                }
            )
        if enable_session_summary and valid_messages:
            rows.append(
                {
                    "kind": "session_summary",
                    "session_id": session_id,
                    "date": date,
                    "messages": [
                        {
                            "role": "user",
                            "content": session_summary_text(session_id, date, valid_messages),
                        }
                    ],
                }
            )
    return rows


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


async def ingest_dataset_item(
    client: httpx.AsyncClient,
    base_url: str,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    turns = iter_ingest_turns(
        item,
        args.ingest_turns,
        args.enable_session_summary,
        args.recent_context_turns,
    )
    semaphore = asyncio.Semaphore(args.session_concurrency)
    applied = 0
    skipped = 0
    edit_operations = 0
    errors: list[str] = []

    async def ingest_one(idx: int, turn: dict[str, Any]) -> None:
        nonlocal applied, skipped, edit_operations
        session_id = turn["session_id"]
        framework_session_id = f"lme_{item.get('question_id')}_{session_id}"
        payload = {
            "session_id": framework_session_id,
            "messages": turn["messages"],
            "top_k": args.top_k,
            "debug_prompt": False,
        }
        try:
            async with semaphore:
                result = await post_json(client, f"{base_url}/sessions/ingest", payload)
            applied += int(result.get("applied") or 0)
            skipped += int(result.get("skipped") or 0)
            edit_operations += len(result.get("edit_operations") or [])
            if args.refresh_user_card_every and (idx + 1) % args.refresh_user_card_every == 0:
                await post_json(client, f"{base_url}/user-card/refresh?limit={args.user_card_limit}", {})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    progress = None
    if not args.no_progress:
        progress = tqdm(
            total=len(turns),
            desc=f"ingest {item.get('question_id')}",
            leave=False,
            unit="turn",
        )

    async def tracked_ingest(idx: int, turn: dict[str, Any]) -> None:
        try:
            await ingest_one(idx, turn)
        finally:
            if progress is not None:
                progress.update(1)

    try:
        await asyncio.gather(*(tracked_ingest(idx, turn) for idx, turn in enumerate(turns)))
    finally:
        if progress is not None:
            progress.close()

    card = await post_json(client, f"{base_url}/user-card/refresh?limit={args.user_card_limit}", {})
    memories = await client.get(f"{base_url}/memories", params={"limit": 1_000_000})
    memories.raise_for_status()
    memory_rows = memories.json().get("memories", [])
    return {
        "ingested_messages": len(turns),
        "ingest_mode": args.ingest_turns,
        "session_summary_enabled": args.enable_session_summary,
        "memory_updates_applied": applied,
        "memory_updates_skipped": skipped,
        "edit_operations": edit_operations,
        "ingest_errors": errors[:20],
        "ingest_error_count": len(errors),
        "memory_count": len(memory_rows),
        "user_card": card,
    }


async def answer_dataset_item(
    client: httpx.AsyncClient,
    base_url: str,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = str(item.get("question", ""))
    question_date = str(item.get("question_date", ""))
    answer_prompt = (
        f"[LongMemEval question_date={question_date}]\n"
        "Answer the following LongMemEval question using only your private Wiki memory. "
        "Use dated memories, session ids, user statements, and assistant-provided recommendations as evidence. "
        "If the evidence is insufficient, answer exactly: I don't know. "
        "Return only the final answer, with no explanation.\n\n"
        f"Question: {question}"
    )
    result = await post_json(
        client,
        f"{base_url}/chat",
        {
            "session_id": f"lme_answer_{item.get('question_id')}",
            "user_input": answer_prompt,
            "recent_messages": [],
            "top_k": args.top_k,
            "async_memory_write": False,
            "debug_prompt": True,
        },
    )
    search = await post_json(
        client,
        f"{base_url}/search",
        {"query": question, "top_k": args.top_k},
    )
    return {
        "reply": result.get("reply", ""),
        "blocked": result.get("blocked", False),
        "retrieval_query": result.get("retrieval_query"),
        "recalled_memories": result.get("recalled_memories", []),
        "edit_operations": result.get("edit_operations", []),
        "debug_prompt": result.get("prompt"),
        "search_retrieval_query": search.get("retrieval_query"),
        "search_memories": search.get("memories", []),
    }


async def process_item(
    dataset_index: int,
    item: dict[str, Any],
    args: argparse.Namespace,
    config: ProviderConfig,
) -> dict[str, Any]:
    question_id = str(item.get("question_id", dataset_index))
    memory_root = item_memory_root(args.run_root, question_id, args.reuse_item_memory)
    base_url = f"http://{args.api_host}:{args.api_port}"
    proc = start_memory_api(args, config, memory_root)
    started = time.time()
    try:
        await wait_for_api(base_url, proc, args.startup_timeout)
        async with httpx.AsyncClient(timeout=args.request_timeout) as client:
            ingest_stats = await ingest_dataset_item(client, base_url, item, args)
            answer = await answer_dataset_item(client, base_url, item, args)
        return {
            "question_id": question_id,
            "dataset_index": dataset_index,
            "question_type": item.get("question_type"),
            "question_date": item.get("question_date"),
            "question": item.get("question"),
            "gold_answer": item.get("answer"),
            "prediction": answer["reply"],
            "error": None,
            "provider": config.provider,
            "model": config.model,
            "top_k": args.top_k,
            "memory_root": str(memory_root),
            "elapsed_seconds": round(time.time() - started, 3),
            **ingest_stats,
            **answer,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "question_id": question_id,
            "dataset_index": dataset_index,
            "question_type": item.get("question_type"),
            "question": item.get("question"),
            "gold_answer": item.get("answer"),
            "prediction": "",
            "error": f"{type(exc).__name__}: {exc}",
            "provider": config.provider,
            "model": config.model,
            "memory_root": str(memory_root),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    finally:
        stop_process(proc)
        if not args.keep_item_memory and not args.reuse_item_memory:
            shutil.rmtree(memory_root, ignore_errors=True)


async def run() -> None:
    args = parse_args()
    if args.session_concurrency < 1:
        raise ValueError("--session-concurrency must be >= 1")
    data = read_json(args.input)
    if not isinstance(data, list):
        raise TypeError(f"Expected list input in {args.input}")
    config = provider_config(args)
    if args.dry_run_plan:
        items = selected_items(data, args.start, args.limit)
        plan = {
            "input": str(args.input),
            "output": str(args.output),
            "items": len(items),
            "provider": config.provider,
            "model": config.model,
            "embedding_url": args.embedding_url,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "ingest_mode": args.ingest_turns,
            "enable_session_summary": args.enable_session_summary,
            "memory_isolation": "one MEMORY_ROOT per dataset item",
            "pipeline": [
                "start isolated memory API",
                "ingest historical user/assistant turns through /sessions/ingest without generating replies",
                "optionally add one session-summary extraction turn per haystack session",
                "trigger safety/privacy/retrieval/edit planner during ingestion",
                "refresh User Card periodically and after ingestion",
                "answer question through /chat with User Card and retrieved Wiki memories",
                "call /search for retrieval diagnostics",
                "write JSONL prediction row",
            ],
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    completed = load_completed_ids(args.output) if args.resume else set()
    args.run_root.mkdir(parents=True, exist_ok=True)
    work_items = selected_items(data, args.start, args.limit)
    if args.resume:
        work_items = [
            (dataset_index, item)
            for dataset_index, item in work_items
            if str(item.get("question_id", dataset_index)) not in completed
        ]
    iterator = work_items
    progress = None
    if not args.no_progress:
        progress = tqdm(iterator, desc="LongMemEval items", unit="item")
        iterator = progress
    try:
        for dataset_index, item in iterator:
            question_id = str(item.get("question_id", dataset_index))
            if question_id in completed:
                continue
            row = await process_item(dataset_index, item, args, config)
            append_jsonl(args.output, row)
            if progress is not None:
                progress.set_postfix_str(f"{question_id} done")
    finally:
        if progress is not None:
            progress.close()


if __name__ == "__main__":
    asyncio.run(run())
