from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LongMemEval.scripts.memory_framework_eval import (  # noqa: E402
    iter_ingest_turns,
    post_json,
    provider_config,
    start_memory_api,
    stop_process,
    wait_for_api,
)


DEFAULT_EXAMPLES_DIR = (
    REPO_ROOT / "result" / "qwen_error_examples_by_type" / "examples"
)
DEFAULT_EXAMPLES_JSONL = (
    REPO_ROOT / "result" / "qwen_error_examples_by_type" / "examples.jsonl"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runtime" / "qwen_error_example_memory_banks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one isolated Memory Wiki data directory per Qwen error example. "
            "The script follows the memory framework ingestion path and does not "
            "ask the final LongMemEval question."
        )
    )
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES_JSONL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--question-type", action="append", default=[])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "qwen"],
        default="qwen",
        help="LLM used by the memory framework during ingestion.",
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
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--ingest-turns",
        choices=["user_with_context", "user_only", "all"],
        default="user_with_context",
    )
    parser.add_argument("--enable-session-summary", action="store_true")
    parser.add_argument("--recent-context-turns", type=int, default=8)
    parser.add_argument(
        "--max-ingest-turns",
        type=int,
        default=None,
        help=(
            "Only ingest the first N historical turns after applying --ingest-turns. "
            "Useful for smoke tests such as building from the first 20 user turns."
        ),
    )
    parser.add_argument("--refresh-user-card-every", type=int, default=25)
    parser.add_argument("--user-card-limit", type=int, default=30)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--api-io-log",
        type=Path,
        default=None,
        help=(
            "Write full /sessions/ingest request and response JSONL here. "
            "Defaults to <memory_root>/runtime/api_io.jsonl."
        ),
    )
    parser.add_argument(
        "--print-api-io",
        action="store_true",
        help="Print a compact request/response summary for each /sessions/ingest call.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if isinstance(row, dict):
                rows.append(row)
            else:
                raise TypeError(f"Expected JSON object on line {line_number} of {path}")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def compact_text(text: str, limit: int = 280) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_api_io_summary(row: dict[str, Any]) -> None:
    request = row.get("request", {})
    response = row.get("response")
    error = row.get("error")
    messages = request.get("messages") or []
    latest = messages[-1]["content"] if messages else ""
    if response is None:
        print(
            "API_IO "
            f"turn={row.get('turn_index')} session={request.get('session_id')} "
            f"status=error error={error} latest={compact_text(latest)!r}",
            flush=True,
        )
        return
    operations = response.get("edit_operations") or []
    print(
        "API_IO "
        f"turn={row.get('turn_index')} session={request.get('session_id')} "
        f"messages={len(messages)} applied={response.get('applied')} "
        f"skipped={response.get('skipped')} ops={len(operations)} "
        f"retrieval_query={response.get('retrieval_query')!r} "
        f"latest={compact_text(latest)!r}",
        flush=True,
    )
    for op_index, operation in enumerate(operations, start=1):
        print(
            "API_IO_OP "
            f"turn={row.get('turn_index')} op={op_index} "
            f"type={operation.get('op')} memory_type={operation.get('memory_type')} "
            f"topic={operation.get('topic')} confidence={operation.get('confidence')} "
            f"content={compact_text(operation.get('content') or '')!r}",
            flush=True,
        )


def iter_example_wrappers(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.examples_dir.exists():
        for path in sorted(args.examples_dir.iterdir()):
            if path.suffix == ".json":
                row = load_json(path)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object in {path}")
                row["_source_path"] = str(path)
                rows.append(row)
            elif path.suffix == ".jsonl":
                for row in load_jsonl(path):
                    row["_source_path"] = str(path)
                    rows.append(row)

    if not rows and args.examples_jsonl.exists():
        for row in load_jsonl(args.examples_jsonl):
            row["_source_path"] = str(args.examples_jsonl)
            rows.append(row)

    rows = [row for row in rows if include_row(row, args)]
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    rows = rows[args.start :]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        rows = rows[: args.limit]
    return rows


def include_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    item = unwrap_dataset_item(row)
    if args.question_id and str(item.get("question_id")) not in set(args.question_id):
        return False
    if args.question_type and str(item.get("question_type")) not in set(args.question_type):
        return False
    return True


def unwrap_dataset_item(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_dataset_item")
    if isinstance(raw, dict):
        return raw
    return row


def completed_ids(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    ids: set[str] = set()
    for row in load_jsonl(index_path):
        if row.get("status") == "built" and row.get("question_id"):
            ids.add(str(row["question_id"]))
    return ids


def slugify(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def memory_root_for(output_root: Path, item: dict[str, Any]) -> Path:
    question_type = slugify(item.get("question_type"), "unknown_type")
    question_id = slugify(item.get("question_id"), "unknown_id")
    return output_root / f"{question_type}__{question_id}"


def ensure_within(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    if parent_resolved == child_resolved or parent_resolved in child_resolved.parents:
        return
    raise ValueError(f"Refusing to operate outside {parent_resolved}: {child_resolved}")


def prepare_memory_root(
    output_root: Path,
    memory_root: Path,
    overwrite: bool,
) -> tuple[bool, str | None]:
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_within(output_root, memory_root)
    if memory_root.exists():
        if not overwrite:
            return False, "memory root already exists; pass --overwrite to rebuild"
        shutil.rmtree(memory_root)
    (memory_root / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (memory_root / "runtime").mkdir(parents=True, exist_ok=True)
    return True, None


def validate_dataset_item(item: dict[str, Any]) -> None:
    required = [
        "question_id",
        "question_type",
        "question",
        "question_date",
        "haystack_sessions",
    ]
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"Dataset item is missing required keys: {missing}")
    if not isinstance(item.get("haystack_sessions"), list):
        raise TypeError("Dataset item haystack_sessions must be a list")


def item_summary(item: dict[str, Any]) -> dict[str, Any]:
    sessions = item.get("haystack_sessions") or []
    message_count = 0
    for session in sessions:
        if isinstance(session, list):
            message_count += sum(
                1
                for message in session
                if isinstance(message, dict) and message.get("role") in {"user", "assistant"}
            )
    return {
        "question_id": item.get("question_id"),
        "question_type": item.get("question_type"),
        "question_date": item.get("question_date"),
        "question": item.get("question"),
        "gold_answer": item.get("answer"),
        "haystack_sessions": len(sessions),
        "haystack_messages": message_count,
    }


def build_manifest(
    wrapper: dict[str, Any],
    item: dict[str, Any],
    memory_root: Path,
    args: argparse.Namespace,
    ingest_stats: dict[str, Any],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "error": error,
        "source_path": wrapper.get("_source_path"),
        "memory_root": str(memory_root),
        "question_id": str(item.get("question_id")),
        "question_type": item.get("question_type"),
        "question_date": item.get("question_date"),
        "question": item.get("question"),
        "gold_answer": item.get("answer"),
        "original_prediction": wrapper.get("prediction"),
        "original_error": wrapper.get("error"),
        "dataset_index": wrapper.get("dataset_index"),
        "input_stats": item_summary(item),
        "build_config": {
            "provider": args.provider,
            "model": args.model,
            "top_k": args.top_k,
            "ingest_turns": args.ingest_turns,
            "max_ingest_turns": args.max_ingest_turns,
            "enable_session_summary": args.enable_session_summary,
            "recent_context_turns": args.recent_context_turns,
            "session_concurrency": args.session_concurrency,
            "embedding_url": args.embedding_url,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
        },
        "ingest_stats": ingest_stats,
    }


async def ingest_dataset_item_limited(
    client: httpx.AsyncClient,
    base_url: str,
    item: dict[str, Any],
    args: argparse.Namespace,
    api_io_log: Path | None = None,
) -> dict[str, Any]:
    turns = iter_ingest_turns(
        item,
        args.ingest_turns,
        args.enable_session_summary,
        args.recent_context_turns,
    )
    total_turns = len(turns)
    if args.max_ingest_turns is not None:
        if args.max_ingest_turns < 0:
            raise ValueError("--max-ingest-turns must be non-negative")
        turns = turns[: args.max_ingest_turns]

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
            row = {
                "turn_index": idx + 1,
                "kind": turn.get("kind"),
                "source_session_id": session_id,
                "date": turn.get("date"),
                "endpoint": "/sessions/ingest",
                "request": payload,
                "response": result,
                "error": None,
            }
            if api_io_log is not None:
                append_jsonl(api_io_log, row)
            if args.print_api_io:
                print_api_io_summary(row)
            applied += int(result.get("applied") or 0)
            skipped += int(result.get("skipped") or 0)
            edit_operations += len(result.get("edit_operations") or [])
            if args.refresh_user_card_every and (idx + 1) % args.refresh_user_card_every == 0:
                await post_json(
                    client,
                    f"{base_url}/user-card/refresh?limit={args.user_card_limit}",
                    {},
                )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            row = {
                "turn_index": idx + 1,
                "kind": turn.get("kind"),
                "source_session_id": session_id,
                "date": turn.get("date"),
                "endpoint": "/sessions/ingest",
                "request": payload,
                "response": None,
                "error": error,
            }
            if api_io_log is not None:
                append_jsonl(api_io_log, row)
            if args.print_api_io:
                print_api_io_summary(row)
            errors.append(error)

    await asyncio.gather(*(ingest_one(idx, turn) for idx, turn in enumerate(turns)))

    card = await post_json(client, f"{base_url}/user-card/refresh?limit={args.user_card_limit}", {})
    memories = await client.get(f"{base_url}/memories", params={"limit": 1_000_000})
    memories.raise_for_status()
    memory_rows = memories.json().get("memories", [])
    return {
        "ingested_messages": len(turns),
        "available_ingest_turns": total_turns,
        "max_ingest_turns": args.max_ingest_turns,
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


async def build_one(
    wrapper: dict[str, Any],
    args: argparse.Namespace,
    config: Any,
    index_path: Path,
) -> dict[str, Any]:
    item = unwrap_dataset_item(wrapper)
    validate_dataset_item(item)
    memory_root = memory_root_for(args.output_root, item)
    created, skip_reason = prepare_memory_root(args.output_root, memory_root, args.overwrite)
    if not created:
        manifest = build_manifest(
            wrapper,
            item,
            memory_root,
            args,
            ingest_stats={},
            status="skipped",
            error=skip_reason,
        )
        append_jsonl(index_path, manifest)
        return manifest

    base_url = f"http://{args.api_host}:{args.api_port}"
    proc = start_memory_api(args, config, memory_root)
    started = time.time()
    try:
        await wait_for_api(base_url, proc, args.startup_timeout)
        async with httpx.AsyncClient(timeout=args.request_timeout) as client:
            api_io_log = args.api_io_log or (memory_root / "runtime" / "api_io.jsonl")
            ingest_stats = await ingest_dataset_item_limited(
                client,
                base_url,
                item,
                args,
                api_io_log=api_io_log,
            )
        ingest_stats["api_io_log"] = str(api_io_log)
        ingest_error_count = int(ingest_stats.get("ingest_error_count") or 0)
        status = "failed" if ingest_error_count else "built"
        error = f"{ingest_error_count} ingest request(s) failed" if ingest_error_count else None
        manifest = build_manifest(
            wrapper,
            item,
            memory_root,
            args,
            ingest_stats=ingest_stats,
            status=status,
            error=error,
        )
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(memory_root / "manifest.json", manifest)
        append_jsonl(index_path, manifest)
        return manifest
    except Exception as exc:  # noqa: BLE001
        manifest = build_manifest(
            wrapper,
            item,
            memory_root,
            args,
            ingest_stats={},
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(memory_root / "manifest.json", manifest)
        append_jsonl(index_path, manifest)
        return manifest
    finally:
        stop_process(proc)


def print_dry_run_plan(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    plan_rows = []
    for wrapper in rows:
        item = unwrap_dataset_item(wrapper)
        validate_dataset_item(item)
        turns = iter_ingest_turns(
            item,
            args.ingest_turns,
            args.enable_session_summary,
            args.recent_context_turns,
        )
        planned_turns = turns[: args.max_ingest_turns] if args.max_ingest_turns is not None else turns
        plan_rows.append(
            {
                **item_summary(item),
                "available_ingest_turns": len(turns),
                "planned_ingest_turns": len(planned_turns),
                "source_path": wrapper.get("_source_path"),
                "memory_root": str(memory_root_for(args.output_root, item)),
            }
        )
    payload = {
        "examples": len(plan_rows),
        "output_root": str(args.output_root),
        "index_path": str(args.index_path or (args.output_root / "build_index.jsonl")),
        "pipeline": [
            "unwrap raw_dataset_item from each error example",
            "start one isolated Memory API process per question_id",
            "set MEMORY_ROOT to that question's memory directory",
            "ingest LongMemEval haystack sessions through /sessions/ingest",
            "optionally limit ingestion to the first --max-ingest-turns turns",
            "refresh /user-card",
            "write per-question manifest.json and a global build_index.jsonl",
            "do not call /chat for the final question",
        ],
        "items": plan_rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


async def run() -> None:
    args = parse_args()
    if args.session_concurrency < 1:
        raise ValueError("--session-concurrency must be >= 1")
    args.output_root = args.output_root.resolve()
    index_path = (args.index_path or (args.output_root / "build_index.jsonl")).resolve()
    rows = iter_example_wrappers(args)
    if args.resume:
        done = completed_ids(index_path)
        rows = [
            row
            for row in rows
            if str(unwrap_dataset_item(row).get("question_id")) not in done
        ]
    if args.dry_run_plan:
        print_dry_run_plan(rows, args)
        return

    config = provider_config(args)
    for wrapper in rows:
        manifest = await build_one(wrapper, args, config, index_path)
        status = manifest["status"]
        question_id = manifest["question_id"]
        memory_count = manifest.get("ingest_stats", {}).get("memory_count")
        print(f"{status}: {question_id} memory_count={memory_count}")


if __name__ == "__main__":
    asyncio.run(run())
