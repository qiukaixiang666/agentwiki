from __future__ import annotations

import argparse
import asyncio
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
from longmemeval_memory_pipeline.common import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_URL,
    append_jsonl,
    ensure_within,
    iter_input_wrappers,
    item_summary,
    memory_root_for,
    read_jsonl,
    selected_wrappers,
    unwrap_dataset_item,
    validate_dataset_item,
    write_json,
)


DEFAULT_INPUT = REPO_ROOT / "LongMemEval" / "longmemeval_s_cleaned.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runtime" / "longmemeval_memory_pipeline" / "memory_banks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: build one isolated memory bank for each LongMemEval question."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
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
        default="deepseek",
        help="LLM used for memory extraction while building banks.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--session-concurrency", type=int, default=1)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=18182)
    parser.add_argument("--embedding-url", default=DEFAULT_EMBEDDING_URL)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--ingest-turns",
        choices=["user_with_context", "user_only", "all"],
        default="user_with_context",
    )
    parser.add_argument("--enable-session-summary", action="store_true")
    parser.add_argument("--recent-context-turns", type=int, default=8)
    parser.add_argument("--max-ingest-turns", type=int, default=None)
    parser.add_argument("--refresh-user-card-every", type=int, default=25)
    parser.add_argument("--user-card-limit", type=int, default=30)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--print-api-io", action="store_true")
    return parser.parse_args()


def completed_ids(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(index_path):
        if row.get("status") == "built" and row.get("question_id"):
            ids.add(str(row["question_id"]))
    return ids


def prepare_memory_root(output_root: Path, memory_root: Path, overwrite: bool) -> tuple[bool, str | None]:
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_within(output_root, memory_root)
    if memory_root.exists():
        if not overwrite:
            return False, "memory root already exists; pass --overwrite to rebuild"
        shutil.rmtree(memory_root)
    (memory_root / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (memory_root / "runtime").mkdir(parents=True, exist_ok=True)
    return True, None


def manifest_for(
    wrapper: dict[str, Any],
    memory_root: Path,
    args: argparse.Namespace,
    ingest_stats: dict[str, Any],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    item = unwrap_dataset_item(wrapper)
    summary = item_summary(wrapper)
    return {
        "status": status,
        "error": error,
        "memory_root": str(memory_root),
        **summary,
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
        "original_prediction": wrapper.get("prediction"),
        "original_error": wrapper.get("error"),
        "question_id": str(item.get("question_id")),
    }


async def ingest_item(
    client: httpx.AsyncClient,
    base_url: str,
    item: dict[str, Any],
    args: argparse.Namespace,
    api_io_log: Path,
) -> dict[str, Any]:
    turns = iter_ingest_turns(
        item,
        args.ingest_turns,
        args.enable_session_summary,
        args.recent_context_turns,
    )
    available_turns = len(turns)
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
        payload = {
            "session_id": f"lme_{item.get('question_id')}_{session_id}",
            "messages": turn["messages"],
            "top_k": args.top_k,
            "debug_prompt": False,
        }
        row = {
            "turn_index": idx + 1,
            "kind": turn.get("kind"),
            "source_session_id": session_id,
            "date": turn.get("date"),
            "endpoint": "/sessions/ingest",
            "request": payload,
            "response": None,
            "error": None,
        }
        try:
            async with semaphore:
                result = await post_json(client, f"{base_url}/sessions/ingest", payload)
            row["response"] = result
            applied += int(result.get("applied") or 0)
            skipped += int(result.get("skipped") or 0)
            edit_operations += len(result.get("edit_operations") or [])
            if args.refresh_user_card_every and (idx + 1) % args.refresh_user_card_every == 0:
                await post_json(client, f"{base_url}/user-card/refresh?limit={args.user_card_limit}", {})
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            row["error"] = error
            errors.append(error)
        append_jsonl(api_io_log, row)
        if args.print_api_io:
            print(
                f"ingest turn={idx + 1} session={session_id} "
                f"applied={applied} skipped={skipped} error={row['error']}",
                flush=True,
            )

    await asyncio.gather(*(ingest_one(idx, turn) for idx, turn in enumerate(turns)))
    card = await post_json(client, f"{base_url}/user-card/refresh?limit={args.user_card_limit}", {})
    memories = await client.get(f"{base_url}/memories", params={"limit": 1_000_000})
    memories.raise_for_status()
    memory_rows = memories.json().get("memories", [])
    return {
        "available_ingest_turns": available_turns,
        "ingested_turns": len(turns),
        "ingest_mode": args.ingest_turns,
        "session_summary_enabled": args.enable_session_summary,
        "memory_updates_applied": applied,
        "memory_updates_skipped": skipped,
        "edit_operations": edit_operations,
        "ingest_errors": errors[:20],
        "ingest_error_count": len(errors),
        "memory_count": len(memory_rows),
        "user_card": card,
        "api_io_log": str(api_io_log),
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
        manifest = manifest_for(wrapper, memory_root, args, {}, "skipped", skip_reason)
        append_jsonl(index_path, manifest)
        return manifest

    base_url = f"http://{args.api_host}:{args.api_port}"
    proc = start_memory_api(args, config, memory_root)
    started = time.time()
    try:
        await wait_for_api(base_url, proc, args.startup_timeout)
        async with httpx.AsyncClient(timeout=args.request_timeout) as client:
            ingest_stats = await ingest_item(
                client,
                base_url,
                item,
                args,
                api_io_log=memory_root / "runtime" / "api_io.jsonl",
            )
        ingest_error_count = int(ingest_stats.get("ingest_error_count") or 0)
        status = "failed" if ingest_error_count else "built"
        error = f"{ingest_error_count} ingest request(s) failed" if ingest_error_count else None
        manifest = manifest_for(wrapper, memory_root, args, ingest_stats, status, error)
    except Exception as exc:  # noqa: BLE001
        manifest = manifest_for(wrapper, memory_root, args, {}, "failed", f"{type(exc).__name__}: {exc}")
    finally:
        stop_process(proc)

    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    write_json(memory_root / "manifest.json", manifest)
    append_jsonl(index_path, manifest)
    return manifest


def dry_run_plan(wrappers: list[dict[str, Any]], args: argparse.Namespace) -> None:
    items = []
    for wrapper in wrappers:
        item = unwrap_dataset_item(wrapper)
        validate_dataset_item(item)
        turns = iter_ingest_turns(
            item,
            args.ingest_turns,
            args.enable_session_summary,
            args.recent_context_turns,
        )
        planned_turns = turns[: args.max_ingest_turns] if args.max_ingest_turns is not None else turns
        items.append(
            {
                **item_summary(wrapper),
                "available_ingest_turns": len(turns),
                "planned_ingest_turns": len(planned_turns),
                "memory_root": str(memory_root_for(args.output_root, item)),
            }
        )
    print(
        json_dumps(
            {
                "input": str(args.input),
                "output_root": str(args.output_root),
                "index_path": str(args.index_path),
                "items": items,
            }
        )
    )


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


async def run() -> None:
    args = parse_args()
    if args.session_concurrency < 1:
        raise ValueError("--session-concurrency must be >= 1")
    args.output_root = args.output_root.resolve()
    args.index_path = (args.index_path or (args.output_root / "build_index.jsonl")).resolve()
    wrappers = selected_wrappers(
        iter_input_wrappers(args.input),
        start=args.start,
        limit=args.limit,
        question_ids=args.question_id,
        question_types=args.question_type,
    )
    if args.resume:
        done = completed_ids(args.index_path)
        wrappers = [
            wrapper
            for wrapper in wrappers
            if str(unwrap_dataset_item(wrapper).get("question_id")) not in done
        ]
    if args.dry_run_plan:
        dry_run_plan(wrappers, args)
        return

    config = provider_config(args)
    for wrapper in wrappers:
        manifest = await build_one(wrapper, args, config, args.index_path)
        print(
            f"{manifest['status']}: {manifest['question_id']} "
            f"memory_count={manifest.get('ingest_stats', {}).get('memory_count')}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(run())

