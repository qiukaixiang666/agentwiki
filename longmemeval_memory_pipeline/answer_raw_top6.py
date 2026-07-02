from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any


from longmemeval_memory_pipeline.common import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_URL,
    build_chat_prompt_dicts,
    call_qwen,
    iter_input_wrappers,
    load_project_env,
    load_store_for,
    memory_root_for,
    qwen_model,
    search_without_reinforcement,
    selected_wrappers,
    unwrap_dataset_item,
    validate_dataset_item,
    write_json,
    write_jsonl,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "LongMemEval" / "longmemeval_s_cleaned.json"
DEFAULT_MEMORY_BANK_ROOT = REPO_ROOT / "runtime" / "longmemeval_memory_pipeline" / "memory_banks"
DEFAULT_OUTPUT = REPO_ROOT / "runtime" / "longmemeval_memory_pipeline" / "raw_top6_qwen_predictions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: retrieve memories with the raw LongMemEval question, "
            "take top-6 by default, and answer with Qwen using the normal chat prompt template."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--memory-bank-root", type=Path, default=DEFAULT_MEMORY_BANK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--question-type", action="append", default=[])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--embedding-url", default=DEFAULT_EMBEDDING_URL)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--aliyun-base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--aliyun-enable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def completed_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    ids: set[str] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                import json

                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("question_id") and not row.get("error"):
                ids.add(str(row["question_id"]))
    return ids


async def recall_raw_question(item: dict[str, Any], memory_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    from memory_system.embedding_client import EmbeddingClient

    store_module, store = load_store_for(memory_root)
    embedder = EmbeddingClient(
        base_url=args.embedding_url,
        provider=args.embedding_provider,
        model=args.embedding_model,
    )
    query = str(item.get("question", ""))
    vector = await embedder.embed_one(query)
    return search_without_reinforcement(
        store_module,
        store,
        query_vector=vector,
        query_text=query,
        temporal_query_text=f"question_date={item.get('question_date', '')} {query}",
        top_k=args.top_k,
        threshold=args.threshold,
    )


async def answer_one(wrapper: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    item = unwrap_dataset_item(wrapper)
    validate_dataset_item(item)
    question_id = str(item.get("question_id"))
    memory_root = memory_root_for(args.memory_bank_root, item)
    started = time.time()
    prediction = ""
    error = None
    llm_metadata: dict[str, Any] = {"model": qwen_model(args.model)}
    memories: list[dict[str, Any]] = []
    prompt: list[dict[str, str]] = []

    try:
        if not memory_root.exists():
            raise FileNotFoundError(f"Memory bank not found: {memory_root}")
        memories = await recall_raw_question(item, memory_root, args)
        prompt = build_chat_prompt_dicts(item, memory_root, memories)
        if args.dry_run:
            prediction = "[DRY_RUN]"
        else:
            prediction, llm_metadata = await call_qwen(
                prompt,
                model=args.model,
                base_url=args.aliyun_base_url,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                enable_thinking=args.aliyun_enable_thinking,
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    return {
        "question_id": question_id,
        "dataset_index": wrapper.get("dataset_index"),
        "question_type": item.get("question_type"),
        "question_date": item.get("question_date"),
        "question": item.get("question"),
        "gold_answer": item.get("answer"),
        "prediction": prediction,
        "error": error,
        "provider": "qwen",
        "model": llm_metadata.get("model") or qwen_model(args.model),
        "thinking_enabled": bool(args.aliyun_enable_thinking),
        "top_k": args.top_k,
        "retrieval_query": item.get("question"),
        "retrieval_mode": "raw_question",
        "recalled_memories": memories,
        "memory_root": str(memory_root),
        "debug_prompt": prompt,
        "prompt_chars": sum(len(message.get("content", "")) for message in prompt),
        "llm_metadata": llm_metadata,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LongMemEval Raw Top-6 Qwen Results",
        "",
        f"- Items: {len(rows)}",
        "- Retrieval mode: raw_question",
        "- Answer prompt: memory_system.prompts.build_chat_prompt",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('question_id')} ({row.get('question_type')})",
                "",
                f"- Question: {row.get('question')}",
                f"- Gold: {row.get('gold_answer')}",
                f"- Prediction: {row.get('prediction')}",
                f"- Error: {row.get('error')}",
                f"- Memories used: {len(row.get('recalled_memories') or [])}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


async def run() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    load_project_env()
    args.memory_bank_root = args.memory_bank_root.resolve()
    wrappers = selected_wrappers(
        iter_input_wrappers(args.input),
        start=args.start,
        limit=args.limit,
        question_ids=args.question_id,
        question_types=args.question_type,
    )
    if args.resume:
        done = completed_ids(args.output)
        wrappers = [
            wrapper
            for wrapper in wrappers
            if str(unwrap_dataset_item(wrapper).get("question_id")) not in done
        ]

    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(wrapper: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            row = await answer_one(wrapper, args)
            print(
                f"answered: {row['question_id']} error={row['error']} prediction={row['prediction'][:80]!r}",
                flush=True,
            )
            return row

    rows = await asyncio.gather(*(guarded(wrapper) for wrapper in wrappers))
    rows = sorted(rows, key=lambda row: (str(row.get("dataset_index")), str(row.get("question_id"))))
    write_jsonl(args.output, rows)
    companion_json = args.output.with_suffix(".json")
    companion_md = args.output.with_suffix(".md")
    write_json(
        companion_json,
        {
            "summary": {
                "count": len(rows),
                "errors": sum(1 for row in rows if row.get("error")),
                "top_k": args.top_k,
                "retrieval_mode": "raw_question",
                "model": qwen_model(args.model),
                "memory_bank_root": str(args.memory_bank_root),
            },
            "items": rows,
        },
    )
    write_text(companion_md, markdown_summary(rows))
    print(f"Wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
