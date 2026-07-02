from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RECALL_DIR = REPO_ROOT / "recall"
DEFAULT_OUTPUT_DIR = DEFAULT_RECALL_DIR / "results"
DEFAULT_BASELINE = REPO_ROOT / "result" / "baseline_eval_llm_qwen.json"
DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_MODEL = "qwen3.5-27b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Qwen comparison experiment using previously recalled memories. "
            "The answer prompt is built with the same chat prompt template used by memory_system.api."
        )
    )
    parser.add_argument("--recall-dir", type=Path, default=DEFAULT_RECALL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output filename stem under --output-dir. Defaults to recall_qwen_<query>_top<k>.",
    )
    parser.add_argument(
        "--query-name",
        choices=["answer_prompt", "raw_question"],
        default="answer_prompt",
        help="Which recalled-memory list from each .recall.json file to use.",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--model", default=None)
    parser.add_argument("--aliyun-base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--aliyun-enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode. Default stays disabled to match baseline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and output metadata without calling Qwen.",
    )
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_baseline_items(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    if isinstance(payload, dict):
        rows = payload.get("items") or payload.get("results") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("question_id"):
            by_id[str(row["question_id"])] = row
    return by_id


def iter_recall_reports(recall_dir: Path) -> list[Path]:
    paths = []
    for path in recall_dir.glob("*/*.recall.json"):
        if "results" in path.parts or "__pycache__" in path.parts:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: (p.parent.name, p.name))


def build_answer_prompt(report: dict[str, Any]) -> str:
    question = str(report.get("question", ""))
    question_date = str(report.get("question_date", ""))
    return (
        f"[LongMemEval question_date={question_date}]\n"
        "Answer the following LongMemEval question using only your private Wiki memory. "
        "Use dated memories, session ids, user statements, and assistant-provided recommendations as evidence. "
        "If the evidence is insufficient, answer exactly: I don't know. "
        "Return only the final answer, with no explanation.\n\n"
        f"Question: {question}"
    )


def load_user_card(memory_root: str | None) -> str:
    if not memory_root:
        return ""
    path = Path(memory_root) / "data" / "user_card.json"
    if not path.exists():
        return ""
    try:
        payload = read_json(path)
    except Exception:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("profile_text") or "")
    return ""


def to_retrieved_memory(memory: dict[str, Any]):
    from memory_system.schemas import RetrievedMemory

    return RetrievedMemory(
        id=str(memory.get("id", "")),
        content=str(memory.get("content", "")),
        memory_type=str(memory.get("memory_type") or "fact"),
        topic=str(memory.get("topic") or "general"),
        confidence=float(memory.get("confidence", 0.8)),
        memory_strength=float(memory.get("memory_strength", 0.8)),
        effective_strength=float(
            memory.get("effective_strength", memory.get("memory_strength", 0.8))
        ),
        tags=list(memory.get("tags") or []),
        score=float(memory.get("score", 0.0)),
        updated_at=str(memory.get("updated_at") or ""),
    )


def chat_prompt_for_report(
    report: dict[str, Any],
    query_name: str,
    top_k: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], str]:
    from memory_system.prompts import build_chat_prompt

    query_result = (report.get("queries") or {}).get(query_name)
    if not isinstance(query_result, dict):
        raise KeyError(f"Report {report.get('question_id')} lacks query {query_name!r}")
    selected_memories = list(query_result.get("memories") or [])[: max(0, top_k)]
    recalled = [to_retrieved_memory(memory) for memory in selected_memories]
    user_input = build_answer_prompt(report)
    user_card = load_user_card(report.get("memory_root"))
    prompt = build_chat_prompt(
        recalled_memories=recalled,
        recent_messages=[],
        user_input=user_input,
        user_card=user_card,
    )
    prompt_dicts = [
        message.model_dump() if hasattr(message, "model_dump") else dict(message)
        for message in prompt
    ]
    return prompt_dicts, selected_memories, query_result.get("query") or user_input


def aliyun_api_key() -> str:
    return (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("ALIYUN_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or ""
    )


def aliyun_base_url(args: argparse.Namespace) -> str:
    return (
        args.aliyun_base_url
        or os.environ.get("ALIYUN_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or DEFAULT_ALIYUN_BASE_URL
    ).rstrip("/")


def aliyun_model(args: argparse.Namespace) -> str:
    return (
        args.model
        or os.environ.get("ALIYUN_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
        or DEFAULT_ALIYUN_MODEL
    )


async def call_qwen(
    prompt: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    api_key = aliyun_api_key()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY, ALIYUN_API_KEY, or QWEN_API_KEY is required.")

    from openai import AsyncOpenAI

    model = aliyun_model(args)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=aliyun_base_url(args),
        timeout=args.timeout,
    )
    response = await client.chat.completions.create(
        model=model,
        messages=prompt,
        stream=False,
        temperature=args.temperature,
        top_p=args.top_p,
        extra_body={
            "result_format": "message",
            "enable_thinking": bool(args.aliyun_enable_thinking),
        },
    )
    usage = response.usage.model_dump() if response.usage is not None else {}
    choice = response.choices[0]
    metadata = {
        "model": model,
        "finish_reason": choice.finish_reason,
        "usage": usage,
    }
    return (choice.message.content or "").strip(), metadata


async def run_one(
    path: Path,
    args: argparse.Namespace,
    baseline_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report = read_json(path)
    question_id = str(report.get("question_id"))
    baseline = baseline_by_id.get(question_id, {})
    prompt, selected_memories, retrieval_query = chat_prompt_for_report(
        report,
        query_name=args.query_name,
        top_k=args.top_k,
    )
    started = time.time()
    prediction = ""
    error = None
    llm_metadata: dict[str, Any] = {"model": aliyun_model(args)}

    if args.dry_run:
        prediction = "[DRY_RUN]"
    else:
        try:
            prediction, llm_metadata = await call_qwen(prompt, args)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    return {
        "question_id": question_id,
        "dataset_index": baseline.get("dataset_index"),
        "question_type": report.get("question_type"),
        "question_date": report.get("question_date"),
        "question": report.get("question"),
        "gold_answer": report.get("gold_answer"),
        "prediction": prediction,
        "reply": prediction,
        "error": error,
        "provider": "aliyun",
        "model": llm_metadata.get("model") or aliyun_model(args),
        "thinking_enabled": bool(args.aliyun_enable_thinking),
        "top_k": args.top_k,
        "query_name": args.query_name,
        "retrieval_query": retrieval_query,
        "recalled_memories": selected_memories,
        "blocked": False,
        "memory_update_status": "skipped_offline_recall_compare",
        "edit_operations": [],
        "debug_prompt": prompt,
        "prompt_chars": sum(len(message.get("content", "")) for message in prompt),
        "memory_count": len(selected_memories),
        "memory_root": report.get("memory_root"),
        "source_recall_report": str(path),
        "baseline_qwen": {
            "prediction": baseline.get("prediction"),
            "error": baseline.get("error"),
            "provider": baseline.get("provider"),
            "model": baseline.get("model"),
            "evaluation": baseline.get("evaluation"),
            "prompt_chars": baseline.get("prompt_chars"),
            "context_chars": baseline.get("context_chars"),
        }
        if baseline
        else None,
        "llm_metadata": llm_metadata,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Recall + Qwen Top-6 Comparison",
        "",
        f"- Items: {len(rows)}",
        "- Prompt: memory_system.prompts.build_chat_prompt",
        "- Query source: answer_prompt recall list unless overridden",
        "",
    ]
    for row in rows:
        baseline_pred = ""
        if row.get("baseline_qwen"):
            baseline_pred = str(row["baseline_qwen"].get("prediction") or "")
        lines.extend(
            [
                f"## {row.get('question_id')} ({row.get('question_type')})",
                "",
                f"- Question: {row.get('question')}",
                f"- Gold: {row.get('gold_answer')}",
                f"- Baseline Qwen: {baseline_pred}",
                f"- Recall+Qwen: {row.get('prediction')}",
                f"- Error: {row.get('error')}",
                f"- Memories used: {row.get('memory_count')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


async def async_main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    load_project_env()

    reports = iter_recall_reports(args.recall_dir)
    if not reports:
        raise FileNotFoundError(f"No .recall.json files found under {args.recall_dir}")

    baseline_by_id = load_baseline_items(args.baseline)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(path: Path) -> dict[str, Any]:
        async with semaphore:
            print(f"Running {path}", flush=True)
            return await run_one(path, args, baseline_by_id)

    rows = await asyncio.gather(*(guarded(path) for path in reports))
    rows = sorted(rows, key=lambda row: (str(row.get("question_type")), str(row.get("question_id"))))

    payload = {
        "summary": {
            "count": len(rows),
            "provider": "aliyun",
            "model": aliyun_model(args),
            "thinking_enabled": bool(args.aliyun_enable_thinking),
            "top_k": args.top_k,
            "query_name": args.query_name,
            "prompt_template": "memory_system.prompts.build_chat_prompt",
            "errors": sum(1 for row in rows if row.get("error")),
            "baseline_file": str(args.baseline),
        },
        "items": rows,
    }
    output_stem = args.output_stem or f"recall_qwen_{args.query_name}_top{args.top_k}"
    write_json(args.output_dir / f"{output_stem}.json", payload)
    write_jsonl(args.output_dir / f"{output_stem}.jsonl", rows)
    write_text(args.output_dir / f"{output_stem}.md", summary_markdown(rows))
    print(f"Wrote {len(rows)} results to {args.output_dir / (output_stem + '.json')}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
