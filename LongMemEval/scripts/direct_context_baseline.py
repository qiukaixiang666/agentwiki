from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = REPO_ROOT / "LongMemEval" / "longmemeval_s_sampled_100.json"
DEFAULT_OUTPUT = REPO_ROOT / "LongMemEval" / "results" / "direct_context_predictions.jsonl"
DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_MODEL = "qwen3.5-27b"


SYSTEM_PROMPT = """You answer LongMemEval questions from the provided long conversation history.

Rules:
- Use only the provided history and the question date.
- The history may include many irrelevant sessions.
- Track dates, session boundaries, user statements, assistant recommendations, and updates over time.
- For count questions, count only evidence that satisfies the question's time range and entity constraints.
- For knowledge-update questions, prefer the newest relevant evidence before the question date.
- For assistant-reference questions, answer from what the assistant previously recommended, said, or concluded.
- Give the shortest answer that fully answers the question.
- If the answer is a number, return only the number unless units are required.
- If the answer is unknown from the history, answer exactly: I don't know.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-context DeepSeek baseline for LongMemEval sampled data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--provider",
        choices=["deepseek", "aliyun"],
        default="deepseek",
        help="LLM provider for answering. Use aliyun for Qwen via DashScope.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help=(
            "0 means pass all haystack sessions. If positive, keep the newest "
            "sessions that fit this character budget."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip question_ids already in output.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent LLM API calls.",
    )
    parser.add_argument("--model", default=None, help="Override LLM_MODEL for this run.")
    parser.add_argument(
        "--aliyun-base-url",
        default=None,
        help=(
            "Aliyun DashScope OpenAI-compatible base URL. Defaults to "
            "ALIYUN_BASE_URL/DASHSCOPE_BASE_URL or the public compatible endpoint."
        ),
    )
    parser.add_argument(
        "--aliyun-enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode for Aliyun. The default is disabled.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=None, help="Override LLM_TIMEOUT_SECONDS.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and write metadata without calling the LLM.",
    )
    return parser.parse_args()


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
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question_id = row.get("question_id")
            if question_id:
                completed.add(str(question_id))
    return completed


def iter_items(data: list[dict[str, Any]], start: int, limit: int | None) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    items = data[start:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        items = items[:limit]
    return items


def format_message(message: dict[str, Any]) -> str:
    role = str(message.get("role", "unknown")).strip() or "unknown"
    content = str(message.get("content", "")).strip()
    return f"{role}: {content}"


def build_session_blocks(item: dict[str, Any]) -> list[str]:
    session_ids = item.get("haystack_session_ids") or []
    dates = item.get("haystack_dates") or []
    sessions = item.get("haystack_sessions") or []
    blocks: list[str] = []

    for idx, session in enumerate(sessions):
        session_id = session_ids[idx] if idx < len(session_ids) else f"session_{idx + 1}"
        date = dates[idx] if idx < len(dates) else "unknown date"
        lines = [
            f"### Session {idx + 1}",
            f"session_id: {session_id}",
            f"date: {date}",
        ]
        if isinstance(session, list):
            lines.extend(format_message(message) for message in session)
        else:
            lines.append(str(session))
        blocks.append("\n".join(lines))
    return blocks


def apply_context_budget(blocks: list[str], max_context_chars: int) -> tuple[str, bool, int]:
    if max_context_chars <= 0:
        return "\n\n".join(blocks), False, len(blocks)

    selected: list[str] = []
    total = 0
    for block in reversed(blocks):
        extra = len(block) + (2 if selected else 0)
        if selected and total + extra > max_context_chars:
            break
        if not selected and len(block) > max_context_chars:
            selected.append(block[-max_context_chars:])
            total = max_context_chars
            break
        selected.append(block)
        total += extra
    selected.reverse()
    return "\n\n".join(selected), len(selected) < len(blocks), len(selected)


def build_prompt(item: dict[str, Any], max_context_chars: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    blocks = build_session_blocks(item)
    context, truncated, kept_sessions = apply_context_budget(blocks, max_context_chars)
    user_prompt = f"""Question date: {item.get("question_date", "")}
Question type: {item.get("question_type", "")}
Question: {item.get("question", "")}

Conversation history:
{context}

Answer the question now. Return only the final answer."""

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    stats = {
        "context_chars": len(context),
        "prompt_chars": sum(len(message["content"]) for message in prompt),
        "num_sessions_total": len(blocks),
        "num_sessions_in_prompt": kept_sessions,
        "truncated": truncated,
    }
    return prompt, stats


def configure_llm_env(args: argparse.Namespace) -> None:
    load_project_env()
    os.environ["LLM_TEMPERATURE"] = str(args.temperature)
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.timeout is not None:
        os.environ["LLM_TIMEOUT_SECONDS"] = str(args.timeout)


def load_project_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def aliyun_api_key() -> str:
    return (
        os.environ.get("ALIYUN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
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


async def call_llm(prompt: list[dict[str, str]]) -> tuple[str, str]:
    from memory_system.llm_client import LLMClient
    from memory_system.schemas import ChatMessage
    from memory_system.config import settings

    client = LLMClient()
    messages = [ChatMessage(role=message["role"], content=message["content"]) for message in prompt]
    prediction = await client.complete(messages, json_mode=False)
    return prediction.strip(), settings.llm_model


async def call_aliyun_llm(
    prompt: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[str, str]:
    api_key = aliyun_api_key()
    if not api_key:
        raise RuntimeError("ALIYUN_API_KEY or DASHSCOPE_API_KEY is required for --provider aliyun.")

    from openai import AsyncOpenAI

    model = args.model or os.environ.get("ALIYUN_MODEL") or os.environ.get("DASHSCOPE_MODEL") or DEFAULT_ALIYUN_MODEL
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
    return (response.choices[0].message.content or "").strip(), model


async def run() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    configure_llm_env(args)

    data = read_json(args.input)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {args.input}")

    completed_ids = load_completed_ids(args.output) if args.resume else set()
    items = iter_items(data, args.start, args.limit)
    work_items: list[tuple[int, dict[str, Any]]] = []

    for local_idx, item in enumerate(items):
        dataset_index = args.start + local_idx
        question_id = str(item.get("question_id", dataset_index))
        if question_id in completed_ids:
            continue
        work_items.append((dataset_index, item))

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def process_one(dataset_index: int, item: dict[str, Any]) -> None:
        async with semaphore:
            question_id = str(item.get("question_id", dataset_index))
            prompt, stats = build_prompt(item, args.max_context_chars)
            started = time.time()
            prediction = ""
            error = None
            model = args.model or ""

            if args.dry_run:
                prediction = "[DRY_RUN]"
            else:
                try:
                    if args.provider == "aliyun":
                        prediction, model = await call_aliyun_llm(prompt, args)
                    else:
                        prediction, model = await call_llm(prompt)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"

            row = {
                "question_id": question_id,
                "dataset_index": dataset_index,
                "question_type": item.get("question_type"),
                "question_date": item.get("question_date"),
                "question": item.get("question"),
                "gold_answer": item.get("answer"),
                "prediction": prediction,
                "error": error,
                "provider": args.provider,
                "model": model,
                "thinking_enabled": bool(args.aliyun_enable_thinking)
                if args.provider == "aliyun"
                else None,
                "max_context_chars": args.max_context_chars,
                "elapsed_seconds": round(time.time() - started, 3),
                **stats,
            }
            async with write_lock:
                append_jsonl(args.output, row)

            if args.sleep > 0 and not args.dry_run:
                await asyncio.sleep(args.sleep)

    pbar = tqdm(total=len(work_items), desc="Evaluating", unit="item")

    async def _wrapped(dataset_index: int, item: dict[str, Any]) -> None:
        try:
            await process_one(dataset_index, item)
        finally:
            pbar.update(1)

    await asyncio.gather(*(_wrapped(dataset_index, item) for dataset_index, item in work_items))
    pbar.close()


if __name__ == "__main__":
    asyncio.run(run())
