from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PREDICTIONS = REPO_ROOT / "LongMemEval" / "results" / "direct_context_predictions.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "LongMemEval" / "results" / "direct_context_eval.json"


JUDGE_SYSTEM_PROMPT = """You are an evaluator for LongMemEval answers.

Decide whether the predicted answer is semantically equivalent to the reference answer
for the question. Be tolerant of harmless wording differences, but do not give credit
when the prediction changes the meaning, misses required items, or gives an incorrect
number/date/name.
For count, date, name, list, and yes/no questions, require the same factual content.
For free-form answers, allow concise paraphrases that preserve all required entities and constraints.
If the prediction says "I don't know", it is correct only when the reference also indicates unknown or unanswerable.

Return a JSON object with:
- correct: boolean
- score: number from 0 to 1
- reason: short string
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LongMemEval baseline predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--judge",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="heuristic is offline; llm calls the configured DeepSeek-compatible API.",
    )
    parser.add_argument(
        "--llm-threshold",
        type=float,
        default=0.5,
        help="Minimum LLM judge score counted as correct.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-item LLM judgements.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent LLM judge API calls.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=None, help="Override LLM_MODEL for judge calls.")
    parser.add_argument("--timeout", type=float, default=None, help="Override LLM_TIMEOUT_SECONDS.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_answer(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: Any) -> list[str]:
    normalized = normalize_answer(value)
    return normalized.split() if normalized else []


def exact_match(prediction: Any, gold: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_score(prediction: Any, gold: Any) -> float:
    pred = normalize_answer(prediction)
    ref = normalize_answer(gold)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    return float(ref in pred or pred in ref)


def extract_numbers(value: Any) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", str(value if value is not None else ""))


def number_match(prediction: Any, gold: Any) -> float | None:
    gold_numbers = extract_numbers(gold)
    if not gold_numbers:
        return None
    pred_numbers = extract_numbers(prediction)
    return float(pred_numbers == gold_numbers)


def heuristic_judge(row: dict[str, Any]) -> dict[str, Any]:
    prediction = row.get("prediction", "")
    gold = row.get("gold_answer")
    em = exact_match(prediction, gold)
    f1 = token_f1(prediction, gold)
    contains = contains_score(prediction, gold)
    num = number_match(prediction, gold)

    correct = bool(em or contains or f1 >= 0.8)
    if num is not None:
        correct = bool(num and (correct or f1 >= 0.5))

    return {
        "correct": correct,
        "score": max(em, contains, f1),
        "exact_match": em,
        "token_f1": f1,
        "contains": contains,
        "number_match": num,
        "reason": "heuristic",
    }


def configure_llm_env(args: argparse.Namespace) -> None:
    import os

    os.environ["LLM_TEMPERATURE"] = str(args.temperature)
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.timeout is not None:
        os.environ["LLM_TIMEOUT_SECONDS"] = str(args.timeout)


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def llm_judge(row: dict[str, Any]) -> dict[str, Any]:
    from memory_system.llm_client import LLMClient
    from memory_system.schemas import ChatMessage

    user_prompt = f"""Question: {row.get("question", "")}

Reference answer:
{row.get("gold_answer")}

Predicted answer:
{row.get("prediction", "")}

Judge whether the predicted answer is correct."""

    client = LLMClient()
    raw = await client.complete(
        [
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ],
        json_mode=True,
    )
    parsed = parse_json_object(raw)
    score = float(parsed.get("score", 0.0))
    return {
        "correct": bool(parsed.get("correct", False)),
        "score": max(0.0, min(1.0, score)),
        "reason": str(parsed.get("reason", "")),
    }


def summarize(rows: list[dict[str, Any]], judge_name: str) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("evaluation")]
    total = len(evaluated)
    correct = sum(1 for row in evaluated if row["evaluation"].get("correct"))
    errored = sum(1 for row in rows if row.get("error"))

    by_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        grouped[str(row.get("question_type", "unknown"))].append(row)

    for question_type, group in sorted(grouped.items()):
        group_correct = sum(1 for row in group if row["evaluation"].get("correct"))
        by_type[question_type] = {
            "count": len(group),
            "accuracy": group_correct / len(group) if group else math.nan,
            "avg_score": sum(float(row["evaluation"].get("score", 0.0)) for row in group) / len(group),
        }

    return {
        "judge": judge_name,
        "count": total,
        "accuracy": correct / total if total else math.nan,
        "avg_score": (
            sum(float(row["evaluation"].get("score", 0.0)) for row in evaluated) / total
            if total
            else math.nan
        ),
        "prediction_errors": errored,
        "by_question_type": by_type,
    }


def load_existing_judgements(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    judgements: dict[str, dict[str, Any]] = {}
    for row in data.get("items", []):
        question_id = row.get("question_id")
        evaluation = row.get("evaluation")
        if question_id and isinstance(evaluation, dict):
            judgements[str(question_id)] = evaluation
    return judgements


async def run() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    rows = read_jsonl(args.predictions)
    if args.limit is not None:
        rows = rows[: args.limit]

    existing = load_existing_judgements(args.output) if args.resume else {}
    if args.judge == "llm":
        configure_llm_env(args)

    evaluated_rows: list[dict[str, Any] | None] = []
    llm_work: list[tuple[int, dict[str, Any]]] = []
    for row_index, row in enumerate(rows):
        row = dict(row)
        question_id = str(row.get("question_id", ""))
        if question_id in existing:
            row["evaluation"] = existing[question_id]
        elif row.get("error"):
            row["evaluation"] = {
                "correct": False,
                "score": 0.0,
                "reason": f"prediction_error: {row.get('error')}",
            }
        elif args.judge == "heuristic":
            row["evaluation"] = heuristic_judge(row)
        else:
            llm_work.append((row_index, row))
        evaluated_rows.append(row)

    if llm_work:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def judge_one(row_index: int, row: dict[str, Any]) -> None:
            async with semaphore:
                try:
                    judgement = await llm_judge(row)
                    judgement["correct"] = bool(
                        judgement["score"] >= args.llm_threshold and judgement["correct"]
                    )
                    row["evaluation"] = judgement
                except Exception as exc:
                    row["evaluation"] = {
                        "correct": False,
                        "score": 0.0,
                        "reason": f"judge_error: {type(exc).__name__}: {exc}",
                    }
                evaluated_rows[row_index] = row

        pbar = tqdm(total=len(llm_work), desc="LLM Judging", unit="item")

        async def _wrapped_judge(row_index: int, row: dict[str, Any]) -> None:
            try:
                await judge_one(row_index, row)
            finally:
                pbar.update(1)

        await asyncio.gather(*(_wrapped_judge(row_index, row) for row_index, row in llm_work))
        pbar.close()

    final_rows = [row for row in evaluated_rows if row is not None]

    result = {
        "summary": summarize(final_rows, args.judge),
        "items": final_rows,
    }
    write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
