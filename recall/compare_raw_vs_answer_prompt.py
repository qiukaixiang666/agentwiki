from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "recall" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw-question and answer-prompt Qwen recall runs.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "recall_qwen_raw_question_top6_real.json",
    )
    parser.add_argument(
        "--answer-prompt",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "recall_qwen_answer_prompt_top6_real.json",
    )
    parser.add_argument(
        "--output-stem",
        default="compare_raw_question_vs_answer_prompt_top6_real",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


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


def items_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("items") if isinstance(payload, dict) else payload
    return {str(row["question_id"]): row for row in rows if isinstance(row, dict) and row.get("question_id")}


def memory_ids(row: dict[str, Any]) -> list[str]:
    return [str(memory.get("id")) for memory in row.get("recalled_memories") or []]


def build_rows(raw_by_id: dict[str, dict[str, Any]], answer_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_id in sorted(set(raw_by_id) | set(answer_by_id)):
        raw = raw_by_id.get(question_id, {})
        answer = answer_by_id.get(question_id, {})
        raw_ids = memory_ids(raw)
        answer_ids = memory_ids(answer)
        rows.append(
            {
                "question_id": question_id,
                "question_type": raw.get("question_type") or answer.get("question_type"),
                "question": raw.get("question") or answer.get("question"),
                "gold_answer": raw.get("gold_answer") or answer.get("gold_answer"),
                "baseline_qwen_prediction": (
                    (raw.get("baseline_qwen") or {}).get("prediction")
                    or (answer.get("baseline_qwen") or {}).get("prediction")
                ),
                "raw_question_prediction": raw.get("prediction"),
                "answer_prompt_prediction": answer.get("prediction"),
                "raw_question_error": raw.get("error"),
                "answer_prompt_error": answer.get("error"),
                "raw_question_memory_ids": raw_ids,
                "answer_prompt_memory_ids": answer_ids,
                "same_memory_order": raw_ids == answer_ids,
                "same_memory_set": set(raw_ids) == set(answer_ids),
                "shared_memory_ids": [memory_id for memory_id in raw_ids if memory_id in set(answer_ids)],
                "raw_only_memory_ids": [memory_id for memory_id in raw_ids if memory_id not in set(answer_ids)],
                "answer_prompt_only_memory_ids": [
                    memory_id for memory_id in answer_ids if memory_id not in set(raw_ids)
                ],
            }
        )
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Raw Question vs Answer Prompt Top-6",
        "",
        f"- Items: {len(rows)}",
        "- Both runs use the same chat prompt template for answering.",
        "- Difference is only which recall list supplies top-6 memories.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['question_id']} ({row.get('question_type')})",
                "",
                f"- Question: {row.get('question')}",
                f"- Gold: {row.get('gold_answer')}",
                f"- Baseline Qwen: {row.get('baseline_qwen_prediction')}",
                f"- Raw-question recall + Qwen: {row.get('raw_question_prediction')}",
                f"- Answer-prompt recall + Qwen: {row.get('answer_prompt_prediction')}",
                f"- Same memory order: {row.get('same_memory_order')}",
                f"- Same memory set: {row.get('same_memory_set')}",
                f"- Raw only memories: {', '.join(row.get('raw_only_memory_ids') or []) or '(none)'}",
                f"- Answer-prompt only memories: {', '.join(row.get('answer_prompt_only_memory_ids') or []) or '(none)'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    raw_by_id = items_by_id(args.raw)
    answer_by_id = items_by_id(args.answer_prompt)
    rows = build_rows(raw_by_id, answer_by_id)
    payload = {
        "summary": {
            "count": len(rows),
            "raw_file": str(args.raw),
            "answer_prompt_file": str(args.answer_prompt),
            "same_memory_order_count": sum(1 for row in rows if row["same_memory_order"]),
            "same_memory_set_count": sum(1 for row in rows if row["same_memory_set"]),
        },
        "items": rows,
    }
    write_json(args.output_dir / f"{args.output_stem}.json", payload)
    write_jsonl(args.output_dir / f"{args.output_stem}.jsonl", rows)
    write_text(args.output_dir / f"{args.output_stem}.md", markdown(rows))
    print(f"Wrote comparison for {len(rows)} items to {args.output_dir / (args.output_stem + '.json')}")


if __name__ == "__main__":
    main()
