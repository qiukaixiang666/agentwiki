from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "LongMemEval" / "longmemeval_s_cleaned.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runtime" / "longmemeval_memory_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convenience wrapper for the two-stage LongMemEval memory pipeline: "
            "build per-question memory banks, then answer with raw-question top-6 retrieval."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--question-type", action="append", default=[])
    parser.add_argument("--build-provider", choices=["deepseek", "qwen"], default="deepseek")
    parser.add_argument("--build-model", default=None)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18083/v1")
    parser.add_argument("--embedding-provider", default="vllm")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--dry-run-answer", action="store_true")
    return parser.parse_args()


def add_common_selection(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(["--start", str(args.start)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    for question_id in args.question_id:
        cmd.extend(["--question-id", question_id])
    for question_type in args.question_type:
        cmd.extend(["--question-type", question_type])


def main() -> None:
    args = parse_args()
    memory_bank_root = args.run_root / "memory_banks"
    output = args.run_root / "raw_top6_qwen_predictions.jsonl"

    build_cmd = [
        sys.executable,
        "-m",
        "longmemeval_memory_pipeline.build_banks",
        "--input",
        str(args.input),
        "--output-root",
        str(memory_bank_root),
        "--provider",
        args.build_provider,
        "--top-k",
        "8",
        "--embedding-url",
        args.embedding_url,
        "--embedding-provider",
        args.embedding_provider,
        "--embedding-model",
        args.embedding_model,
    ]
    if args.build_model:
        build_cmd.extend(["--model", args.build_model])
    if args.overwrite:
        build_cmd.append("--overwrite")
    if args.resume:
        build_cmd.append("--resume")
    if args.dry_run_plan:
        build_cmd.append("--dry-run-plan")
    add_common_selection(build_cmd, args)

    answer_cmd = [
        sys.executable,
        "-m",
        "longmemeval_memory_pipeline.answer_raw_top6",
        "--input",
        str(args.input),
        "--memory-bank-root",
        str(memory_bank_root),
        "--output",
        str(output),
        "--top-k",
        str(args.top_k),
        "--embedding-url",
        args.embedding_url,
        "--embedding-provider",
        args.embedding_provider,
        "--embedding-model",
        args.embedding_model,
    ]
    if args.answer_model:
        answer_cmd.extend(["--model", args.answer_model])
    if args.resume:
        answer_cmd.append("--resume")
    if args.dry_run_answer:
        answer_cmd.append("--dry-run")
    add_common_selection(answer_cmd, args)

    subprocess.run(build_cmd, cwd=REPO_ROOT, check=True)
    if not args.dry_run_plan:
        subprocess.run(answer_cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

