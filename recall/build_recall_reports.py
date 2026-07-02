from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_EXAMPLES_DIR = REPO_ROOT / "result" / "qwen_error_examples_by_type" / "examples"
DEFAULT_MEMORY_BANK_ROOT = REPO_ROOT / "runtime" / "deepseek_v4_flash_six_memory_banks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "recall"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recall memories for each LongMemEval error example question using the "
            "project's current embedding + hybrid MemoryStore retrieval logic."
        )
    )
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR)
    parser.add_argument("--memory-bank-root", type=Path, default=DEFAULT_MEMORY_BANK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18083/v1")
    parser.add_argument("--embedding-provider", default="vllm")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument(
        "--query-mode",
        choices=["raw", "answer_prompt", "both"],
        default="both",
        help=(
            "raw matches /search on the original question; answer_prompt matches the "
            "/chat retrieval query used by LongMemEval evaluation; both writes both."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def unwrap_dataset_item(wrapper: dict[str, Any]) -> dict[str, Any]:
    raw = wrapper.get("raw_dataset_item")
    return raw if isinstance(raw, dict) else wrapper


def memory_root_for(memory_bank_root: Path, item: dict[str, Any]) -> Path:
    return memory_bank_root / f"{slugify(item.get('question_type'))}__{slugify(item.get('question_id'))}"


def build_answer_prompt(item: dict[str, Any]) -> str:
    question = str(item.get("question", ""))
    question_date = str(item.get("question_date", ""))
    return (
        f"[LongMemEval question_date={question_date}]\n"
        "Answer the following LongMemEval question using only your private Wiki memory. "
        "Use dated memories, session ids, user statements, and assistant-provided recommendations as evidence. "
        "If the evidence is insufficient, answer exactly: I don't know. "
        "Return only the final answer, with no explanation.\n\n"
        f"Question: {question}"
    )


def load_store_for(memory_root: Path):
    os.environ["MEMORY_ROOT"] = str(memory_root.resolve())
    import memory_system.config as config_module
    import memory_system.store as store_module

    importlib.reload(config_module)
    store_module = importlib.reload(store_module)
    return store_module, store_module.MemoryStore()


def search_without_reinforcement(
    store_module,
    store,
    query_vector: list[float],
    query_text: str,
    top_k: int,
    threshold: float,
) -> list[dict[str, Any]]:
    with store._lock:
        ids = [
            memory_id
            for memory_id, record in store._memories.items()
            if memory_id in store._vectors and store._is_recallable(record)
        ]
        if not ids:
            return []

        matrix = np.asarray([store._vectors[memory_id] for memory_id in ids], dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
            return []

        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return []

        semantic_scores = matrix @ query / np.maximum(matrix_norms * query_norm, 1e-12)
        bm25_scores = store._bm25_scores(query_text, ids) if query_text else {}
        entity_boosts = store._entity_boosts(query_text, ids) if query_text else {}
        has_bm25 = bool(bm25_scores)
        has_entity = bool(entity_boosts)
        max_possible = (
            1.0
            + (1.0 if has_bm25 else 0.0)
            + (store_module.ENTITY_BOOST_WEIGHT if has_entity else 0.0)
        )

        scored: list[dict[str, Any]] = []
        for idx, memory_id in enumerate(ids):
            record = store._memories[memory_id]
            semantic_score = float(semantic_scores[idx])
            if semantic_score < threshold:
                continue
            bm25_score = float(bm25_scores.get(memory_id, 0.0))
            entity_boost = float(entity_boosts.get(memory_id, 0.0))
            effective_strength = float(store._effective_strength(record))
            raw_hybrid = semantic_score + bm25_score + entity_boost
            hybrid_score = min(raw_hybrid / max_possible, 1.0)
            final_score = min(
                1.0,
                hybrid_score
                + store_module.STRENGTH_SCORE_WEIGHT * effective_strength
                + store_module.CONFIDENCE_SCORE_WEIGHT * record.confidence,
            )
            scored.append(
                {
                    "id": record.id,
                    "content": record.content,
                    "memory_type": record.memory_type,
                    "topic": record.topic,
                    "confidence": record.confidence,
                    "memory_strength": record.memory_strength
                    or store._default_strength(record.memory_type),
                    "effective_strength": effective_strength,
                    "tags": record.tags,
                    "score": float(final_score),
                    "semantic_score": semantic_score,
                    "bm25_score": bm25_score,
                    "entity_boost": entity_boost,
                    "updated_at": record.updated_at,
                    "created_at": record.created_at,
                    "source_session_id": record.source_session_id,
                    "status": record.status,
                }
            )

        scored.sort(key=lambda row: row["score"], reverse=True)
        for rank, row in enumerate(scored[: max(0, top_k)], start=1):
            row["rank"] = rank
        return scored[: max(0, top_k)]


def markdown_for_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['question_id']} ({report['question_type']})",
        "",
        f"- Status: {report.get('manifest_status')}",
        f"- Question date: {report.get('question_date')}",
        f"- Top k: {report.get('top_k')}",
        f"- Gold answer: {report.get('gold_answer')}",
        "",
        "## Question",
        "",
        str(report.get("question", "")),
        "",
    ]
    for query_name, result in report["queries"].items():
        lines.extend(
            [
                f"## {query_name}",
                "",
                "```text",
                result["query"],
                "```",
                "",
            ]
        )
        memories = result.get("memories") or []
        if not memories:
            lines.extend(["No memories recalled.", ""])
            continue
        for memory in memories:
            lines.extend(
                [
                    f"### {memory['rank']}. {memory['id']} score={memory['score']:.4f}",
                    "",
                    f"- Type: {memory['memory_type']}",
                    f"- Topic: {memory['topic']}",
                    f"- Confidence: {memory['confidence']}",
                    f"- Effective strength: {memory['effective_strength']:.4f}",
                    f"- Semantic: {memory['semantic_score']:.4f}",
                    f"- BM25: {memory['bm25_score']:.4f}",
                    f"- Entity boost: {memory['entity_boost']:.4f}",
                    f"- Tags: {', '.join(memory.get('tags') or [])}",
                    f"- Source session: {memory.get('source_session_id')}",
                    "",
                    memory["content"],
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


async def build_reports(args: argparse.Namespace) -> list[dict[str, Any]]:
    from memory_system.embedding_client import EmbeddingClient

    embedder = EmbeddingClient(
        base_url=args.embedding_url,
        provider=args.embedding_provider,
        model=args.embedding_model,
    )

    examples = sorted(args.examples_dir.glob("*.json"))
    reports: list[dict[str, Any]] = []
    query_modes = ["raw_question", "answer_prompt"] if args.query_mode == "both" else [args.query_mode]

    for path in examples:
        wrapper = load_json(path)
        item = unwrap_dataset_item(wrapper)
        question_id = str(item.get("question_id"))
        question_type = str(item.get("question_type"))
        memory_root = memory_root_for(args.memory_bank_root, item)
        if not memory_root.exists():
            raise FileNotFoundError(f"Memory bank not found for {question_id}: {memory_root}")

        manifest_path = memory_root / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = load_json(manifest_path)

        store_module, store = load_store_for(memory_root)

        queries: dict[str, dict[str, Any]] = {}
        for query_name in query_modes:
            if query_name == "raw_question" or query_name == "raw":
                query_text = str(item.get("question", ""))
                output_name = "raw_question"
            elif query_name == "answer_prompt":
                query_text = build_answer_prompt(item)
                output_name = "answer_prompt"
            else:
                raise ValueError(f"Unknown query mode: {query_name}")

            query_vector = await embedder.embed_one(query_text)
            memories = search_without_reinforcement(
                store_module,
                store,
                query_vector=query_vector,
                query_text=query_text,
                top_k=args.top_k,
                threshold=args.threshold,
            )
            queries[output_name] = {
                "query": query_text,
                "memory_count": len(memories),
                "memories": memories,
            }

        report = {
            "question_id": question_id,
            "question_type": question_type,
            "question_date": item.get("question_date"),
            "question": item.get("question"),
            "gold_answer": item.get("answer"),
            "source_path": str(path),
            "memory_root": str(memory_root),
            "manifest_status": manifest.get("status"),
            "manifest_memory_count": (manifest.get("ingest_stats") or {}).get("memory_count"),
            "top_k": args.top_k,
            "threshold": args.threshold,
            "queries": queries,
        }
        reports.append(report)

        type_dir = args.output_dir / slugify(question_type)
        json_path = type_dir / f"{slugify(question_id)}.recall.json"
        md_path = type_dir / f"{slugify(question_id)}.recall.md"
        write_json(json_path, report)
        write_text(md_path, markdown_for_report(report))

    return reports


def write_index(args: argparse.Namespace, reports: list[dict[str, Any]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    lines = ["# Recall Summary", ""]
    for report in reports:
        row = {
            "question_id": report["question_id"],
            "question_type": report["question_type"],
            "question": report["question"],
            "gold_answer": report["gold_answer"],
            "manifest_status": report["manifest_status"],
            "manifest_memory_count": report["manifest_memory_count"],
            "top_k": report["top_k"],
            "query_memory_counts": {
                key: value.get("memory_count", 0)
                for key, value in report.get("queries", {}).items()
            },
            "output_json": str(
                args.output_dir
                / slugify(report["question_type"])
                / f"{slugify(report['question_id'])}.recall.json"
            ),
            "output_markdown": str(
                args.output_dir
                / slugify(report["question_type"])
                / f"{slugify(report['question_id'])}.recall.md"
            ),
        }
        summary_rows.append(row)
        lines.extend(
            [
                f"## {report['question_id']} ({report['question_type']})",
                "",
                f"- Status: {report['manifest_status']}",
                f"- Memory bank size: {report['manifest_memory_count']}",
                f"- Query counts: {row['query_memory_counts']}",
                f"- File: `{row['output_markdown']}`",
                "",
            ]
        )

    write_json(args.output_dir / "summary.json", {"reports": summary_rows})
    append_jsonl(args.output_dir / "summary.jsonl", summary_rows)
    write_text(args.output_dir / "summary.md", "\n".join(lines).rstrip() + "\n")


async def async_main() -> None:
    args = parse_args()
    reports = await build_reports(args)
    write_index(args, reports)
    print(f"Wrote {len(reports)} recall reports to {args.output_dir}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
