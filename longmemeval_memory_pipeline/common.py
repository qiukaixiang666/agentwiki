from __future__ import annotations

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

DEFAULT_EMBEDDING_URL = "http://127.0.0.1:18083/v1"
DEFAULT_EMBEDDING_PROVIDER = "vllm"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.5-27b"


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def iter_input_wrappers(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_dir():
        rows: list[dict[str, Any]] = []
        for child in sorted(input_path.iterdir()):
            if child.suffix.lower() not in {".json", ".jsonl"}:
                continue
            rows.extend(iter_input_wrappers(child))
        return rows

    if input_path.suffix.lower() == ".jsonl":
        rows = read_jsonl(input_path)
    elif input_path.suffix.lower() == ".json":
        payload = read_json(input_path)
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            raise TypeError(f"Expected object/list JSON in {input_path}")
    else:
        raise ValueError(f"Unsupported input path suffix: {input_path}")

    wrapped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("dataset_index", index)
        item["_source_path"] = str(input_path)
        wrapped.append(item)
    return wrapped


def unwrap_dataset_item(wrapper: dict[str, Any]) -> dict[str, Any]:
    raw = wrapper.get("raw_dataset_item")
    return raw if isinstance(raw, dict) else wrapper


def selected_wrappers(
    wrappers: list[dict[str, Any]],
    start: int,
    limit: int | None,
    question_ids: list[str] | None = None,
    question_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    question_id_set = set(question_ids or [])
    question_type_set = set(question_types or [])
    filtered = []
    for wrapper in wrappers:
        item = unwrap_dataset_item(wrapper)
        if question_id_set and str(item.get("question_id")) not in question_id_set:
            continue
        if question_type_set and str(item.get("question_type")) not in question_type_set:
            continue
        filtered.append(wrapper)
    filtered = filtered[start:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        filtered = filtered[:limit]
    return filtered


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


def slugify(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def memory_root_for(output_root: Path, item: dict[str, Any]) -> Path:
    return output_root / f"{slugify(item.get('question_type'))}__{slugify(item.get('question_id'))}"


def ensure_within(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    if parent_resolved == child_resolved or parent_resolved in child_resolved.parents:
        return
    raise ValueError(f"Refusing to operate outside {parent_resolved}: {child_resolved}")


def item_summary(wrapper: dict[str, Any]) -> dict[str, Any]:
    item = unwrap_dataset_item(wrapper)
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
        "dataset_index": wrapper.get("dataset_index"),
        "source_path": wrapper.get("_source_path"),
        "haystack_sessions": len(sessions),
        "haystack_messages": message_count,
    }


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
    threshold: float = 0.1,
    temporal_query_text: str = "",
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
        temporal_text = temporal_query_text or query_text
        temporal_boosts = store._temporal_boosts(temporal_text, ids) if temporal_text else {}
        max_possible = (
            1.0
            + (1.0 if bm25_scores else 0.0)
            + (store_module.ENTITY_BOOST_WEIGHT if entity_boosts else 0.0)
            + (store_module.TEMPORAL_BOOST_WEIGHT if temporal_boosts else 0.0)
        )

        scored: list[dict[str, Any]] = []
        for idx, memory_id in enumerate(ids):
            record = store._memories[memory_id]
            semantic_score = float(semantic_scores[idx])
            if semantic_score < threshold:
                continue
            bm25_score = float(bm25_scores.get(memory_id, 0.0))
            entity_boost = float(entity_boosts.get(memory_id, 0.0))
            temporal_boost = float(temporal_boosts.get(memory_id, 0.0))
            effective_strength = float(store._effective_strength(record))
            raw_hybrid = semantic_score + bm25_score + entity_boost + temporal_boost
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
                    "temporal_boost": temporal_boost,
                    "updated_at": record.updated_at,
                    "created_at": record.created_at,
                    "observed_at": record.observed_at,
                    "source_session_id": record.source_session_id,
                    "status": record.status,
                }
            )

        scored.sort(key=lambda row: row["score"], reverse=True)
        for rank, row in enumerate(scored[: max(0, top_k)], start=1):
            row["rank"] = rank
        return scored[: max(0, top_k)]


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
        observed_at=str(memory.get("observed_at") or "") or None,
    )


def load_user_card(memory_root: Path) -> str:
    path = memory_root / "data" / "user_card.json"
    if not path.exists():
        return ""
    try:
        payload = read_json(path)
    except Exception:
        return ""
    return str(payload.get("profile_text") or "") if isinstance(payload, dict) else ""


def build_chat_prompt_dicts(
    item: dict[str, Any],
    memory_root: Path,
    memories: list[dict[str, Any]],
) -> list[dict[str, str]]:
    from memory_system.prompts import build_chat_prompt

    prompt = build_chat_prompt(
        recalled_memories=[to_retrieved_memory(memory) for memory in memories],
        recent_messages=[],
        user_input=build_answer_prompt(item),
        user_card=load_user_card(memory_root),
    )
    return [
        message.model_dump() if hasattr(message, "model_dump") else dict(message)
        for message in prompt
    ]


def aliyun_api_key() -> str:
    return (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("ALIYUN_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or ""
    )


def aliyun_base_url(value: str | None = None) -> str:
    return (
        value
        or os.environ.get("ALIYUN_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or DEFAULT_ALIYUN_BASE_URL
    ).rstrip("/")


def qwen_model(value: str | None = None) -> str:
    return (
        value
        or os.environ.get("ALIYUN_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
        or DEFAULT_QWEN_MODEL
    )


async def call_qwen(
    prompt: list[dict[str, str]],
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    top_p: float = 0.8,
    timeout: float = 180.0,
    enable_thinking: bool = False,
) -> tuple[str, dict[str, Any]]:
    api_key = aliyun_api_key()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY, ALIYUN_API_KEY, or QWEN_API_KEY is required.")

    from openai import AsyncOpenAI

    resolved_model = qwen_model(model)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=aliyun_base_url(base_url),
        timeout=timeout,
    )
    response = await client.chat.completions.create(
        model=resolved_model,
        messages=prompt,
        stream=False,
        temperature=temperature,
        top_p=top_p,
        extra_body={
            "result_format": "message",
            "enable_thinking": bool(enable_thinking),
        },
    )
    choice = response.choices[0]
    usage = response.usage.model_dump() if response.usage is not None else {}
    return (
        (choice.message.content or "").strip(),
        {
            "model": resolved_model,
            "finish_reason": choice.finish_reason,
            "usage": usage,
        },
    )
