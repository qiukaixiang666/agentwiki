from __future__ import annotations

import json
import math
import re
import threading
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import ensure_directories, settings
from .schemas import ChatMessage, EditOperation, MemoryRecord, RetrievedMemory
from .time_utils import normalize_observed_at, parse_observed_date
from .utils import append_jsonl, atomic_write_text, load_jsonl, model_to_dict, utc_now

ACTIVE_STATUS = "active"
DUPLICATE_SIMILARITY = 0.92
TOPIC_CONFLICT_SIMILARITY = 0.55
MIN_RECALL_STRENGTH = 0.15
ENTITY_BOOST_WEIGHT = 0.5
TEMPORAL_BOOST_WEIGHT = 0.18
STRENGTH_SCORE_WEIGHT = 0.15
CONFIDENCE_SCORE_WEIGHT = 0.05

TYPE_MEMORY_DEFAULTS = {
    "preference": {"strength": 0.85, "decay_rate": 0.002, "reinforce_delta": 0.03},
    "fact": {"strength": 0.80, "decay_rate": 0.003, "reinforce_delta": 0.03},
    "relation": {"strength": 0.85, "decay_rate": 0.002, "reinforce_delta": 0.03},
    "task": {"strength": 0.75, "decay_rate": 0.020, "reinforce_delta": 0.05},
    "event": {"strength": 0.65, "decay_rate": 0.010, "reinforce_delta": 0.02},
}

_SPACY_FULL = None
_SPACY_LEMMA = None
_SPACY_FULL_FAILED = False
_SPACY_LEMMA_FAILED = False


class MemoryStore:
    def __init__(self) -> None:
        ensure_directories()
        self._lock = threading.RLock()
        self.wiki_path = settings.data_dir / "wiki.jsonl"
        self.vector_ids_path = settings.data_dir / "vector_ids.json"
        self.vectors_path = settings.data_dir / "vectors.npy"
        self.audit_path = settings.data_dir / "audit.jsonl"
        self.privacy_events_path = settings.data_dir / "privacy_events.jsonl"
        self.sessions_dir = settings.data_dir / "sessions"
        self._memories: dict[str, MemoryRecord] = {}
        self._vectors: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            self._memories.clear()
            for item in load_jsonl(self.wiki_path):
                record = MemoryRecord(**item)
                self._ensure_strength_fields(record)
                if self._is_expired(record):
                    record.status = "expired"
                self._memories[record.id] = record

            self._vectors.clear()
            if self.vector_ids_path.exists() and self.vectors_path.exists():
                ids = json.loads(self.vector_ids_path.read_text(encoding="utf-8"))
                matrix = np.load(self.vectors_path)
                for memory_id, vector in zip(ids, matrix):
                    if memory_id in self._memories:
                        self._vectors[memory_id] = vector.astype(np.float32).tolist()

    def _persist(self) -> None:
        records = sorted(self._memories.values(), key=lambda item: item.created_at)
        wiki_text = "".join(
            json.dumps(model_to_dict(record), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        atomic_write_text(self.wiki_path, wiki_text)

        ids = [record.id for record in records if record.id in self._vectors]
        atomic_write_text(self.vector_ids_path, json.dumps(ids, ensure_ascii=False, indent=2))

        if ids:
            matrix = np.asarray([self._vectors[memory_id] for memory_id in ids], dtype=np.float32)
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)
        tmp = self.vectors_path.with_suffix(".npy.tmp")
        with tmp.open("wb") as handle:
            np.save(handle, matrix)
        tmp.replace(self.vectors_path)

    def list_memories(self, limit: int = 100) -> list[MemoryRecord]:
        with self._lock:
            records = sorted(
                (record for record in self._memories.values() if self._is_recallable(record)),
                key=lambda item: (self._effective_strength(item), item.updated_at),
                reverse=True,
            )
            return records[:limit]

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            return self._memories.get(memory_id)

    def search_by_embedding(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_text: str = "",
        threshold: float = 0.1,
        reinforce: bool = True,
        temporal_query_text: str = "",
    ) -> list[RetrievedMemory]:
        with self._lock:
            if not self._memories or not self._vectors:
                return []
            ids = [
                memory_id
                for memory_id, record in self._memories.items()
                if memory_id in self._vectors and self._is_recallable(record)
            ]
            if not ids:
                return []
            matrix = np.asarray([self._vectors[memory_id] for memory_id in ids], dtype=np.float32)
            query = np.asarray(query_vector, dtype=np.float32)
            if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
                return []
            matrix_norms = np.linalg.norm(matrix, axis=1)
            query_norm = float(np.linalg.norm(query))
            if query_norm == 0:
                return []
            semantic_scores = matrix @ query / np.maximum(matrix_norms * query_norm, 1e-12)
            bm25_scores = self._bm25_scores(query_text, ids) if query_text else {}
            entity_boosts = self._entity_boosts(query_text, ids) if query_text else {}
            temporal_text = " ".join(part for part in [query_text, temporal_query_text] if part)
            temporal_boosts = self._temporal_boosts(temporal_text, ids) if temporal_text else {}
            has_bm25 = bool(bm25_scores)
            has_entity = bool(entity_boosts)
            has_temporal = bool(temporal_boosts)
            max_possible = (
                1.0
                + (1.0 if has_bm25 else 0.0)
                + (ENTITY_BOOST_WEIGHT if has_entity else 0.0)
                + (TEMPORAL_BOOST_WEIGHT if has_temporal else 0.0)
            )
            final_scores = []
            for idx, memory_id in enumerate(ids):
                record = self._memories[memory_id]
                semantic_score = float(semantic_scores[idx])
                if semantic_score < threshold:
                    final_scores.append(-1.0)
                    continue
                effective_strength = self._effective_strength(record)
                raw_hybrid = (
                    semantic_score
                    + bm25_scores.get(memory_id, 0.0)
                    + entity_boosts.get(memory_id, 0.0)
                    + temporal_boosts.get(memory_id, 0.0)
                )
                hybrid_score = min(raw_hybrid / max_possible, 1.0)
                final_score = min(
                    1.0,
                    hybrid_score
                    + STRENGTH_SCORE_WEIGHT * effective_strength
                    + CONFIDENCE_SCORE_WEIGHT * record.confidence,
                )
                final_scores.append(final_score)
            order = np.argsort(-np.asarray(final_scores, dtype=np.float32))[: max(0, top_k)]
            results: list[RetrievedMemory] = []
            changed = False
            now = utc_now()
            for idx in order:
                memory_id = ids[int(idx)]
                if final_scores[int(idx)] < 0:
                    continue
                record = self._memories[memory_id]
                effective_strength = self._effective_strength(record)
                results.append(
                    RetrievedMemory(
                        id=record.id,
                        content=record.content,
                        memory_type=record.memory_type,
                        topic=record.topic,
                        confidence=record.confidence,
                        memory_strength=record.memory_strength or self._default_strength(record.memory_type),
                        effective_strength=effective_strength,
                        tags=record.tags,
                        score=float(final_scores[int(idx)]),
                        updated_at=record.updated_at,
                        observed_at=record.observed_at,
                    )
                )
                if reinforce:
                    self._reinforce(record, now)
                    changed = True
            if reinforce and changed:
                self._persist()
            return results

    def apply_operations(
        self,
        operations: list[EditOperation],
        embeddings_by_content: dict[str, list[float]],
        actor: str,
        session_id: str | None = None,
    ) -> tuple[int, int, list[MemoryRecord]]:
        applied = 0
        skipped = 0
        changed: list[MemoryRecord] = []
        now = utc_now()
        with self._lock:
            for operation in operations:
                if operation.op == "add":
                    if not operation.content:
                        skipped += 1
                        continue
                    operation.topic = self._normalize_topic(operation.topic)
                    operation.confidence = self._clamp_confidence(operation.confidence)
                    if operation.confidence < 0.6:
                        skipped += 1
                        continue
                    duplicate = self._find_duplicate(operation, embeddings_by_content)
                    if duplicate is not None:
                        skipped += 1
                        continue
                    superseded = []
                    if "mem0_additive" not in operation.tags:
                        superseded = self._supersede_topic_conflicts(
                            operation,
                            now,
                            embeddings_by_content,
                        )
                    changed.extend(superseded)
                    memory_id = "mem_" + uuid.uuid4().hex[:12]
                    record = MemoryRecord(
                        id=memory_id,
                        content=operation.content.strip(),
                        memory_type=operation.memory_type,
                        topic=operation.topic,
                        confidence=operation.confidence,
                        status=ACTIVE_STATUS,
                        expires_at=operation.expires_at or self._default_expires_at(operation, now),
                        memory_strength=self._default_strength(operation.memory_type),
                        decay_rate=self._default_decay_rate(operation.memory_type),
                        last_reinforced_at=now,
                        tags=operation.tags,
                        metadata={
                            **operation.metadata,
                            "reason": operation.reason,
                            "actor": actor,
                            "intent": operation.intent,
                        },
                        source_session_id=session_id,
                        observed_at=normalize_observed_at(operation.observed_at)
                        or normalize_observed_at(operation.metadata.get("observed_at")),
                        created_at=now,
                        updated_at=now,
                    )
                    self._memories[memory_id] = record
                    self._vectors[memory_id] = embeddings_by_content[operation.content]
                    changed.append(record)
                    applied += 1
                elif operation.op == "replace":
                    if not operation.memory_id or not operation.content:
                        skipped += 1
                        continue
                    old = self._memories.get(operation.memory_id)
                    if old is None:
                        skipped += 1
                        continue
                    operation.topic = self._normalize_topic(operation.topic or old.topic)
                    operation.confidence = self._clamp_confidence(operation.confidence)
                    if operation.confidence < 0.6:
                        skipped += 1
                        continue
                    record = MemoryRecord(
                        id=old.id,
                        content=operation.content.strip(),
                        memory_type=operation.memory_type or old.memory_type,
                        topic=operation.topic,
                        confidence=operation.confidence,
                        status=ACTIVE_STATUS,
                        expires_at=operation.expires_at
                        or old.expires_at
                        or self._default_expires_at(operation, now),
                        memory_strength=max(
                            self._effective_strength(old),
                            self._default_strength(operation.memory_type or old.memory_type),
                        ),
                        decay_rate=self._default_decay_rate(operation.memory_type or old.memory_type),
                        last_reinforced_at=now,
                        tags=operation.tags or old.tags,
                        metadata={
                            **old.metadata,
                            **operation.metadata,
                            "reason": operation.reason,
                            "actor": actor,
                            "intent": operation.intent,
                        },
                        source_session_id=old.source_session_id or session_id,
                        observed_at=normalize_observed_at(operation.observed_at)
                        or normalize_observed_at(operation.metadata.get("observed_at"))
                        or old.observed_at,
                        created_at=old.created_at,
                        updated_at=now,
                        revision=old.revision + 1,
                    )
                    self._memories[record.id] = record
                    self._vectors[record.id] = embeddings_by_content[operation.content]
                    changed.append(record)
                    applied += 1
                elif operation.op == "delete":
                    if not operation.memory_id or operation.memory_id not in self._memories:
                        skipped += 1
                        continue
                    record = self._memories[operation.memory_id]
                    record.status = "deleted"
                    record.updated_at = now
                    changed.append(record)
                    applied += 1
                else:
                    skipped += 1

            if applied:
                self._persist()
            append_jsonl(
                self.audit_path,
                {
                    "at": now,
                    "actor": actor,
                    "session_id": session_id,
                    "applied": applied,
                    "skipped": skipped,
                    "operations": [model_to_dict(op) for op in operations],
                },
            )
        return applied, skipped, changed

    def append_session_message(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            return
        path = self.sessions_dir / f"{session_id}.jsonl"
        append_jsonl(path, {"at": utc_now(), "role": role, "content": content})

    def get_recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        path = self.sessions_dir / f"{session_id}.jsonl"
        items = load_jsonl(path)[-limit:]
        return [ChatMessage(role=item["role"], content=item["content"]) for item in items]

    def append_privacy_event(self, event: dict[str, Any]) -> None:
        append_jsonl(self.privacy_events_path, event)

    def select_user_card_memories(self, limit: int = 20) -> list[MemoryRecord]:
        with self._lock:
            candidates = [
                record
                for record in self._memories.values()
                if self._is_recallable(record)
                and record.memory_type in {"preference", "fact", "relation"}
                and record.confidence >= 0.7
                and self._effective_strength(record) >= 0.35
            ]
            type_weight = {"preference": 1.0, "fact": 0.9, "relation": 0.8}
            candidates.sort(
                key=lambda record: (
                    self._effective_strength(record) * 0.5
                    + record.confidence * 0.3
                    + type_weight.get(record.memory_type, 0.3) * 0.2
                ),
                reverse=True,
            )
            return candidates[:limit]

    def _is_recallable(self, record: MemoryRecord) -> bool:
        return (
            record.status == ACTIVE_STATUS
            and not self._is_expired(record)
            and self._effective_strength(record) >= MIN_RECALL_STRENGTH
        )

    def _is_expired(self, record: MemoryRecord) -> bool:
        if not record.expires_at:
            return False
        try:
            expires_at = datetime.fromisoformat(record.expires_at)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        return expires_at <= datetime.now(timezone.utc)

    def _find_duplicate(
        self,
        operation: EditOperation,
        embeddings_by_content: dict[str, list[float]],
    ) -> MemoryRecord | None:
        normalized_content = self._normalize_text(operation.content or "")
        operation_vector = embeddings_by_content.get(operation.content or "")
        for record in self._memories.values():
            if not self._is_recallable(record):
                continue
            if self._normalize_text(record.content) == normalized_content:
                return record
            if record.memory_type != operation.memory_type or record.topic != operation.topic:
                continue
            if operation_vector is not None and record.id in self._vectors:
                similarity = self._cosine_similarity(operation_vector, self._vectors[record.id])
                if similarity >= DUPLICATE_SIMILARITY:
                    return record
        return None

    def _supersede_topic_conflicts(
        self,
        operation: EditOperation,
        now: str,
        embeddings_by_content: dict[str, list[float]],
    ) -> list[MemoryRecord]:
        changed: list[MemoryRecord] = []
        operation_vector = embeddings_by_content.get(operation.content or "")
        for record in self._memories.values():
            if not self._is_recallable(record):
                continue
            if record.memory_type != operation.memory_type or record.topic != operation.topic:
                continue
            if operation.topic != "general":
                record.status = "superseded"
                record.updated_at = now
                changed.append(record)
                continue
            if operation_vector is not None and record.id in self._vectors:
                similarity = self._cosine_similarity(operation_vector, self._vectors[record.id])
                if similarity < TOPIC_CONFLICT_SIMILARITY:
                    continue
            record.status = "superseded"
            record.updated_at = now
            changed.append(record)
        return changed

    def _default_expires_at(self, operation: EditOperation, now: str) -> str | None:
        return operation.expires_at

    def _normalize_topic(self, topic: str) -> str:
        text = topic.strip().lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "general"

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
        return " ".join(text.split())

    def _lemmatize_for_bm25(self, text: str) -> str:
        nlp = _get_spacy_lemma()
        if nlp is None:
            return self._normalize_text(text)
        doc = nlp(text.lower())
        tokens = []
        for token in doc:
            if token.is_punct or token.is_stop:
                continue
            lemma = token.lemma_
            if lemma.isalnum():
                tokens.append(lemma)
            if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
                tokens.append(token.text)
        return " ".join(tokens)

    def _bm25_scores(self, query_text: str, ids: list[str]) -> dict[str, float]:
        query_tokens = self._lemmatize_for_bm25(query_text).split()
        if not query_tokens:
            return {}
        docs = [
            self._lemmatize_for_bm25(self._memories[memory_id].content).split()
            for memory_id in ids
        ]
        if not docs:
            return {}
        doc_freq: Counter[str] = Counter()
        for doc in docs:
            doc_freq.update(set(doc))
        avgdl = sum(len(doc) for doc in docs) / max(len(docs), 1)
        if avgdl <= 0:
            return {}
        midpoint, steepness = self._bm25_params(query_tokens)
        scores: dict[str, float] = {}
        n_docs = len(docs)
        for memory_id, doc in zip(ids, docs):
            if not doc:
                continue
            counts = Counter(doc)
            raw = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if tf <= 0:
                    continue
                df = doc_freq.get(token, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                raw += idf * (tf * 2.2) / (tf + 1.2 * (1.0 - 0.75 + 0.75 * len(doc) / avgdl))
            if raw > 0:
                scores[memory_id] = self._normalize_bm25(raw, midpoint, steepness)
        return scores

    def _entity_boosts(self, query_text: str, ids: list[str]) -> dict[str, float]:
        query_entities = self._extract_entities(query_text)
        if not query_entities:
            return {}
        boosts: dict[str, float] = {}
        query_entity_set = {_normalize_entity(entity) for entity in query_entities[:8]}
        for memory_id in ids:
            record = self._memories[memory_id]
            candidates = set(record.tags)
            candidates.add(record.topic)
            candidates.update(self._extract_entities(record.content))
            normalized_candidates = {_normalize_entity(item) for item in candidates if item}
            overlap = query_entity_set & normalized_candidates
            if not overlap:
                continue
            boost = min(ENTITY_BOOST_WEIGHT, 0.18 * len(overlap))
            boosts[memory_id] = max(boosts.get(memory_id, 0.0), boost)
        return boosts

    def _temporal_boosts(self, query_text: str, ids: list[str]) -> dict[str, float]:
        constraints = _extract_temporal_constraints(query_text)
        if not constraints:
            return {}
        boosts: dict[str, float] = {}
        for memory_id in ids:
            record = self._memories[memory_id]
            observed_date = parse_observed_date(record.observed_at)
            if observed_date is None:
                continue
            best = max(_temporal_constraint_score(observed_date, constraint) for constraint in constraints)
            if best > 0:
                boosts[memory_id] = best
        return boosts

    def _extract_entities(self, text: str) -> list[str]:
        nlp = _get_spacy_full()
        if nlp is not None:
            doc = nlp(text)
            values = [
                ent.text
                for ent in doc.ents
                if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT"}
            ]
            if values:
                return _dedupe_texts(values)
        values = re.findall(r"\b[A-Z][A-Za-z0-9&_.-]*(?:\s+[A-Z][A-Za-z0-9&_.-]*){0,3}\b", text)
        return _dedupe_texts(value for value in values if value.lower() not in {"user", "assistant"})

    def _bm25_params(self, query_tokens: list[str]) -> tuple[float, float]:
        num_terms = len(query_tokens) or 1
        if num_terms <= 3:
            return 5.0, 0.7
        if num_terms <= 6:
            return 7.0, 0.6
        if num_terms <= 9:
            return 9.0, 0.5
        if num_terms <= 15:
            return 10.0, 0.5
        return 12.0, 0.5

    def _normalize_bm25(self, raw_score: float, midpoint: float, steepness: float) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))

    def _clamp_confidence(self, confidence: float) -> float:
        return max(0.0, min(1.0, float(confidence)))

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        left_vector = np.asarray(left, dtype=np.float32)
        right_vector = np.asarray(right, dtype=np.float32)
        denom = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denom == 0:
            return 0.0
        return float(left_vector @ right_vector / denom)

    def _ensure_strength_fields(self, record: MemoryRecord) -> None:
        if record.memory_strength is None:
            record.memory_strength = self._default_strength(record.memory_type)
        if record.decay_rate is None:
            record.decay_rate = self._default_decay_rate(record.memory_type)
        if record.last_reinforced_at is None:
            record.last_reinforced_at = record.updated_at or record.created_at
        if record.observed_at is None:
            record.observed_at = normalize_observed_at(record.metadata.get("observed_at"))

    def _effective_strength(self, record: MemoryRecord) -> float:
        self._ensure_strength_fields(record)
        try:
            reinforced_at = datetime.fromisoformat(record.last_reinforced_at or record.updated_at)
            if reinforced_at.tzinfo is None:
                reinforced_at = reinforced_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return self._clamp_confidence(record.memory_strength or 0.0)
        elapsed_days = max(
            0.0,
            (datetime.now(timezone.utc) - reinforced_at).total_seconds() / 86400.0,
        )
        strength = (record.memory_strength or 0.0) * math.exp(
            -(record.decay_rate or 0.0) * elapsed_days
        )
        return self._clamp_confidence(strength)

    def _reinforce(self, record: MemoryRecord, now: str) -> None:
        current = self._effective_strength(record)
        delta = self._default_reinforce_delta(record.memory_type)
        record.memory_strength = self._clamp_confidence(current + delta)
        record.last_reinforced_at = now

    def _default_strength(self, memory_type: str) -> float:
        return TYPE_MEMORY_DEFAULTS.get(memory_type, TYPE_MEMORY_DEFAULTS["fact"])["strength"]

    def _default_decay_rate(self, memory_type: str) -> float:
        return TYPE_MEMORY_DEFAULTS.get(memory_type, TYPE_MEMORY_DEFAULTS["fact"])["decay_rate"]

    def _default_reinforce_delta(self, memory_type: str) -> float:
        return TYPE_MEMORY_DEFAULTS.get(memory_type, TYPE_MEMORY_DEFAULTS["fact"])["reinforce_delta"]


def _normalize_entity(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def _dedupe_texts(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = " ".join(str(value).split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _extract_temporal_constraints(text: str) -> list[dict[str, date | str]]:
    text = text or ""
    anchor = _extract_anchor_date(text)
    constraints: list[dict[str, date | str]] = []
    text_without_anchor = re.sub(
        r"\bquestion_date\s*=\s*[0-9]{4}[-/][0-9]{2}[-/][0-9]{2}",
        "",
        text,
        flags=re.I,
    )
    seen = set()
    for match in re.finditer(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", text_without_anchor):
        normalized = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        parsed = parse_observed_date(normalized)
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        constraints.append({"kind": "near", "date": parsed})
    if anchor is None:
        return constraints

    lowered = text.lower()
    if "yesterday" in lowered:
        constraints.append({"kind": "near", "date": anchor - _days(1)})
    if "today" in lowered:
        constraints.append({"kind": "near", "date": anchor})
    if "last week" in lowered or "past week" in lowered or "previous week" in lowered:
        constraints.append({"kind": "range", "start": anchor - _days(7), "end": anchor})
    if "last month" in lowered or "past month" in lowered or "previous month" in lowered:
        constraints.append({"kind": "range", "start": anchor - _days(31), "end": anchor})
    if "last year" in lowered or "past year" in lowered or "previous year" in lowered:
        constraints.append({"kind": "range", "start": anchor - _days(366), "end": anchor})

    relative_patterns = [
        (r"\bsince\s+(\d{4}[-/]\d{2}[-/]\d{2})", "after"),
        (r"\bafter\s+(\d{4}[-/]\d{2}[-/]\d{2})", "after"),
        (r"\bbefore\s+(\d{4}[-/]\d{2}[-/]\d{2})", "before"),
        (r"\bon\s+(\d{4}[-/]\d{2}[-/]\d{2})", "near"),
    ]
    for pattern, kind in relative_patterns:
        for match in re.finditer(pattern, lowered):
            parsed = parse_observed_date(match.group(1))
            if parsed is None:
                continue
            if kind == "after":
                constraints.append({"kind": "after", "date": parsed})
            elif kind == "before":
                constraints.append({"kind": "before", "date": parsed})
            else:
                constraints.append({"kind": "near", "date": parsed})
    return constraints


def _extract_anchor_date(text: str) -> date | None:
    question_date_match = re.search(
        r"\bquestion_date\s*=\s*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2})",
        text,
        flags=re.I,
    )
    if question_date_match:
        return parse_observed_date(question_date_match.group(1))
    dates = [
        parse_observed_date(match.group(0))
        for match in re.finditer(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)
    ]
    dates = [item for item in dates if item is not None]
    return max(dates) if dates else None


def _temporal_constraint_score(observed_date: date, constraint: dict[str, date | str]) -> float:
    kind = constraint.get("kind")
    if kind == "near":
        target = constraint.get("date")
        if not isinstance(target, date):
            return 0.0
        delta_days = abs((observed_date - target).days)
        if delta_days == 0:
            return TEMPORAL_BOOST_WEIGHT
        if delta_days <= 7:
            return TEMPORAL_BOOST_WEIGHT * 0.6
        if delta_days <= 31:
            return TEMPORAL_BOOST_WEIGHT * 0.3
        return 0.0
    if kind == "range":
        start = constraint.get("start")
        end = constraint.get("end")
        if not isinstance(start, date) or not isinstance(end, date):
            return 0.0
        return TEMPORAL_BOOST_WEIGHT if start <= observed_date <= end else 0.0
    if kind == "after":
        target = constraint.get("date")
        return TEMPORAL_BOOST_WEIGHT * 0.8 if isinstance(target, date) and observed_date >= target else 0.0
    if kind == "before":
        target = constraint.get("date")
        return TEMPORAL_BOOST_WEIGHT * 0.8 if isinstance(target, date) and observed_date <= target else 0.0
    return 0.0


def _days(count: int):
    from datetime import timedelta

    return timedelta(days=count)


def _get_spacy_full():
    global _SPACY_FULL, _SPACY_FULL_FAILED
    if _SPACY_FULL_FAILED:
        return None
    if _SPACY_FULL is not None:
        return _SPACY_FULL
    try:
        import spacy

        if not spacy.util.is_package("en_core_web_sm"):
            _SPACY_FULL_FAILED = True
            return None
        _SPACY_FULL = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
        return _SPACY_FULL
    except Exception:
        _SPACY_FULL_FAILED = True
        return None


def _get_spacy_lemma():
    global _SPACY_LEMMA, _SPACY_LEMMA_FAILED
    if _SPACY_LEMMA_FAILED:
        return None
    if _SPACY_LEMMA is not None:
        return _SPACY_LEMMA
    try:
        import spacy

        if not spacy.util.is_package("en_core_web_sm"):
            _SPACY_LEMMA_FAILED = True
            return None
        _SPACY_LEMMA = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        return _SPACY_LEMMA
    except Exception:
        _SPACY_LEMMA_FAILED = True
        return None
