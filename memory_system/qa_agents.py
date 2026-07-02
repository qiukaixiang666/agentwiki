from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .llm_client import LLMClient
from .prompts import (
    build_memory_triage_prompt,
    build_profile_decision_prompt,
    build_qa_summary_prompt,
)
from .schemas import (
    ChatMessage,
    EditOperation,
    MemoryTriage,
    ProfileUseDecision,
    QAPairEvidence,
    RetrievedMemory,
)
from .utils import model_to_dict


class QAPairSummarizer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def summarize(
        self,
        user_message: ChatMessage,
        assistant_message: ChatMessage | None,
    ) -> QAPairEvidence:
        evidence, _debug = await self.summarize_with_debug(user_message, assistant_message)
        return evidence

    async def summarize_with_debug(
        self,
        user_message: ChatMessage,
        assistant_message: ChatMessage | None,
    ) -> tuple[QAPairEvidence, dict]:
        raw = await self.llm.complete(
            build_qa_summary_prompt(user_message, assistant_message),
            json_mode=True,
        )
        payload = _safe_json_object(raw)
        debug = {"raw": raw, "parsed": payload, "fallback": False, "error": ""}
        if not payload:
            evidence = _fallback_evidence(user_message, assistant_message)
            debug.update({"parsed": _model_dump(evidence), "fallback": True, "error": "malformed_or_non_object_json"})
            return evidence, debug
        try:
            evidence = QAPairEvidence(**payload)
        except ValidationError as exc:
            evidence = _fallback_evidence(user_message, assistant_message)
            debug.update({"parsed": _model_dump(evidence), "fallback": True, "error": str(exc)})
            return evidence, debug
        evidence.raw_user = evidence.raw_user.strip() or user_message.content.strip()
        evidence.raw_assistant = evidence.raw_assistant.strip()
        if assistant_message and not evidence.raw_assistant:
            evidence.raw_assistant = assistant_message.content.strip()
        if not evidence.summary.strip():
            evidence.summary = _fallback_summary(evidence.raw_user, evidence.raw_assistant)
        if not evidence.date or not evidence.source_session_id:
            date, source_session_id = _extract_longmemeval_metadata(evidence.raw_user)
            evidence.date = evidence.date or date
            evidence.source_session_id = evidence.source_session_id or source_session_id
        evidence.entities = _dedupe_texts(evidence.entities)
        evidence.evidence_items = _dedupe_texts(evidence.evidence_items)
        debug["normalized"] = _model_dump(evidence)
        return evidence, debug


class MemoryTriageAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def triage(
        self,
        evidence: QAPairEvidence,
        recalled_memories: list[RetrievedMemory],
    ) -> MemoryTriage:
        triage, _debug = await self.triage_with_debug(evidence, recalled_memories)
        return triage

    async def triage_with_debug(
        self,
        evidence: QAPairEvidence,
        recalled_memories: list[RetrievedMemory],
    ) -> tuple[MemoryTriage, dict]:
        evidence_json = _json_dumps(evidence)
        raw = await self.llm.complete(
            build_memory_triage_prompt(evidence_json, recalled_memories),
            json_mode=True,
        )
        payload = _safe_json_object(raw)
        debug = {"raw": raw, "parsed": payload, "fallback": False, "error": ""}
        if not payload:
            triage = _fallback_triage(evidence)
            debug.update({"parsed": _model_dump(triage), "fallback": True, "error": "malformed_or_non_object_json"})
            return triage, debug
        try:
            triage = MemoryTriage(**payload)
        except ValidationError as exc:
            triage = _fallback_triage(evidence)
            debug.update({"parsed": _model_dump(triage), "fallback": True, "error": str(exc)})
            return triage, debug
        triage.category = _clean_label(triage.category) or "not_useful"
        triage.memory_type = _clean_memory_type(triage.memory_type)
        triage.topic = _normalize_topic(triage.topic)
        triage.retrieval_query = " ".join(triage.retrieval_query.split())
        if not triage.retrieval_query:
            triage.retrieval_query = _fallback_retrieval_query(evidence)
        debug["normalized"] = _model_dump(triage)
        return triage, debug


class ProfileUseAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def decide(
        self,
        evidence: QAPairEvidence,
        operations: list[EditOperation],
    ) -> ProfileUseDecision:
        decision, _debug = await self.decide_with_debug(evidence, operations)
        return decision

    async def decide_with_debug(
        self,
        evidence: QAPairEvidence,
        operations: list[EditOperation],
    ) -> tuple[ProfileUseDecision, dict]:
        if not operations:
            decision = ProfileUseDecision(usable=False, reason="no changed memories")
            return decision, {
                "raw": "",
                "parsed": _model_dump(decision),
                "fallback": False,
                "error": "no_operations",
            }
        raw = await self.llm.complete(
            build_profile_decision_prompt(_json_dumps(evidence), _json_dumps(operations)),
            json_mode=True,
        )
        payload = _safe_json_object(raw)
        debug = {"raw": raw, "parsed": payload, "fallback": False, "error": ""}
        if not payload:
            decision = _fallback_profile_decision(evidence, operations)
            debug.update({"parsed": _model_dump(decision), "fallback": True, "error": "malformed_or_non_object_json"})
            return decision, debug
        try:
            decision = ProfileUseDecision(**payload)
        except ValidationError as exc:
            decision = _fallback_profile_decision(evidence, operations)
            debug.update({"parsed": _model_dump(decision), "fallback": True, "error": str(exc)})
            return decision, debug
        debug["normalized"] = _model_dump(decision)
        return decision, debug


def pair_user_assistant_messages(messages: list[ChatMessage]) -> list[tuple[ChatMessage, ChatMessage | None]]:
    pairs: list[tuple[ChatMessage, ChatMessage | None]] = []
    idx = 0
    while idx < len(messages):
        message = messages[idx]
        if message.role != "user":
            idx += 1
            continue
        assistant_message: ChatMessage | None = None
        if idx + 1 < len(messages) and messages[idx + 1].role == "assistant":
            assistant_message = messages[idx + 1]
            idx += 2
        else:
            idx += 1
        pairs.append((message, assistant_message))
    return pairs


def evidence_to_json(evidence: QAPairEvidence) -> str:
    return _json_dumps(evidence)


def triage_to_json(triage: MemoryTriage) -> str:
    return _json_dumps(triage)


def _safe_json_object(text: str) -> dict:
    try:
        payload = _parse_json_object(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json_object(text: str) -> object:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_evidence(
    user_message: ChatMessage,
    assistant_message: ChatMessage | None,
) -> QAPairEvidence:
    raw_user = user_message.content.strip()
    raw_assistant = assistant_message.content.strip() if assistant_message else ""
    date, source_session_id = _extract_longmemeval_metadata(raw_user)
    return QAPairEvidence(
        summary=_fallback_summary(raw_user, raw_assistant),
        raw_user=raw_user,
        raw_assistant=raw_assistant,
        date=date,
        source_session_id=source_session_id,
        entities=_fallback_entities(raw_user + "\n" + raw_assistant),
        evidence_items=[_fallback_summary(raw_user, raw_assistant)],
        reason="fallback summary after malformed model JSON",
    )


def _fallback_triage(evidence: QAPairEvidence) -> MemoryTriage:
    text = f"{evidence.summary}\n{evidence.raw_user}\n{evidence.raw_assistant}".lower()
    category = "user_question"
    memory_type = "fact"
    if any(term in text for term in ["prefer", "like", "dislike", "value", "feel better", "希望", "喜欢", "偏好"]):
        category = "user_preference"
        memory_type = "preference"
    elif evidence.date:
        category = "event"
        memory_type = "event"
    elif evidence.raw_assistant.strip():
        category = "assistant_evidence"
        memory_type = "fact"
    return MemoryTriage(
        category=category,
        memory_type=memory_type,
        topic=_normalize_topic("_".join(evidence.entities[:3]) or "longmemeval_evidence"),
        retrieval_query=_fallback_retrieval_query(evidence),
        should_attempt_edit=True,
        related_memory_policy="replace_if_refined",
        reason="fallback triage after malformed model JSON",
    )


def _fallback_profile_decision(
    evidence: QAPairEvidence,
    operations: list[EditOperation],
) -> ProfileUseDecision:
    profile_types = {"preference", "relation"}
    usable = any(operation.memory_type in profile_types for operation in operations)
    if not usable:
        combined = f"{evidence.summary}\n{evidence.raw_user}".lower()
        usable = any(
            term in combined
            for term in ["prefer", "like", "dislike", "often", "regularly", "希望", "喜欢", "偏好", "经常"]
        )
    return ProfileUseDecision(
        usable=usable,
        profile_aspect="fallback_profile_signal" if usable else "",
        reason="fallback profile decision after malformed model JSON",
    )


def _fallback_summary(raw_user: str, raw_assistant: str) -> str:
    user_core = _strip_metadata(raw_user)
    assistant_core = _strip_metadata(raw_assistant)
    if assistant_core:
        return f"用户询问：{_truncate(user_core, 180)}；助手回答：{_truncate(assistant_core, 220)}"
    return f"用户询问：{_truncate(user_core, 260)}"


def _fallback_retrieval_query(evidence: QAPairEvidence) -> str:
    parts = []
    if evidence.date:
        parts.append(evidence.date)
    if evidence.source_session_id:
        parts.append(evidence.source_session_id)
    parts.extend(evidence.entities[:6])
    parts.append(_strip_metadata(evidence.summary))
    return " ".join(part for part in parts if part).strip() or "longmemeval historical qa evidence"


def _extract_longmemeval_metadata(text: str) -> tuple[str | None, str | None]:
    date_match = re.search(r"\bdate=([0-9]{4}/[0-9]{2}/[0-9]{2})", text)
    session_match = re.search(r"\b(?:LongMemEval\s+)?session_id=([A-Za-z0-9_.:-]+)", text)
    return (
        date_match.group(1) if date_match else None,
        session_match.group(1).rstrip(";]") if session_match else None,
    )


def _fallback_entities(text: str) -> list[str]:
    entities = re.findall(r"\b[A-Z][A-Za-z0-9&_.-]*(?:\s+[A-Z][A-Za-z0-9&_.-]*){0,4}\b", text)
    return _dedupe_texts(entity for entity in entities if len(entity) > 2)[:8]


def _clean_label(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")


def _clean_memory_type(value: str) -> str:
    value = _clean_label(value)
    if value in {"fact", "preference", "task", "relation", "event"}:
        return value
    return "fact"


def _normalize_topic(topic: str) -> str:
    text = str(topic).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "general"


def _strip_metadata(text: str) -> str:
    text = re.sub(r"^\[LongMemEval[^\]]+\]\s*", "", text.strip(), flags=re.I)
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _dedupe_texts(values) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _json_dumps(value) -> str:
    if isinstance(value, list):
        payload = [model_to_dict(item) if hasattr(item, "model_dump") else item for item in value]
    elif hasattr(value, "model_dump"):
        payload = model_to_dict(value)
    else:
        payload = value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _model_dump(value) -> dict:
    return model_to_dict(value)
