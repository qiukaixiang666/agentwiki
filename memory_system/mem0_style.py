from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .llm_client import LLMClient
from .prompts import build_mem0_additive_extraction_prompt
from .schemas import ChatMessage, EditOperation, RetrievedMemory
from .time_utils import observed_at_from_messages, strip_longmemeval_prefixes, utc_now_date
from .utils import model_to_dict

_SPACY_ENTITY = None
_SPACY_ENTITY_FAILED = False


@dataclass
class Mem0ExtractionDebug:
    prompt: list[ChatMessage]
    raw: str = ""
    parsed: dict[str, Any] | None = None
    extracted: list[dict[str, Any]] = field(default_factory=list)
    operations: list[EditOperation] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str = ""


class Mem0AdditiveExtractor:
    """Mem0-style ADD-only memory extraction over the existing Wiki store."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def extract_with_debug(
        self,
        new_messages: list[ChatMessage],
        existing_memories: list[RetrievedMemory],
        *,
        last_messages: list[ChatMessage] | None = None,
        profile_summary: str = "",
        custom_instructions: str = "",
        observation_date: str | None = None,
    ) -> tuple[list[EditOperation], Mem0ExtractionDebug]:
        observed_at = observation_date or observed_at_from_messages(new_messages) or utc_now_date()
        prompt_messages = strip_longmemeval_prefixes(new_messages)
        prompt_last_messages = strip_longmemeval_prefixes(last_messages or [])
        prompt = build_mem0_additive_extraction_prompt(
            new_messages=prompt_messages,
            existing_memories=existing_memories,
            last_messages=prompt_last_messages,
            profile_summary=profile_summary,
            recently_extracted=[],
            custom_instructions=custom_instructions,
        )
        debug = Mem0ExtractionDebug(prompt=prompt)
        raw = await self.llm.complete(prompt, json_mode=True)
        debug.raw = raw

        try:
            payload = _parse_json_object(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            debug.parse_error = f"model returned malformed JSON: {exc}"
            return [], debug
        if not isinstance(payload, dict):
            debug.parse_error = "model returned non-object JSON"
            return [], debug

        debug.parsed = payload
        extracted = payload.get("memory", [])
        if not isinstance(extracted, list):
            debug.parse_error = "JSON object did not contain a list-valued memory field"
            return [], debug
        debug.extracted = [item for item in extracted if isinstance(item, dict)]

        operations: list[EditOperation] = []
        dropped: list[dict[str, Any]] = []
        for item in debug.extracted:
            text = " ".join(str(item.get("text", "")).split())
            if not text:
                dropped.append({"item": item, "reason": "missing text"})
                continue
            try:
                operation = EditOperation(
                    op="add",
                    content=text,
                    memory_type=_infer_memory_type(text),
                    topic=_topic_from_memory(text),
                    confidence=_confidence_from_memory(text),
                    tags=_tags_from_memory(text, item),
                    reason="mem0 additive extraction",
                    intent="Extract every durable and queryable memory from the current conversation turn.",
                    metadata={"observed_at": observed_at} if observed_at else {},
                    observed_at=observed_at,
                )
            except ValidationError as exc:
                dropped.append({"item": item, "reason": str(exc)})
                continue
            operations.append(operation)

        debug.operations = operations
        debug.dropped = dropped
        return operations, debug


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

def _infer_memory_type(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["prefers", "preference", "likes", "dislikes", "enjoys", "values "]):
        return "preference"
    if any(term in lowered for term in ["wife", "husband", "daughter", "son", "mother", "father", "friend", "colleague"]):
        return "relation"
    if any(term in lowered for term in ["plans to", "intends to", "needs to", "wants to", "is scheduled to", "reminder"]):
        return "task"
    if re.search(r"\b(on|around|during|in)\s+\d{4}[-/]\d{2}[-/]\d{2}\b", lowered):
        return "event"
    return "fact"


def _confidence_from_memory(text: str) -> float:
    if re.search(r"\b(user|the user|assistant|the assistant)\b", text, flags=re.I):
        return 0.9
    if re.search(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b|\bsession\s+[A-Za-z0-9_.:-]+", text):
        return 0.9
    return 0.8


def _tags_from_memory(text: str, item: dict[str, Any]) -> list[str]:
    tags = ["mem0_additive"]
    lowered = text.lower()
    if "recommended" in lowered or "suggested" in lowered or "the assistant" in lowered:
        tags.append("assistant_evidence")
    if "longmemeval" in lowered or "session " in lowered:
        tags.append("longmemeval")
    linked = item.get("linked_memory_ids")
    if isinstance(linked, list) and linked:
        tags.append("linked")
    return tags


def _topic_from_memory(text: str) -> str:
    entities = _extract_entities(text)
    if entities:
        return _normalize_topic("_".join(entities[:3]))
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text.lower())
        if token not in _TOPIC_STOPWORDS
    ]
    return _normalize_topic("_".join(tokens[:4]) or "general")


def _extract_entities(text: str) -> list[str]:
    nlp = _get_spacy_entity()
    if nlp is not None:
        doc = nlp(text)
        values = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"}]
        if values:
            return _dedupe(values)
    values = re.findall(r"\b[A-Z][A-Za-z0-9&_.-]*(?:\s+[A-Z][A-Za-z0-9&_.-]*){0,3}\b", text)
    return _dedupe(value for value in values if value.lower() not in {"user", "assistant"})


def _get_spacy_entity():
    global _SPACY_ENTITY, _SPACY_ENTITY_FAILED
    if _SPACY_ENTITY_FAILED:
        return None
    if _SPACY_ENTITY is not None:
        return _SPACY_ENTITY
    try:
        import spacy

        if not spacy.util.is_package("en_core_web_sm"):
            _SPACY_ENTITY_FAILED = True
            return None
        _SPACY_ENTITY = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
        return _SPACY_ENTITY
    except Exception:
        _SPACY_ENTITY_FAILED = True
        return None


def _dedupe(values) -> list[str]:
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


def _normalize_topic(topic: str) -> str:
    text = topic.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "general"


def debug_to_dict(debug: Mem0ExtractionDebug) -> dict[str, Any]:
    return {
        "prompt": [model_to_dict(message) for message in debug.prompt],
        "raw": debug.raw,
        "parsed": debug.parsed,
        "extracted": debug.extracted,
        "operations": [model_to_dict(operation) for operation in debug.operations],
        "dropped": debug.dropped,
        "parse_error": debug.parse_error,
    }


_TOPIC_STOPWORDS = {
    "the",
    "user",
    "assistant",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "their",
    "there",
    "were",
    "was",
    "has",
    "have",
    "had",
    "and",
    "for",
    "recommended",
    "suggested",
}
