from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .llm_client import LLMClient
from .schemas import ChatMessage, EditOperation, EditPlan, EditPlanDebug, RetrievedMemory
from .utils import model_to_dict

MIN_MEMORY_CONFIDENCE = 0.6

_CONTEXT_DEPENDENT_PATTERNS = [
    re.compile(r"\b(it|this|that|these|those|same|above|previous|earlier)\b", re.I),
    re.compile(
        r"(\u8fd9\u4e2a|\u90a3\u4e2a|\u8fd9\u4e9b|\u90a3\u4e9b|"
        r"\u4e0a\u9762|\u524d\u9762|\u521a\u624d|\u5b83|\u5176)"
    ),
]


class EditPlanner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(
        self,
        prompt_messages: list[ChatMessage],
        recalled_memories: list[RetrievedMemory],
        latest_user_input: str = "",
    ) -> EditPlan:
        plan, _debug = await self.plan_with_debug(prompt_messages, recalled_memories, latest_user_input)
        return plan

    async def plan_with_debug(
        self,
        prompt_messages: list[ChatMessage],
        recalled_memories: list[RetrievedMemory],
        latest_user_input: str = "",
    ) -> tuple[EditPlan, EditPlanDebug]:
        raw = await self.llm.complete(prompt_messages, json_mode=True)
        debug = EditPlanDebug(raw=raw)
        try:
            payload = _parse_json_object(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            debug.parse_error = "model returned malformed JSON"
            plan = EditPlan(operations=[], rationale=debug.parse_error)
            debug.rationale = plan.rationale
            return plan, debug
        if not isinstance(payload, dict):
            debug.parse_error = "model returned non-object JSON"
            plan = EditPlan(operations=[], rationale=debug.parse_error)
            debug.rationale = plan.rationale
            return plan, debug
        debug.parsed = payload
        try:
            plan = EditPlan(**payload)
        except ValidationError as exc:
            debug.parse_error = str(exc)
            plan = EditPlan(operations=[], rationale="model returned invalid edit plan schema")
            debug.rationale = plan.rationale
            return plan, debug

        debug.before_filter = list(plan.operations)
        safe_operations: list[EditOperation] = []
        dropped: list[dict] = []
        known_ids = {memory.id for memory in recalled_memories}
        for operation in plan.operations:
            if operation.op in {"replace", "delete"} and operation.memory_id not in known_ids:
                dropped.append(_drop_debug(operation, "replace/delete memory_id was not in recalled memories"))
                continue
            if operation.op in {"add", "replace"}:
                if not operation.content:
                    dropped.append(_drop_debug(operation, "missing content"))
                    continue
                operation.topic = _normalize_topic(operation.topic)
                operation.confidence = max(0.0, min(1.0, float(operation.confidence)))
                if operation.confidence < MIN_MEMORY_CONFIDENCE:
                    dropped.append(_drop_debug(operation, "confidence below threshold"))
                    continue
                if not _is_specific_intent(operation.intent):
                    dropped.append(_drop_debug(operation, "intent too generic or too short"))
                    continue
                if not _is_complete_fact_statement(operation.content):
                    dropped.append(_drop_debug(operation, "content is not a complete standalone memory statement"))
                    continue
                if _looks_like_direct_latest_input_rewrite(
                    operation.content,
                    latest_user_input,
                ) and not _allows_direct_memory(operation, latest_user_input):
                    dropped.append(_drop_debug(operation, "content looked like a direct latest-input rewrite"))
                    continue
            safe_operations.append(operation)
        plan.operations = safe_operations
        debug.after_filter = list(safe_operations)
        debug.dropped = dropped
        debug.rationale = plan.rationale
        return plan, debug


def _parse_json_object(text: str) -> dict:
    text = text.strip()
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


def _is_complete_fact_statement(content: str) -> bool:
    text = " ".join(content.strip().split())
    if len(text) < 12:
        return False
    if any(pattern.search(text) for pattern in _CONTEXT_DEPENDENT_PATTERNS):
        return False
    if not re.search(
        r"\b(the user|user's|the project|the memory system)\b|"
        r"\b(the assistant|assistant-provided|assistant recommended|assistant suggested|assistant concluded)\b|"
        r"\b(on \d{4}/\d{2}/\d{2}|in session [a-z0-9_\-]+)\b|"
        r"\bLongMemEval\b|\bQA pair\b|\bhistorical conversation\b|"
        r"(\u7528\u6237|\u9879\u76ee|\u8bb0\u5fc6\u7cfb\u7edf|\u52a9\u624b)",
        text,
        flags=re.I,
    ):
        return False
    return True


def _is_specific_intent(intent: str) -> bool:
    text = " ".join(intent.strip().lower().split())
    if len(text) < 16:
        return False
    generic_phrases = {
        "rewrite user intent into a standalone memory fact",
        "durable memory",
        "user intent",
        "memory update",
    }
    return text not in generic_phrases


def _normalize_topic(topic: str) -> str:
    text = topic.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "general"


def _allows_direct_memory(operation: EditOperation, latest_user_input: str) -> bool:
    latest = latest_user_input.lower()
    if operation.memory_type in {"preference", "task"}:
        return True
    if "longmemeval session_id=" in latest:
        return True
    if "qa pair evidence" in latest or "summarized qa pair" in latest:
        return True
    if operation.memory_type in {"fact", "relation", "event"} and any(
        marker in latest
        for marker in [
            "remember",
            "note that",
            "my ",
            "i am ",
            "i'm ",
            "i asked ",
            "i need ",
            "i have ",
            "\u8bb0\u4f4f",
            "\u6211\u662f",
            "\u6211\u5728",
            "\u6211\u7684",
        ]
    ):
        return True
    return False


def _looks_like_direct_latest_input_rewrite(content: str, latest_user_input: str) -> bool:
    content_norm = _normalize_for_copy_check(content)
    latest_norm = _normalize_for_copy_check(latest_user_input)
    if not content_norm or not latest_norm:
        return False

    if len(latest_norm) >= 20 and (latest_norm in content_norm or content_norm in latest_norm):
        return True

    core = _latest_user_core(latest_user_input)
    if core and len(core) >= 8 and core in content_norm:
        return True

    core_tokens = _meaningful_tokens(core)
    content_tokens = set(_meaningful_tokens(content_norm))
    if len(core_tokens) >= 2:
        overlap = sum(1 for token in core_tokens if token in content_tokens)
        if overlap / len(core_tokens) >= 0.8:
            return True
    return False


def _latest_user_core(text: str) -> str:
    normalized = _normalize_for_copy_check(text)
    normalized = re.sub(
        r"^(please\s+)?(remember\s+that\s+|note\s+that\s+|save\s+that\s+)?",
        "",
        normalized,
    )
    normalized = re.sub(
        r"^(i|we)\s+(prefer|like|want|need|hope|wish|am|are|use|build|work)\s+",
        "",
        normalized,
    )
    normalized = re.sub(r"^my\s+", "", normalized)
    return normalized.strip()


def _normalize_for_copy_check(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def _meaningful_tokens(text: str) -> list[str]:
    stopwords = {
        "the",
        "user",
        "users",
        "prefers",
        "prefer",
        "wants",
        "want",
        "needs",
        "need",
        "likes",
        "like",
        "that",
        "this",
        "with",
        "from",
        "into",
        "about",
        "more",
        "care",
    }
    return [
        token
        for token in _normalize_for_copy_check(text).split()
        if len(token) > 2 and token not in stopwords
    ]


def _drop_debug(operation: EditOperation, reason: str) -> dict:
    return {"reason": reason, "operation": model_to_dict(operation)}
