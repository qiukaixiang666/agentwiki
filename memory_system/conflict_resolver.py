from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .llm_client import LLMClient
from .prompts import build_memory_conflict_prompt
from .schemas import ChatMessage, EditOperation, RetrievedMemory
from .utils import model_to_dict


class ConflictDecision(BaseModel):
    has_conflict: bool = False
    target_memory_id: str | None = None
    rewritten_memory: str | None = None
    reason: str = ""


@dataclass
class ConflictResolverDebug:
    candidate: EditOperation
    related_memories: list[RetrievedMemory]
    prompt: list[ChatMessage] = field(default_factory=list)
    raw: str = ""
    parsed: dict[str, Any] | None = None
    decision: ConflictDecision | None = None
    fallback: bool = False
    parse_error: str = ""


class MemoryConflictResolver:
    """Narrow post-extraction module: only decide conflict vs no conflict."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def resolve_with_debug(
        self,
        candidate: EditOperation,
        related_memories: list[RetrievedMemory],
    ) -> tuple[EditOperation, ConflictResolverDebug]:
        debug = ConflictResolverDebug(candidate=candidate, related_memories=related_memories)
        if candidate.op != "add" or not candidate.content or not related_memories:
            debug.decision = ConflictDecision(
                has_conflict=False,
                reason="No related memories to compare, or candidate is not an add operation.",
            )
            return candidate, debug

        prompt = build_memory_conflict_prompt(candidate, related_memories)
        debug.prompt = prompt
        raw = await self.llm.complete(prompt, json_mode=True)
        debug.raw = raw

        try:
            payload = _parse_json_object(raw)
            if not isinstance(payload, dict):
                raise TypeError("model returned non-object JSON")
            decision = ConflictDecision(**payload)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            debug.fallback = True
            debug.parse_error = f"model returned malformed conflict decision: {exc}"
            decision = ConflictDecision(has_conflict=False, reason=debug.parse_error)

        debug.parsed = payload if "payload" in locals() and isinstance(payload, dict) else None
        debug.decision = decision
        if not decision.has_conflict:
            return candidate, debug

        valid_targets = {memory.id for memory in related_memories}
        if not decision.target_memory_id or decision.target_memory_id not in valid_targets:
            debug.fallback = True
            debug.parse_error = "conflict decision target_memory_id was missing or not in related memories"
            return candidate, debug
        rewritten = " ".join(str(decision.rewritten_memory or "").split())
        if not rewritten:
            debug.fallback = True
            debug.parse_error = "conflict decision did not include rewritten_memory"
            return candidate, debug

        return (
            EditOperation(
                op="replace",
                memory_id=decision.target_memory_id,
                content=rewritten,
                memory_type=candidate.memory_type,
                topic=candidate.topic,
                confidence=candidate.confidence,
                expires_at=candidate.expires_at,
                tags=candidate.tags,
                reason=f"conflict resolver replace: {decision.reason}",
                intent=candidate.intent,
                metadata=candidate.metadata,
                observed_at=candidate.observed_at,
            ),
            debug,
        )


def debug_to_dict(debug: ConflictResolverDebug) -> dict[str, Any]:
    return {
        "candidate": model_to_dict(debug.candidate),
        "related_memories": [model_to_dict(memory) for memory in debug.related_memories],
        "prompt": [model_to_dict(message) for message in debug.prompt],
        "raw": debug.raw,
        "parsed": debug.parsed,
        "decision": model_to_dict(debug.decision) if debug.decision else None,
        "fallback": debug.fallback,
        "parse_error": debug.parse_error,
    }


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
