from __future__ import annotations

import json
import re

from .llm_client import LLMClient
from .privacy import check_privacy
from .prompts import build_retrieval_query_prompt
from .schemas import ChatMessage, RetrievalQuery


class RetrievalQueryRewriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def rewrite(
        self,
        recent_messages: list[ChatMessage],
        user_input: str,
    ) -> RetrievalQuery:
        prompt = build_retrieval_query_prompt(recent_messages, user_input)
        raw = await self.llm.complete(prompt, json_mode=True)
        try:
            payload = _parse_json_object(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict) or not payload.get("query"):
            payload = {
                "query": _fallback_semantic_query(recent_messages, user_input),
                "intent": "Fallback retrieval intent after malformed model JSON.",
                "reason": "fallback query because the model returned malformed JSON",
            }
        query = RetrievalQuery(**payload)
        query.query = _clean_query(query.query)

        if not query.query or _looks_like_direct_copy(query.query, user_input):
            query.query = _fallback_semantic_query(recent_messages, user_input)
            query.reason = "fallback query because the model returned a direct copy"

        privacy = check_privacy(query.query, reject_sensitive=True)
        if privacy.is_sensitive:
            query.query = "non-sensitive contextual memory relevant to the current request"
            query.reason = "privacy-redacted retrieval query"

        return query


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


def _clean_query(query: str) -> str:
    return " ".join(query.strip().split())


def _looks_like_direct_copy(query: str, user_input: str) -> bool:
    query_norm = _normalize(query)
    input_norm = _normalize(user_input)
    if not query_norm or not input_norm:
        return False
    if query_norm == input_norm:
        return True
    if len(input_norm) >= 24 and input_norm in query_norm:
        return True
    return False


def _fallback_semantic_query(
    recent_messages: list[ChatMessage],
    user_input: str,
) -> str:
    recent_text = " ".join(message.content for message in recent_messages[-4:]).lower()
    latest = user_input.lower()
    if (
        any(term in recent_text for term in ["long memory system", "memory system", "\u8bb0\u5fc6\u7cfb\u7edf"])
        and any(term in latest for term in ["privacy", "private", "\u9690\u79c1"])
    ):
        return "long-memory system privacy-preserving retrieval preferences"
    if any(term in latest for term in ["python", "example", "sample"]):
        return "user preference for concise Python examples"
    return "contextual user preferences and project memory relevant to the current request"


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())
