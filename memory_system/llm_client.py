from __future__ import annotations

import json
from typing import Any

from .config import settings
from .schemas import ChatMessage
from .utils import model_to_dict


class LLMConfigurationError(RuntimeError):
    pass


class LLMClient:
    async def complete(self, messages: list[ChatMessage], json_mode: bool = False) -> str:
        if settings.llm_provider == "mock":
            return self._mock_complete(messages, json_mode=json_mode)
        if not settings.llm_api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY or DEEPSEEK_API_KEY is required unless LLM_PROVIDER=mock."
            )

        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": [model_to_dict(message) for message in messages],
            "temperature": settings.llm_temperature,
        }
        if settings.llm_top_p is not None:
            kwargs["top_p"] = settings.llm_top_p
        if settings.llm_extra_body_json:
            kwargs["extra_body"] = json.loads(settings.llm_extra_body_json)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _mock_complete(self, messages: list[ChatMessage], json_mode: bool = False) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if json_mode:
            first_system = next((m.content for m in messages if m.role == "system"), "")
            if "harmful-content safety classifier" in first_system.lower():
                return json.dumps(_mock_safety_decision(last_user))
            if "memory-retrieval queries" in first_system.lower():
                return json.dumps(_mock_retrieval_query(last_user))
            remembered = _extract_mock_memory(last_user)
            if remembered:
                return json.dumps(
                    {
                        "operations": [
                            {
                                "op": "add",
                                "content": remembered,
                                "tags": ["mock"],
                                "intent": (
                                    "The user is refining the long-memory system behavior "
                                    "to emphasize privacy-preserving retrieval."
                                ),
                                "reason": "mock durable memory rewrite",
                            }
                        ],
                        "rationale": "mock rewrote one durable memory",
                    }
                )
            return json.dumps({"operations": [], "rationale": "mock found no durable memory"})
        return "Mock reply: " + (last_user.strip() or "I am ready.")


def _extract_mock_memory(text: str) -> str | None:
    recent_dialog = _extract_section(text, "Recent dialog:", "\n\nuser:")
    latest_user = _extract_after(text, "\n\nuser:") or _extract_after(text, "\nuser:") or text
    combined = f"{recent_dialog}\n{latest_user}".lower()
    latest_lower = latest_user.lower()

    if (
        any(
            term in combined
            for term in [
                "long memory system",
                "memory system",
                "\u8bb0\u5fc6\u7cfb\u7edf",
            ]
        )
        and any(term in latest_lower for term in ["privacy", "private", "\u9690\u79c1"])
        and any(
            term in latest_lower
            for term in ["prefer", "want", "\u5e0c\u671b", "\u66f4\u91cd\u89c6"]
        )
    ):
        return "The user wants the long-memory system to prioritize privacy-preserving retrieval."
    return None


def _mock_retrieval_query(text: str) -> dict[str, str]:
    recent_dialog = _extract_section(text, "Recent dialog:", "\n\nlatest user input:")
    latest_user = _extract_after(text, "\n\nlatest user input:") or text
    combined = f"{recent_dialog}\n{latest_user}".lower()
    latest_lower = latest_user.lower()

    if (
        any(
            term in combined
            for term in [
                "long memory system",
                "memory system",
                "\u8bb0\u5fc6\u7cfb\u7edf",
            ]
        )
        and any(term in latest_lower for term in ["privacy", "private", "\u9690\u79c1"])
    ):
        return {
            "query": "long-memory system privacy-preserving retrieval preferences",
            "intent": "Retrieve memories about the user's preferred privacy behavior for the long-memory system.",
            "reason": "mock resolved contextual reference to the long-memory system",
        }

    if any(term in latest_lower for term in ["python", "example", "sample"]):
        return {
            "query": "user preference for concise Python examples",
            "intent": "Retrieve the user's preferences about Python example style.",
            "reason": "mock rewrote a Python-example request for semantic memory retrieval",
        }

    return {
        "query": "contextual user preferences and project memory relevant to the current request",
        "intent": "Retrieve generally relevant non-sensitive memories for the current request.",
        "reason": "mock generic retrieval rewrite",
    }


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        return text[start:].strip()
    return text[start:end].strip()


def _extract_after(text: str, marker: str) -> str:
    idx = text.rfind(marker)
    if idx < 0:
        return ""
    tail = text[idx + len(marker) :].strip()
    assistant_idx = tail.find("\nassistant:")
    if assistant_idx >= 0:
        tail = tail[:assistant_idx].strip()
    return tail


def _mock_safety_decision(text: str) -> dict[str, Any]:
    lowered = text.lower()
    rules = [
        ("weapon", ["build a bomb", "make a bomb", "explosive", "weapon"]),
        ("cyber_abuse", ["steal password", "phishing", "malware", "ransomware"]),
        ("fraud", ["fake id", "credit card fraud", "launder money"]),
        ("self_harm", ["kill myself", "suicide instructions"]),
        ("violence", ["how to poison", "hide a body"]),
    ]
    categories = [category for category, terms in rules if any(term in lowered for term in terms)]
    if categories:
        return {
            "is_harmful": True,
            "action": "refuse",
            "categories": categories,
            "reason": "mock safety classifier matched harmful request terms",
        }
    return {"is_harmful": False, "action": "allow", "categories": [], "reason": ""}
