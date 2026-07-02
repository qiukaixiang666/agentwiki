from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .llm_client import LLMClient
from .schemas import ChatMessage, SafetyDecision


SAFETY_SYSTEM = """You are a harmful-content safety classifier.
Return JSON only, with this shape:
{"is_harmful":true|false,"action":"allow|refuse","categories":["short labels"],"reason":"brief reason"}

Refuse content that requests, plans, enables, or meaningfully facilitates harm, including violence, weapons, self-harm, cyber abuse, fraud, credential theft, evading law enforcement, illegal drug production, or other dangerous wrongdoing.
Allow benign, educational, defensive, high-level, or safety-seeking content.
When uncertain, choose the safer classification."""


class LLMSafetyChecker:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def check(self, user_input: str) -> SafetyDecision:
        messages = [
            ChatMessage(role="system", content=SAFETY_SYSTEM),
            ChatMessage(role="user", content=user_input),
        ]
        raw = await self.llm.complete(messages, json_mode=True)
        payload = _parse_json_object(raw)
        decision = _coerce_safety_decision(payload)
        if decision.is_harmful:
            decision.action = "refuse"
        return decision


def harmful_refusal_message(decision: SafetyDecision) -> str:
    category_text = ", ".join(decision.categories) if decision.categories else "harmful content"
    return (
        "I can't help with that request because it appears to involve harmful content "
        f"({category_text}). I also will not store it in memory."
    )


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


def _coerce_safety_decision(payload: dict) -> SafetyDecision:
    try:
        return SafetyDecision(**payload)
    except ValidationError:
        return _fallback_safety_decision(payload)


def _fallback_safety_decision(payload: dict) -> SafetyDecision:
    text = json.dumps(payload, ensure_ascii=False).lower()
    harmful_terms = {
        "violence": ["weapon", "bomb", "explosive", "poison", "kill", "hide a body"],
        "cyber_abuse": ["malware", "phishing", "steal password", "credential theft"],
        "fraud": ["fake id", "credit card fraud", "launder money"],
        "self_harm": ["suicide instructions", "kill myself"],
        "drug_production": ["illegal drug production"],
    }
    categories = [
        category
        for category, terms in harmful_terms.items()
        if any(term in text for term in terms)
    ]
    if categories:
        return SafetyDecision(
            is_harmful=True,
            action="refuse",
            categories=categories,
            reason="Fallback safety classifier matched harmful terms after malformed LLM JSON.",
        )
    return SafetyDecision(
        is_harmful=False,
        action="allow",
        categories=[],
        reason="Allowed by fallback after malformed safety-classifier JSON.",
    )
