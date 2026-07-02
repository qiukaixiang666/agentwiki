from __future__ import annotations

import re

from .schemas import PrivacyDecision


_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("private_key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
    ("api_key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|[A-Za-z0-9_]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,})\b")),
    ("password", re.compile(r"\b(password|passwd|pwd|secret|token)\s*[:=]\s*\S+", re.I)),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("china_id", re.compile(r"\b\d{17}[\dXx]\b")),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[- ]?)?(?:\d{3,4}[- ]?){2,3}\d{3,4}\b")),
]

_HIGH_RISK = {"private_key", "api_key", "password", "credit_card", "ssn", "china_id"}


def check_privacy(text: str, reject_sensitive: bool = True) -> PrivacyDecision:
    categories: list[str] = []
    compact = text.strip()
    for name, pattern in _PATTERNS:
        if pattern.search(compact):
            categories.append(name)

    if not categories:
        return PrivacyDecision(is_sensitive=False, action="allow", categories=[], reason="")

    action = "reject" if reject_sensitive or any(c in _HIGH_RISK for c in categories) else "mark"
    reason = "Input contains sensitive data categories: " + ", ".join(sorted(set(categories)))
    return PrivacyDecision(
        is_sensitive=True,
        action=action,
        categories=sorted(set(categories)),
        reason=reason,
    )


def refusal_message(decision: PrivacyDecision) -> str:
    categories = ", ".join(decision.categories) if decision.categories else "sensitive data"
    return (
        "This message appears to contain sensitive information "
        f"({categories}). I will not embed it, send it to the Wiki memory flow, "
        "or store it. Please remove the sensitive parts and try again."
    )
