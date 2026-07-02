from __future__ import annotations

import re
from datetime import date, datetime, timezone

from .schemas import ChatMessage


_LONGMEMEVAL_PREFIX_RE = re.compile(
    r"^\s*\[LongMemEval\s+(?P<meta>[^\]]+)\]\s*\n?",
    flags=re.I,
)


def extract_longmemeval_metadata(text: str) -> dict[str, str | None]:
    """Extract source metadata from a LongMemEval message prefix."""
    match = _LONGMEMEVAL_PREFIX_RE.match(text or "")
    if not match:
        return {"observed_at": None, "source_session_id": None}
    meta = match.group("meta")
    date_match = re.search(r"\bdate=([^;]+)", meta)
    session_match = re.search(r"\bsession_id=([^;]+)", meta)
    return {
        "observed_at": normalize_observed_at(date_match.group(1).strip()) if date_match else None,
        "source_session_id": session_match.group(1).strip() if session_match else None,
    }


def strip_longmemeval_prefix(text: str) -> str:
    return _LONGMEMEVAL_PREFIX_RE.sub("", text or "", count=1).strip()


def strip_longmemeval_prefixes(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [
        ChatMessage(role=message.role, content=strip_longmemeval_prefix(message.content))
        for message in messages
    ]


def observed_at_from_messages(messages: list[ChatMessage]) -> str | None:
    for message in messages:
        observed_at = extract_longmemeval_metadata(message.content).get("observed_at")
        if observed_at:
            return observed_at
    return None


def source_session_id_from_messages(messages: list[ChatMessage]) -> str | None:
    for message in messages:
        source_session_id = extract_longmemeval_metadata(message.content).get("source_session_id")
        if source_session_id:
            return source_session_id
    return None


def normalize_observed_at(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    date_match = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", text)
    if date_match:
        return f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    text = text.replace("/", "-")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def parse_observed_date(value: str | None) -> date | None:
    normalized = normalize_observed_at(value)
    if not normalized:
        return None
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def utc_now_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()
