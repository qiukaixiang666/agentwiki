from __future__ import annotations

import json
import re

from .config import settings
from .llm_client import LLMClient
from .schemas import ChatMessage, MemoryRecord, UserCard
from .utils import atomic_write_text, model_to_dict, utc_now


USER_CARD_SYSTEM = """You update a compact evidence profile card for LongMemEval-style long-history QA.
Return JSON only, with this shape:
{"profile_text":"80-150 Chinese characters","used_memory_ids":["memory ids"]}

Rules:
- Combine the previous user card with selected high-value active memories.
- Preserve previous card information only when it is still supported by active memories.
- Summarize stable user facts/preferences and recurring themes, plus important dated evidence that may help answer future dataset questions.
- Remove information that is contradicted, superseded, deleted, expired, or merely temporary.
- Include concrete activities, preferences, plans, assistant recommendations, and notable counts only when supported by active memories.
- Write concise Chinese text. Do not mention memory ids in profile_text."""


class UserCardManager:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.path = settings.data_dir / "user_card.json"

    def load(self) -> UserCard:
        if not self.path.exists():
            return UserCard()
        try:
            return UserCard(**json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return UserCard()

    def save(self, card: UserCard) -> None:
        atomic_write_text(
            self.path,
            json.dumps(model_to_dict(card), ensure_ascii=False, indent=2, sort_keys=True),
        )

    async def refresh(self, memories: list[MemoryRecord]) -> UserCard:
        previous = self.load()
        if not memories:
            previous.updated_at = utc_now()
            self.save(previous)
            return previous

        messages = [
            ChatMessage(role="system", content=USER_CARD_SYSTEM),
            ChatMessage(
                role="user",
                content="\n".join(
                    [
                        "Previous User Card:",
                        previous.profile_text or "(none)",
                        "",
                        "Selected high-value active memories:",
                        _render_memories(memories),
                    ]
                ),
            ),
        ]
        raw = await self.llm.complete(messages, json_mode=True)
        try:
            payload = _parse_json_object(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {
                "profile_text": previous.profile_text,
                "used_memory_ids": previous.source_memory_ids,
            }
        used_ids = [
            memory_id
            for memory_id in payload.get("used_memory_ids", [])
            if any(memory.id == memory_id for memory in memories)
        ]
        card = UserCard(
            profile_text=str(payload.get("profile_text", "")).strip(),
            source_memory_ids=used_ids,
            updated_at=utc_now(),
        )
        self.save(card)
        return card


def _render_memories(memories: list[MemoryRecord]) -> str:
    lines = []
    for memory in memories:
        strength = memory.memory_strength if memory.memory_strength is not None else 0.0
        lines.append(
            f"- [{memory.id}] type={memory.memory_type} topic={memory.topic} "
            f"confidence={memory.confidence:.2f} strength={strength:.2f}: {memory.content}"
        )
    return "\n".join(lines)


def _parse_json_object(text: str):
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
