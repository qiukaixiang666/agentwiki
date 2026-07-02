from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class PrivacyDecision(BaseModel):
    is_sensitive: bool
    action: Literal["allow", "reject", "mark"] = "allow"
    categories: list[str] = Field(default_factory=list)
    reason: str = ""


class SafetyDecision(BaseModel):
    is_harmful: bool
    action: Literal["allow", "refuse"] = "allow"
    categories: list[str] = Field(default_factory=list)
    reason: str = ""


class RetrievalQuery(BaseModel):
    query: str
    intent: str = ""
    reason: str = ""


class MemoryRecord(BaseModel):
    id: str
    content: str
    memory_type: Literal["fact", "preference", "task", "relation", "event"] = "fact"
    topic: str = "general"
    confidence: float = 0.8
    status: Literal["active", "superseded", "expired", "deleted"] = "active"
    expires_at: str | None = None
    memory_strength: float | None = None
    decay_rate: float | None = None
    last_reinforced_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_session_id: str | None = None
    observed_at: str | None = None
    created_at: str
    updated_at: str
    revision: int = 1


class RetrievedMemory(BaseModel):
    id: str
    content: str
    memory_type: str = "fact"
    topic: str = "general"
    confidence: float = 0.8
    memory_strength: float = 0.8
    effective_strength: float = 0.8
    tags: list[str] = Field(default_factory=list)
    score: float
    updated_at: str
    observed_at: str | None = None


class EditOperation(BaseModel):
    op: Literal["add", "replace", "delete"]
    memory_id: str | None = None
    content: str | None = None
    memory_type: Literal["fact", "preference", "task", "relation", "event"] = "fact"
    topic: str = "general"
    confidence: float = 0.8
    expires_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    reason: str = ""
    intent: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    observed_at: str | None = None


class EditPlan(BaseModel):
    operations: list[EditOperation] = Field(default_factory=list)
    rationale: str = ""


class EditPlanDebug(BaseModel):
    raw: str = ""
    parsed: dict[str, Any] | None = None
    before_filter: list[EditOperation] = Field(default_factory=list)
    after_filter: list[EditOperation] = Field(default_factory=list)
    dropped: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    parse_error: str = ""


class QAPairEvidence(BaseModel):
    summary: str = ""
    raw_user: str = ""
    raw_assistant: str = ""
    date: str | None = None
    source_session_id: str | None = None
    entities: list[str] = Field(default_factory=list)
    evidence_items: list[str] = Field(default_factory=list)
    reason: str = ""


class MemoryTriage(BaseModel):
    category: str = "not_useful"
    memory_type: str = "fact"
    topic: str = "general"
    retrieval_query: str = ""
    should_attempt_edit: bool = True
    related_memory_policy: str = ""
    reason: str = ""


class ProfileUseDecision(BaseModel):
    usable: bool = False
    profile_aspect: str = ""
    reason: str = ""


class ChatRequest(BaseModel):
    user_input: str
    session_id: str | None = None
    recent_messages: list[ChatMessage] = Field(default_factory=list)
    top_k: int = 5
    async_memory_write: bool = False
    debug_prompt: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    blocked: bool = False
    privacy: PrivacyDecision
    safety: SafetyDecision | None = None
    retrieval_query: str | None = None
    recalled_memories: list[RetrievedMemory] = Field(default_factory=list)
    memory_update_status: str = "skipped"
    edit_operations: list[EditOperation] = Field(default_factory=list)
    prompt: list[ChatMessage] | None = None


class IngestSessionRequest(BaseModel):
    session_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    top_k: int = 5
    debug_prompt: bool = False


class IngestSessionResponse(BaseModel):
    session_id: str
    applied: int = 0
    skipped: int = 0
    edit_operations: list[EditOperation] = Field(default_factory=list)
    recalled_memories: list[RetrievedMemory] = Field(default_factory=list)
    retrieval_query: str | None = None
    prompt: list[ChatMessage] | None = None


class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    memories: list[RetrievedMemory]
    privacy: PrivacyDecision
    retrieval_query: str | None = None


class ManualEditRequest(BaseModel):
    operations: list[EditOperation]
    actor: str = "manual"


class ManualEditResponse(BaseModel):
    applied: int
    skipped: int
    memories: list[MemoryRecord]


class HealthResponse(BaseModel):
    ok: bool
    service: str
    details: dict[str, Any] = Field(default_factory=dict)


class UserCard(BaseModel):
    profile_text: str = ""
    source_memory_ids: list[str] = Field(default_factory=list)
    updated_at: str | None = None
