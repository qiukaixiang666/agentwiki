from __future__ import annotations

import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException

from .config import ensure_directories, settings
from .conflict_resolver import MemoryConflictResolver, debug_to_dict as conflict_debug_to_dict
from .embedding_client import EmbeddingClient
from .llm_client import LLMClient, LLMConfigurationError
from .mem0_style import Mem0AdditiveExtractor, debug_to_dict
from .privacy import check_privacy, refusal_message
from .prompts import build_chat_prompt
from .qa_agents import pair_user_assistant_messages
from .safety import LLMSafetyChecker, harmful_refusal_message
from .schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EditOperation,
    HealthResponse,
    IngestSessionRequest,
    IngestSessionResponse,
    ManualEditRequest,
    ManualEditResponse,
    SearchRequest,
    SearchResponse,
)
from .store import MemoryStore
from .time_utils import observed_at_from_messages, source_session_id_from_messages
from .user_card import UserCardManager
from .utils import append_jsonl, model_to_dict, sha256_text, utc_now

app = FastAPI(title="Memory Wiki API")
store = MemoryStore()
embedder = EmbeddingClient()
llm = LLMClient()
safety_checker = LLMSafetyChecker(llm)
user_cards = UserCardManager(llm)
mem0_extractor = Mem0AdditiveExtractor(llm)
conflict_resolver = MemoryConflictResolver(llm)


@app.on_event("startup")
def startup() -> None:
    ensure_directories()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    details: dict[str, Any] = {
        "memory_root": str(settings.memory_root),
        "records": len(store.list_memories(limit=1_000_000)),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "memory_extraction": "mem0_additive_with_conflict_resolver",
        "memory_retrieval": "mem0_hybrid_embedding_bm25_entity_temporal",
    }
    try:
        details["embedding"] = await embedder.health()
        ok = True
    except Exception as exc:
        details["embedding_error"] = str(exc)
        ok = False
    return HealthResponse(ok=ok, service="memory_api", details=details)


@app.post("/privacy/check")
async def privacy_check(payload: dict[str, str]) -> dict[str, Any]:
    text = payload.get("text", "")
    decision = check_privacy(text, reject_sensitive=settings.privacy_reject_sensitive)
    return model_to_dict(decision)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    decision = check_privacy(request.query, reject_sensitive=settings.privacy_reject_sensitive)
    if decision.is_sensitive:
        store.append_privacy_event(
            {
                "at": utc_now(),
                "action": "search_rejected",
                "hash": sha256_text(request.query),
                "categories": decision.categories,
            }
        )
        return SearchResponse(memories=[], privacy=decision, retrieval_query=None)
    retrieval_query = request.query
    vector = await embedder.embed_one(retrieval_query)
    memories = store.search_by_embedding(
        vector,
        top_k=request.top_k,
        query_text=retrieval_query,
    )
    return SearchResponse(
        memories=memories,
        privacy=decision,
        retrieval_query=retrieval_query,
    )


@app.get("/memories")
async def list_memories(limit: int = 100) -> dict[str, Any]:
    return {"memories": [model_to_dict(item) for item in store.list_memories(limit=limit)]}


@app.get("/user-card")
async def get_user_card() -> dict[str, Any]:
    return model_to_dict(user_cards.load())


@app.post("/user-card/refresh")
async def refresh_user_card(limit: int = 20) -> dict[str, Any]:
    memories = store.select_user_card_memories(limit=limit)
    card = await user_cards.refresh(memories)
    return model_to_dict(card)


@app.post("/memories/forget", response_model=ManualEditResponse)
async def forget_memory(payload: dict[str, Any]) -> ManualEditResponse:
    memory_id = payload.get("memory_id")
    query = payload.get("query")
    top_k = int(payload.get("top_k", 3))
    operations = []

    if memory_id:
        operations.append(
            {
                "op": "delete",
                "memory_id": str(memory_id),
                "reason": "user requested forgetting by memory id",
                "intent": "forget explicit memory id",
            }
        )
    elif query:
        decision = check_privacy(str(query), reject_sensitive=settings.privacy_reject_sensitive)
        if decision.is_sensitive:
            store.append_privacy_event(
                {
                    "at": utc_now(),
                    "action": "forget_rejected_sensitive_query",
                    "hash": sha256_text(str(query)),
                    "categories": decision.categories,
                }
            )
            return ManualEditResponse(applied=0, skipped=1, memories=[])
        retrieval_query = str(query)
        vector = await embedder.embed_one(retrieval_query)
        memories = store.search_by_embedding(
            vector,
            top_k=top_k,
            query_text=retrieval_query,
        )
        operations.extend(
            {
                "op": "delete",
                "memory_id": memory.id,
                "reason": f"user requested forgetting by query: {retrieval_query}",
                "intent": "forget semantically matched memories",
            }
            for memory in memories
        )
    else:
        raise HTTPException(status_code=400, detail="memory_id or query is required")

    edit_operations = [EditOperation(**operation) for operation in operations]
    applied, skipped, changed = store.apply_operations(
        edit_operations,
        {},
        actor="user_forget",
    )
    return ManualEditResponse(applied=applied, skipped=skipped, memories=changed)


@app.post("/wiki/edit", response_model=ManualEditResponse)
async def manual_edit(request: ManualEditRequest) -> ManualEditResponse:
    safe_operations = []
    for operation in request.operations:
        if operation.op in {"add", "replace"}:
            if not operation.content:
                continue
            decision = check_privacy(operation.content, reject_sensitive=True)
            if decision.is_sensitive:
                continue
        safe_operations.append(operation)

    embeddings_by_content = await _embed_operation_contents(safe_operations)
    applied, skipped, changed = store.apply_operations(
        safe_operations,
        embeddings_by_content,
        actor=request.actor,
    )
    if any(item.memory_type in {"preference", "fact", "relation"} for item in changed):
        await user_cards.refresh(store.select_user_card_memories())
    return ManualEditResponse(applied=applied, skipped=skipped, memories=changed)


@app.post("/sessions/ingest", response_model=IngestSessionResponse)
async def ingest_session(request: IngestSessionRequest) -> IngestSessionResponse:
    session_id = request.session_id or "session_" + uuid.uuid4().hex[:12]
    messages = [message for message in request.messages if message.role in {"user", "assistant"}]
    if not messages:
        return IngestSessionResponse(session_id=session_id)

    latest_user_input = next((message.content for message in reversed(messages) if message.role == "user"), "")
    retrieval_seed = "\n".join(f"{message.role}: {message.content}" for message in messages[-8:])
    privacy = check_privacy(retrieval_seed, reject_sensitive=settings.privacy_reject_sensitive)
    if privacy.is_sensitive:
        store.append_privacy_event(
            {
                "at": utc_now(),
                "action": "historical_ingest_rejected_privacy",
                "hash": sha256_text(retrieval_seed),
                "categories": privacy.categories,
            }
        )
        return IngestSessionResponse(session_id=session_id, skipped=len(messages))
    try:
        safety = await safety_checker.check(retrieval_seed)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if safety.is_harmful or safety.action == "refuse":
        store.append_privacy_event(
            {
                "at": utc_now(),
                "action": "historical_ingest_rejected_safety",
                "hash": sha256_text(retrieval_seed),
                "safety_categories": safety.categories,
            }
        )
        return IngestSessionResponse(session_id=session_id, skipped=len(messages))

    applied = 0
    skipped = 0
    all_operations: list[EditOperation] = []
    all_recalled = []
    retrieval_queries: list[str] = []
    debug_prompts: list[ChatMessage] = []
    should_refresh_user_card = False

    try:
        qa_pairs = pair_user_assistant_messages(messages)
        for user_message, assistant_message in qa_pairs:
            result = await _process_qa_pair_memory(
                user_message,
                assistant_message,
                session_id=session_id,
                top_k=request.top_k or settings.memory_top_k,
                actor="historical_session_ingest",
                disable_context=False,
                disable_profile_context=True,
            )
            applied += result["applied"]
            skipped += result["skipped"]
            all_operations.extend(result["operations"])
            all_recalled.extend(result["recalled"])
            if result["retrieval_query"]:
                retrieval_queries.append(result["retrieval_query"])
            if request.debug_prompt:
                debug_prompts.extend(result["debug_prompts"])
            should_refresh_user_card = should_refresh_user_card or result["profile_usable"]
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for message in messages:
        store.append_session_message(session_id, message.role, message.content)
    if should_refresh_user_card:
        await user_cards.refresh(store.select_user_card_memories())

    return IngestSessionResponse(
        session_id=session_id,
        applied=applied,
        skipped=skipped,
        edit_operations=all_operations,
        recalled_memories=all_recalled,
        retrieval_query=" | ".join(retrieval_queries) if retrieval_queries else None,
        prompt=debug_prompts if request.debug_prompt else None,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    session_id = request.session_id or "session_" + uuid.uuid4().hex[:12]
    privacy = check_privacy(request.user_input, reject_sensitive=settings.privacy_reject_sensitive)
    if privacy.is_sensitive:
        privacy.action = "mark"

    try:
        safety = await safety_checker.check(request.user_input)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if safety.is_harmful or safety.action == "refuse":
        block_reason = "chat_rejected_safety_with_privacy" if privacy.is_sensitive else "chat_rejected_safety"
        store.append_privacy_event(
            {
                "at": utc_now(),
                "action": block_reason,
                "hash": sha256_text(request.user_input),
                "categories": privacy.categories,
                "safety_categories": safety.categories,
            }
        )
        return ChatResponse(
            session_id=session_id,
            reply=harmful_refusal_message(safety),
            blocked=True,
            privacy=privacy,
            safety=safety,
            memory_update_status="blocked_by_safety",
        )

    if privacy.is_sensitive:
        store.append_privacy_event(
            {
                "at": utc_now(),
                "action": "chat_answered_with_privacy_filtered_memory",
                "hash": sha256_text(request.user_input),
                "categories": privacy.categories,
            }
        )
        top_k = request.top_k or settings.memory_top_k
        recent = _merge_recent_messages(session_id, request.recent_messages)
        retrieval_query = request.user_input
        recalled = await _retrieve_memories(retrieval_query, top_k=top_k)
        prompt = build_chat_prompt(
            recalled,
            recent,
            request.user_input,
            user_card=user_cards.load().profile_text,
        )
        try:
            reply = await llm.complete(prompt, json_mode=False)
        except LLMConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        store.append_session_message(session_id, "user", request.user_input)
        store.append_session_message(session_id, "assistant", reply)
        result = await _process_qa_pair_memory(
            ChatMessage(role="user", content=request.user_input),
            ChatMessage(role="assistant", content=reply),
            session_id=session_id,
            top_k=top_k,
            actor="mem0_additive_extractor_privacy_filtered",
            fallback_recalled=recalled,
            fallback_retrieval_query=retrieval_query,
            privacy_filtered=True,
            privacy_categories=privacy.categories,
        )
        if result["profile_usable"]:
            await user_cards.refresh(store.select_user_card_memories())
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            blocked=False,
            privacy=privacy,
            safety=safety,
            retrieval_query=retrieval_query,
            recalled_memories=recalled,
            memory_update_status="privacy_filtered_applied",
            edit_operations=result["operations"],
            prompt=None,
        )

    top_k = request.top_k or settings.memory_top_k
    recent = _merge_recent_messages(session_id, request.recent_messages)
    retrieval_query = request.user_input
    recalled = await _retrieve_memories(retrieval_query, top_k=top_k)
    prompt = build_chat_prompt(
        recalled,
        recent,
        request.user_input,
        user_card=user_cards.load().profile_text,
    )

    try:
        reply = await llm.complete(prompt, json_mode=False)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    store.append_session_message(session_id, "user", request.user_input)
    store.append_session_message(session_id, "assistant", reply)

    if request.async_memory_write:
        background_tasks.add_task(
            _process_qa_pair_memory,
            ChatMessage(role="user", content=request.user_input),
            ChatMessage(role="assistant", content=reply),
            session_id,
            top_k,
            "mem0_additive_extractor",
            None,
            None,
            True,
        )
        update_status = "scheduled"
        operations = []
    else:
        result = await _process_qa_pair_memory(
            ChatMessage(role="user", content=request.user_input),
            ChatMessage(role="assistant", content=reply),
            session_id=session_id,
            top_k=top_k,
            actor="mem0_additive_extractor",
            fallback_recalled=recalled,
            fallback_retrieval_query=retrieval_query,
        )
        if result["profile_usable"]:
            await user_cards.refresh(store.select_user_card_memories())
        operations = result["operations"]
        update_status = "applied"

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        blocked=False,
        privacy=privacy,
        safety=safety,
        retrieval_query=retrieval_query,
        recalled_memories=recalled,
        memory_update_status=update_status,
        edit_operations=operations,
        prompt=prompt if request.debug_prompt else None,
    )


def _merge_recent_messages(session_id: str, request_messages: list[ChatMessage]) -> list[ChatMessage]:
    stored = store.get_recent_messages(session_id, settings.recent_turns)
    merged = stored + request_messages
    return merged[-settings.recent_turns :]


async def _retrieve_memories(retrieval_query: str, top_k: int) -> list:
    query_vector = await embedder.embed_one(retrieval_query)
    return store.search_by_embedding(
        query_vector,
        top_k=top_k,
        query_text=retrieval_query,
    )


async def _embed_operation_contents(operations: list) -> dict[str, list[float]]:
    contents = [
        operation.content
        for operation in operations
        if operation.op in {"add", "replace"} and operation.content
    ]
    if not contents:
        return {}
    vectors = await embedder.embed(contents, normalize=True)
    return dict(zip(contents, vectors))


async def _process_qa_pair_memory(
    user_message: ChatMessage,
    assistant_message: ChatMessage | None,
    session_id: str,
    top_k: int,
    actor: str,
    fallback_recalled: list | None = None,
    fallback_retrieval_query: str | None = None,
    refresh_user_card: bool = False,
    privacy_filtered: bool = False,
    privacy_categories: list[str] | None = None,
    disable_context: bool = False,
    disable_profile_context: bool = False,
) -> dict[str, Any]:
    new_messages = [user_message]
    if assistant_message is not None:
        new_messages.append(assistant_message)
    observed_at = observed_at_from_messages(new_messages)
    source_session_id = source_session_id_from_messages(new_messages)
    initial_query = fallback_retrieval_query or _messages_to_query(new_messages)
    if disable_context:
        initial_recalled = []
    elif fallback_recalled is None:
        initial_recalled = await _retrieve_memories(initial_query, top_k=top_k)
    else:
        initial_recalled = fallback_recalled

    operations, extraction_debug = await mem0_extractor.extract_with_debug(
        new_messages,
        initial_recalled,
        last_messages=[] if disable_context or disable_profile_context else store.get_recent_messages(session_id, settings.recent_turns),
        profile_summary="" if disable_context or disable_profile_context else user_cards.load().profile_text,
        custom_instructions=_privacy_filtered_extraction_instruction(privacy_categories)
        if privacy_filtered
        else "",
    )
    if privacy_filtered:
        operations = _drop_sensitive_extracted_operations(operations)
        extraction_debug.operations = operations
    operations, conflict_debugs = await _resolve_operation_conflicts(operations)
    embeddings_by_content = await _embed_operation_contents(operations)
    applied, skipped, _changed = store.apply_operations(
        operations,
        embeddings_by_content,
        actor=actor,
        session_id=source_session_id or session_id,
    )
    profile_usable = _should_refresh_profile_from_operations(operations)
    if profile_usable and refresh_user_card:
        await user_cards.refresh(store.select_user_card_memories())
    _append_agent_debug(
        session_id,
        actor,
        user_message,
        assistant_message,
        debug_to_dict(extraction_debug),
        [conflict_debug_to_dict(debug) for debug in conflict_debugs],
        initial_query,
        initial_query,
        initial_recalled,
        applied=applied,
        skipped=skipped,
        skip_reason="" if operations else (extraction_debug.parse_error or "mem0 additive extractor produced no memories"),
        privacy_filtered=privacy_filtered,
        privacy_categories=privacy_categories or [],
    )
    return {
        "applied": applied,
        "skipped": skipped,
        "operations": operations,
        "recalled": initial_recalled,
        "retrieval_query": initial_query,
        "debug_prompts": extraction_debug.prompt,
        "profile_usable": profile_usable,
        "observed_at": observed_at,
    }


def _append_agent_debug(
    session_id: str,
    actor: str,
    user_message: ChatMessage,
    assistant_message: ChatMessage | None,
    mem0_debug: dict,
    conflict_debugs: list[dict],
    initial_query: str,
    retrieval_query: str,
    recalled: list,
    applied: int,
    skipped: int,
    skip_reason: str,
    privacy_filtered: bool = False,
    privacy_categories: list[str] | None = None,
) -> None:
    path = settings.memory_root / "runtime" / "agent_debug.jsonl"
    append_jsonl(
        path,
        {
            "at": utc_now(),
            "actor": actor,
            "session_id": session_id,
            "source": {
                "user": user_message.content,
                "assistant": assistant_message.content if assistant_message else "",
            },
            "initial_query": initial_query,
            "retrieval_query": retrieval_query,
            "recalled_memories": [model_to_dict(memory) for memory in recalled],
            "mem0_additive_extractor": mem0_debug,
            "memory_conflict_resolver": conflict_debugs,
            "result": {
                "applied": applied,
                "skipped": skipped,
                "skip_reason": skip_reason,
                "privacy_filtered": privacy_filtered,
                "privacy_categories": privacy_categories or [],
            },
        },
    )


def _messages_to_query(messages: list[ChatMessage]) -> str:
    return " ".join(message.content for message in messages if message.content).strip()


async def _resolve_operation_conflicts(
    operations: list[EditOperation],
) -> tuple[list[EditOperation], list]:
    resolved: list[EditOperation] = []
    debug_rows = []
    for operation in operations:
        if operation.op != "add" or not operation.content:
            resolved.append(operation)
            continue
        query_vector = await embedder.embed_one(operation.content)
        related = store.search_by_embedding(
            query_vector,
            top_k=min(settings.memory_top_k, 5),
            query_text=operation.content,
            threshold=0.1,
            reinforce=False,
        )
        resolved_operation, debug = await conflict_resolver.resolve_with_debug(operation, related)
        resolved.append(resolved_operation)
        debug_rows.append(debug)
    return resolved, debug_rows


def _should_refresh_profile_from_operations(operations: list[EditOperation]) -> bool:
    return any(
        operation.op in {"add", "replace"}
        and operation.memory_type in {"preference", "relation"}
        for operation in operations
    )


def _privacy_filtered_extraction_instruction(categories: list[str] | None) -> str:
    category_text = ", ".join(categories or []) or "sensitive_or_private_information"
    return (
        "This chat turn was marked sensitive by the privacy checker. During memory extraction, "
        f"do not save any content belonging to these sensitive categories: {category_text}. "
        "Ignore secrets, credentials, contact details, precise identifiers, private medical/legal/financial details, "
        "and any other sensitive spans. However, still extract useful non-sensitive memories from the same user query "
        "and assistant answer when they are separable, such as harmless preferences, project goals, high-level interests, "
        "non-sensitive plans, or concrete assistant recommendations. Do not mention that sensitive content was filtered."
    )


def _drop_sensitive_extracted_operations(operations: list[EditOperation]) -> list[EditOperation]:
    safe_operations: list[EditOperation] = []
    for operation in operations:
        if operation.op not in {"add", "replace"} or not operation.content:
            safe_operations.append(operation)
            continue
        decision = check_privacy(operation.content, reject_sensitive=True)
        if decision.is_sensitive:
            continue
        safe_operations.append(operation)
    return safe_operations


def main() -> None:
    import uvicorn

    uvicorn.run(
        "memory_system.api:app",
        host=settings.memory_api_host,
        port=settings.memory_api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
