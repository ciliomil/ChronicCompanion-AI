"""Layered retrieval over mid memory: planned MemoryQuery, ranked needs, events, pack.

Orchestrates:

1. **Memory query** — :class:`~src.retrieval.memory_query_builder.LLMMemoryQueryBuilder`
   (rule fallback) produces a :class:`~src.retrieval.schemas.MemoryQuery`.
2. **Need recall** — cosine similarity on ``current_need`` / ``current_context`` plus
   tag Jaccard over ``related_tags``.
3. **Cluster preferences** — ``preference_principle`` from ``need_preferences`` by
   ``cluster_id`` on ranked items.
4. **Event bundle** — provenance ids from ``MemoryQuery.relevant_claims`` (each
   claim’s ``source_event_ids``, including those resolved from selected basic_info
   claims) and need ``context_event_ids``, then tag-overlap fill.

Backend: any embedder satisfying :class:`src.llm.embedder.Embedder`.
"""

from __future__ import annotations

from typing import Any

from src.llm.embedder import Embedder, cosine_similarity, get_default_embedder
from src.llm.llm import LLMClient
from src.memory.schemas import utc_now_iso
from src.retrieval.memory_query_builder import (
    LLMMemoryQueryBuilder,
    MEMORY_QUERY_BUILD_QUERY_KEY,
)
from src.retrieval.schemas import MemoryPack, RelevantClaim

# ---------------------------------------------------------------------------
# Tag / embedding helpers
# ---------------------------------------------------------------------------


def _normalize_tag_set(tags: Any) -> set[str]:
    if tags is None:
        return set()
    if isinstance(tags, str):
        raw = [tags]
    elif isinstance(tags, list):
        raw = tags
    else:
        return set()
    return {str(t).strip().lower() for t in raw if str(t).strip()}


def _tag_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


# ---------------------------------------------------------------------------
# Need ranking
# ---------------------------------------------------------------------------


def _rank_need_items(
    current_need: str,
    current_context: str,
    current_tags: list[str],
    items: list[dict[str, Any]],
    embedder: Embedder,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if not items:
        return []
    q_need = embedder.embed_text((current_need or " ").strip() or " ")
    q_ctx = embedder.embed_text((current_context or " ").strip() or " ")
    tag_set = _normalize_tag_set(current_tags)

    need_texts: list[str] = []
    ctx_texts: list[str] = []
    for it in items:
        need_texts.append(str(it.get("inferred_need") or "").strip() or " ")
        ctx_texts.append(str(it.get("context") or "").strip() or " ")
    v_need = embedder.embed_batch(need_texts)
    v_ctx = embedder.embed_batch(ctx_texts)

    scored: list[tuple[float, dict[str, Any]]] = []
    for it, vn, vc in zip(items, v_need, v_ctx):
        sim_n = cosine_similarity(q_need, vn)
        sim_c = cosine_similarity(q_ctx, vc)
        item_tags = _normalize_tag_set(it.get("related_tags"))
        j = _tag_jaccard(tag_set, item_tags)
        score = 0.45 * sim_n + 0.35 * sim_c + 0.20 * j
        scored.append((score, it))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:top_k]]


def _preference_principles_for_items(
    need_preferences: list[dict[str, Any]],
    ranked_items: list[dict[str, Any]],
) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for it in ranked_items:
        cid = it.get("cluster_id")
        if not cid:
            continue
        s = str(cid).strip()
        if s and s not in seen:
            seen.add(s)
            order.append(s)
    by_cluster: dict[str, str] = {}
    for row in need_preferences:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("cluster_id") or "").strip()
        if not cid:
            continue
        prin = str(row.get("preference_principle") or "").strip()
        if prin:
            by_cluster[cid] = prin
    out: list[str] = []
    seen_prin: set[str] = set()
    for cid in order:
        p = by_cluster.get(cid, "")
        if p and p not in seen_prin:
            seen_prin.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------


def _ordered_event_ids(
    relevant_claims: list[RelevantClaim],
    need_items: list[dict[str, Any]],
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    def push(eid: str) -> None:
        s = eid.strip()
        if s and s not in seen:
            seen.add(s)
            ids.append(s)

    for sl in relevant_claims:
        for eid in sl.source_event_ids:
            push(str(eid))
    for it in need_items:
        for eid in it.get("context_event_ids") or []:
            push(str(eid))
    return ids


def _events_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        eid = str(ev.get("event_id") or "").strip()
        if eid:
            idx[eid] = ev
    return idx


def _select_relevant_events(
    events: list[dict[str, Any]],
    want_ids: list[str],
    current_tags: list[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_id = _events_by_id(events)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for eid in want_ids:
        ev = by_id.get(eid)
        if ev and eid not in seen:
            seen.add(eid)
            picked.append(ev)
        if len(picked) >= limit:
            return picked

    tag_set = _normalize_tag_set(current_tags)
    scored_aux: list[tuple[float, dict[str, Any]]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        eid = str(ev.get("event_id") or "").strip()
        if not eid or eid in seen:
            continue
        ev_tags = _normalize_tag_set(ev.get("tags"))
        j = _tag_jaccard(tag_set, ev_tags)
        risk_bonus = 0.15 if "safety_risk" in ev_tags else 0.0
        scored_aux.append((j + risk_bonus, ev))
    scored_aux.sort(key=lambda x: -x[0])
    for _, ev in scored_aux:
        eid = str(ev.get("event_id") or "").strip()
        if eid in seen:
            continue
        seen.add(eid)
        picked.append(ev)
        if len(picked) >= limit:
            break
    return picked[:limit]


# ---------------------------------------------------------------------------
# Layered retriever
# ---------------------------------------------------------------------------


class LayeredRetriever:
    """Compose :class:`~src.retrieval.schemas.MemoryPack` for generation."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        memory_query_builder: LLMMemoryQueryBuilder | None = None,
    ) -> None:
        self._embedder = embedder
        self._memory_query_builder = memory_query_builder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_default_embedder()
        return self._embedder

    @property
    def memory_query_builder(self) -> LLMMemoryQueryBuilder:
        if self._memory_query_builder is None:
            self._memory_query_builder = LLMMemoryQueryBuilder()
        return self._memory_query_builder

    def build_memory_pack(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_query: str,
        dialogue_context: str,
        profile: dict[str, Any],
        need_solution_items: list[dict[str, Any]],
        events: list[dict[str, Any]],
        llm: LLMClient | None = None,
        top_k_needs: int = 8,
        top_k_events: int = 14,
    ) -> MemoryPack:
        recent = profile.get("recent_status") or {}
        if not isinstance(recent, dict):
            recent = {}
        builder = LLMMemoryQueryBuilder(client=llm) if llm is not None else self.memory_query_builder
        planned = builder.build(user_query, dialogue_context, profile)
        query = planned[MEMORY_QUERY_BUILD_QUERY_KEY]

        ranked = _rank_need_items(
            query.current_need,
            query.current_context,
            query.current_tags,
            need_solution_items,
            self.embedder,
            top_k=top_k_needs,
        )
        prefs = profile.get("need_preferences") or []
        if not isinstance(prefs, list):
            prefs = []
        principles = _preference_principles_for_items(prefs, ranked)
        event_ids = _ordered_event_ids(query.relevant_claims, ranked)
        rel_events = _select_relevant_events(
            events,
            event_ids,
            query.current_tags,
            limit=top_k_events,
        )
        return MemoryPack(
            session_id=session_id,
            turn_id=turn_id,
            timestamp=utc_now_iso(),
            user_query=user_query,
            current_query=query,
            preference_principles=principles,
            relevant_needs=ranked,
            relevant_events=rel_events,
            recent_status=recent,
            safety_notes=[],
        )


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

_DEFAULT_RETRIEVER: LayeredRetriever | None = None


def _get_default_retriever() -> LayeredRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = LayeredRetriever()
    return _DEFAULT_RETRIEVER


def reset_default_layered_retriever() -> None:
    """Drop the cached default retriever (useful for tests)."""
    global _DEFAULT_RETRIEVER
    _DEFAULT_RETRIEVER = None


def build_memory_pack(
    *,
    session_id: str,
    turn_id: str,
    user_query: str,
    dialogue_context: str,
    profile: dict[str, Any],
    need_solution_items: list[dict[str, Any]],
    events: list[dict[str, Any]],
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    top_k_needs: int = 8,
    top_k_events: int = 14,
) -> MemoryPack:
    """Build a :class:`MemoryPack` using the process-wide default :class:`LayeredRetriever`."""
    retr = _get_default_retriever()
    if embedder is not None:
        retr = LayeredRetriever(
            embedder=embedder,
            memory_query_builder=retr.memory_query_builder if llm is None else None,
        )
    return retr.build_memory_pack(
        session_id=session_id,
        turn_id=turn_id,
        user_query=user_query,
        dialogue_context=dialogue_context,
        profile=profile,
        need_solution_items=need_solution_items,
        events=events,
        llm=llm,
        top_k_needs=top_k_needs,
        top_k_events=top_k_events,
    )


# ---------------------------------------------------------------------------
# Backward-compatible bundle
# ---------------------------------------------------------------------------


def retrieve_layered_context(
    need_result: dict[str, Any],
    events: list[dict[str, Any]],
    need_solution_items: list[dict[str, Any]],
    raw_turns: list[dict[str, Any]],
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Legacy dict bundle; prefer :func:`build_memory_pack` for new code."""
    profile = need_result.get("profile")
    if not isinstance(profile, dict):
        profile = {}
    user_query = str(need_result.get("user_query") or need_result.get("primary_need") or "").strip()
    if not user_query:
        user_query = "日常陪伴"
    pack = build_memory_pack(
        session_id=str(need_result.get("session_id") or "session"),
        turn_id=str(need_result.get("turn_id") or "turn"),
        user_query=user_query,
        dialogue_context=str(need_result.get("dialogue_context") or ""),
        profile=profile,
        need_solution_items=need_solution_items,
        events=events,
        llm=llm,
        embedder=embedder,
    )
    source_turn_ids = {
        tid
        for it in pack.relevant_needs
        for tid in (it.get("source_turn_ids") or [])
        if tid
    }
    raw_evidence = [t for t in raw_turns if t.get("turn_id") in source_turn_ids][-4:]
    return {
        "primary_need": pack.current_query.current_need,
        "matched_need_items": pack.relevant_needs,
        "matched_events": pack.relevant_events,
        "raw_evidence": raw_evidence,
        "memory_pack": pack.to_dict(),
    }
