"""Stage 2 of retrieval — deterministic memory pack assembly.

Consumes a :class:`CurrentQueryFrame` + :class:`MemoryRetrievalPlan` produced
by :class:`LLMRetrievalPlanner` and assembles a :class:`MemoryPack` by:

- pulling the basic_info claims listed in ``plan.selected_basic_info_claims``
  with their LLM-assigned application roles;
- pulling preference principles for ``plan.target_need_domains``;
- ranking historical NeedItems via embedder cosine on
  ``frame.inferred_need / frame.current_context`` plus tag overlap with
  ``frame.tags`` (NeedItem-mirror match);
- selecting events by seeding from the chosen claims' ``source_event_ids`` and
  filling up to ``plan.event_top_k`` via tag overlap on ``frame.tags``;
- slicing ``recent_status`` by ``plan.include_recent_status_fields``.

No LLM is invoked at this stage. The retriever is purely deterministic so the
plan fully drives "how to fetch memory" while the frame supplies semantic
content.
"""

from __future__ import annotations

from typing import Any

from src.llm.embedder import Embedder, cosine_similarity, get_default_embedder
from src.llm.llm import LLMClient
from src.memory.schemas import utc_now_iso
from src.retrieval.retrieval_planner import LLMRetrievalPlanner
from src.retrieval.schemas import (
    CurrentQueryFrame,
    MemoryPack,
    MemoryRetrievalPlan,
    SelectedBasicInfoClaim,
)


# ---------------------------------------------------------------------------
# Tag helpers
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
    union = len(a | b)
    if not union:
        return 0.0
    return float(len(a & b)) / float(union)


def _basic_info_claim_index(
    profile: dict[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Return ``claim_id -> (category, claim_dict)`` for every claim in basic_info."""
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    basic = profile.get("basic_info") if isinstance(profile, dict) else None
    if not isinstance(basic, dict):
        return out
    for sec, payload in basic.items():
        if not isinstance(payload, dict):
            continue
        for c in payload.get("claims") or []:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("claim_id") or "").strip()
            if cid:
                out[cid] = (str(sec), c)
    return out


# ---------------------------------------------------------------------------
# Channel-level resolvers
# ---------------------------------------------------------------------------


def _resolve_selected_claims(
    plan: MemoryRetrievalPlan,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    index = _basic_info_claim_index(profile)
    out: list[dict[str, Any]] = []
    for sel in plan.selected_basic_info_claims:
        if not isinstance(sel, dict):
            continue
        cid = str(sel.get("claim_id") or "").strip()
        role = str(sel.get("assigned_role") or "background_context").strip()
        entry = index.get(cid)
        if entry is None:
            continue
        section, claim = entry
        out.append(
            SelectedBasicInfoClaim(
                claim_id=cid,
                category=section,
                content=str(claim.get("content") or ""),
                claim_type=str(claim.get("claim_type") or ""),
                tags=list(claim.get("tags") or []),
                source_event_ids=list(claim.get("source_event_ids") or []),
                assigned_role=role,
                confidence=claim.get("confidence"),
                status=str(claim.get("status") or "active"),
            ).to_dict()
        )
    return out


def _resolve_preference_principles(
    plan: MemoryRetrievalPlan,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    if not plan.target_need_domains:
        return []
    domains = set(plan.target_need_domains)
    prefs = profile.get("need_preferences") or []
    if not isinstance(prefs, list):
        return []
    out: list[dict[str, Any]] = []
    for row in prefs:
        if not isinstance(row, dict):
            continue
        nd = str(row.get("need_domain") or "").strip()
        if nd not in domains:
            continue
        principle = str(row.get("preference_principle") or "").strip()
        if not principle:
            continue
        out.append(
            {
                "cluster_id": str(row.get("cluster_id") or "").strip(),
                "need_domain": nd,
                "preference_principle": principle,
            }
        )
    return out


def _rank_need_items(
    frame: CurrentQueryFrame,
    domains: set[str],
    need_items: list[dict[str, Any]],
    embedder: Embedder,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not need_items:
        return []

    q_need = embedder.embed_text((frame.inferred_need or " ").strip() or " ")
    q_ctx = embedder.embed_text((frame.current_context or " ").strip() or " ")
    tag_set = _normalize_tag_set(frame.tags)

    need_texts = [str(it.get("inferred_need") or "").strip() or " " for it in need_items]
    ctx_texts = [str(it.get("context") or "").strip() or " " for it in need_items]
    v_need = embedder.embed_batch(need_texts)
    v_ctx = embedder.embed_batch(ctx_texts)

    scored: list[tuple[float, dict[str, Any]]] = []
    for it, vn, vc in zip(need_items, v_need, v_ctx):
        sim_n = cosine_similarity(q_need, vn)
        sim_c = cosine_similarity(q_ctx, vc)
        item_tags = _normalize_tag_set(it.get("related_tags"))
        j = _tag_jaccard(tag_set, item_tags)
        domain_bonus = 0.0
        if domains and str(it.get("need_domain") or "").strip() in domains:
            domain_bonus = 0.15
        score = 0.40 * sim_n + 0.30 * sim_c + 0.20 * j + domain_bonus
        scored.append((score, it))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:top_k]]


def _select_relevant_events(
    frame: CurrentQueryFrame,
    selected_claims: list[dict[str, Any]],
    ranked_needs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not events:
        return []

    by_id: dict[str, dict[str, Any]] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        eid = str(ev.get("event_id") or "").strip()
        if eid:
            by_id[eid] = ev

    seed: list[str] = []
    seen_seed: set[str] = set()

    def _push(eid: str) -> None:
        s = eid.strip()
        if s and s not in seen_seed:
            seen_seed.add(s)
            seed.append(s)

    for c in selected_claims:
        for eid in c.get("source_event_ids") or []:
            _push(str(eid))
    for it in ranked_needs:
        for eid in it.get("context_event_ids") or []:
            _push(str(eid))

    picked: list[dict[str, Any]] = []
    picked_ids: set[str] = set()
    for eid in seed:
        ev = by_id.get(eid)
        if ev and eid not in picked_ids:
            picked_ids.add(eid)
            picked.append(ev)
            if len(picked) >= limit:
                return picked

    tag_set = _normalize_tag_set(frame.tags)
    aux: list[tuple[float, dict[str, Any]]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        eid = str(ev.get("event_id") or "").strip()
        if not eid or eid in picked_ids:
            continue
        ev_tags = _normalize_tag_set(ev.get("tags"))
        j = _tag_jaccard(tag_set, ev_tags)
        risk_bonus = 0.15 if "safety_risk" in ev_tags else 0.0
        aux.append((j + risk_bonus, ev))
    aux.sort(key=lambda x: -x[0])
    for _, ev in aux:
        eid = str(ev.get("event_id") or "").strip()
        if eid in picked_ids:
            continue
        picked_ids.add(eid)
        picked.append(ev)
        if len(picked) >= limit:
            break
    return picked[:limit]


def _slice_recent_status(
    plan: MemoryRetrievalPlan,
    profile: dict[str, Any],
) -> dict[str, Any]:
    if not plan.include_recent_status_fields:
        return {}
    rs = profile.get("recent_status") or {}
    if not isinstance(rs, dict):
        return {}
    out: dict[str, Any] = {}
    for f in plan.include_recent_status_fields:
        if f in rs:
            out[f] = rs[f]
    return out


# ---------------------------------------------------------------------------
# LayeredRetriever
# ---------------------------------------------------------------------------


class LayeredRetriever:
    """Compose a :class:`MemoryPack` from a frame + plan + profile."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        planner: LLMRetrievalPlanner | None = None,
    ) -> None:
        self._embedder = embedder
        self._planner = planner

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_default_embedder()
        return self._embedder

    @property
    def planner(self) -> LLMRetrievalPlanner:
        if self._planner is None:
            self._planner = LLMRetrievalPlanner()
        return self._planner

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
    ) -> MemoryPack:
        planner = LLMRetrievalPlanner(client=llm) if llm is not None else self.planner
        frame, plan = planner.plan(user_query, dialogue_context, profile)

        return self.assemble_pack(
            session_id=session_id,
            turn_id=turn_id,
            user_query=user_query,
            frame=frame,
            plan=plan,
            profile=profile,
            need_solution_items=need_solution_items,
            events=events,
        )

    def assemble_pack(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_query: str,
        frame: CurrentQueryFrame,
        plan: MemoryRetrievalPlan,
        profile: dict[str, Any],
        need_solution_items: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> MemoryPack:
        selected_claims = _resolve_selected_claims(plan, profile)
        principles = _resolve_preference_principles(plan, profile)
        ranked_needs = _rank_need_items(
            frame,
            set(plan.target_need_domains),
            need_solution_items,
            self.embedder,
            top_k=plan.need_top_k,
        )
        relevant_events = _select_relevant_events(
            frame,
            selected_claims,
            ranked_needs,
            events,
            limit=plan.event_top_k,
        )
        rs_slice = _slice_recent_status(plan, profile)

        return MemoryPack(
            session_id=session_id,
            turn_id=turn_id,
            timestamp=utc_now_iso(),
            user_query=user_query,
            current_query=frame,
            plan=plan,
            selected_basic_info_claims=selected_claims,
            preference_principles=principles,
            relevant_needs=ranked_needs,
            relevant_events=relevant_events,
            recent_status_slice=rs_slice,
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


_DEFAULT_RETRIEVER: LayeredRetriever | None = None


def _get_default_retriever() -> LayeredRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = LayeredRetriever()
    return _DEFAULT_RETRIEVER


def reset_default_layered_retriever() -> None:
    """Drop the cached default retriever (useful in tests)."""
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
) -> MemoryPack:
    """Build a :class:`MemoryPack` via the default :class:`LayeredRetriever`."""
    if embedder is not None:
        retr = LayeredRetriever(embedder=embedder)
    else:
        retr = _get_default_retriever()
    return retr.build_memory_pack(
        session_id=session_id,
        turn_id=turn_id,
        user_query=user_query,
        dialogue_context=dialogue_context,
        profile=profile,
        need_solution_items=need_solution_items,
        events=events,
        llm=llm,
    )
