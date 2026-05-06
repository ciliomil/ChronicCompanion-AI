"""Stage 1 of retrieval — produce the frame + plan in a single LLM call.

The planner's contract is::

    plan(user_query, dialogue_context, profile)
        -> (CurrentQueryFrame, MemoryRetrievalPlan)

The frame carries semantic content (NeedItem-mirror fields + reply-control
labels). The plan carries operational knobs (selected claim ids with assigned
roles, top-k budgets, recent_status field selection, safety flag).

Both pieces are produced by a single LLM call so the LLM can reason
"frame first, then plan" internally without two round trips. The prompt is
explicit about that ordering. On any LLM / parsing failure, a deterministic
:class:`RuleRetrievalPlanner` keeps the pipeline alive with conservative
defaults (no claims, no recall, recent_status passthrough).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.llm.llm import LLMClient, get_default_llm_client
from src.llm.prompt_loader import load_prompt
from src.memory.ontology import BASIC_INFO_CATEGORIES, MEMORY_TAGS, NEED_DOMAINS
from src.retrieval.ontology import (
    APPLICATION_ROLES,
    APPLICATION_ROLE_DESCRIPTIONS,
    INTENT_TYPES,
    MEDICAL_RELEVANCE_LEVELS,
    RECENT_STATUS_FIELD_KEYS,
    RISK_LEVELS,
)
from src.retrieval.schemas import CurrentQueryFrame, MemoryRetrievalPlan

_logger = logging.getLogger(__name__)


# Frozen lookups for fast validation.
_INTENT_SET = frozenset(INTENT_TYPES)
_REL_SET = frozenset(MEDICAL_RELEVANCE_LEVELS)
_RISK_SET = frozenset(RISK_LEVELS)
_NEED_DOMAIN_SET = frozenset(NEED_DOMAINS.keys())
_MEMORY_TAG_SET = frozenset(MEMORY_TAGS)
_APPLICATION_ROLE_SET = frozenset(APPLICATION_ROLES)
_RECENT_STATUS_FIELD_SET = frozenset(RECENT_STATUS_FIELD_KEYS)

_MAX_FRAME_TAGS = 5
_MAX_RISK_TRIGGERS = 3
_MAX_SELECTED_CLAIMS = 8
_MAX_NEED_TOP_K = 8
_MAX_EVENT_TOP_K = 14


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _vocab_str(values: tuple[str, ...] | frozenset[str]) -> str:
    if isinstance(values, frozenset):
        return ", ".join(sorted(values))
    return ", ".join(values)


def _format_descriptions(d: dict[str, str]) -> str:
    return "\n".join(f"  - {k}：{v}" for k, v in d.items())


def _format_need_domains() -> str:
    return "\n".join(f"  - {k}：{v}" for k, v in NEED_DOMAINS.items())


RETRIEVAL_PLAN_SYSTEM: str = load_prompt(
    "retrieval/retrieval_plan/system",
    need_domains=_format_need_domains(),
    intent_types=_vocab_str(INTENT_TYPES),
    application_role_descriptions=_format_descriptions(APPLICATION_ROLE_DESCRIPTIONS),
    recent_status_fields=_vocab_str(RECENT_STATUS_FIELD_KEYS),
)


def build_retrieval_plan_prompt(
    *,
    user_query: str,
    dialogue_context: str,
    basic_info: dict[str, Any],
    recent_status: dict[str, Any],
) -> str:
    return load_prompt(
        "retrieval/retrieval_plan/user",
        user_query=user_query.strip() or "(空)",
        dialogue_context=(dialogue_context or "").strip() or "(无)",
        basic_info_json=json.dumps(
            _compact_basic_info(basic_info), ensure_ascii=False
        ),
        recent_status_json=json.dumps(recent_status or {}, ensure_ascii=False),
    )


def _compact_basic_info(basic_info: dict[str, Any]) -> dict[str, Any]:
    """Render basic_info into per-section summary + active/uncertain claims."""
    out: dict[str, Any] = {}
    for sec in BASIC_INFO_CATEGORIES:
        s = basic_info.get(sec) if isinstance(basic_info, dict) else None
        if not isinstance(s, dict):
            out[sec] = {"summary": "", "claims": []}
            continue
        summary = str(s.get("summary") or "").strip()
        claims: list[dict[str, Any]] = []
        for c in s.get("claims") or []:
            if not isinstance(c, dict):
                continue
            status = str(c.get("status") or "").strip()
            if status not in ("active", "uncertain"):
                continue
            cid = str(c.get("claim_id") or "").strip()
            if not cid:
                continue
            claims.append(
                {
                    "claim_id": cid,
                    "claim_type": str(c.get("claim_type") or ""),
                    "content": str(c.get("content") or "")[:300],
                    "tags": list(c.get("tags") or []),
                    "source_event_ids": list(c.get("source_event_ids") or []),
                    "status": status,
                }
            )
        out[sec] = {"summary": summary, "claims": claims}
    return out


def _all_active_claim_ids(basic_info: dict[str, Any]) -> set[str]:
    """active+uncertain claim ids across all sections (validates plan output)."""
    ids: set[str] = set()
    if not isinstance(basic_info, dict):
        return ids
    for sec in BASIC_INFO_CATEGORIES:
        s = basic_info.get(sec) or {}
        if not isinstance(s, dict):
            continue
        for c in s.get("claims") or []:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("claim_id") or "").strip()
            status = str(c.get("status") or "").strip()
            if cid and status in ("active", "uncertain"):
                ids.add(cid)
    return ids


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_frame(raw: Any, *, user_query: str) -> CurrentQueryFrame:
    if not isinstance(raw, dict):
        raw = {}

    intent_type = str(raw.get("intent_type") or "").strip()
    if intent_type not in _INTENT_SET:
        intent_type = "casual_chat"

    medical_relevance = str(raw.get("medical_relevance") or "").strip()
    if medical_relevance not in _REL_SET:
        medical_relevance = "none"

    risk_level = str(raw.get("risk_level") or "").strip()
    if risk_level not in _RISK_SET:
        risk_level = "normal"

    need_domain = str(raw.get("need_domain") or "").strip()
    if need_domain not in _NEED_DOMAIN_SET:
        need_domain = "other"

    tags = _coerce_tag_list(raw.get("tags"), allowed=_MEMORY_TAG_SET, limit=_MAX_FRAME_TAGS)

    risk_triggers: list[str] = []
    seen_rt: set[str] = set()
    rt_raw = raw.get("risk_triggers") if isinstance(raw.get("risk_triggers"), list) else []
    for t in rt_raw:
        s = str(t).strip()
        if s and s not in seen_rt:
            seen_rt.add(s)
            risk_triggers.append(s)
        if len(risk_triggers) >= _MAX_RISK_TRIGGERS:
            break

    return CurrentQueryFrame(
        user_query=str(raw.get("user_query") or user_query or "").strip(),
        query_summary=str(raw.get("query_summary") or "").strip(),
        inferred_need=str(raw.get("inferred_need") or "").strip() or "日常陪伴",
        need_domain=need_domain,
        need_object=str(raw.get("need_object") or "").strip(),
        tags=tags,
        current_context=str(raw.get("current_context") or "").strip(),
        intent_type=intent_type,
        medical_relevance=medical_relevance,
        risk_level=risk_level,
        risk_triggers=risk_triggers,
    )


def _parse_plan(
    raw: Any,
    *,
    valid_claim_ids: set[str],
    frame: CurrentQueryFrame,
) -> MemoryRetrievalPlan:
    if not isinstance(raw, dict):
        raw = {}

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for r in raw.get("selected_basic_info_claims") or []:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("claim_id") or "").strip()
        if not cid or cid not in valid_claim_ids or cid in seen_ids:
            continue
        role = str(r.get("assigned_role") or "").strip()
        if role not in _APPLICATION_ROLE_SET:
            role = "background_context"
        seen_ids.add(cid)
        selected.append({"claim_id": cid, "assigned_role": role})
        if len(selected) >= _MAX_SELECTED_CLAIMS:
            break

    target_domains: list[str] = []
    seen_d: set[str] = set()
    for d in raw.get("target_need_domains") or []:
        s = str(d).strip()
        if s in _NEED_DOMAIN_SET and s not in seen_d:
            seen_d.add(s)
            target_domains.append(s)
    if not target_domains and frame.need_domain in _NEED_DOMAIN_SET and frame.need_domain != "other":
        target_domains = [frame.need_domain]

    need_top_k = _coerce_int(raw.get("need_top_k"), lo=0, hi=_MAX_NEED_TOP_K)
    event_top_k = _coerce_int(raw.get("event_top_k"), lo=0, hi=_MAX_EVENT_TOP_K)

    rs_fields: list[str] = []
    seen_f: set[str] = set()
    for f in raw.get("include_recent_status_fields") or []:
        s = str(f).strip()
        if s in _RECENT_STATUS_FIELD_SET and s not in seen_f:
            seen_f.add(s)
            rs_fields.append(s)

    safety_sensitive = bool(raw.get("safety_sensitive"))
    if frame.risk_level == "urgent" or frame.risk_triggers:
        safety_sensitive = True

    return MemoryRetrievalPlan(
        selected_basic_info_claims=selected,
        target_need_domains=target_domains,
        need_top_k=need_top_k,
        event_top_k=event_top_k,
        include_recent_status_fields=rs_fields,
        safety_sensitive=safety_sensitive,
    )


def _coerce_tag_list(value: Any, *, allowed: frozenset[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in value:
        s = str(v).strip()
        if not s or s in seen:
            continue
        if s not in allowed:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _coerce_int(value: Any, *, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


# ---------------------------------------------------------------------------
# Rule fallback — deterministic, no network.
# ---------------------------------------------------------------------------


class RuleRetrievalPlanner:
    """Conservative no-LLM fallback.

    Returns an empty plan (no basic_info claim selection, no need/event recall)
    plus a recent_status passthrough for any nonempty fields. Keeps the
    pipeline alive without inventing semantic content.
    """

    def plan(
        self,
        user_query: str,
        dialogue_context: str,
        profile: dict[str, Any],
    ) -> tuple[CurrentQueryFrame, MemoryRetrievalPlan]:
        del dialogue_context
        recent_status = profile.get("recent_status") if isinstance(profile, dict) else None
        if not isinstance(recent_status, dict):
            recent_status = {}

        frame = CurrentQueryFrame(
            user_query=user_query.strip(),
            query_summary=user_query.strip()[:80],
            inferred_need="日常陪伴",
            need_domain="other",
            need_object="",
            tags=[],
            current_context="",
            intent_type="casual_chat",
            medical_relevance="none",
            risk_level="normal",
            risk_triggers=[],
        )

        nonempty_fields: list[str] = []
        for f in RECENT_STATUS_FIELD_KEYS:
            v = recent_status.get(f)
            if isinstance(v, str) and v.strip():
                nonempty_fields.append(f)
            elif isinstance(v, list) and v:
                nonempty_fields.append(f)

        plan = MemoryRetrievalPlan(
            selected_basic_info_claims=[],
            target_need_domains=[],
            need_top_k=0,
            event_top_k=0,
            include_recent_status_fields=nonempty_fields,
            safety_sensitive=False,
        )
        return frame, plan


# ---------------------------------------------------------------------------
# LLM-driven planner.
# ---------------------------------------------------------------------------


class LLMRetrievalPlanner:
    """LLM-driven planner with deterministic fallback."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        fallback: RuleRetrievalPlanner | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or RuleRetrievalPlanner()

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_default_llm_client()
        return self._client

    def plan(
        self,
        user_query: str,
        dialogue_context: str,
        profile: dict[str, Any],
    ) -> tuple[CurrentQueryFrame, MemoryRetrievalPlan]:
        basic_info = profile.get("basic_info") if isinstance(profile, dict) else None
        if not isinstance(basic_info, dict):
            basic_info = {}
        recent_status = profile.get("recent_status") if isinstance(profile, dict) else None
        if not isinstance(recent_status, dict):
            recent_status = {}

        try:
            prompt = build_retrieval_plan_prompt(
                user_query=user_query,
                dialogue_context=dialogue_context,
                basic_info=basic_info,
                recent_status=recent_status,
            )
            response = self.client.generate_json(
                prompt,
                system_prompt=RETRIEVAL_PLAN_SYSTEM,
            )
            if not isinstance(response, dict):
                raise ValueError("retrieval_plan response is not a dict")

            frame = _parse_frame(
                response.get("current_query_frame"),
                user_query=user_query,
            )
            plan = _parse_plan(
                response.get("memory_retrieval_plan"),
                valid_claim_ids=_all_active_claim_ids(basic_info),
                frame=frame,
            )
            return frame, plan
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "LLMRetrievalPlanner.plan failed (%s); falling back to rule planner.",
                err,
            )
            return self._fallback.plan(user_query, dialogue_context, profile)


# ---------------------------------------------------------------------------
# Module-level convenience.
# ---------------------------------------------------------------------------


_DEFAULT_PLANNER: LLMRetrievalPlanner | None = None


def _get_default_planner() -> LLMRetrievalPlanner:
    global _DEFAULT_PLANNER
    if _DEFAULT_PLANNER is None:
        _DEFAULT_PLANNER = LLMRetrievalPlanner()
    return _DEFAULT_PLANNER


def reset_default_retrieval_planner() -> None:
    """Drop the cached default planner (useful in tests)."""
    global _DEFAULT_PLANNER
    _DEFAULT_PLANNER = None


def build_retrieval_plan(
    user_query: str,
    dialogue_context: str,
    profile: dict[str, Any],
    *,
    llm: LLMClient | None = None,
) -> tuple[CurrentQueryFrame, MemoryRetrievalPlan]:
    """Module-level entry point (delegates to default or injected planner)."""
    if llm is None:
        return _get_default_planner().plan(user_query, dialogue_context, profile)
    return LLMRetrievalPlanner(client=llm).plan(user_query, dialogue_context, profile)
