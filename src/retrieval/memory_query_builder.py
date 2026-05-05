"""Memory query planning for retrieval-time context packing.

Two builders are provided:

- :class:`RuleMemoryQueryBuilder` — lightweight keyword heuristics over the
  user query and ``recent_status`` narrative fields; no network.
- :class:`LLMMemoryQueryBuilder` — LLM-driven structured planning with prompts
  under ``src/prompts/retrieval/memory_query/``.

Both expose the same surface:

    build(user_query, dialogue_context, profile) -> dict[str, Any]

Return dict has key ``\"query\"`` (:class:`~src.retrieval.schemas.MemoryQuery`).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.llm.llm import LLMClient, get_default_llm_client
from src.llm.prompt_loader import load_prompt
from src.memory.ontology import BASIC_INFO_CATEGORIES
from src.memory.update.need_solution_extractor import RuleNeedSolutionExtractor
from src.retrieval.schemas import MemoryQuery, RelevantClaim

_logger = logging.getLogger(__name__)

# Keys on the dict returned by ``RuleMemoryQueryBuilder.build`` /
# ``LLMMemoryQueryBuilder.build`` / :func:`build_memory_query`.
MEMORY_QUERY_BUILD_QUERY_KEY = "query"


def _memory_query_build_dict(query: MemoryQuery) -> dict[str, Any]:
    return {MEMORY_QUERY_BUILD_QUERY_KEY: query}


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

RECENT_FIELD_TO_CATEGORY: dict[str, str] = {
    "health_status": "health",
    "self_management_status": "health",
    "mental_status": "health",
    "family_social_status": "family",
    "interest_changes": "leisure",
    "risk_flags": "health",
}

ALLOWED_USE_ROLES: frozenset[str] = frozenset(
    {"context", "constraint", "preference", "care_context", "risk_relevant"}
)

ALLOWED_CATEGORIES: frozenset[str] = frozenset(BASIC_INFO_CATEGORIES)


def _vocab_str(values: tuple[str, ...] | frozenset[str]) -> str:
    if isinstance(values, frozenset):
        return ", ".join(sorted(values))
    return ", ".join(values)


# ---------------------------------------------------------------------------
# Loaded system prompts
# ---------------------------------------------------------------------------

MEMORY_QUERY_SYSTEM: str = load_prompt(
    "retrieval/memory_query/system",
    basic_info_categories=_vocab_str(BASIC_INFO_CATEGORIES),
    allowed_use_roles=_vocab_str(ALLOWED_USE_ROLES),
)


# ---------------------------------------------------------------------------
# Helpers for building per-call user prompts
# ---------------------------------------------------------------------------


def _basic_info_claims_payload(profile: dict[str, Any]) -> dict[str, Any]:
    """Per-section claim lists (claim_id, content, claim_type, source_event_ids) for the LLM."""
    out: dict[str, Any] = {}
    basic = profile.get("basic_info") or {}
    if not isinstance(basic, dict):
        return {k: [] for k in BASIC_INFO_CATEGORIES}
    for key in BASIC_INFO_CATEGORIES:
        sec = basic.get(key)
        rows: list[dict[str, Any]] = []
        if isinstance(sec, dict):
            claims_raw = sec.get("claims")
            if isinstance(claims_raw, list):
                for c in claims_raw:
                    if not isinstance(c, dict):
                        continue
                    cid = str(c.get("claim_id") or "").strip()
                    if not cid:
                        continue
                    rows.append(
                        {
                            "claim_id": cid,
                            "content": str(c.get("content") or "").strip(),
                            "claim_type": str(c.get("claim_type") or "").strip(),
                        }
                    )
        out[key] = rows
    return out


def build_memory_query_prompt(
    *,
    user_query: str,
    dialogue_context: str,
    basic_info_summaries: dict[str, str],
    basic_info_claims: dict[str, Any],
    recent_status: dict[str, Any],
) -> str:
    """Render the user message for the memory-query LLM call."""
    return load_prompt(
        "retrieval/memory_query/user",
        user_query=user_query.strip() or "(空)",
        dialogue_context=(dialogue_context or "").strip() or "(无)",
        basic_info_summaries_json=json.dumps(basic_info_summaries, ensure_ascii=False),
        basic_info_claims_json=json.dumps(basic_info_claims, ensure_ascii=False),
        recent_status_json=json.dumps(recent_status, ensure_ascii=False),
    )


def _claim_type_to_use_role(claim_type: str) -> str:
    ct = (claim_type or "").strip()
    if ct == "long_term_constraint":
        return "constraint"
    if ct == "long_term_preference":
        return "preference"
    if ct == "care_context":
        return "care_context"
    if ct == "recurring_pattern":
        return "preference"
    return "context"


def _find_claim_dict(profile: dict[str, Any], claim_id: str) -> tuple[str, dict[str, Any]] | None:
    if not claim_id:
        return None
    basic = profile.get("basic_info") or {}
    if not isinstance(basic, dict):
        return None
    for cat in BASIC_INFO_CATEGORIES:
        sec = basic.get(cat)
        if not isinstance(sec, dict):
            continue
        for c in sec.get("claims") or []:
            if isinstance(c, dict) and str(c.get("claim_id") or "").strip() == claim_id:
                return cat, c
    return None


def _relevant_claim_from_claim_id(profile: dict[str, Any], claim_id: str) -> RelevantClaim | None:
    found = _find_claim_dict(profile, claim_id)
    if not found:
        return None
    cat, c = found
    content = str(c.get("content") or "").strip()
    if not content:
        return None
    role = _claim_type_to_use_role(str(c.get("claim_type") or ""))
    sids = c.get("source_event_ids")
    ids: list[str] = (
        [str(x).strip() for x in sids if str(x).strip()] if isinstance(sids, list) else []
    )
    return RelevantClaim(
        category=cat,
        content=content[:800],
        use_role=role,
        source_event_ids=ids,
    )


def _default_recent_status_slices(recent_status: dict[str, Any]) -> list[RelevantClaim]:
    out: list[RelevantClaim] = []
    fse = recent_status.get("field_source_event_ids") or {}
    if not isinstance(fse, dict):
        fse = {}
    for key in ("health_status", "self_management_status", "mental_status", "family_social_status"):
        content = str(recent_status.get(key) or "").strip()
        if not content:
            continue
        raw_ids = fse.get(key)
        ids = (
            [str(x).strip() for x in raw_ids if str(x).strip()]
            if isinstance(raw_ids, list)
            else []
        )
        cat = RECENT_FIELD_TO_CATEGORY.get(key, "health")
        out.append(
            RelevantClaim(
                category=cat,
                content=content[:500],
                use_role="context",
                source_event_ids=ids,
            )
        )
    for key in ("interest_changes", "risk_flags"):
        val = recent_status.get(key)
        parts: list[str] = []
        if isinstance(val, list):
            parts = [str(x).strip() for x in val if str(x).strip()]
        elif isinstance(val, str) and val.strip():
            parts = [val.strip()]
        if not parts:
            continue
        content = "; ".join(parts)[:500]
        role = "risk_relevant" if key == "risk_flags" else "context"
        raw_ids = fse.get(key)
        ids = (
            [str(x).strip() for x in raw_ids if str(x).strip()]
            if isinstance(raw_ids, list)
            else []
        )
        cat = RECENT_FIELD_TO_CATEGORY.get(key, "health")
        out.append(
            RelevantClaim(
                category=cat,
                content=content,
                use_role=role,
                source_event_ids=ids,
            )
        )
    return out


def _keyword_tags_from_query(text: str) -> list[str]:
    """Loosely mirror event-tag vocabulary for offline fallback ranking."""
    t = text or ""
    tags: list[str] = []
    if any(x in t for x in ("吃", "饮食", "餐", "血糖", "糖")):
        tags.append("diet")
    if any(x in t for x in ("运动", "散步", "锻炼")):
        tags.append("activity")
    if any(x in t for x in ("睡", "失眠")):
        tags.append("sleep")
    if any(x in t for x in ("药", "胰岛素", "用药")):
        tags.extend(["medication", "medical_visit"])
    if any(x in t for x in ("担心", "焦虑", "难过", "孤独")):
        tags.append("emotion")
    return tags[:12] if tags else ["other"]


def _looks_like_flat_memory_query_payload(d: dict[str, Any]) -> bool:
    return bool(
        isinstance(d.get("current_need"), str)
        and isinstance(d.get("current_context"), str)
        and isinstance(d.get("current_tags"), list)
    )


def _parse_relevant_claim_row(raw: Any, profile: dict[str, Any]) -> RelevantClaim | None:
    if not isinstance(raw, dict):
        return None
    cat = str(raw.get("category") or "").strip()
    fn = str(raw.get("field_name") or "").strip()
    if not cat and fn:
        cat = RECENT_FIELD_TO_CATEGORY.get(fn, "")
    if cat not in ALLOWED_CATEGORIES:
        return None
    content = str(raw.get("content") or "").strip()
    if not content:
        return None
    role = str(raw.get("use_role") or "context").strip()
    if role not in ALLOWED_USE_ROLES:
        role = "context"
    sids = raw.get("source_event_ids")
    ids: list[str] = (
        [str(x).strip() for x in sids if str(x).strip()] if isinstance(sids, list) else []
    )
    ref = str(raw.get("basic_info_claim_id") or raw.get("claim_id") or "").strip()
    if ref:
        resolved = _relevant_claim_from_claim_id(profile, ref)
        if resolved:
            if not ids:
                ids = list(resolved.source_event_ids)
            return RelevantClaim(
                category=resolved.category,
                content=resolved.content[:800],
                use_role=role,
                source_event_ids=ids or list(resolved.source_event_ids),
            )
    return RelevantClaim(
        category=cat,
        content=content[:800],
        use_role=role,
        source_event_ids=ids,
    )


def _dedupe_relevant_claims(claims: list[RelevantClaim]) -> list[RelevantClaim]:
    seen: set[tuple[str, str]] = set()
    out: list[RelevantClaim] = []
    for c in claims:
        key = (c.category, (c.content or "")[:240])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


class RuleMemoryQueryBuilder:
    """Deterministic planner when no LLM is available or the LLM path fails."""

    def build(
        self,
        user_query: str,
        dialogue_context: str,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        recent_status = profile.get("recent_status") or {}
        if not isinstance(recent_status, dict):
            recent_status = {}
        rule = RuleNeedSolutionExtractor()
        need_pack = rule.infer_need(user_query, None, None)
        current_need = str(need_pack.get("primary_need") or "日常陪伴").strip()
        ctx_bits: list[str] = []
        if (dialogue_context or "").strip():
            ctx_bits.append(dialogue_context.strip()[:400])
        for key in ("health_status", "self_management_status", "mental_status", "family_social_status"):
            s = str(recent_status.get(key) or "").strip()
            if s:
                ctx_bits.append(s[:200])
        current_context = " ".join(ctx_bits).strip()[:600]
        tags_list = _keyword_tags_from_query(user_query)
        slices = _default_recent_status_slices(recent_status)
        return _memory_query_build_dict(
            MemoryQuery(
                current_need=current_need,
                current_context=current_context,
                current_tags=tags_list,
                relevant_claims=slices,
            ),
        )


# ---------------------------------------------------------------------------
# LLM-driven builder
# ---------------------------------------------------------------------------


class LLMMemoryQueryBuilder:
    """LLM-driven memory query planner; :meth:`build` returns a small result dict."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        fallback: RuleMemoryQueryBuilder | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or RuleMemoryQueryBuilder()

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_default_llm_client()
        return self._client

    def build(
        self,
        user_query: str,
        dialogue_context: str,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        recent_status = profile.get("recent_status") or {}
        if not isinstance(recent_status, dict):
            recent_status = {}
        summaries = self._basic_info_summaries(profile)
        claims_payload = _basic_info_claims_payload(profile)
        prompt = build_memory_query_prompt(
            user_query=user_query,
            dialogue_context=dialogue_context,
            basic_info_summaries=summaries,
            basic_info_claims=claims_payload,
            recent_status=recent_status,
        )
        try:
            response = self.client.generate_json(
                prompt,
                system_prompt=MEMORY_QUERY_SYSTEM,
            )
            if not isinstance(response, dict):
                raise ValueError("memory query response is not a dict")
            mq_raw = self._parse_memory_query_payload(response)
            query = self._build_memory_query_from_dict(mq_raw, profile, recent_status)
            return _memory_query_build_dict(query)
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "LLMMemoryQueryBuilder failed (%s); falling back to rule builder.",
                err,
            )
            return self._fallback.build(user_query, dialogue_context, profile)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _basic_info_summaries(profile: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        basic = profile.get("basic_info") or {}
        if not isinstance(basic, dict):
            return {k: "" for k in BASIC_INFO_CATEGORIES}
        for key in BASIC_INFO_CATEGORIES:
            section = basic.get(key)
            if isinstance(section, dict):
                out[key] = str(section.get("summary") or "").strip()
            else:
                out[key] = ""
        return out

    def _parse_memory_query_payload(self, response: dict[str, Any]) -> dict[str, Any]:
        nested = response.get("memory_query")
        if isinstance(nested, dict):
            return nested
        if _looks_like_flat_memory_query_payload(response):
            return response
        return {}

    def _build_memory_query_from_dict(
        self,
        mq: dict[str, Any],
        profile: dict[str, Any],
        recent_status: dict[str, Any],
    ) -> MemoryQuery:
        claims: list[RelevantClaim] = []
        for cid in mq.get("basic_info_claim_ids") or []:
            s = str(cid).strip()
            if not s:
                continue
            rc = _relevant_claim_from_claim_id(profile, s)
            if rc:
                claims.append(rc)
        for row in mq.get("relevant_claims") or []:
            pr = _parse_relevant_claim_row(row, profile)
            if pr:
                claims.append(pr)
        claims = _dedupe_relevant_claims(claims)
        if not claims:
            claims = _default_recent_status_slices(recent_status)
        tags_raw = mq.get("current_tags") or []
        tags = [str(t).strip() for t in tags_raw if str(t).strip()][:24]
        if not tags:
            tags = ["other"]
        return MemoryQuery(
            current_need=str(mq.get("current_need") or "").strip() or "日常陪伴",
            current_context=str(mq.get("current_context") or "").strip(),
            current_tags=tags,
            relevant_claims=claims,
        )


# ---------------------------------------------------------------------------
# Backward-compatible module API
# ---------------------------------------------------------------------------

_DEFAULT_BUILDER: LLMMemoryQueryBuilder | None = None


def _get_default_builder() -> LLMMemoryQueryBuilder:
    global _DEFAULT_BUILDER
    if _DEFAULT_BUILDER is None:
        _DEFAULT_BUILDER = LLMMemoryQueryBuilder()
    return _DEFAULT_BUILDER


def reset_default_memory_query_builder() -> None:
    """Drop the cached default builder (useful for tests)."""
    global _DEFAULT_BUILDER
    _DEFAULT_BUILDER = None


def build_memory_query(
    user_query: str,
    dialogue_context: str,
    profile: dict[str, Any],
    *,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Delegate to :class:`LLMMemoryQueryBuilder` (optional injected ``llm`` client)."""
    if llm is None:
        return _get_default_builder().build(
            user_query,
            dialogue_context,
            profile,
        )
    return LLMMemoryQueryBuilder(client=llm).build(
        user_query,
        dialogue_context,
        profile,
    )
