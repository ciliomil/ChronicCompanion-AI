"""Session-end basic_info update via a three-stage LLM pipeline.

Stages
------
1. **Propose** — Given this session's :class:`EventItem` and :class:`NeedItem`
   evidence, the LLM emits *candidate* long-term claims (no awareness of the
   existing profile). Tags are constrained to a 1~3 conservative subset of the
   union of cited evidence's tags.
2. **Consolidate** — Given the existing active / uncertain claims and the
   candidates, the LLM emits a per-candidate ``action`` in
   ``{add, update, supersede, ignore}``. Code applies the actions to obtain the
   merged per-section claim list (existing claims that aren't referenced are
   preserved as-is).
3. **Summarize** — Given the merged active / uncertain claims, the LLM emits a
   1~3 sentence Chinese summary and ``summary_source_claim_ids`` per section.

Two updaters are exposed:

- :class:`RuleBasicInfoUpdater` — deterministic no-op fallback (keeps previous
  basic_info untouched).
- :class:`LLMBasicInfoUpdater` — LLM-driven pipeline above with the rule-based
  fallback used on any per-stage parse / API failure.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.llm.llm import LLMClient, get_default_llm_client
from src.memory.ontology import BASIC_INFO_CLAIM_TYPES, BASIC_INFO_STATUS, MEMORY_TAGS
from src.memory.update.prompts import (
    BASIC_INFO_CONSOLIDATE_SYSTEM,
    BASIC_INFO_PROPOSE_SYSTEM,
    BASIC_INFO_SUMMARIZE_SYSTEM,
    build_basic_info_consolidate_prompt,
    build_basic_info_propose_prompt,
    build_basic_info_summarize_prompt,
)
from src.utils.clock import Clock, RealClock

_logger = logging.getLogger(__name__)


BASIC_INFO_SECTION_KEYS: tuple[str, ...] = (
    "medical_care",
    "family",
    "health",
    "leisure",
)

_CLAIM_TYPE_SET = frozenset(BASIC_INFO_CLAIM_TYPES)
_STATUS_SET = frozenset(BASIC_INFO_STATUS)
_MEMORY_TAG_SET = frozenset(MEMORY_TAGS)

_DEFAULT_CLAIM_TYPE = "stable_life_background"
_DEFAULT_STATUS = "active"

_VALID_ACTIONS: frozenset[str] = frozenset({"add", "update", "supersede", "ignore"})
# Stage 1 candidates can only be active or uncertain; supersede comes from
# stage 2 actions, never from the proposer.
_PROPOSE_STATUS_SET: frozenset[str] = frozenset({"active", "uncertain"})


# ---------------------------------------------------------------------------
# Public normalization helper
# ---------------------------------------------------------------------------


def normalize_basic_info(
    raw: dict[str, Any] | None,
    *,
    clock: Clock,
) -> dict[str, Any]:
    """Return a valid basic_info dict with exactly the four section keys."""
    now = clock.now_iso()

    root: dict[str, Any]
    if isinstance(raw, dict) and isinstance(raw.get("basic_info"), dict):
        root = dict(raw["basic_info"])
    elif isinstance(raw, dict):
        root = dict(raw)
    else:
        root = {}

    out: dict[str, Any] = {}
    for section_key in BASIC_INFO_SECTION_KEYS:
        sec = root.get(section_key)
        if not isinstance(sec, dict):
            out[section_key] = _empty_section(now)
            continue

        claims_out: list[dict[str, Any]] = []
        claims_raw = sec.get("claims")
        if isinstance(claims_raw, list):
            for raw_claim in claims_raw:
                sanitized = _sanitize_claim(
                    raw_claim,
                    section_key=section_key,
                    clock_iso=now,
                    strict_evidence=False,
                )
                if sanitized is not None:
                    claims_out.append(sanitized)

        out[section_key] = {
            "summary": str(sec.get("summary", "") or ""),
            "claims": claims_out,
            "summary_source_claim_ids": _coerce_str_list(
                sec.get("summary_source_claim_ids"),
            ),
            "updated_at": str(sec.get("updated_at", "") or now),
        }

    return _finalize_basic_info(out, clock_iso=now)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


class RuleBasicInfoUpdater:
    """Safe deterministic fallback.

    We deliberately do not create long-term claims from rules here.  Basic info
    should be conservative; when the LLM fails, keeping the previous profile is
    safer than generating weak long-term background claims.
    """

    def update(
        self,
        old_basic_info: dict[str, Any] | None,
        session_events: list[dict[str, Any]],
        session_need_items: list[dict[str, Any]],
        *,
        clock: Clock | None = None,
    ) -> dict[str, Any]:
        del session_events, session_need_items
        return normalize_basic_info(old_basic_info, clock=clock or RealClock())


# ---------------------------------------------------------------------------
# LLM-driven updater
# ---------------------------------------------------------------------------


class LLMBasicInfoUpdater:
    """LLM-driven basic_info updater (propose → consolidate → summarize)."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        fallback: RuleBasicInfoUpdater | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or RuleBasicInfoUpdater()

    @property
    def client(self) -> LLMClient:
        """Lazy default-client resolution so tests can monkeypatch easily."""
        if self._client is None:
            self._client = get_default_llm_client()
        return self._client

    def update(
        self,
        old_basic_info: dict[str, Any] | None,
        session_events: list[dict[str, Any]],
        session_need_items: list[dict[str, Any]],
        *,
        clock: Clock | None = None,
    ) -> dict[str, Any]:
        clk = clock or RealClock()
        baseline = normalize_basic_info(old_basic_info, clock=clk)

        events = _to_dict_list(session_events)
        needs = _to_dict_list(session_need_items)

        # No mid-layer evidence → nothing to propose; keep previous profile.
        if not events and not needs:
            return baseline

        try:
            candidates = self._propose_candidates(events, needs, clk)
            merged_claims = self._consolidate_claims(
                baseline=baseline,
                candidates=candidates,
                clock=clk,
            )
            sections = self._summarize_sections(merged_claims, clock=clk)
            return _assemble_basic_info(
                merged_claims=merged_claims,
                sections=sections,
                clock=clk,
            )
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "LLMBasicInfoUpdater.update failed (%s); keeping previous basic_info.",
                err,
            )
            return self._fallback.update(
                old_basic_info,
                session_events,
                session_need_items,
                clock=clk,
            )

    # -----------------------------------------------------------------
    # Stage 1: propose candidates
    # -----------------------------------------------------------------

    def _propose_candidates(
        self,
        events: list[dict[str, Any]],
        needs: list[dict[str, Any]],
        clock: Clock,
    ) -> list[dict[str, Any]]:
        prompt = build_basic_info_propose_prompt(
            session_events=events,
            session_need_items=needs,
        )
        response = self.client.generate_json(
            prompt,
            system_prompt=BASIC_INFO_PROPOSE_SYSTEM,
        )

        known_event_ids, known_need_item_ids, known_turn_ids = _collect_evidence_ids(
            session_events=events,
            session_need_items=needs,
        )
        evidence_tags_by_event = _evidence_tags_index(events, key="event_id", tag_key="tags")
        evidence_tags_by_need = _evidence_tags_index(
            needs, key="item_id", tag_key="related_tags"
        )

        candidates = _parse_candidates(
            response,
            clock=clock,
            known_event_ids=known_event_ids,
            known_need_item_ids=known_need_item_ids,
            known_turn_ids=known_turn_ids,
            evidence_tags_by_event=evidence_tags_by_event,
            evidence_tags_by_need=evidence_tags_by_need,
        )
        return candidates

    # -----------------------------------------------------------------
    # Stage 2: consolidate
    # -----------------------------------------------------------------

    def _consolidate_claims(
        self,
        *,
        baseline: dict[str, Any],
        candidates: list[dict[str, Any]],
        clock: Clock,
    ) -> dict[str, list[dict[str, Any]]]:
        existing_claims_by_section = {
            sec: list(baseline.get(sec, {}).get("claims") or [])
            for sec in BASIC_INFO_SECTION_KEYS
        }

        if not candidates:
            return existing_claims_by_section

        active_claims_by_section = {
            sec: [
                c
                for c in claims
                if str(c.get("status", "")).strip() in ("active", "uncertain")
            ]
            for sec, claims in existing_claims_by_section.items()
        }

        prompt = build_basic_info_consolidate_prompt(
            existing_claims=_render_existing_claims(active_claims_by_section),
            candidates=_render_candidates(candidates),
        )
        response = self.client.generate_json(
            prompt,
            system_prompt=BASIC_INFO_CONSOLIDATE_SYSTEM,
        )
        decisions = _parse_decisions(
            response,
            candidates=candidates,
            existing_claims_by_section=existing_claims_by_section,
        )
        return _apply_decisions(
            existing_claims_by_section=existing_claims_by_section,
            candidates=candidates,
            decisions=decisions,
            clock=clock,
        )

    # -----------------------------------------------------------------
    # Stage 3: summarize sections
    # -----------------------------------------------------------------

    def _summarize_sections(
        self,
        merged_claims: dict[str, list[dict[str, Any]]],
        *,
        clock: Clock,
    ) -> dict[str, dict[str, Any]]:
        # Empty-everywhere shortcut: skip the LLM call.
        active_per_section = {
            sec: [
                c
                for c in claims
                if str(c.get("status", "")).strip() in ("active", "uncertain")
            ]
            for sec, claims in merged_claims.items()
        }
        if not any(active_per_section.values()):
            return {
                sec: {"summary": "", "summary_source_claim_ids": []}
                for sec in BASIC_INFO_SECTION_KEYS
            }

        prompt = build_basic_info_summarize_prompt(
            section_claims=_render_section_claims(active_per_section),
        )
        response = self.client.generate_json(
            prompt,
            system_prompt=BASIC_INFO_SUMMARIZE_SYSTEM,
        )
        return _parse_section_summaries(
            response,
            active_claims_by_section=active_per_section,
        )


# ---------------------------------------------------------------------------
# Stage 1 parsing
# ---------------------------------------------------------------------------


def _parse_candidates(
    response: Any,
    *,
    clock: Clock,
    known_event_ids: set[str],
    known_need_item_ids: set[str],
    known_turn_ids: set[str],
    evidence_tags_by_event: dict[str, set[str]],
    evidence_tags_by_need: dict[str, set[str]],
) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    raw_list = response.get("candidates")
    if not isinstance(raw_list, list):
        return []

    now = clock.now_iso()
    out: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str]] = set()

    for raw in raw_list:
        if not isinstance(raw, dict):
            continue

        category = str(raw.get("category", "") or "").strip()
        if category not in BASIC_INFO_SECTION_KEYS:
            continue

        content = str(raw.get("content", "") or "").strip()
        if not content:
            continue

        claim_type = str(raw.get("claim_type", "") or "").strip()
        if claim_type not in _CLAIM_TYPE_SET:
            continue

        status = str(raw.get("status", "") or "").strip()
        if status not in _PROPOSE_STATUS_SET:
            status = "active"

        source_event_ids = _filter_known_ids(
            _coerce_str_list(raw.get("source_event_ids")),
            known_event_ids,
        )
        source_need_item_ids = _filter_known_ids(
            _coerce_str_list(raw.get("source_need_item_ids")),
            known_need_item_ids,
        )
        source_turn_ids = _filter_known_ids(
            _coerce_str_list(raw.get("source_turn_ids")),
            known_turn_ids,
        )

        if not (source_event_ids or source_need_item_ids):
            # Stage 1 candidates must cite at least one EventItem or NeedItem.
            continue

        allowed_tags: set[str] = set()
        for eid in source_event_ids:
            allowed_tags |= evidence_tags_by_event.get(eid, set())
        for nid in source_need_item_ids:
            allowed_tags |= evidence_tags_by_need.get(nid, set())
        tags = _coerce_evidence_tags(raw.get("tags"), allowed_tags)

        sig = (category, content)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        candidate_id = f"cand-{len(out) + 1:03d}"
        out.append(
            {
                "candidate_id": candidate_id,
                "category": category,
                "claim_type": claim_type,
                "content": content,
                "tags": tags,
                "source_event_ids": source_event_ids,
                "source_need_item_ids": source_need_item_ids,
                "source_turn_ids": source_turn_ids,
                "first_seen": str(raw.get("first_seen", "") or now),
                "last_seen": str(raw.get("last_seen", "") or now),
                "confidence": _coerce_unit_float(raw.get("confidence")),
                "status": status,
            }
        )

    return out


# ---------------------------------------------------------------------------
# Stage 2 parsing
# ---------------------------------------------------------------------------


def _parse_decisions(
    response: Any,
    *,
    candidates: list[dict[str, Any]],
    existing_claims_by_section: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Return ``candidate_id -> decision`` mapping.

    Candidates that are missing / malformed default to ``ignore`` so that no
    short-term content slips into basic_info via under-specified responses.
    """
    candidate_ids = {c["candidate_id"] for c in candidates}
    candidate_categories = {c["candidate_id"]: c["category"] for c in candidates}

    valid_target_ids: set[str] = set()
    target_section: dict[str, str] = {}
    for sec, claims in existing_claims_by_section.items():
        for claim in claims:
            cid = str(claim.get("claim_id", "") or "").strip()
            if not cid:
                continue
            valid_target_ids.add(cid)
            target_section[cid] = sec

    decisions: dict[str, dict[str, Any]] = {}

    if isinstance(response, dict):
        raw_list = response.get("decisions")
        if isinstance(raw_list, list):
            for raw in raw_list:
                if not isinstance(raw, dict):
                    continue
                cand_id = str(raw.get("candidate_id", "") or "").strip()
                if cand_id not in candidate_ids or cand_id in decisions:
                    continue

                action = str(raw.get("action", "") or "").strip()
                if action not in _VALID_ACTIONS:
                    decisions[cand_id] = {"action": "ignore"}
                    continue

                target_id = str(raw.get("target_claim_id", "") or "").strip()

                if action in ("update", "supersede"):
                    if target_id not in valid_target_ids:
                        decisions[cand_id] = {"action": "ignore"}
                        continue
                    if action == "update":
                        # update requires the target to live in the same section,
                        # otherwise downgrade to add to keep section invariants.
                        if target_section.get(target_id) != candidate_categories.get(cand_id):
                            decisions[cand_id] = {"action": "add"}
                            continue

                decision: dict[str, Any] = {"action": action}
                if action in ("update", "supersede"):
                    decision["target_claim_id"] = target_id

                if action == "update":
                    if "merged_content" in raw:
                        decision["merged_content"] = str(
                            raw.get("merged_content", "") or ""
                        ).strip()
                    if "merged_status" in raw:
                        decision["merged_status"] = str(
                            raw.get("merged_status", "") or ""
                        ).strip()
                    if "merged_confidence" in raw:
                        decision["merged_confidence"] = _coerce_unit_float(
                            raw.get("merged_confidence")
                        )
                    if "merged_tags" in raw:
                        decision["merged_tags"] = _coerce_str_list(raw.get("merged_tags"))

                decisions[cand_id] = decision

    for cand_id in candidate_ids:
        decisions.setdefault(cand_id, {"action": "ignore"})
    return decisions


def _apply_decisions(
    *,
    existing_claims_by_section: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    clock: Clock,
) -> dict[str, list[dict[str, Any]]]:
    """Apply per-candidate actions to the existing per-section claim lists."""
    now = clock.now_iso()
    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    # Deep-ish copy so we don't mutate the baseline objects.
    section_claims: dict[str, list[dict[str, Any]]] = {
        sec: [dict(c) for c in claims]
        for sec, claims in existing_claims_by_section.items()
    }

    target_index: dict[str, dict[str, Any]] = {}
    for claims in section_claims.values():
        for c in claims:
            cid = str(c.get("claim_id", "") or "").strip()
            if cid:
                target_index[cid] = c

    for cand in candidates:
        cand_id = cand["candidate_id"]
        decision = decisions.get(cand_id, {"action": "ignore"})
        action = decision.get("action", "ignore")
        section = cand["category"]

        if action == "ignore":
            continue

        if action == "add":
            section_claims[section].append(_candidate_to_new_claim(cand, now))
            continue

        target_id = decision.get("target_claim_id", "")
        target = target_index.get(target_id)
        if target is None:
            section_claims[section].append(_candidate_to_new_claim(cand, now))
            continue

        if action == "update":
            _merge_into_target(target, cand, decision, now)
            continue

        if action == "supersede":
            target["status"] = "superseded"
            target["updated_at"] = now
            new_claim = _candidate_to_new_claim(cand, now)
            existing_super = list(new_claim.get("supersedes") or [])
            if target_id and target_id not in existing_super:
                existing_super.append(target_id)
            new_claim["supersedes"] = existing_super
            section_claims[section].append(new_claim)

    return section_claims


def _merge_into_target(
    target: dict[str, Any],
    candidate: dict[str, Any],
    decision: dict[str, Any],
    now: str,
) -> None:
    """In-place evidence + field merge for an ``update`` decision."""
    target["source_event_ids"] = _merge_str_lists(
        target.get("source_event_ids"), candidate.get("source_event_ids")
    )
    target["source_need_item_ids"] = _merge_str_lists(
        target.get("source_need_item_ids"), candidate.get("source_need_item_ids")
    )
    target["source_turn_ids"] = _merge_str_lists(
        target.get("source_turn_ids"), candidate.get("source_turn_ids")
    )

    cand_last = str(candidate.get("last_seen", "") or "")
    if cand_last:
        target["last_seen"] = cand_last

    merged_content = decision.get("merged_content")
    if isinstance(merged_content, str) and merged_content.strip():
        target["content"] = merged_content.strip()

    merged_status = decision.get("merged_status")
    if isinstance(merged_status, str) and merged_status in _STATUS_SET:
        target["status"] = merged_status

    if "merged_confidence" in decision:
        merged_conf = decision["merged_confidence"]
        if merged_conf is not None:
            target["confidence"] = merged_conf

    merged_tags = decision.get("merged_tags")
    if isinstance(merged_tags, list):
        allowed = set(target.get("tags") or []) | set(candidate.get("tags") or [])
        target["tags"] = _coerce_evidence_tags(merged_tags, allowed)

    target["updated_at"] = now


def _candidate_to_new_claim(candidate: dict[str, Any], now: str) -> dict[str, Any]:
    """Convert a stage-1 candidate into a new claim dict (id assigned later)."""
    return {
        "claim_id": "",
        "category": candidate["category"],
        "claim_type": candidate.get("claim_type") or _DEFAULT_CLAIM_TYPE,
        "content": str(candidate.get("content", "") or "").strip(),
        "tags": list(candidate.get("tags") or []),
        "source_event_ids": list(candidate.get("source_event_ids") or []),
        "source_need_item_ids": list(candidate.get("source_need_item_ids") or []),
        "source_turn_ids": list(candidate.get("source_turn_ids") or []),
        "first_seen": str(candidate.get("first_seen", "") or now),
        "last_seen": str(candidate.get("last_seen", "") or now),
        "confidence": candidate.get("confidence"),
        "status": str(candidate.get("status", "") or _DEFAULT_STATUS),
        "supersedes": [],
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# Stage 3 parsing
# ---------------------------------------------------------------------------


def _parse_section_summaries(
    response: Any,
    *,
    active_claims_by_section: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        sec: {"summary": "", "summary_source_claim_ids": []}
        for sec in BASIC_INFO_SECTION_KEYS
    }
    if not isinstance(response, dict):
        return out

    sections = response.get("sections")
    if not isinstance(sections, dict):
        # Tolerate flat ``{section_key: {...}}`` shape too.
        if any(k in response for k in BASIC_INFO_SECTION_KEYS):
            sections = response
        else:
            return out

    for sec in BASIC_INFO_SECTION_KEYS:
        node = sections.get(sec)
        if not isinstance(node, dict):
            continue
        summary = str(node.get("summary", "") or "").strip()
        valid_ids = {
            str(c.get("claim_id", "")).strip()
            for c in active_claims_by_section.get(sec, [])
            if str(c.get("claim_id", "")).strip()
        }
        source_ids = [
            cid
            for cid in _coerce_str_list(node.get("summary_source_claim_ids"))
            if cid in valid_ids
        ]
        out[sec] = {
            "summary": summary,
            "summary_source_claim_ids": source_ids,
        }
    return out


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------


def _assemble_basic_info(
    *,
    merged_claims: dict[str, list[dict[str, Any]]],
    sections: dict[str, dict[str, Any]],
    clock: Clock,
) -> dict[str, Any]:
    now = clock.now_iso()

    out: dict[str, Any] = {}
    for sec in BASIC_INFO_SECTION_KEYS:
        section_payload = sections.get(sec) or {}
        out[sec] = {
            "summary": section_payload.get("summary", "") or "",
            "claims": merged_claims.get(sec, []),
            "summary_source_claim_ids": section_payload.get(
                "summary_source_claim_ids", []
            )
            or [],
            "updated_at": now,
        }

    return _finalize_basic_info(out, clock_iso=now)


# ---------------------------------------------------------------------------
# Sanitization shared by normalize_basic_info()
# ---------------------------------------------------------------------------


def _sanitize_claim(
    raw: Any,
    *,
    section_key: str,
    clock_iso: str,
    strict_evidence: bool,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    content = str(raw.get("content", "") or "").strip()
    if not content:
        return None

    claim_type = str(raw.get("claim_type", "") or "").strip()
    if claim_type not in _CLAIM_TYPE_SET:
        if strict_evidence:
            return None
        claim_type = _DEFAULT_CLAIM_TYPE

    status = str(raw.get("status", "") or "").strip()
    if status not in _STATUS_SET:
        if strict_evidence:
            return None
        status = _DEFAULT_STATUS

    source_event_ids = _coerce_str_list(raw.get("source_event_ids"))
    source_need_item_ids = _coerce_str_list(raw.get("source_need_item_ids"))
    source_turn_ids = _coerce_str_list(raw.get("source_turn_ids"))
    tags = _coerce_global_tags(raw.get("tags"))

    if status in ("active", "uncertain"):
        has_evidence = bool(source_event_ids or source_need_item_ids or source_turn_ids)
        if not has_evidence:
            return None

    claim_id = str(raw.get("claim_id", "") or "").strip()

    return {
        "claim_id": claim_id,
        "category": section_key,
        "claim_type": claim_type,
        "content": content,
        "tags": tags,
        "source_event_ids": source_event_ids,
        "source_need_item_ids": source_need_item_ids,
        "source_turn_ids": source_turn_ids,
        "first_seen": str(raw.get("first_seen", "") or ""),
        "last_seen": str(raw.get("last_seen", "") or ""),
        "confidence": _coerce_unit_float(raw.get("confidence")),
        "status": status,
        "supersedes": _coerce_str_list(raw.get("supersedes")),
        "updated_at": str(raw.get("updated_at", "") or clock_iso),
    }


def _finalize_basic_info(
    basic_info: dict[str, Any],
    *,
    clock_iso: str,
) -> dict[str, Any]:
    """Assign missing claim ids and repair summary_source_claim_ids."""
    out: dict[str, Any] = {}

    for section_key in BASIC_INFO_SECTION_KEYS:
        sec = basic_info.get(section_key)
        if not isinstance(sec, dict):
            out[section_key] = _empty_section(clock_iso)
            continue

        claims_out: list[dict[str, Any]] = []
        for raw_claim in sec.get("claims") or []:
            if not isinstance(raw_claim, dict):
                continue
            claim = dict(raw_claim)
            claim_id = str(claim.get("claim_id", "") or "").strip()
            if not claim_id:
                claim_id = f"claim-{uuid.uuid4().hex[:16]}"
            claim["claim_id"] = claim_id
            claim["category"] = section_key
            claim["updated_at"] = str(claim.get("updated_at", "") or clock_iso)
            claims_out.append(claim)

        valid_claim_ids = {
            str(c.get("claim_id", "")).strip()
            for c in claims_out
            if str(c.get("claim_id", "")).strip()
        }

        summary_source_claim_ids = [
            cid
            for cid in _coerce_str_list(sec.get("summary_source_claim_ids"))
            if cid in valid_claim_ids
        ]

        # If the summary stage didn't tag any source ids, conservatively link
        # the summary to all active / uncertain claims in that section.
        summary = str(sec.get("summary", "") or "").strip()
        if summary and not summary_source_claim_ids:
            summary_source_claim_ids = [
                str(c.get("claim_id", "")).strip()
                for c in claims_out
                if str(c.get("status", "")).strip() in ("active", "uncertain")
                and str(c.get("claim_id", "")).strip()
            ]

        out[section_key] = {
            "summary": summary,
            "claims": claims_out,
            "summary_source_claim_ids": summary_source_claim_ids,
            "updated_at": str(sec.get("updated_at", "") or clock_iso),
        }

    return out


# ---------------------------------------------------------------------------
# Rendering helpers (per-stage user prompt building blocks)
# ---------------------------------------------------------------------------


def _render_existing_claims(
    claims_by_section: dict[str, list[dict[str, Any]]],
) -> str:
    blocks: list[str] = []
    for sec in BASIC_INFO_SECTION_KEYS:
        claims = claims_by_section.get(sec, [])
        blocks.append(f"{sec}:")
        if not claims:
            blocks.append("(无)")
            continue
        for claim in claims:
            blocks.append(_render_claim_block(claim, include_id=True))
    return "\n".join(blocks)


def _render_candidates(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "(无)"
    blocks: list[str] = []
    for cand in candidates:
        header = (
            f"- [{cand['candidate_id']} | {cand['category']} | "
            f"{cand.get('claim_type', '')} | tags: {_join_tags(cand.get('tags'))}]"
        )
        blocks.append(header)
        blocks.append(f"  内容: {cand.get('content', '')}")
        blocks.append(f"  evidence: {_render_evidence(cand)}")
    return "\n".join(blocks)


def _render_section_claims(
    claims_by_section: dict[str, list[dict[str, Any]]],
) -> str:
    blocks: list[str] = []
    for sec in BASIC_INFO_SECTION_KEYS:
        claims = claims_by_section.get(sec, [])
        blocks.append(f"{sec}:")
        if not claims:
            blocks.append("(无)")
            continue
        for claim in claims:
            cid = str(claim.get("claim_id", "") or "")
            status = str(claim.get("status", "") or "")
            claim_type = str(claim.get("claim_type", "") or "")
            blocks.append(f"- [{cid} | {status} | {claim_type}]")
            blocks.append(f"  内容: {claim.get('content', '')}")
    return "\n".join(blocks)


def _render_claim_block(claim: dict[str, Any], *, include_id: bool) -> str:
    cid = str(claim.get("claim_id", "") or "")
    status = str(claim.get("status", "") or "")
    claim_type = str(claim.get("claim_type", "") or "")
    head_id = cid if include_id else ""
    head_parts = [head_id, status, claim_type, f"tags: {_join_tags(claim.get('tags'))}"]
    head = "- [" + " | ".join(p for p in head_parts if p) + "]"
    return "\n".join(
        [
            head,
            f"  内容: {claim.get('content', '')}",
            f"  evidence: {_render_evidence(claim)}",
        ]
    )


def _render_evidence(item: dict[str, Any]) -> str:
    events = ",".join(item.get("source_event_ids") or [])
    needs = ",".join(item.get("source_need_item_ids") or [])
    turns = ",".join(item.get("source_turn_ids") or [])
    return f"events={events}; needs={needs}; turns={turns}"


def _join_tags(tags: Any) -> str:
    if not isinstance(tags, list) or not tags:
        return ""
    return ", ".join(str(t) for t in tags)


# ---------------------------------------------------------------------------
# Backward-compatible module API
# ---------------------------------------------------------------------------


_DEFAULT_UPDATER: LLMBasicInfoUpdater | None = None


def _get_default_updater() -> LLMBasicInfoUpdater:
    global _DEFAULT_UPDATER
    if _DEFAULT_UPDATER is None:
        _DEFAULT_UPDATER = LLMBasicInfoUpdater()
    return _DEFAULT_UPDATER


def reset_default_updater() -> None:
    """Drop the cached default updater, useful in tests."""
    global _DEFAULT_UPDATER
    _DEFAULT_UPDATER = None


def update_basic_info(
    old_basic_info: dict[str, Any] | None,
    session_events: list[dict[str, Any]],
    session_need_items: list[dict[str, Any]],
    *,
    llm_client: LLMClient | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Session-end basic_info update.

    On any LLM / parsing failure, returns the normalized previous basic_info.
    """
    updater = (
        LLMBasicInfoUpdater(client=llm_client)
        if llm_client is not None
        else _get_default_updater()
    )
    return updater.update(
        old_basic_info,
        session_events,
        session_need_items,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# Internal coercion helpers
# ---------------------------------------------------------------------------


def _empty_section(clock_iso: str) -> dict[str, Any]:
    return {
        "summary": "",
        "claims": [],
        "summary_source_claim_ids": [],
        "updated_at": clock_iso,
    }


def _to_dict_list(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(dict(row))
            continue
        to_dict = getattr(row, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
            if isinstance(value, dict):
                out.append(value)
    return out


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        seen: set[str] = set()
        for v in value:
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_global_tags(value: Any) -> list[str]:
    """Tag coercion against the full MEMORY_TAGS vocabulary (legacy / restore path)."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in _coerce_str_list(value):
        if tag not in _MEMORY_TAG_SET or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out[:8]


def _coerce_evidence_tags(value: Any, allowed: set[str]) -> list[str]:
    """Tag coercion against an evidence-derived allow-list (1~3 conservative).

    Only tags that are also in MEMORY_TAGS *and* in ``allowed`` are kept; output
    is capped at 3 entries to enforce the conservative-selection rule.
    """
    seen: set[str] = set()
    out: list[str] = []
    for tag in _coerce_str_list(value):
        if tag not in _MEMORY_TAG_SET or tag not in allowed or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= 3:
            break
    return out


def _filter_known_ids(
    ids: list[str],
    known_ids: set[str] | None,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item_id in ids:
        if item_id in seen:
            continue
        if known_ids is not None and known_ids and item_id not in known_ids:
            continue
        seen.add(item_id)
        out.append(item_id)
    return out


def _merge_str_lists(*sources: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for src in sources:
        for v in _coerce_str_list(src):
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
    return out


def _coerce_unit_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _collect_evidence_ids(
    *,
    session_events: list[dict[str, Any]],
    session_need_items: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    event_ids: set[str] = set()
    need_item_ids: set[str] = set()
    turn_ids: set[str] = set()

    for event in session_events:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id", "") or "").strip()
        if event_id:
            event_ids.add(event_id)
        turn_ids.update(_coerce_str_list(event.get("source_turn_ids")))

    for item in session_need_items:
        if not isinstance(item, dict):
            continue

        item_id = str(item.get("item_id", "") or "").strip()
        if item_id:
            need_item_ids.add(item_id)

        turn_ids.update(_coerce_str_list(item.get("source_turn_ids")))

        solutions = item.get("solutions")
        if isinstance(solutions, list):
            for sol in solutions:
                if not isinstance(sol, dict):
                    continue
                turn_ids.update(_coerce_str_list(sol.get("feedback_turn_ids")))

    return event_ids, need_item_ids, turn_ids


def _evidence_tags_index(
    items: list[dict[str, Any]],
    *,
    key: str,
    tag_key: str,
) -> dict[str, set[str]]:
    """Build ``id -> set(tags ∩ MEMORY_TAGS)`` index for stage 1 candidates."""
    out: dict[str, set[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get(key, "") or "").strip()
        if not item_id:
            continue
        raw_tags = item.get(tag_key)
        if not isinstance(raw_tags, list):
            continue
        tags: set[str] = set()
        for t in raw_tags:
            ts = str(t).strip()
            if ts and ts in _MEMORY_TAG_SET:
                tags.add(ts)
        out[item_id] = tags
    return out


# Re-export for callers that previously imported these names from this module.
__all__ = [
    "BASIC_INFO_SECTION_KEYS",
    "LLMBasicInfoUpdater",
    "RuleBasicInfoUpdater",
    "normalize_basic_info",
    "reset_default_updater",
    "update_basic_info",
]
