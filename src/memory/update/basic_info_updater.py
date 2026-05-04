"""Session-end basic_info update from mid-layer evidence.

Two updaters are provided:

- :class:`RuleBasicInfoUpdater` — deterministic no-op fallback.
- :class:`LLMBasicInfoUpdater` — LLM-driven profile update from session-level
  EventItem and NeedItem evidence.

The updater expects the LLM to return a complete ``basic_info`` object, then
post-processes it to:

- ensure the four section keys exist;
- validate controlled vocabularies;
- remove unsupported / hallucinated evidence ids when possible;
- assign ``claim_id`` for new claims;
- keep safe fallback behavior on LLM or parsing failure.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.llm.llm import LLMClient, get_default_llm_client
from src.memory.schemas import BASIC_INFO_CLAIM_TYPES, BASIC_INFO_STATUS
from src.memory.update.prompts import (
    BASIC_INFO_UPDATE_SYSTEM,
    build_basic_info_update_prompt,
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

_DEFAULT_CLAIM_TYPE = "stable_fact"
_DEFAULT_STATUS = "active"


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
                    known_event_ids=None,
                    known_need_item_ids=None,
                    known_turn_ids=None,
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
    """LLM-driven basic_info updater with deterministic fallback."""

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

        try:
            events = _to_dict_list(session_events)
            needs = _to_dict_list(session_need_items)
            prompt = build_basic_info_update_prompt(
                old_basic_info=baseline,
                session_events=events,
                session_need_items=needs,
            )
            response = self.client.generate_json(
                prompt,
                system_prompt=BASIC_INFO_UPDATE_SYSTEM,
            )
            parsed = _parse_llm_basic_info(
                response,
                baseline=baseline,
                session_events=events,
                session_need_items=needs,
                clock=clk,
            )
            if parsed is not None:
                return parsed

            _logger.warning("LLMBasicInfoUpdater.update: unusable JSON shape.")
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


# ---------------------------------------------------------------------------
# Parsing / sanitization
# ---------------------------------------------------------------------------


def _parse_llm_basic_info(
    response: Any,
    *,
    baseline: dict[str, Any],
    session_events: list[dict[str, Any]],
    session_need_items: list[dict[str, Any]],
    clock: Clock,
) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None

    root = response.get("basic_info")
    if not isinstance(root, dict):
        # Tolerate bare section object:
        # {"medical_care": {...}, "family": {...}, ...}
        if any(k in response for k in BASIC_INFO_SECTION_KEYS):
            root = response
        else:
            return None

    now = clock.now_iso()
    known_event_ids, known_need_item_ids, known_turn_ids = _collect_known_ids(
        baseline=baseline,
        session_events=session_events,
        session_need_items=session_need_items,
    )

    out: dict[str, Any] = {}

    for section_key in BASIC_INFO_SECTION_KEYS:
        sec = root.get(section_key)

        # If the model omitted a section, preserve the previous normalized
        # section instead of accidentally deleting long-term memory.
        if not isinstance(sec, dict):
            out[section_key] = dict(baseline.get(section_key) or _empty_section(now))
            continue

        claims_out: list[dict[str, Any]] = []
        claims_raw = sec.get("claims")
        if isinstance(claims_raw, list):
            seen: set[tuple[str, str, str]] = set()
            for raw_claim in claims_raw:
                sanitized = _sanitize_claim(
                    raw_claim,
                    section_key=section_key,
                    clock_iso=now,
                    known_event_ids=known_event_ids,
                    known_need_item_ids=known_need_item_ids,
                    known_turn_ids=known_turn_ids,
                    strict_evidence=True,
                )
                if sanitized is None:
                    continue

                sig = (
                    str(sanitized.get("claim_id", "")),
                    str(sanitized.get("status", "")),
                    str(sanitized.get("content", "")),
                )
                if sig in seen:
                    continue
                seen.add(sig)
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


def _sanitize_claim(
    raw: Any,
    *,
    section_key: str,
    clock_iso: str,
    known_event_ids: set[str] | None,
    known_need_item_ids: set[str] | None,
    known_turn_ids: set[str] | None,
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

    # Active / uncertain claims must be evidence-backed. Superseded claims are
    # allowed to survive even if evidence is imperfect, because they may be old
    # historical trace entries.
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

        # If the LLM omitted summary sources, conservatively link the summary to
        # all active / uncertain claims in that section.
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
# Known-id collection
# ---------------------------------------------------------------------------


def _collect_known_ids(
    *,
    baseline: dict[str, Any],
    session_events: list[dict[str, Any]],
    session_need_items: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    event_ids: set[str] = set()
    need_item_ids: set[str] = set()
    turn_ids: set[str] = set()

    # Existing evidence ids should remain valid; otherwise old claims would lose
    # their evidence when the LLM returns complete basic_info.
    for sec in baseline.values():
        if not isinstance(sec, dict):
            continue
        for claim in sec.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            event_ids.update(_coerce_str_list(claim.get("source_event_ids")))
            need_item_ids.update(_coerce_str_list(claim.get("source_need_item_ids")))
            turn_ids.update(_coerce_str_list(claim.get("source_turn_ids")))

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

    On LLM / parsing failure, returns normalized previous basic_info.
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
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


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
