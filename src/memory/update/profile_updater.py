"""Periodic profile updater from mid-layer memory.

Three periodic updates are implemented:

1. **Recent status summary** — LLM-driven summary over events within a
   sliding ``window_days`` window. Events are split into health, self-management,
   mental, family/social, interest, and risk-hint buckets; the resulting
   ``recent_status`` matches :func:`src.memory.schemas._empty_recent_status`.
2. **Need preference clusters** — Domain-based assignment. Each ``need_domain``
   maps to one cluster; items without ``cluster_id`` are assigned by domain and
   preference principles are updated via ``preference_principle_update`` prompt.
3. **Basic info (long-term background)** — Last: when the caller passes this
   session's event and need rows, an LLM merges them into evidence-backed
   ``basic_info`` claims (see :mod:`src.memory.update.basic_info_updater`).

The orchestrator returns ``(updated_profile, item_id_to_cluster_id)``. The
caller is responsible for persisting ``cluster_id`` back onto each
:class:`~src.memory.schemas.NeedItem` via
:meth:`NeedSolutionStore.update_item`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.llm.embedder import Embedder, get_default_embedder
from src.llm.llm import LLMClient, get_default_llm_client
from src.memory.schemas import (
    NeedCluster,
    UserProfile,
    _empty_recent_status,
)
from src.memory.update.basic_info_updater import normalize_basic_info, update_basic_info
from src.memory.update.need_clustering import NeedClusterer
from src.memory.update.prompts import (
    RECENT_STATUS_SUMMARIZE_SYSTEM,
    build_recent_status_summarize_prompt,
    split_events_by_bucket,
)
from src.utils.clock import Clock, RealClock

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time-window helpers
# ---------------------------------------------------------------------------


def _parse_iso_timestamp(ts: str) -> datetime | None:
    """Best-effort ISO-8601 parser; tolerates trailing ``Z`` and missing tz."""
    if not ts:
        return None
    text = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _filter_events_by_window(
    events: list[dict[str, Any]],
    *,
    window_days: int,
    clock: Clock,
) -> tuple[list[dict[str, Any]], str, str]:
    """Keep events whose timestamp falls within the last ``window_days``.

    The ``window_days`` are anchored at ``clock.now()`` rather than wall-clock
    time so dataset-replay runs use the sample's ``dialogue_timestamp`` as the
    cut-off.

    Returns ``(filtered, window_start_iso, window_end_iso)``. Empty result
    yields empty strings for the window endpoints.
    """
    if not events:
        return [], "", ""
    now = clock.now()
    cutoff = now - timedelta(days=window_days)
    filtered: list[dict[str, Any]] = []
    earliest: datetime | None = None
    latest: datetime | None = None
    for ev in events:
        dt = _parse_iso_timestamp(str(ev.get("timestamp", "")))
        if dt is None or dt < cutoff:
            continue
        filtered.append(ev)
        if earliest is None or dt < earliest:
            earliest = dt
        if latest is None or dt > latest:
            latest = dt
    if not filtered:
        return [], "", ""
    return (
        filtered,
        earliest.isoformat() if earliest else "",
        latest.isoformat() if latest else "",
    )


# ---------------------------------------------------------------------------
# Recent status summary
# ---------------------------------------------------------------------------


def _event_id_list(ev_list: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for ev in ev_list:
        eid = ev.get("event_id")
        if eid:
            out.append(str(eid))
    return out


def _join_event_summaries(evlist: list[dict[str, Any]], *, max_chars: int = 200) -> str:
    parts = [
        str(ev.get("event_summary", "")).strip()
        for ev in evlist[-3:]
        if ev.get("event_summary")
    ]
    joined = "; ".join(parts)
    return joined[:max_chars] if joined else ""


def summarize_recent_status(
    profile_dict: dict[str, Any],
    recent_events: list[dict[str, Any]],
    *,
    llm_client: LLMClient,
    window_days: int = 14,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Return a structured ``recent_status`` dict for the given profile.

    Strategy: filter events to the last ``window_days`` (anchored at
    ``clock.now()``), split them into per-field buckets, call the LLM, and
    fall back to stitching event summaries per bucket if the model fails.
    """

    clock = clock or RealClock()
    in_window, window_start, window_end = _filter_events_by_window(
        recent_events, window_days=window_days, clock=clock,
    )
    (
        health_events,
        self_management_events,
        mental_events,
        family_social_events,
        interest_events,
        risk_hint_events,
    ) = split_events_by_bucket(in_window)
    old_recent = profile_dict.get("recent_status") or {}

    summary: dict[str, Any] = _empty_recent_status()
    for key in (
        "health_status",
        "self_management_status",
        "mental_status",
        "family_social_status",
    ):
        if old_recent.get(key):
            summary[key] = str(old_recent[key])
    if isinstance(old_recent.get("interest_changes"), list):
        summary["interest_changes"] = [
            str(x).strip() for x in old_recent["interest_changes"] if str(x).strip()
        ][:12]
    if isinstance(old_recent.get("risk_flags"), list):
        summary["risk_flags"] = [
            str(x).strip() for x in old_recent["risk_flags"] if str(x).strip()
        ][:12]

    summary["window_start"] = window_start
    summary["window_end"] = window_end
    summary["field_source_event_ids"] = {
        "health_status": _event_id_list(health_events),
        "self_management_status": _event_id_list(self_management_events),
        "mental_status": _event_id_list(mental_events),
        "family_social_status": _event_id_list(family_social_events),
        "interest_changes": _event_id_list(interest_events),
        "risk_flags": _event_id_list(risk_hint_events),
    }

    if not in_window:
        return summary

    old_for_prompt = {
        "health_status": summary["health_status"],
        "self_management_status": summary["self_management_status"],
        "mental_status": summary["mental_status"],
        "family_social_status": summary["family_social_status"],
        "interest_changes": summary["interest_changes"],
        "risk_flags": summary["risk_flags"],
    }

    try:
        prompt = build_recent_status_summarize_prompt(
            old_recent_status=old_for_prompt,
            health_events=health_events,
            self_management_events=self_management_events,
            mental_events=mental_events,
            family_social_events=family_social_events,
            interest_events=interest_events,
            risk_hint_events=risk_hint_events,
            window_days=window_days,
        )
        resp = llm_client.generate_json(prompt, system_prompt=RECENT_STATUS_SUMMARIZE_SYSTEM)
        if isinstance(resp, dict):
            for key in (
                "health_status",
                "self_management_status",
                "mental_status",
                "family_social_status",
            ):
                if key in resp:
                    summary[key] = str(resp.get(key) or "")[:200]
            ic = resp.get("interest_changes")
            if isinstance(ic, list):
                summary["interest_changes"] = [
                    str(x).strip() for x in ic if str(x).strip()
                ][:12]
            rf = resp.get("risk_flags")
            if isinstance(rf, list):
                summary["risk_flags"] = [
                    str(x).strip() for x in rf if str(x).strip()
                ][:12]
            return summary
    except Exception as err:  # noqa: BLE001 — defensive fallback
        _logger.warning(
            "summarize_recent_status: LLM failed (%s); using rule fallback.", err,
        )

    if health_events:
        t = _join_event_summaries(health_events)
        if t:
            summary["health_status"] = t
    if self_management_events:
        t = _join_event_summaries(self_management_events)
        if t:
            summary["self_management_status"] = t
    if mental_events:
        t = _join_event_summaries(mental_events)
        if t:
            summary["mental_status"] = t
    if family_social_events:
        t = _join_event_summaries(family_social_events)
        if t:
            summary["family_social_status"] = t
    if interest_events:
        phrases = [
            str(ev.get("event_summary", "")).strip()[:10]
            for ev in interest_events[-5:]
            if ev.get("event_summary")
        ]
        phrases = [p for p in phrases if p][:8]
        if phrases:
            summary["interest_changes"] = phrases
    if risk_hint_events:
        phrases = [
            str(ev.get("event_summary", "")).strip()[:12]
            for ev in risk_hint_events[-5:]
            if ev.get("event_summary")
        ]
        phrases = [p for p in phrases if p][:8]
        if phrases:
            summary["risk_flags"] = phrases

    return summary


# ---------------------------------------------------------------------------
# Cluster (de)serialisation
# ---------------------------------------------------------------------------


def _cluster_from_dict(d: dict[str, Any], *, clock: Clock) -> NeedCluster:
    return NeedCluster(
        cluster_id=str(d.get("cluster_id", "")),
        need_domain=str(d.get("need_domain", "") or "other"),
        preference_principle=str(d.get("preference_principle", "")),
        member_item_ids=[str(x) for x in (d.get("member_item_ids") or [])],
        size=int(d.get("size", 0) or 0),
        updated_at=str(d.get("updated_at", "") or clock.now_iso()),
    )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def update_profile_from_mid_memory(
    profile_dict: dict[str, Any],
    recent_events: list[dict[str, Any]],
    need_solution_items: list[dict[str, Any]],
    *,
    embedder: Embedder | None = None,
    llm_client: LLMClient | None = None,
    clock: Clock | None = None,
    window_days: int = 14,
    session_events: list[dict[str, Any]] | None = None,
    session_need_items: list[dict[str, Any]] | None = None,
) -> tuple[UserProfile, dict[str, str]]:
    """Periodic top-layer update.
    Returns:
        ``(updated_profile, item_id_to_cluster_id)``. The caller should
        persist the cluster-id mapping onto the corresponding
        :class:`NeedItem`s via :meth:`NeedSolutionStore.update_item`.

    Behaviour:
        - When the profile has no clusters yet, all items are grouped by domain
          via :meth:`NeedClusterer.initialize`.
        - Otherwise only items without a ``cluster_id`` are assigned by domain
          and used to update the affected domain principles.
        - **basic_info** (after recent status and need clustering): when
          ``session_events`` and/or ``session_need_items`` are not ``None``
          (ingest passes this session's extracted rows), they are passed to
          :func:`update_basic_info`; a missing side is treated as ``[]``. The
          LLM runs only if at least one row is present. When both are ``None``,
          ``basic_info`` is only normalised, not LLM-updated.
    """

    llm = llm_client or get_default_llm_client()
    emb = embedder or get_default_embedder()
    clk: Clock = clock or RealClock()

    recent_status = summarize_recent_status(
        profile_dict, recent_events,
        llm_client=llm, window_days=window_days, clock=clk,
    )

    existing_clusters: list[NeedCluster] = [
        _cluster_from_dict(c, clock=clk)
        for c in (profile_dict.get("need_preferences") or [])
        if isinstance(c, dict)
    ]
    needs_full_init = not existing_clusters

    clusterer = NeedClusterer(
        embedder=emb,
        llm_client=llm,
        now_iso=clk.now_iso(),
    )

    if needs_full_init:
        clusters, mapping = clusterer.initialize(need_solution_items)
    else:
        unassigned = [it for it in need_solution_items if not it.get("cluster_id")]
        clusters, mapping = clusterer.assign_and_refine(existing_clusters, unassigned)

    bio_in = profile_dict.get("basic_info")
    if session_events is not None or session_need_items is not None:
        ev_rows = session_events or []
        need_rows = session_need_items or []
        if ev_rows or need_rows:
            basic_info = update_basic_info(
                bio_in,
                ev_rows,
                need_rows,
                llm_client=llm,
                clock=clk,
            )
        else:
            basic_info = normalize_basic_info(bio_in, clock=clk)
    else:
        basic_info = normalize_basic_info(bio_in, clock=clk)

    profile = UserProfile(
        basic_info=basic_info,
        recent_status=recent_status,
        need_preferences=[c.to_dict() for c in clusters],
        updated_at=clk.now_iso(),
    )
    return profile, mapping
