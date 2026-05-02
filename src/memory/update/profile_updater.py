"""Periodic profile updater from mid-layer memory.

Two periodic updates are implemented:

1. **Recent status summary** — LLM-driven summary over events within a
   sliding ``window_days`` window, with health-related and life/family
   buckets handled separately so the resulting ``recent_status`` foregrounds
   the patient's recent disease trajectory.
2. **Need preference clusters** — Items previously without a ``cluster_id`` are
   incrementally assigned & refined; clusters are bootstrapped from scratch
   via :meth:`NeedClusterer.initialize` when none exist yet (or when the
   caller forces a re-cluster).

The orchestrator returns ``(updated_profile, item_id_to_cluster_id)``. The
caller is responsible for persisting ``cluster_id`` back onto each
:class:`~src.memory.schemas.NeedSolutionItem` via
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
)
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
    ``clock.now()``), split them into a medical bucket and a life bucket,
    ask the LLM for a structured summary, and fall back to a deterministic
    rule summary if the LLM call fails.
    """

    clock = clock or RealClock()
    in_window, window_start, window_end = _filter_events_by_window(
        recent_events, window_days=window_days, clock=clock,
    )
    medical, life = split_events_by_bucket(in_window)
    old_recent = profile_dict.get("recent_status") or {}

    # Start from the previous summary so we keep historical text the new
    # window doesn't refresh (per the prompt's "if no new info, keep old").
    summary: dict[str, Any] = {
        "health_status": str(old_recent.get("health_status", "")),
        "family_status": str(old_recent.get("family_status", "")),
        "interest_changes": list(old_recent.get("interest_changes", []) or []),
        "window_start": window_start,
        "window_end": window_end,
        "source_event_ids": [
            ev.get("event_id") for ev in in_window if ev.get("event_id")
        ],
    }

    if not in_window:
        return summary

    try:
        prompt = build_recent_status_summarize_prompt(
            old_recent_status={
                "health_status": summary["health_status"],
                "family_status": summary["family_status"],
                "interest_changes": summary["interest_changes"],
            },
            medical_events=medical,
            life_events=life,
            window_days=window_days,
        )
        resp = llm_client.generate_json(prompt, system_prompt=RECENT_STATUS_SUMMARIZE_SYSTEM)
        if isinstance(resp, dict):
            summary["health_status"] = str(
                resp.get("health_status", summary["health_status"]) or ""
            )[:200]
            summary["family_status"] = str(
                resp.get("family_status", summary["family_status"]) or ""
            )[:200]
            ic = resp.get("interest_changes")
            if isinstance(ic, list):
                summary["interest_changes"] = [
                    str(x).strip() for x in ic if str(x).strip()
                ][:8]
            return summary
    except Exception as err:  # noqa: BLE001 — defensive fallback
        _logger.warning(
            "summarize_recent_status: LLM failed (%s); using rule fallback.", err,
        )

    # Rule fallback: stitch the latest summaries from each bucket together.
    if medical:
        joined = "; ".join(
            ev.get("event_summary", "").strip() for ev in medical[-3:] if ev.get("event_summary")
        )
        if joined:
            summary["health_status"] = joined[:200]
    if life:
        joined = "; ".join(
            ev.get("event_summary", "").strip() for ev in life[-3:] if ev.get("event_summary")
        )
        if joined:
            summary["family_status"] = joined[:200]
    return summary


# ---------------------------------------------------------------------------
# Cluster (de)serialisation
# ---------------------------------------------------------------------------


def _cluster_from_dict(d: dict[str, Any], *, clock: Clock) -> NeedCluster:
    return NeedCluster(
        cluster_id=str(d.get("cluster_id", "")),
        need_type=str(d.get("need_type", "")),
        preference_principle=str(d.get("preference_principle", "")),
        centroid=[float(x) for x in (d.get("centroid") or [])],
        member_item_ids=[str(x) for x in (d.get("member_item_ids") or [])],
        sample_needs=[str(x) for x in (d.get("sample_needs") or [])],
        sample_preferences=[str(x) for x in (d.get("sample_preferences") or [])],
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
    n_clusters: int = 5,
    window_days: int = 14,
    min_items_to_cluster: int = 5,
    force_recluster: bool = False,
) -> tuple[UserProfile, dict[str, str]]:
    """Periodic top-layer update.

    Args:
        clock: Source of "now" for the recent-status window cutoff,
            ``cluster.updated_at`` and ``profile.updated_at``. Defaults to
            :class:`RealClock`; pass a :class:`FixedClock` (anchored at the
            session's ``dialogue_timestamp``) when replaying the dataset.

    Returns:
        ``(updated_profile, item_id_to_cluster_id)``. The caller should
        persist the cluster-id mapping onto the corresponding
        :class:`NeedSolutionItem`s via :meth:`NeedSolutionStore.update_item`.

    Behaviour:
        - When the profile has no clusters yet (or ``force_recluster=True``),
          all items are re-clustered from scratch via
          :meth:`NeedClusterer.initialize`.
        - Otherwise only items without a ``cluster_id`` are incrementally
          assigned and used to refine the affected clusters.
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
    needs_full_init = force_recluster or not existing_clusters

    clusterer = NeedClusterer(
        embedder=emb,
        llm_client=llm,
        n_clusters=n_clusters,
        min_items_to_cluster=min_items_to_cluster,
        now_iso=clk.now_iso(),
    )

    if needs_full_init:
        clusters, mapping = clusterer.initialize(need_solution_items)
    else:
        unassigned = [it for it in need_solution_items if not it.get("cluster_id")]
        clusters, mapping = clusterer.assign_and_refine(existing_clusters, unassigned)

    profile = UserProfile(
        basic_info=profile_dict.get("basic_info", {}) or {},
        recent_status=recent_status,
        need_preferences=[c.to_dict() for c in clusters],
        updated_at=clk.now_iso(),
    )
    return profile, mapping
