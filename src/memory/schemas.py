"""Shared memory schemas for raw/mid/profile layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RawTurn:
    session_id: str
    turn_id: str
    role: str
    text: str
    timestamp: str = field(default_factory=utc_now_iso)
    topic_id: str = "general"
    chunk_id: str = "chunk-0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventItem:
    event_id: str
    event_type: str
    timestamp: str
    source_turn_ids: list[str]
    event_summary: str
    tags: list[str]
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NeedSolutionItem:
    """A single (need, AI solution, user preference, holistic quality) trace.

    ``inferred_need`` is a free-form short Chinese phrase.
    ``preference`` summarizes both (a) solution-shape preferences inferred from the
    user turn and (b) any in-window behavioural feedback evidenced by subsequent
    user turns, produced in **one** LLM pass together with ``quality_score``.

    ``cluster_id`` assigns the trace to :class:`UserProfile.need_preferences`.
    """

    item_id: str
    timestamp: str
    source_turn_ids: list[str]
    inferred_need: str
    ai_solution_summary: str
    related_tags: list[str]
    preference: str = ""
    quality_score: float | None = None
    feedback_turn_ids: list[str] = field(default_factory=list)
    cluster_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Top-layer profile schemas
# ---------------------------------------------------------------------------


def _empty_recent_status() -> dict[str, Any]:
    """Default value for :attr:`UserProfile.recent_status`.

    Splits "body / disease" status from family / interest ones so downstream
    prompts can foreground recent disease trajectory.
    """
    return {
        "health_status": "",
        "family_status": "",
        "interest_changes": [],
        "window_start": "",
        "window_end": "",
        "source_event_ids": [],
    }


@dataclass
class NeedCluster:
    """
    - ``need_type`` is gamma_i (LLM-labelled requirement type, e.g. "控糖饮食建议").
    - ``preference_principle`` is rho_i (LLM-summarised preference principle).
    - ``centroid`` is the L2-normalised mean embedding of cluster members,
      used for incremental nearest-cluster assignment.
    - ``sample_needs`` / ``sample_preferences`` are kept aligned by index and
      capped to the latest few for use in re-labelling prompts.
    """

    cluster_id: str
    need_type: str
    preference_principle: str
    centroid: list[float]
    member_item_ids: list[str] = field(default_factory=list)
    sample_needs: list[str] = field(default_factory=list)
    sample_preferences: list[str] = field(default_factory=list)
    size: int = 0
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UserProfile:
    basic_info: dict[str, Any] = field(default_factory=dict)
    recent_status: dict[str, Any] = field(default_factory=_empty_recent_status)
    # List of NeedCluster.to_dict() dicts. Stored as plain dicts so that
    # JsonStore round-trips don't need a bespoke decoder.
    need_preferences: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Session-time topic windowing
# ---------------------------------------------------------------------------


@dataclass
class TopicWindow:
    """One contiguous topic-coherent slice of a session's dialogue.

    Produced by :mod:`src.memory.update.topic_segmenter`. ``turn_ids`` lists
    the :class:`RawTurn` ids that fall into this window in dialogue order;
    ``window_id`` is what gets stored as :attr:`RawTurn.chunk_id` (and is
    also used as ``cluster_id`` scoping at retrieval time).
    """

    window_id: str
    topic_label: str
    turn_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

