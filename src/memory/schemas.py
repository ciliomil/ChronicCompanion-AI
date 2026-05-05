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
    chunk_id: str = "chunk-0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventItem:
    event_id: str
    timestamp: str
    source_turn_ids: list[str]
    event_summary: str
    tags: list[str]
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NeedSolutionProposal:
    """One assistant proposal addressing a :class:`NeedItem` need."""

    ai_solution_summary: str = ""
    feedback_turn_ids: list[str] = field(default_factory=list)
    fit_score: float | None = None
    revealed_preference: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NeedItem:
    """One user ``inferred_need`` with one or more assistant solution traces.

    ``context`` / ``context_event_ids`` summarise recent background grounded in
    this session’s extracted events (filled during ingest). Clustering still keys
    only on ``inferred_need``; each ``solutions`` row is a
    :class:`NeedSolutionProposal`.

    ``item_id`` is assigned when persisting via
    :func:`src.memory.update.session_ingest.ingest_session` as
    ``"need-{session_id}-{n}"`` (``n`` session-local).
    """

    item_id: str
    timestamp: str
    source_turn_ids: list[str]

    inferred_need: str
    need_domain: str 
    need_object: str 
    
    related_tags: list[str]
    context: str 
    context_event_ids: list[str] = field(default_factory=list)

    solutions: list[NeedSolutionProposal] = field(default_factory=list)

    cluster_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Top-layer profile schemas
# ---------------------------------------------------------------------------

@dataclass
class BasicInfoClaim:
    """One evidence-backed long-term background claim.
    claim_id:
        Existing claims keep their id. New claims can be assigned during ingest.

    category:
        One of BASIC_INFO_CATEGORIES.

    claim_type:
        One of BASIC_INFO_CLAIM_TYPES.

    content:
        A concise Chinese sentence describing a long-term user background fact,
        pattern, constraint, preference, or care context.

    source_event_ids / source_need_item_ids / source_turn_ids:
        Evidence chain. At least one should be non-empty for active/uncertain claims.

    first_seen / last_seen:
        ISO timestamps or session timestamps. Useful for aging and conflict handling.

    status:
        active: currently valid.
        uncertain: plausible but weak evidence.
        superseded: contradicted or replaced by newer evidence; kept for traceability.
    """

    claim_id: str
    category: str
    claim_type: str
    content: str

    source_event_ids: list[str] = field(default_factory=list)
    source_need_item_ids: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)

    first_seen: str = ""
    last_seen: str = ""
    confidence: float | None = None

    status: str = "active"
    supersedes: list[str] = field(default_factory=list)

    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class BasicInfoSection:
    """Long-term background section with evidence-backed claims."""
    summary: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)
    # Claim ids used to produce the current summary.
    summary_source_claim_ids: list[str] = field(default_factory=list)

    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _empty_basic_info() -> dict[str, Any]:
    return {
        "medical_care": BasicInfoSection().to_dict(),
        "family": BasicInfoSection().to_dict(),
        "health": BasicInfoSection().to_dict(),
        "leisure": BasicInfoSection().to_dict(),
    }

def _empty_recent_status() -> dict[str, Any]:
    """Default value for :attr:`UserProfile.recent_status`.

    Four narrative strings (health / self-management / mental / family-social),
    ``interest_changes`` / ``risk_flags`` phrases, time-window endpoints, and
    ``field_source_event_ids`` provenance aligned with those fields.
    """
    return {
        "health_status": "",
        "self_management_status": "",
        "mental_status": "",
        "family_social_status": "",
        "interest_changes": [],
        "risk_flags": [],
        "window_start": "",
        "window_end": "",
        "field_source_event_ids": {
            "health_status": [],
            "self_management_status": [],
            "mental_status": [],
            "family_social_status": [],
            "interest_changes": [],
            "risk_flags": [],
        }
    }

@dataclass
class NeedClusterSample:
    """A representative item kept inside a need-preference cluster.

    This is not the full :class:`NeedItem`; it is a compact copy used for
    cluster labelling / refinement and debugging.
    """
    item_id: str
    inferred_need: str
    revealed_preference: str = ""
    timestamp: str = ""
    fit_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class NeedCluster:
    """
    - ``need_domain`` is the ontology key for this cluster (one domain => one cluster).
    - ``preference_principle`` is rho_i (LLM-summarised preference principle).
    - ``member_item_ids`` tracks NeedItems assigned to this domain cluster.
    """

    cluster_id: str
    need_domain: str
    preference_principle: str

    member_item_ids: list[str] = field(default_factory=list)

    size: int = 0
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass
class UserProfile:
    basic_info: dict[str, Any] = field(default_factory=_empty_basic_info)
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
    ``window_id`` identifies the slice and during ingestion is written to both
    :attr:`RawTurn.chunk_id` for chunk-aligned
    scoping on raw turns (no separate free-text topic label field).
    """

    window_id: str
    turn_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
