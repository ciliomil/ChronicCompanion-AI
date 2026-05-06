"""Schemas for retrieval-time artifacts.

Three dataclasses drive the retrieval pipeline:

- :class:`CurrentQueryFrame` — semantic understanding of the current turn. Its
  NeedItem-mirror fields (``inferred_need / need_domain / need_object / tags /
  current_context``) are deliberately shaped like :class:`memory.schemas.NeedItem`
  so the retriever can match the current turn against historical NeedItems
  with no field translation. The ``intent_type / medical_relevance / risk_*``
  fields are reply-side signals and don't participate in retrieval ranking.
- :class:`MemoryRetrievalPlan` — pure operational layer (which claim ids,
  which need_domains, top-k budgets, recent_status field selection,
  safety flag). Carries no semantic content; the retriever reads semantics
  from the frame.
- :class:`MemoryPack` — final bundle handed to the response generator,
  containing the frame, the plan (for audit), and the resolved memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CurrentQueryFrame:
    """NeedItem-mirror + reply-control representation of the current turn."""

    user_query: str
    query_summary: str

    # NeedItem-mirror block (used as the query against historical NeedItems).
    inferred_need: str
    need_domain: str
    need_object: str
    tags: list[str]
    current_context: str

    # Reply-control block (consumed by the response generator).
    intent_type: str = "casual_chat"
    medical_relevance: str = "none"
    risk_level: str = "normal"
    risk_triggers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectedBasicInfoClaim:
    """A basic_info claim picked by the planner with its application role."""

    claim_id: str
    category: str
    content: str
    claim_type: str
    tags: list[str]
    source_event_ids: list[str]
    assigned_role: str
    confidence: float | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRetrievalPlan:
    """Operational knobs for retrieval. Semantic content lives on the frame."""

    # basic_info: planner picks claims and tags each one with an application role.
    selected_basic_info_claims: list[dict[str, Any]] = field(default_factory=list)

    # need_preferences: which clusters' principles to surface.
    target_need_domains: list[str] = field(default_factory=list)

    # Historical NeedItem recall (semantic signal comes from the frame).
    need_top_k: int = 0

    # Event recall (seeded by selected_basic_info_claims.source_event_ids;
    # frame.tags drives tag-overlap fill).
    event_top_k: int = 0

    # recent_status slice.
    include_recent_status_fields: list[str] = field(default_factory=list)

    safety_sensitive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryPack:
    """Final bundle handed to the response generator."""

    session_id: str
    turn_id: str
    timestamp: str
    user_query: str

    current_query: CurrentQueryFrame
    plan: MemoryRetrievalPlan

    selected_basic_info_claims: list[dict[str, Any]] = field(default_factory=list)
    preference_principles: list[dict[str, Any]] = field(default_factory=list)
    relevant_needs: list[dict[str, Any]] = field(default_factory=list)
    relevant_events: list[dict[str, Any]] = field(default_factory=list)
    recent_status_slice: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
