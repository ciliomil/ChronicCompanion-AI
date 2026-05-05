"""Schemas for retrieval outputs (e.g. context bundles for generation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RelevantClaim:
    category: str
    # medical_care / family / health / leisure
    content: str

    use_role: str = "context"
    # context / constraint / preference / care_context / risk_relevant
    source_event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryQuery:
    current_need: str
    current_context: str
    current_tags: list[str]

    relevant_claims: list[RelevantClaim] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryPack:
    session_id: str
    turn_id: str
    timestamp: str
    user_query: str

    # 1. 当前请求的理解
    current_query: MemoryQuery

    # 2. 由 cluster 得到的 preference principle
    preference_principles: list[str] = field(default_factory=list)

    # 3. mid 层证据和可以借鉴的信息
    relevant_needs: list[dict[str, Any]] = field(default_factory=list)
    relevant_events: list[dict[str, Any]] = field(default_factory=list)

    # 4. 近期状态
    recent_status: dict[str, Any] = field(default_factory=dict)

    # 5. 特别关注点
    safety_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
