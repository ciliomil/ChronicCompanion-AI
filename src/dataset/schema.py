"""Dataset schema definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetTurn:
    session_id: str
    role: str
    text: str
    timestamp: str


@dataclass
class CandidateSolution:
    solution: str
    drawbacks: str = ""


@dataclass
class TopicSample:
    topic_id: str
    user_query: str
    requirement: str
    candidate_solutions: list[CandidateSolution] = field(default_factory=list)


@dataclass
class DialogueSample:
    sample_id: str
    dialogue_timestamp: str
    dialogue_text: str
    logs: list[dict[str, Any]] = field(default_factory=list)
    topics: list[TopicSample] = field(default_factory=list)


@dataclass
class UserDataset:
    user_id: str
    history: list[DialogueSample] = field(default_factory=list)
    query: list[DialogueSample] = field(default_factory=list)


@dataclass
class QueryTopicTask:
    user_id: str
    dialogue_id: str
    topic_id: str
    dialogue_text: str
    user_query: str
    requirement: str
    candidate_solutions: list[CandidateSolution] = field(default_factory=list)
