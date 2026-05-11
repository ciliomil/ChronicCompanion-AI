"""Dataset loading + iteration for the ChronicCompanion eval set.

Input file is a dict::

    {
      "0000": {"history": [...], "query": [<sample>, ...]},
      "0001": {...},
      ...
    }

Each query ``sample`` has::

    {
      "sample_id": "0000_sample24",
      "dialogue_timestamp": "...",
      "dialogue": {...},
      "topics": {
        "topic-1": {
          "user_query": "...",
          "implicit_needs": [...],
          "requirement": "...",
          "solution": {"pos": [s1, s2], "neg": [s1, s2]},
          "candidate_solutions": [{"solution": "...", "feedback": "pos|neu|neg"}, ...8],
        },
        "topic-2": {...},
      }
    }

We iterate at the (user_id, sample_id, topic_id) granularity since each topic
is one independent eval unit for all three tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TopicSample:
    user_id: str
    sample_id: str
    topic_id: str
    dialogue_timestamp: str
    user_query: str
    requirement: str
    positive_solutions: list[str]
    negative_solutions: list[str]
    candidate_solutions: list[dict[str, str]]
    gold_pos_indices: list[int]

    @property
    def topic_key(self) -> tuple[str, str, str]:
        return (self.user_id, self.sample_id, self.topic_id)


def load_input(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_topic_samples(
    data: dict[str, Any],
    *,
    user_ids: list[str] | None = None,
    max_samples: int | None = None,
) -> Iterable[TopicSample]:
    """Yield :class:`TopicSample` per (user, sample, topic) triple.

    ``user_ids`` filters the user dimension; ``max_samples`` caps the number
    of *samples* (not topics) per user.
    """
    selected = _select_user_ids(data, user_ids)
    for uid in selected:
        user_payload = data.get(uid)
        if not isinstance(user_payload, dict):
            continue
        queries = user_payload.get("query") or []
        if not isinstance(queries, list):
            continue
        if max_samples is not None:
            queries = queries[:max_samples]

        for sample in queries:
            if not isinstance(sample, dict):
                continue
            sample_id = str(sample.get("sample_id") or "")
            timestamp = str(sample.get("dialogue_timestamp") or "")
            topics = sample.get("topics") or {}
            if not isinstance(topics, dict):
                continue
            for topic_id in sorted(topics.keys()):
                topic = topics[topic_id]
                if not isinstance(topic, dict):
                    continue
                yield _build_topic_sample(uid, sample_id, timestamp, str(topic_id), topic)


def _select_user_ids(
    data: dict[str, Any],
    user_ids: list[str] | None,
) -> list[str]:
    all_ids = sorted(k for k in data.keys() if isinstance(data.get(k), dict))
    if user_ids is None:
        return all_ids
    keep = [uid for uid in user_ids if uid in data]
    missing = [uid for uid in user_ids if uid not in data]
    if missing:
        raise KeyError(f"user_ids not found in input: {missing}")
    return keep


def _build_topic_sample(
    user_id: str,
    sample_id: str,
    timestamp: str,
    topic_id: str,
    topic: dict[str, Any],
) -> TopicSample:
    candidates = topic.get("candidate_solutions") or []
    norm_candidates: list[dict[str, str]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        norm_candidates.append(
            {
                "solution": str(c.get("solution") or ""),
                "feedback": str(c.get("feedback") or "").strip().lower(),
            }
        )

    gold_pos_indices = [
        i for i, c in enumerate(norm_candidates) if c["feedback"] == "pos"
    ]

    solution = topic.get("solution") or {}
    pos = solution.get("pos") if isinstance(solution, dict) else []
    neg = solution.get("neg") if isinstance(solution, dict) else []

    return TopicSample(
        user_id=user_id,
        sample_id=sample_id,
        topic_id=topic_id,
        dialogue_timestamp=timestamp,
        user_query=str(topic.get("user_query") or ""),
        requirement=str(topic.get("requirement") or ""),
        positive_solutions=[str(s) for s in (pos or [])],
        negative_solutions=[str(s) for s in (neg or [])],
        candidate_solutions=norm_candidates,
        gold_pos_indices=gold_pos_indices,
    )
