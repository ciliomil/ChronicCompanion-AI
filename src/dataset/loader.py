"""Load dataset files for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.dataset.schema import (
    CandidateSolution,
    DialogueSample,
    QueryTopicTask,
    TopicSample,
    UserDataset,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stringify_dialogue(dialogue: dict[str, Any]) -> str:
    if not dialogue:
        return ""

    lines: list[str] = []
    turn_keys = sorted(dialogue.keys(), key=lambda key: int(str(key).split("_")[-1]))
    for turn_key in turn_keys:
        turn_item = dialogue.get(turn_key, {})
        user_text = turn_item.get("user", {}).get("content", "")
        assistant_text = turn_item.get("assistant", {}).get("content", "")
        if user_text:
            lines.append(f"user: {user_text}")
        if assistant_text:
            lines.append(f"assistant: {assistant_text}")
    return "\n".join(lines)


def _parse_topic(topic_id: str, topic_data: dict[str, Any]) -> TopicSample:
    candidates = [
        CandidateSolution(
            solution=str(item.get("solution", "")).strip(),
            drawbacks=str(item.get("drawbacks", "")).strip(),
        )
        for item in topic_data.get("candidate_solutions", [])
        if str(item.get("solution", "")).strip()
    ]
    requirement = str(topic_data.get("requirement", "")).strip()
    user_query = str(topic_data.get("user_query", "")).strip()
    return TopicSample(
        topic_id=topic_id,
        user_query=user_query,
        requirement=requirement,
        candidate_solutions=candidates,
    )


def _parse_dialogue_sample(item: dict[str, Any]) -> DialogueSample:
    topics_map = item.get("topics", {})
    topics = [
        _parse_topic(topic_id=topic_id, topic_data=topic_data)
        for topic_id, topic_data in topics_map.items()
        if isinstance(topic_data, dict)
    ]
    return DialogueSample(
        sample_id=str(item.get("sample_id", "")).strip(),
        dialogue_timestamp=str(item.get("dialogue_timestamp", "")).strip(),
        dialogue_text=stringify_dialogue(item.get("dialogue", {})),
        logs=item.get("logs", []),
        topics=topics,
    )


def load_mempal_style_dataset(path: Path) -> dict[str, UserDataset]:
    raw = load_json(path)
    parsed: dict[str, UserDataset] = {}
    for user_id, user_data in raw.items():
        if not isinstance(user_data, dict):
            continue
        history = [_parse_dialogue_sample(item) for item in user_data.get("history", [])]
        query = [_parse_dialogue_sample(item) for item in user_data.get("query", [])]
        parsed[user_id] = UserDataset(user_id=user_id, history=history, query=query)
    return parsed


def build_history_chunks(user_data: UserDataset) -> list[dict[str, str]]:
    def _slice_text(text: str, max_chars: int = 320) -> list[str]:
        compact = text.strip()
        if not compact:
            return []
        if len(compact) <= max_chars:
            return [compact]
        parts: list[str] = []
        start = 0
        while start < len(compact):
            parts.append(compact[start : start + max_chars])
            start += max_chars
        return parts

    chunks: list[dict[str, str]] = []
    for sample in user_data.history:
        for segment in _slice_text(sample.dialogue_text):
            chunks.append(
                {
                    "text": segment,
                    "source_type": "history_dialogue",
                    "sample_id": sample.sample_id,
                    "timestamp": sample.dialogue_timestamp,
                }
            )
        for idx, log in enumerate(sample.logs):
            content = str(log.get("content", "")).strip()
            if not content:
                continue
            chunks.append(
                {
                    "text": content,
                    "source_type": "history_log",
                    "sample_id": sample.sample_id,
                    "timestamp": str(log.get("timestamp", sample.dialogue_timestamp)),
                    "log_idx": str(idx),
                }
            )
    return chunks


def iter_query_topic_tasks(user_data: UserDataset) -> list[QueryTopicTask]:
    tasks: list[QueryTopicTask] = []
    for dialogue in user_data.query:
        for topic in dialogue.topics:
            tasks.append(
                QueryTopicTask(
                    user_id=user_data.user_id,
                    dialogue_id=dialogue.sample_id,
                    topic_id=topic.topic_id,
                    dialogue_text=dialogue.dialogue_text,
                    user_query=topic.user_query,
                    requirement=topic.requirement,
                    candidate_solutions=topic.candidate_solutions,
                )
            )
    return tasks
