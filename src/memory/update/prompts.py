"""Prompts and controlled vocabularies for mid- and top-layer memory extraction.

Prompt files live under ``src/prompts/memory/<task>/``. Each task owns two
templates:

- ``system.md``: role, constraints, controlled vocabularies, examples.
- ``user.md``: per-call input rendering.

Variant switching still works per file: for example
``LONGMEM_PROMPT_VARIANT=v2`` will prefer ``...system.v2.md`` /
``...user.v2.md`` when present, falling back to the default file otherwise.

Note on controlled vocabularies
-------------------------------
``inferred_need`` / ``primary_need`` are now LLM-produced free-form Chinese
phrases (no controlled vocabulary). The ``NEED_TAXONOMY`` constant below is
kept around purely for the deterministic rule fallback in
:mod:`src.memory.update.need_solution_extractor`; it is **not** injected into
any LLM prompt.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from src.llm.prompt_loader import load_prompt

# ---------------------------------------------------------------------------
# Controlled vocabularies (rule-fallback only for needs)
# ---------------------------------------------------------------------------

NEED_TAXONOMY: tuple[str, ...] = (
    "diet_support",
    "activity_support",
    "sleep_support",
    "medication_support",
    "emotional_support",
    "knowledge_explanation",
    "general_companionship",
)

EVENT_TYPES: tuple[str, ...] = (
    "health_medical",
    "self_management",
    "family_social",
    "mental_emotional",
    "interest_activity",
    "daily_life",
    "other",
)

EVENT_TAGS: tuple[str, ...] = (
    "glucose",
    "medication",
    "medical_visit",
    "symptom",
    "complication",
    "diet",
    "sleep",
    "activity",
    "monitoring",
    "adherence",
    "family",
    "caregiver",
    "living_alone",
    "social",
    "emotion",
    "stress",
    "hobby",
    "safety_risk",
    "other",
)

def _vocab_str(values: tuple[str, ...]) -> str:
    return ", ".join(values)


# ---------------------------------------------------------------------------
# Loaded system prompts
# ---------------------------------------------------------------------------

EVENT_EXTRACT_SYSTEM: str = load_prompt(
    "memory/event_extract/system",
    event_types=_vocab_str(EVENT_TYPES),
    event_tags=_vocab_str(EVENT_TAGS),
)

# need_infer no longer takes a controlled need taxonomy.
NEED_INFER_SYSTEM: str = load_prompt("memory/need_infer/system")

# need extract: inferred_need free-form; tags controlled. items[] ≈ NeedItem:
# inferred_need, related_tags, context, context_event_ids, solutions[] rows
# (ai_solution_summary, feedback_turn_ids, quality_score, preference, confidence).
NEED_SOLUTION_EXTRACT_SYSTEM: str = load_prompt(
    "memory/need_solution_extract/system",
    event_tags=_vocab_str(EVENT_TAGS),
)

RECENT_STATUS_SUMMARIZE_SYSTEM: str = load_prompt("memory/recent_status_summarize/system")
BASIC_INFO_UPDATE_SYSTEM: str = load_prompt("memory/basic_info_update/system")
NEED_CLUSTER_LABEL_SYSTEM: str = load_prompt("memory/need_cluster_label/system")
NEED_CLUSTER_REFINE_SYSTEM: str = load_prompt("memory/need_cluster_refine/system")
TOPIC_SEGMENT_SYSTEM: str = load_prompt("memory/topic_segment/system")


# ---------------------------------------------------------------------------
# Helpers for building per-call user prompts
# ---------------------------------------------------------------------------

def render_turns(turns: list[dict[str, Any]]) -> str:
    """Render a list of raw-turn dicts into a compact, turn-id-tagged transcript.

    Each turn is formatted as ``[turn_id|role|timestamp] text``. Used by
    extractors when they need to ground LLM output back to specific turns.
    """
    lines: list[str] = []
    for turn in turns:
        turn_id = str(turn.get("turn_id", "?"))
        role = str(turn.get("role", "?"))
        timestamp = str(turn.get("timestamp", ""))
        text = str(turn.get("text", "")).strip().replace("\n", " ")
        lines.append(f"[{turn_id}|{role}|{timestamp}] {text}")
    return "\n".join(lines)


def render_event_lines(events: Iterable[dict[str, Any]]) -> str:
    """Render events as ``[event_id|YYYY-MM-DD] summary`` lines.

    Returns the literal string ``(无)`` when empty so the LLM doesn't get an
    awkward blank section.
    """
    lines: list[str] = []
    for ev in events:
        eid = str(ev.get("event_id", "?"))
        ts = str(ev.get("timestamp", "")).split("T")[0]
        summary = str(ev.get("event_summary", "")).strip().replace("\n", " ")
        lines.append(f"[{eid}|{ts}] {summary}")
    return "\n".join(lines) if lines else "(无)"


def _event_tags(ev: dict[str, Any]) -> set[str]:
    raw = ev.get("tags")
    if not isinstance(raw, list):
        return set()
    return {str(t).strip() for t in raw if str(t).strip()}


def _event_type(ev: dict[str, Any]) -> str:
    return str(ev.get("event_type", "")).strip()


def split_events_by_bucket(
    events: Iterable[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Split events for :func:`summarize_recent_status`.

    An event may appear in multiple lists. Returns:

    ``(health, self_management, mental, family_social, interest, risk_hint)``

    ``risk_hint`` lists events tagged ``safety_risk`` to ground ``risk_flags``.
    """
    health_events: list[dict[str, Any]] = []
    self_management_events: list[dict[str, Any]] = []
    mental_events: list[dict[str, Any]] = []
    family_social_events: list[dict[str, Any]] = []
    interest_events: list[dict[str, Any]] = []
    risk_hint_events: list[dict[str, Any]] = []

    health_tags = {"glucose", "medication", "medical_visit", "symptom", "complication"}
    self_tags = {"diet", "sleep", "activity", "monitoring", "adherence"}
    mental_tags = {"emotion", "stress"}
    family_tags = {"family", "caregiver", "living_alone", "social"}
    interest_tags = {"hobby", "activity"}

    for ev in events:
        if not isinstance(ev, dict):
            continue
        tags = _event_tags(ev)
        et = _event_type(ev)

        if et == "health_medical" or (tags & health_tags):
            health_events.append(ev)
        if et == "self_management" or (tags & self_tags):
            self_management_events.append(ev)
        if et == "mental_emotional" or (tags & mental_tags):
            mental_events.append(ev)
        if et == "family_social" or (tags & family_tags):
            family_social_events.append(ev)
        if et == "interest_activity" or (tags & interest_tags):
            interest_events.append(ev)
        if "safety_risk" in tags:
            risk_hint_events.append(ev)

    return (
        health_events,
        self_management_events,
        mental_events,
        family_social_events,
        interest_events,
        risk_hint_events,
    )


def build_event_extract_prompt(turns: list[dict[str, Any]]) -> str:
    transcript = render_turns(turns)
    return load_prompt("memory/event_extract/user", transcript=transcript)


def build_need_infer_prompt(
    query: str,
    profile_hint: dict[str, Any] | None = None,
    top_needs: list[str] | None = None,
) -> str:
    profile_text = profile_hint or {}
    top_needs_text = top_needs or []
    return load_prompt(
        "memory/need_infer/user",
        query=query,
        profile_hint=profile_text,
        top_needs=top_needs_text,
    )


def build_need_solution_extract_prompt(
    window_turns: list[dict[str, Any]],
    *,
    session_events: list[dict[str, Any]] | None = None,
) -> str:
    transcript = render_turns(window_turns)
    return load_prompt(
        "memory/need_solution_extract/user",
        window_transcript=transcript,
        session_events=render_event_lines(session_events or []),
    )



def build_basic_info_update_prompt(
    *,
    old_basic_info: dict[str, Any],
    session_events: list[dict[str, Any]],
    session_need_items: list[dict[str, Any]],
) -> str:
    return load_prompt(
        "memory/basic_info_update/user",
        old_basic_info=json.dumps(old_basic_info, ensure_ascii=False),
        session_events=json.dumps(session_events, ensure_ascii=False),
        session_need_items=json.dumps(session_need_items, ensure_ascii=False),
    )


def build_recent_status_summarize_prompt(
    *,
    old_recent_status: dict[str, Any],
    health_events: list[dict[str, Any]],
    self_management_events: list[dict[str, Any]],
    mental_events: list[dict[str, Any]],
    family_social_events: list[dict[str, Any]],
    interest_events: list[dict[str, Any]],
    risk_hint_events: list[dict[str, Any]],
    window_days: int,
) -> str:
    return load_prompt(
        "memory/recent_status_summarize/user",
        old_recent_status=json.dumps(old_recent_status, ensure_ascii=False),
        health_events=render_event_lines(health_events),
        self_management_events=render_event_lines(self_management_events),
        mental_events=render_event_lines(mental_events),
        family_social_events=render_event_lines(family_social_events),
        interest_events=render_event_lines(interest_events),
        risk_hint_events=render_event_lines(risk_hint_events),
        window_days=window_days,
    )


def build_need_cluster_label_prompt(samples: list[tuple[str, str]]) -> str:
    """Render same-cluster (need, preference) samples for the labelling LLM."""
    rendered = "\n".join(
        f"- 需求：\"{r}\" / 偏好：\"{p}\"" for r, p in samples
    ) or "(无样本)"
    return load_prompt("memory/need_cluster_label/user", samples=rendered)


def build_topic_segment_prompt(turns: list[dict[str, Any]]) -> str:
    """Render dialogue turns for the topic segmentation LLM call."""
    transcript = render_turns(turns)
    return load_prompt("memory/topic_segment/user", transcript=transcript)


def build_need_cluster_refine_prompt(
    *,
    prev_need_type: str,
    prev_preference_principle: str,
    history_samples: list[tuple[str, str]],
    new_need: str,
    new_preference: str,
) -> str:
    rendered = "\n".join(
        f"- 需求：\"{r}\" / 偏好：\"{p}\"" for r, p in history_samples
    ) or "(无历史样本)"
    return load_prompt(
        "memory/need_cluster_refine/user",
        prev_need_type=prev_need_type,
        prev_preference_principle=prev_preference_principle,
        history_samples=rendered,
        new_need=new_need,
        new_preference=new_preference,
    )
