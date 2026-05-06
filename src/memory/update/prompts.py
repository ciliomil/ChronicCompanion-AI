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
from src.memory.ontology import (
    BASIC_INFO_CLAIM_TYPE_DESCRIPTIONS,
    MEMORY_TAGS,
    NEED_DOMAINS,
)

# ---------------------------------------------------------------------------
# Controlled vocabularies (rule-fallback only for needs)
# ---------------------------------------------------------------------------

NEED_TAXONOMY: tuple[str, ...] = tuple(NEED_DOMAINS.keys())
EVENT_TAGS: tuple[str, ...] = MEMORY_TAGS


def _vocab_str(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def _claim_type_descriptions_str() -> str:
    """Render BASIC_INFO_CLAIM_TYPE_DESCRIPTIONS as a bulleted list."""
    lines: list[str] = []
    for key, desc in BASIC_INFO_CLAIM_TYPE_DESCRIPTIONS.items():
        lines.append(f"- {key}：{desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loaded system prompts
# ---------------------------------------------------------------------------

EVENT_EXTRACT_SYSTEM: str = load_prompt(
    "memory/event_extract/system",
    event_tags=_vocab_str(EVENT_TAGS),
)

# need_infer no longer takes a controlled need taxonomy.
NEED_INFER_SYSTEM: str = load_prompt("memory/need_infer/system")
NEED_SOLUTION_EXTRACT_SYSTEM: str = load_prompt(
    "memory/need_solution_extract/system",
    need_domains=_vocab_str(tuple(NEED_DOMAINS.keys())),
    memory_tags=_vocab_str(MEMORY_TAGS),
)
RECENT_STATUS_SUMMARIZE_SYSTEM: str = load_prompt("memory/recent_status_summarize/system")

# basic_info update is split into a three-stage pipeline:
# 1. propose candidate long-term claims from this session's events / needs;
# 2. consolidate candidates against existing claims (add | update | supersede | ignore);
# 3. re-summarize each section based on the merged active / uncertain claims.
BASIC_INFO_PROPOSE_SYSTEM: str = load_prompt(
    "memory/basic_info_propose/system",
    claim_type_descriptions=_claim_type_descriptions_str(),
)
BASIC_INFO_CONSOLIDATE_SYSTEM: str = load_prompt(
    "memory/basic_info_consolidate/system",
    claim_type_descriptions=_claim_type_descriptions_str(),
)
BASIC_INFO_SUMMARIZE_SYSTEM: str = load_prompt("memory/basic_info_summarize/system")

PREFERENCE_PRINCIPLE_UPDATE_SYSTEM: str = load_prompt("memory/preference_principle_update/system")
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
        if tags & health_tags:
            health_events.append(ev)
        if tags & self_tags:
            self_management_events.append(ev)
        if tags & mental_tags:
            mental_events.append(ev)
        if tags & family_tags:
            family_social_events.append(ev)
        if tags & interest_tags:
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



def render_event_lines_with_tags(events: Iterable[dict[str, Any]]) -> str:
    """Render events as ``[event_id|YYYY-MM-DD] summary [tags: a, b]`` lines.

    Used by the basic_info propose stage so the LLM can constrain candidate
    tags to the union of cited evidence's tags. Returns ``(无)`` when empty.
    """
    lines: list[str] = []
    for ev in events:
        eid = str(ev.get("event_id", "?"))
        ts = str(ev.get("timestamp", "")).split("T")[0]
        summary = str(ev.get("event_summary", "")).strip().replace("\n", " ")
        raw_tags = ev.get("tags")
        tags: list[str] = []
        if isinstance(raw_tags, list):
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        tag_part = f" [tags: {', '.join(tags)}]" if tags else " [tags: ]"
        lines.append(f"[{eid}|{ts}] {summary}{tag_part}")
    return "\n".join(lines) if lines else "(无)"


def build_basic_info_propose_prompt(
    *,
    session_events: list[dict[str, Any]],
    session_need_items: list[dict[str, Any]],
) -> str:
    return load_prompt(
        "memory/basic_info_propose/user",
        session_events=render_event_lines_with_tags(session_events or []),
        session_need_items=json.dumps(session_need_items or [], ensure_ascii=False),
    )


def build_basic_info_consolidate_prompt(
    *,
    existing_claims: str,
    candidates: str,
) -> str:
    return load_prompt(
        "memory/basic_info_consolidate/user",
        existing_claims=existing_claims,
        candidates=candidates,
    )


def build_basic_info_summarize_prompt(
    *,
    section_claims: str,
) -> str:
    return load_prompt(
        "memory/basic_info_summarize/user",
        section_claims=section_claims,
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


def build_preference_principle_update_prompt(
    *,
    need_domain: str,
    existing_preference_principle: str,
    need_item: dict[str, Any],
) -> str:
    return load_prompt(
        "memory/preference_principle_update/user",
        need_domain=need_domain,
        existing_preference_principle=existing_preference_principle or "",
        need_item_json=json.dumps(need_item, ensure_ascii=False),
    )



def build_topic_segment_prompt(turns: list[dict[str, Any]]) -> str:
    """Render dialogue turns for the topic segmentation LLM call."""
    transcript = render_turns(turns)
    return load_prompt("memory/topic_segment/user", transcript=transcript)