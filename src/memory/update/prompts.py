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
    "medical_visit",
    "life_event",
)

EVENT_TAGS: tuple[str, ...] = (
    "medical",
    "lifestyle",
    "family",
    "emotion",
    "diet",
    "sleep",
    "activity",
    "other",
)

# Tags that should land in the "medical / body" bucket when summarising
# recent_status. Diet / sleep / activity also affect blood-sugar control so we
# count them as health-relevant for diabetic users.
_MEDICAL_BUCKET_TAGS = {"medical", "diet", "sleep", "activity"}
_LIFE_BUCKET_TAGS = {"family", "emotion", "lifestyle", "other"}


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

# need_solution inferred_need are free-form; tags still controlled. Prompt is
# topic-window — one call covers every user→assistant pair in that window plus
# preference / quality_score / feedback_turn_ids in the same structured output.
NEED_SOLUTION_EXTRACT_SYSTEM: str = load_prompt(
    "memory/need_solution_extract/system",
    event_tags=_vocab_str(EVENT_TAGS),
)

FEEDBACK_ANALYZE_SYSTEM: str = load_prompt("memory/feedback_analyze/system")

RECENT_STATUS_SUMMARIZE_SYSTEM: str = load_prompt("memory/recent_status_summarize/system")
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


def split_events_by_bucket(
    events: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split events into ``(medical_bucket, life_bucket)`` lists.

    Buckets are decided by ``event_type`` first (``medical_visit`` always goes
    medical) and by tag overlap with :data:`_MEDICAL_BUCKET_TAGS` otherwise.
    Events that match neither bucket-tag set fall into the life bucket.
    """
    medical: list[dict[str, Any]] = []
    life: list[dict[str, Any]] = []
    for ev in events:
        tags = set(ev.get("tags", []) or [])
        et = str(ev.get("event_type", ""))
        if et == "medical_visit" or (tags & _MEDICAL_BUCKET_TAGS):
            medical.append(ev)
        else:
            life.append(ev)
    return medical, life


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
) -> str:
    transcript = render_turns(window_turns)
    return load_prompt(
        "memory/need_solution_extract/user",
        window_transcript=transcript,
    )


def build_feedback_analyze_prompt(
    item: dict[str, Any],
    follow_up_turns: list[dict[str, Any]],
) -> str:
    inferred_need = item.get("inferred_need", "")
    ai_solution = item.get("ai_solution_summary", "")
    item_id = item.get("item_id", "")
    follow_up = render_turns(follow_up_turns)
    return load_prompt(
        "memory/feedback_analyze/user",
        item_id=item_id,
        inferred_need=inferred_need,
        ai_solution_summary=ai_solution,
        follow_up=follow_up,
    )


def build_recent_status_summarize_prompt(
    *,
    old_recent_status: dict[str, Any],
    medical_events: list[dict[str, Any]],
    life_events: list[dict[str, Any]],
    window_days: int,
) -> str:
    return load_prompt(
        "memory/recent_status_summarize/user",
        old_recent_status=json.dumps(old_recent_status, ensure_ascii=False),
        medical_events=render_event_lines(medical_events),
        life_events=render_event_lines(life_events),
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
