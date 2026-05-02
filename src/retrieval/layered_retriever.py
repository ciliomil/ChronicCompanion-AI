"""Layered retrieval over mid memory with raw evidence fallback."""

from __future__ import annotations


def retrieve_layered_context(
    need_result: dict,
    events: list[dict],
    need_solution_items: list[dict],
    raw_turns: list[dict],
) -> dict:
    primary_need = need_result.get("primary_need", "general_companionship")

    matched_need_items = [
        item for item in need_solution_items if item.get("inferred_need") == primary_need
    ]
    matched_need_items = matched_need_items[-3:]

    matched_events = []
    for event in reversed(events):
        tags = event.get("tags", [])
        if primary_need.split("_")[0] in "_".join(tags) or "general" in tags:
            matched_events.append(event)
        if len(matched_events) >= 2:
            break
    matched_events.reverse()

    source_turn_ids = {
        turn_id
        for item in matched_need_items
        for turn_id in item.get("source_turn_ids", [])
    }
    raw_evidence = [turn for turn in raw_turns if turn.get("turn_id") in source_turn_ids][-4:]

    return {
        "primary_need": primary_need,
        "matched_need_items": matched_need_items,
        "matched_events": matched_events,
        "raw_evidence": raw_evidence,
    }
