"""Event extraction for the mid-layer memory.

Two extractors are provided:

- :class:`RuleEventExtractor` — lightweight keyword heuristics, no network.
- :class:`LLMEventExtractor` — LLM-driven structured extraction over a
  controlled vocabulary defined in :mod:`src.memory.update.prompts`.

Both expose the same surface:

    extract_from_turn(turn) -> EventItem | None
    extract_from_session(turns) -> list[EventItem]

"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from src.llm.llm import LLMClient, get_default_llm_client
from src.memory.schemas import EventItem, RawTurn
from src.memory.update.prompts import (
    EVENT_EXTRACT_SYSTEM,
    EVENT_TAGS,
    EVENT_TYPES,
    build_event_extract_prompt,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


class RuleEventExtractor:
    """Naive keyword-based extractor used as a deterministic fallback.

    Mirrors the original prototype logic so behaviour is unchanged when no
    LLM is available (offline tests, CI without network, etc.).
    """

    def extract_from_turn(self, turn: RawTurn) -> EventItem | None:
        if turn.role != "user":
            return None

        text = turn.text.strip()
        if not text:
            return None

        event_type = "daily_life"
        tags: list[str] = []

        if any(token in text for token in ("医院", "复诊", "看病", "化验", "就诊")):
            event_type = "health_medical"
            tags.append("medical_visit")
        if any(token in text for token in ("散步", "运动")):
            tags.append("activity")
        if any(token in text for token in ("饮食",)):
            tags.append("diet")
        if any(token in text for token in ("睡眠",)):
            tags.append("sleep")
        if any(token in text for token in ("家人", "孩子", "老伴")):
            tags.append("family")

        tags = [t for t in tags if t in EVENT_TAGS]
        if not tags:
            tags = ["other"]

        return EventItem(
            event_id=f"event-{turn.turn_id}",
            event_type=event_type,
            timestamp=turn.timestamp,
            source_turn_ids=[turn.turn_id],
            event_summary=text[:80],
            tags=tags,
            confidence=None,
        )

    def extract_from_session(
        self,
        turns: list[RawTurn],
        *,
        window_id: str | None = None,
    ) -> list[EventItem]:
        del window_id  # rule path doesn't need a session/window anchor — kept for parity with :class:`LLMEventExtractor`.
        events: list[EventItem] = []
        seen_summaries: set[str] = set()
        for turn in turns:
            event = self.extract_from_turn(turn)
            if event is None:
                continue
            if event.event_summary in seen_summaries:
                continue
            seen_summaries.add(event.event_summary)
            events.append(event)
        return events


# ---------------------------------------------------------------------------
# LLM-driven extractor
# ---------------------------------------------------------------------------


class LLMEventExtractor:
    """LLM-driven event extractor that returns schema-validated EventItems."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        fallback: RuleEventExtractor | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or RuleEventExtractor()

    @property
    def client(self) -> LLMClient:
        """Lazy default-client resolution so tests can monkeypatch easily."""
        if self._client is None:
            self._client = get_default_llm_client()
        return self._client

    def extract_from_turn(self, turn: RawTurn) -> EventItem | None:
        if turn.role != "user" or not turn.text.strip():
            return None
        events = self.extract_from_session([turn])
        return events[0] if events else None

    def extract_from_session(
        self,
        turns: list[RawTurn],
        *,
        session_id: str | None = None,
    ) -> list[EventItem]:
        turn_dicts = [t.to_dict() for t in turns if t.text.strip()]
        if not turn_dicts:
            return []

        try:
            prompt = build_event_extract_prompt(turn_dicts)
            response = self.client.generate_json(
                prompt,
                system_prompt=EVENT_EXTRACT_SYSTEM,
            )
            return self._build_events(response, turns, session_id=session_id)
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "LLMEventExtractor failed (%s); falling back to rule extractor.",
                err,
            )
            return self._fallback.extract_from_session(list(turns))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _build_events(
        response: dict[str, Any],
        turns: list[RawTurn],
        *,
        session_id: str | None,
    ) -> list[EventItem]:
        raw_events = response.get("events") if isinstance(response, dict) else None
        if not isinstance(raw_events, list):
            return []

        valid_turn_ids = {t.turn_id for t in turns}
        timestamp_by_turn = {t.turn_id: t.timestamp for t in turns}
        last_timestamp = turns[-1].timestamp if turns else ""

        seen_signatures: set[tuple[str, frozenset[str]]] = set()
        out: list[EventItem] = []

        for index, raw in enumerate(raw_events):
            event = LLMEventExtractor._coerce_event(
                raw,
                index=index,
                session_id=session_id,
                valid_turn_ids=valid_turn_ids,
                timestamp_by_turn=timestamp_by_turn,
                last_timestamp=last_timestamp,
            )
            if event is None:
                continue
            sig = (event.event_summary.strip(), frozenset(event.source_turn_ids))
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            out.append(event)
        return out

    @staticmethod
    def _coerce_event(
        raw: Any,
        *,
        index: int,
        session_id: str | None,
        valid_turn_ids: set[str],
        timestamp_by_turn: dict[str, str],
        last_timestamp: str,
    ) -> EventItem | None:
        if not isinstance(raw, dict):
            return None

        event_type = str(raw.get("event_type", "")).strip()
        summary = str(raw.get("event_summary", "")).strip()
        if event_type not in EVENT_TYPES or not summary:
            return None

        source_turn_ids = _coerce_str_list(raw.get("source_turn_ids"))
        # Restrict source_turn_ids to those actually present in the session.
        # If the model hallucinated ids, drop them; if everything is dropped,
        # fall back to the latest turn's id when available.
        filtered = [tid for tid in source_turn_ids if tid in valid_turn_ids]
        if not filtered and valid_turn_ids:
            filtered = [next(iter(valid_turn_ids))]
        if not filtered:
            return None

        tags = [t for t in _coerce_str_list(raw.get("tags")) if t in EVENT_TAGS]
        if not tags:
            tags = ["other"]

        confidence = _coerce_float(raw.get("confidence"))

        # Pick the latest timestamp among source turns; gives the event a
        # sensible position on the user's timeline.
        timestamp = max(
            (timestamp_by_turn.get(tid, "") for tid in filtered),
            default=last_timestamp,
        ) or last_timestamp

        primary = filtered[0]
        suffix = f"-{index+1}" if index >= 0 else ""
        anchor = session_id or primary
        event_id = f"event-{anchor}{suffix}"

        return EventItem(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            source_turn_ids=filtered,
            event_summary=summary[:200],
            tags=tags,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Backward-compatible module API
# ---------------------------------------------------------------------------


_DEFAULT_EXTRACTOR: LLMEventExtractor | None = None


def _get_default_extractor() -> LLMEventExtractor:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = LLMEventExtractor()
    return _DEFAULT_EXTRACTOR


def reset_default_extractor() -> None:
    """Drop the cached default extractor (useful in tests)."""
    global _DEFAULT_EXTRACTOR
    _DEFAULT_EXTRACTOR = None


def extract_event_from_turn(turn: RawTurn) -> EventItem | None:
    """Backward-compatible entry point — delegates to the default extractor."""
    return _get_default_extractor().extract_from_turn(turn)


def extract_events_from_window(
    turns: Iterable[RawTurn],
    *,
    window_id: str | None = None,
) -> list[EventItem]:
    """Convenience wrapper for window-level extraction."""
    return _get_default_extractor().extract_from_session(list(turns), window_id=window_id)


# ---------------------------------------------------------------------------
# Internal coercion helpers
# ---------------------------------------------------------------------------


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f
