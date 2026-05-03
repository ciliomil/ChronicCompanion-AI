"""Topic-window segmentation for session ingestion.

Three implementations are provided:

- :class:`LLMTopicSegmenter` — one LLM call over the whole session, returning
  a list of contiguous, non-overlapping :class:`TopicWindow`s.
- :class:`RuleTopicSegmenter` — deterministic fallback that groups every
  ``window_size`` user/assistant pairs into one window. Used when the LLM
  call fails or when running offline.
- :func:`windows_from_dataset_topics` — adapter that turns the dataset's
  per-sample ``topics`` block into matching :class:`TopicWindow`s, mapping
  each topic's ``user_query`` to the closest user turn via character-bigram
  Jaccard similarity. Lets evaluation runs use gold topic boundaries to
  isolate downstream behaviour from segmenter quality.

All three converge on the same surface so the orchestrator can swap them.
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm.llm import LLMClient, get_default_llm_client
from src.memory.schemas import RawTurn, TopicWindow
from src.memory.update.prompts import (
    TOPIC_SEGMENT_SYSTEM,
    build_topic_segment_prompt,
)

_logger = logging.getLogger(__name__)

_DEFAULT_RULE_WINDOW = 4  # turns per fallback chunk (≈2 user/assistant pairs)


# ---------------------------------------------------------------------------
# Helpers shared by all paths
# ---------------------------------------------------------------------------


def _coerce_window_payload(
    response: Any,
    *,
    valid_turn_ids: list[str],
) -> list[TopicWindow] | None:
    """Validate the LLM-produced segmentation against the input turn list.

    Returns ``None`` (rather than partial windows) if the response fails any
    invariant; the caller can then fall back to the rule segmenter and avoid
    silently dropping turns. Invariants:

    - response must be a dict with a list ``windows``;
    - each window has ``window_id`` and ``turn_ids: list[str]``;
    - turn_ids span exactly the input set (same ids, no extras, no missing);
    - turn order across windows must match the input order.
    """
    if not isinstance(response, dict):
        return None
    raw_windows = response.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        return None

    seen: list[str] = []
    out: list[TopicWindow] = []
    for idx, raw in enumerate(raw_windows):
        if not isinstance(raw, dict):
            return None
        ids_field = raw.get("turn_ids")
        if not isinstance(ids_field, list) or not ids_field:
            return None
        ids = [str(t) for t in ids_field if str(t).strip()]
        if not ids:
            return None
        for tid in ids:
            if tid not in valid_turn_ids:
                return None
            if tid in seen:
                return None
            seen.append(tid)
        out.append(
            TopicWindow(
                window_id=str(raw.get("window_id") or f"w-{idx + 1}").strip(),
                turn_ids=ids,
            )
        )

    if seen != valid_turn_ids:
        return None
    return out


# ---------------------------------------------------------------------------
# LLM segmenter
# ---------------------------------------------------------------------------


class LLMTopicSegmenter:
    """Whole-session LLM segmenter; falls back to :class:`RuleTopicSegmenter`."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        fallback: "RuleTopicSegmenter | None" = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or RuleTopicSegmenter()

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_default_llm_client()
        return self._client

    def segment(self, turns: list[RawTurn]) -> list[TopicWindow]:
        if not turns:
            return []
        if len(turns) <= 2:
            # Too short to LLM-segment usefully.
            return self._fallback.segment(turns)

        try:
            prompt = build_topic_segment_prompt([t.to_dict() for t in turns])
            response = self.client.generate_json(prompt, system_prompt=TOPIC_SEGMENT_SYSTEM)
            windows = _coerce_window_payload(
                response, valid_turn_ids=[t.turn_id for t in turns]
            )
            if windows:
                return windows
            _logger.warning(
                "LLMTopicSegmenter response failed validation; falling back to rule.",
            )
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "LLMTopicSegmenter failed (%s); falling back to rule segmenter.", err,
            )
        return self._fallback.segment(turns)


# ---------------------------------------------------------------------------
# Rule fallback
# ---------------------------------------------------------------------------


class RuleTopicSegmenter:
    """Group every ``window_size`` turns into one chunk; deterministic."""

    def __init__(self, window_size: int = _DEFAULT_RULE_WINDOW) -> None:
        self._window_size = max(2, window_size)

    def segment(self, turns: list[RawTurn]) -> list[TopicWindow]:
        if not turns:
            return []
        windows: list[TopicWindow] = []
        i = 0
        idx = 1
        while i < len(turns):
            chunk = turns[i : i + self._window_size]
            windows.append(
                TopicWindow(
                    window_id=f"w-{idx}",
                    turn_ids=[t.turn_id for t in chunk],
                )
            )
            i += self._window_size
            idx += 1
        return windows


# ---------------------------------------------------------------------------
# Gold topics adapter (dataset-driven)
# ---------------------------------------------------------------------------


def _char_bigrams(text: str) -> set[str]:
    text = "".join(text.split())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _char_bigrams(a), _char_bigrams(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def windows_from_dataset_topics(
    turns: list[RawTurn],
    topics_block: dict[str, dict[str, Any]],
) -> list[TopicWindow]:
    """Build :class:`TopicWindow`s from the dataset's gold ``topics`` block.

    The dataset annotates each topic with a paraphrased ``user_query`` that
    opens it but does not say which raw turn started which topic. We locate
    each topic's start by char-bigram Jaccard similarity over the user-side
    content of every turn, then slice contiguously between consecutive
    topic starts.

    If the resulting starts are non-monotone (typically because two topics
    paraphrase very similar questions and best-match the same turn), we
    fall back to evenly slicing the user-turn count across the topics, so
    callers always get exactly ``len(topics_block)`` non-empty windows.
    """
    if not turns:
        return []
    ordered_topics = sorted(topics_block.items())  # topic-1, topic-2, ...
    if not ordered_topics:
        return []

    user_indices: list[int] = [i for i, t in enumerate(turns) if t.role == "user"]
    if not user_indices:
        return []

    starts: list[int] = []
    for _, payload in ordered_topics:
        query = str(payload.get("user_query", "")).strip()
        if not query:
            starts.append(-1)
            continue
        best_idx = -1
        best_sim = -1.0
        for ui in user_indices:
            sim = _jaccard(query, turns[ui].text)
            if sim > best_sim:
                best_sim = sim
                best_idx = ui
        starts.append(best_idx if best_sim > 0.0 else -1)

    monotone = all(
        starts[i] != -1 and starts[i + 1] != -1 and starts[i + 1] > starts[i]
        for i in range(len(starts) - 1)
    ) and (not starts or starts[0] != -1)

    if not monotone:
        # Fall back to even-split over user-turn indices.
        chunk = max(1, len(user_indices) // len(ordered_topics))
        starts = [user_indices[min(i * chunk, len(user_indices) - 1)] for i in range(len(ordered_topics))]
        starts[0] = user_indices[0]  # first window always starts at the first user turn

    # First window must absorb any leading non-user turns (rare for this
    # dataset but cheap to be defensive about).
    starts[0] = 0

    # Build windows from contiguous index ranges.
    boundaries = list(starts) + [len(turns)]
    windows: list[TopicWindow] = []
    for idx, ((_topic_key, _payload), lo) in enumerate(zip(ordered_topics, starts), start=1):
        hi = boundaries[idx] if idx < len(boundaries) - 1 else len(turns)
        if hi <= lo:
            hi = lo + 1
        slice_turns = turns[lo:hi]
        if not slice_turns:
            continue
        windows.append(
            TopicWindow(
                window_id=f"w-{idx}",
                turn_ids=[t.turn_id for t in slice_turns],
            )
        )

    if not windows:
        return RuleTopicSegmenter().segment(turns)
    return windows
