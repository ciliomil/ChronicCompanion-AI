"""Need / need-solution extraction for the mid-layer memory.

Two extractors are provided, mirroring :mod:`src.memory.update.event_extractor`:

- :class:`RuleNeedSolutionExtractor` — lightweight keyword heuristics, no
  network (deterministic fallback).
- :class:`LLMNeedSolutionExtractor` — one structured LLM call per **topic
  window transcript** via :meth:`~LLMNeedSolutionExtractor.extract_pair`. The
  model chooses how many anchored (user→assistant) rows to emit; the rule
  fallback still derives one heuristic row per adjacent pair.

Both expose the same surface:

    infer_need(query, profile_hint=None, top_needs=None)
        -> {"primary_need": str, "candidate_needs": list[str]}
    extract_pair(window_turns: list[RawTurn]) -> list[NeedSolutionItem]

Module helpers :func:`infer_need` and :func:`extract_need_solution` keep simple
signatures for :mod:`src.retrieval.need_inference` etc.
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm.llm import LLMClient, get_default_llm_client
from src.memory.schemas import NeedSolutionItem, RawTurn
from src.memory.update.prompts import (
    EVENT_TAGS,
    NEED_INFER_SYSTEM,
    NEED_SOLUTION_EXTRACT_SYSTEM,
    build_need_infer_prompt,
    build_need_solution_extract_prompt,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free-form need phrase shaping
# ---------------------------------------------------------------------------
#
# The LLM is now allowed to return arbitrary short Chinese phrases for
# ``inferred_need`` / ``primary_need``. We still want to keep them roughly the
# same shape as the prompt asks for: a 4–12-character phrase, no whitespace,
# no quote noise. The constants below cap the length and provide a sensible
# default when the model returns nothing usable.

_DEFAULT_NEED = "日常陪伴"
_MAX_NEED_LEN = 24
_MAX_PREFERENCE_LEN = 80
_MAX_SOLUTION_LEN = 120

# Used by the rule extractor to map a heuristic need to a sensible event tag.
# Keys are now Chinese phrases (matching the new free-form schema); we keep a
# small, focused mapping so the rule path still produces useful tags offline.
_NEED_TO_TAG: dict[str, str] = {
    "饮食支持": "diet",
    "活动支持": "activity",
    "睡眠支持": "sleep",
    "用药支持": "medical",
    "情绪支持": "emotion",
    "知识解释": "other",
    "日常陪伴": "other",
}


def _shape_need_phrase(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.strip(" \t\r\n\"'`“”‘’《》（）()[]【】")
    text = " ".join(text.split())
    if not text:
        return ""
    return text[:_MAX_NEED_LEN]


def _shape_preference(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:_MAX_PREFERENCE_LEN]



# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

class RuleNeedSolutionExtractor:
    def infer_need(
        self,
        query: str,
        profile_hint: dict[str, Any] | None = None,
        top_needs: list[str] | None = None,
    ) -> dict[str, Any]:
        primary = self._keyword_need(query)
        return {"primary_need": primary, "candidate_needs": [primary]}

    def extract_pair(self, window_turns: list[RawTurn]) -> list[NeedSolutionItem]:
        return [self._item_from_rule_pair(u, a) for u, a in _iter_user_assistant_pairs(window_turns)]

    def _item_from_rule_pair(self, user_turn: RawTurn, assistant_turn: RawTurn) -> NeedSolutionItem:
        need = self._keyword_need(user_turn.text)
        tag = _NEED_TO_TAG.get(need, "other")
        return NeedSolutionItem(
            item_id=f"need-{assistant_turn.turn_id}",
            timestamp=assistant_turn.timestamp,
            source_turn_ids=[user_turn.turn_id, assistant_turn.turn_id],
            inferred_need=need,
            ai_solution_summary=assistant_turn.text.strip()[:100],
            related_tags=[tag],
            preference="",
            quality_score=0.5,
            feedback_turn_ids=[],
            cluster_id=None,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _keyword_need(text: str) -> str:
        if not text:
            return _DEFAULT_NEED
        if any(token in text for token in ("吃", "饮食", "餐", "夜宵", "血糖")):
            return "饮食支持"
        if any(token in text for token in ("运动", "散步", "锻炼")):
            return "活动支持"
        if any(token in text for token in ("睡", "失眠", "休息")):
            return "睡眠支持"
        if any(token in text for token in ("药", "服药", "用药", "胰岛素")):
            return "用药支持"
        if any(token in text for token in ("担心", "焦虑", "害怕", "难过", "孤独")):
            return "情绪支持"
        return _DEFAULT_NEED


# ---------------------------------------------------------------------------
# LLM-driven extractor
# ---------------------------------------------------------------------------


class LLMNeedSolutionExtractor:
    """LLM-driven need / need-solution extractor with rule-based fallback.

    On any LLM / parsing failure the extractor falls back to
    :class:`RuleNeedSolutionExtractor` so a single bad call cannot break the
    surrounding pipeline.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        fallback: RuleNeedSolutionExtractor | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or RuleNeedSolutionExtractor()

    @property
    def client(self) -> LLMClient:
        """Lazy default-client resolution so tests can monkeypatch easily."""
        if self._client is None:
            self._client = get_default_llm_client()
        return self._client

    # -- need inference -----------------------------------------------------

    def infer_need(
        self,
        query: str,
        profile_hint: dict[str, Any] | None = None,
        top_needs: list[str] | None = None,
    ) -> dict[str, Any]:
        if not query or not query.strip():
            return self._fallback.infer_need(query, profile_hint, top_needs)

        try:
            prompt = build_need_infer_prompt(
                query=query,
                profile_hint=profile_hint,
                top_needs=top_needs,
            )
            response = self.client.generate_json(
                prompt,
                system_prompt=NEED_INFER_SYSTEM,
            )
            return self._coerce_need_inference(response)
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "LLMNeedSolutionExtractor.infer_need failed (%s); "
                "falling back to rule extractor.",
                err,
            )
            return self._fallback.infer_need(query, profile_hint, top_needs)

    def extract_pair(self, window_turns: list[RawTurn]) -> list[NeedSolutionItem]:
        pairs = self._iter_user_assistant_pairs(window_turns)
        if not pairs:
            return []

        nonempty = all(u.text.strip() and a.text.strip() for u, a in pairs)
        if not nonempty:
            return [
                self._fallback._item_from_rule_pair(u, a)
                for u, a in pairs
            ]

        try:
            turn_dicts = [t.to_dict() for t in window_turns if t.text.strip()]
            prompt = build_need_solution_extract_prompt(turn_dicts)
            response = self.client.generate_json(
                prompt,
                system_prompt=NEED_SOLUTION_EXTRACT_SYSTEM,
            )
            return self._build_item(response, user_turn, assistant_turn)
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "LLMNeedSolutionExtractor.extract_pair failed (%s); "
                "falling back to rule extractor.",
                err,
            )
            return self._fallback.extract_pair(user_turn, assistant_turn)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _coerce_quality_score(self, value: Any) -> float | None:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f
    
    def _coerce_need_inference(self, response: Any) -> dict[str, Any]:
        if not isinstance(response, dict):
            return {"primary_need": _DEFAULT_NEED, "candidate_needs": [_DEFAULT_NEED]}

        primary = _shape_need_phrase(response.get("primary_need"))
        if not primary:
            primary = _DEFAULT_NEED

        candidates_raw = self._coerce_str_list(response.get("candidate_needs"))
        # Dedup while preserving order, ensure primary is first.
        seen: set[str] = set()
        candidates: list[str] = []
        for need in [primary, *candidates_raw]:
            shaped = _shape_need_phrase(need)
            if not shaped or shaped in seen:
                continue
            seen.add(shaped)
            candidates.append(shaped)
        if not candidates:
            candidates = [_DEFAULT_NEED]

        return {"primary_need": primary, "candidate_needs": candidates[:3]}



    def _build_item(
        self,
        response: Any,
        user_turn: RawTurn,
        assistant_turn: RawTurn,
    ) -> NeedSolutionItem:
        if not isinstance(response, dict):
            return self._fallback.extract_pair(user_turn, assistant_turn)

        need = _shape_need_phrase(response.get("inferred_need"))
        if not need:
            need = _DEFAULT_NEED

        solution = str(response.get("ai_solution_summary", "")).strip()
        if not solution:
            # Last-ditch: take the first 100 chars of the assistant turn so we
            # never produce an empty summary.
            solution = assistant_turn.text.strip()[:100]
        else:
            solution = solution[:200]

        preference = _shape_preference(response.get("preference"))

        tags = [
            t for t in self._coerce_str_list(response.get("related_tags")) if t in EVENT_TAGS
        ]
        if not tags:
            tags = [_NEED_TO_TAG.get(need, "other")]
        # Preserve order, dedup, cap at 3.
        seen_tags: set[str] = set()
        unique_tags: list[str] = []
        for tag in tags:
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            unique_tags.append(tag)
        tags = unique_tags[:3]

        return NeedSolutionItem(
            item_id=f"need-{assistant_turn.turn_id}",
            timestamp=assistant_turn.timestamp,
            source_turn_ids=[user_turn.turn_id, assistant_turn.turn_id],
            inferred_need=need,
            ai_solution_summary=solution,
            related_tags=tags,
            preference=preference,
            quality_score=None,
            feedback_turn_ids=[],
            cluster_id=None,
        )

    def _iter_user_assistant_pairs(turns: list[RawTurn]) -> list[tuple[RawTurn, RawTurn]]:
        pairs: list[tuple[RawTurn, RawTurn]] = []
        pending_user: RawTurn | None = None
        for t in turns:
            if t.role == "user":
                pending_user = t
                continue
            if t.role == "assistant" and pending_user is not None:
                pairs.append((pending_user, t))
                pending_user = None
            elif t.role != "assistant":
                pending_user = None
        return pairs


# ---------------------------------------------------------------------------
# Backward-compatible module API
# ---------------------------------------------------------------------------


_DEFAULT_EXTRACTOR: LLMNeedSolutionExtractor | None = None


def _get_default_extractor() -> LLMNeedSolutionExtractor:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = LLMNeedSolutionExtractor()
    return _DEFAULT_EXTRACTOR


def reset_default_extractor() -> None:
    """Drop the cached default extractor (useful in tests)."""
    global _DEFAULT_EXTRACTOR
    _DEFAULT_EXTRACTOR = None


def infer_need(text: str) -> str:
    """Backward-compatible entry point — returns only ``primary_need``.

    Used by :mod:`src.retrieval.need_inference`. The returned phrase is now
    free-form (no longer constrained to ``NEED_TAXONOMY``); callers that need
    candidate alternatives should instantiate :class:`LLMNeedSolutionExtractor`
    directly.
    """
    result = _get_default_extractor().infer_need(text)
    primary = _shape_need_phrase(result.get("primary_need"))
    return primary or _DEFAULT_NEED


def extract_need_solution(window_turns: list[RawTurn]) -> list[NeedSolutionItem]:
    """Delegates to the default extractor's window-scoped extraction."""
    return _get_default_extractor().extract_pair(window_turns)

