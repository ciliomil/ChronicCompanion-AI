"""Need / need-solution extraction for the mid-layer memory.

Two extractors are provided, mirroring :mod:`src.memory.update.event_extractor`:

- :class:`RuleNeedSolutionExtractor` — lightweight keyword heuristics, no
  network (deterministic fallback).
- :class:`LLMNeedSolutionExtractor` — one structured LLM call per **topic
  window transcript** via :meth:`~LLMNeedSolutionExtractor.extract_item`. The
  model returns :class:`~src.memory.schemas.NeedItem` rows; each may list
  several ``solutions`` (assistant proposals). Optional ``session_events`` (this
  session’s extracted events) is passed from ingest for ``context`` /
  ``context_event_ids``. The rule fallback still emits one item per adjacent
  user→assistant pair (empty context).

Both expose the same surface:

    infer_need(query, profile_hint=None, top_needs=None)
        -> {"primary_need": str, "candidate_needs": list[str]}
    extract_item(window_turns, *, session_events=None) -> list[NeedItem]

Module helpers :func:`infer_need` and :func:`extract_need_solution` keep simple
signatures for :mod:`src.retrieval.need_inference` etc.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from src.llm.llm import LLMClient, get_default_llm_client
from src.memory.ontology import NEED_DOMAINS
from src.memory.schemas import NeedItem, NeedSolutionProposal, RawTurn
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

_NEED_ITEM_ID_PLACEHOLDER = "need-pending-ingest"

_DEFAULT_NEED = "日常陪伴"
_MAX_NEED_LEN = 24
_MAX_NEED_OBJECT_LEN = 15
_MAX_PREFERENCE_LEN = 80
_MAX_SOLUTION_LEN = 120
_MAX_CONTEXT_LEN = 120

_NEED_DOMAIN_KEYS: tuple[str, ...] = tuple(NEED_DOMAINS.keys())
_DEFAULT_NEED_DOMAIN = "other"

# Used by the rule extractor to map a heuristic need to a sensible event tag.
# Keys are now Chinese phrases (matching the new free-form schema); we keep a
# small, focused mapping so the rule path still produces useful tags offline.
_NEED_TO_TAG: dict[str, str] = {
    "饮食支持": "diet",
    "活动支持": "activity",
    "睡眠支持": "sleep",
    "用药支持": "medication",
    "情绪支持": "emotion",
    "知识解释": "other",
    "日常陪伴": "other",
}

_NEED_TO_DOMAIN: dict[str, str] = {
    "饮食支持": "diet_glucose_management",
    "活动支持": "activity_safety",
    "睡眠支持": "routine_habit_adherence",
    "用药支持": "medication_adherence",
    "情绪支持": "emotional_motivation",
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


def _shape_need_object(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:_MAX_NEED_OBJECT_LEN]


def _shape_context(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:_MAX_CONTEXT_LEN]


def _filter_context_event_ids(
    raw_ids: list[str],
    session_events: list[dict[str, Any]] | None,
) -> list[str]:
    """Keep only ids present in this session's extracted events (max 5)."""
    if not session_events:
        return []
    valid = {str(e.get("event_id")) for e in session_events if e.get("event_id")}
    out = [i for i in raw_ids if i in valid]
    return out[:5]


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

    def extract_item(
        self,
        window_turns: list[RawTurn],
        *,
        session_events: list[dict[str, Any]] | None = None,
    ) -> list[NeedItem]:
        del session_events  # rule path does not use session events
        return [self._item_from_rule_pair(u, a) for u, a in self._iter_user_assistant_pairs(window_turns)]

    def _item_from_rule_pair(self, user_turn: RawTurn, assistant_turn: RawTurn) -> NeedItem:
        need = self._keyword_need(user_turn.text)
        tag = _NEED_TO_TAG.get(need, "other")
        need_domain = _NEED_TO_DOMAIN.get(need, _DEFAULT_NEED_DOMAIN)
        need_object = need[:_MAX_NEED_OBJECT_LEN]
        return NeedItem(
            item_id=_NEED_ITEM_ID_PLACEHOLDER,
            timestamp=user_turn.timestamp,
            source_turn_ids=[user_turn.turn_id],
            inferred_need=need,
            need_domain=need_domain,
            need_object=need_object,
            related_tags=[tag],
            context="",
            context_event_ids=[],
            solutions=[
                NeedSolutionProposal(
                    ai_solution_summary=assistant_turn.text.strip()[:100],
                    feedback_turn_ids=[],
                    fit_score=0.5,
                    revealed_preference="",
                    confidence=None,
                ),
            ],
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
    
    @staticmethod
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
# LLM-driven extractor
# ---------------------------------------------------------------------------


class LLMNeedSolutionExtractor:
    """LLM-driven :class:`NeedItem` extractor with rule-based fallback.

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

    def extract_item(
        self,
        window_turns: list[RawTurn],
        *,
        session_events: list[dict[str, Any]] | None = None,
    ) -> list[NeedItem]:
        if not window_turns:
            return []
        nonempty = all(t.text.strip() for t in window_turns)
        if not nonempty:
            return self._fallback.extract_item(window_turns, session_events=session_events)

        try:
            turn_dicts = [t.to_dict() for t in window_turns if t.text.strip()]
            prompt = build_need_solution_extract_prompt(
                turn_dicts,
                session_events=session_events,
            )
            response = self.client.generate_json(
                prompt,
                system_prompt=NEED_SOLUTION_EXTRACT_SYSTEM,
            )
            return self._merge_items(response, window_turns, session_events=session_events)
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "LLMNeedSolutionExtractor.extract_item failed (%s); "
                "falling back to rule extractor.",
                err,
            )
            return self._fallback.extract_item(window_turns, session_events=session_events)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _coerce_unit_float(self, value: Any) -> float | None:
        """Clamp to [0, 1]; used for ``fit_score`` and ``confidence``."""
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


    def _parse_solution_rows(
        self,
        raw: dict[str, Any],
        *,
        window_turns: list[RawTurn],
        fallback_assistant_text: str,
    ) -> list[NeedSolutionProposal]:
        rows_raw = raw.get("solutions")
        parsed: list[NeedSolutionProposal] = []
        if isinstance(rows_raw, list):
            for row in rows_raw:
                if not isinstance(row, dict):
                    continue
                summ = str(row.get("ai_solution_summary", "")).strip()
                if not summ:
                    summ = fallback_assistant_text
                else:
                    summ = summ[:_MAX_SOLUTION_LEN]
                parsed.append(
                    NeedSolutionProposal(
                        ai_solution_summary=summ,
                        feedback_turn_ids=self._coerce_str_list(row.get("feedback_turn_ids")),
                        fit_score=self._coerce_unit_float(row.get("fit_score")),
                        revealed_preference=_shape_preference(row.get("revealed_preference")),
                        confidence=self._coerce_unit_float(row.get("confidence")),
                    ),
                )
        return parsed

    def _build_item(
        self,
        item: dict[str, Any],
        window_turns: list[RawTurn],
        *,
        session_events: list[dict[str, Any]] | None,
    ) -> NeedItem:
        need = _shape_need_phrase(item.get("inferred_need"))
        if not need:
            need = _DEFAULT_NEED
        need_domain = str(item.get("need_domain") or "").strip()
        if need_domain not in _NEED_DOMAIN_KEYS:
            need_domain = _NEED_TO_DOMAIN.get(need, _DEFAULT_NEED_DOMAIN)
        need_object = _shape_need_object(item.get("need_object"))
        if not need_object:
            need_object = need[:_MAX_NEED_OBJECT_LEN]

        fallback_assistant = window_turns[-1].text.strip()[:100]

        tags = [
            t for t in self._coerce_str_list(item.get("related_tags")) if t in EVENT_TAGS
        ]
        if not tags:
            tags = [_NEED_TO_TAG.get(need, "other")]
        seen_tags: set[str] = set()
        unique_tags: list[str] = []
        for tag in tags:
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            unique_tags.append(tag)
        tags = unique_tags[:3]

        solutions = self._parse_solution_rows(
            item,
            window_turns=window_turns,
            fallback_assistant_text=fallback_assistant,
        )

        ctx_ids = _filter_context_event_ids(
            self._coerce_str_list(item.get("context_event_ids")),
            session_events,
        )

        user_turns = [t for t in window_turns if t.role == "user" and t.text.strip()]
        source_turn_ids = self._coerce_str_list(item.get("source_turn_ids"))
        valid_user_ids = {t.turn_id for t in user_turns}
        source_turn_ids = [tid for tid in source_turn_ids if tid in valid_user_ids]
        if not source_turn_ids:
            source_turn_ids = [t.turn_id for t in user_turns]
        if not source_turn_ids:
            source_turn_ids = [window_turns[0].turn_id]
        ts_by_turn = {t.turn_id: t.timestamp for t in window_turns}
        timestamp = min((ts_by_turn.get(tid, "") for tid in source_turn_ids), default="") or window_turns[0].timestamp

        return NeedItem(
            item_id=_NEED_ITEM_ID_PLACEHOLDER,
            timestamp=timestamp,
            source_turn_ids=source_turn_ids,
            inferred_need=need,
            need_domain=need_domain,
            need_object=need_object,
            related_tags=tags,
            context=_shape_context(item.get("context")),
            context_event_ids=ctx_ids,
            solutions=solutions,
            cluster_id=None,
        )

    def _merge_items(
        self,
        response: Any,
        window_turns: list[RawTurn],
        *,
        session_events: list[dict[str, Any]] | None,
    ) -> list[NeedItem]:
        if not isinstance(response, dict):
            return []
        items = response.get("items") if isinstance(response, dict) else None
        if not isinstance(items, list):
            return []
        out: list[NeedItem] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            out.append(self._build_item(raw, window_turns, session_events=session_events))
        return out



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


def extract_need_solution(window_turns: list[RawTurn]) -> list[NeedItem]:
    """Window-scoped extraction with one-based ``need-{n}`` ids for this call only."""
    xs = _get_default_extractor().extract_item(window_turns, session_events=None)
    return [replace(it, item_id=f"need-{i}") for i, it in enumerate(xs, start=1)]

