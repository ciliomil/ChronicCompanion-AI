"""Task 3 — solution selection.

Inputs gold ``requirement`` + a ``memory_context`` + 8 candidate solutions
(shuffled). Output strictly ``{"selected_indices": [a, b]}`` with exactly two
distinct 0-based indices. Metrics:

- ``gold_pos_indices``: indices of candidates with ``feedback == "pos"`` (2)
- ``selected_indices``: model output (validated)
- ``selected_feedback``: candidate feedback labels at ``selected_indices``
- ``hit_pos_count``: number of selected indices whose feedback == "pos"
- ``exact_match``: ``set(selected) == set(gold)``
- ``selection_score``: ``hit_pos_count / 2 * 100`` ∈ {0, 50, 100}
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from src.evaluation.dataset import TopicSample
from src.evaluation.memory_context import MemoryContextProvider
from src.llm.llm import LLMClient
from src.llm.prompt_loader import load_prompt

_logger = logging.getLogger(__name__)


SOLUTION_SELECTION_SYSTEM: str = load_prompt("eval/solution_selection/system")


@dataclass
class SolutionSelectionResult:
    user_id: str
    sample_id: str
    topic_id: str
    requirement_ref: str
    memory_context: str
    candidates: list[dict[str, str]] = field(default_factory=list)
    raw_response: str = ""
    gold_pos_indices: list[int] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    selected_feedback: list[str] = field(default_factory=list)
    hit_pos_count: int = 0
    exact_match: bool = False
    selection_score: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _render_candidates(candidates: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates):
        lines.append(f"[{i}] {str(c.get('solution') or '').strip()}")
    return "\n".join(lines)


def _parse_selected_indices(payload: Any) -> list[int]:
    """Validate and normalize the model's selected_indices output.

    Returns ``[]`` when the payload is malformed or violates the
    "exactly 2 distinct ints in [0,7]" constraint.
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("selected_indices")
    if not isinstance(raw, list) or len(raw) != 2:
        return []
    try:
        idx = [int(x) for x in raw]
    except (TypeError, ValueError):
        return []
    if len(set(idx)) != 2:
        return []
    if not all(0 <= i <= 7 for i in idx):
        return []
    return idx


class SolutionSelectionTask:
    name = "solution_selection"

    def __init__(
        self,
        llm: LLMClient,
        provider: MemoryContextProvider,
    ) -> None:
        self._llm = llm
        self._provider = provider

    def run(self, sample: TopicSample) -> SolutionSelectionResult:
        result = SolutionSelectionResult(
            user_id=sample.user_id,
            sample_id=sample.sample_id,
            topic_id=sample.topic_id,
            requirement_ref=sample.requirement,
            memory_context="",
            candidates=list(sample.candidate_solutions),
            gold_pos_indices=list(sample.gold_pos_indices),
        )
        try:
            memory_context = self._provider.for_task23(
                sample.user_id,
                query=sample.requirement,
            )
            result.memory_context = memory_context

            prompt = load_prompt(
                "eval/solution_selection/user",
                memory_context=memory_context.strip() or "（无）",
                requirement_ref=sample.requirement.strip() or "(空)",
                candidates=_render_candidates(sample.candidate_solutions),
            )
            payload = self._llm.generate_json(
                prompt,
                system_prompt=SOLUTION_SELECTION_SYSTEM,
            )
            result.raw_response = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
            selected = _parse_selected_indices(payload)
            if not selected:
                result.error = "invalid selected_indices payload"
                return result

            result.selected_indices = selected
            result.selected_feedback = [
                str(sample.candidate_solutions[i].get("feedback") or "")
                for i in selected
            ]
            result.hit_pos_count = sum(1 for fb in result.selected_feedback if fb == "pos")
            result.exact_match = set(selected) == set(sample.gold_pos_indices)
            result.selection_score = result.hit_pos_count / 2.0 * 100.0
        except Exception as err:  # noqa: BLE001
            _logger.warning(
                "[task3] %s/%s/%s failed: %s",
                sample.user_id,
                sample.sample_id,
                sample.topic_id,
                err,
            )
            result.error = f"{type(err).__name__}: {err}"
        return result


__all__ = [
    "SolutionSelectionResult",
    "SolutionSelectionTask",
    "_parse_selected_indices",
]
