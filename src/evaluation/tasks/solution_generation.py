"""Task 2 — solution generation.

Inputs the gold ``requirement`` plus a ``memory_context`` rendered from the
selected ``memory_strategy``. Output a single solution. Reference for BLEU is
``solution.pos`` (multi-reference, both pos solutions).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from src.evaluation.bleu import sentence_bleu
from src.evaluation.dataset import TopicSample
from src.evaluation.memory_context import MemoryContextProvider
from src.llm.llm import LLMClient
from src.llm.prompt_loader import load_prompt

_logger = logging.getLogger(__name__)


SOLUTION_GENERATION_SYSTEM: str = load_prompt("eval/solution_generation/system")


@dataclass
class SolutionGenerationResult:
    user_id: str
    sample_id: str
    topic_id: str
    requirement_ref: str
    memory_context: str
    prediction: str = ""
    references: list[str] = field(default_factory=list)
    bleu: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SolutionGenerationTask:
    name = "solution_generation"

    def __init__(
        self,
        llm: LLMClient,
        provider: MemoryContextProvider,
        *,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm
        self._provider = provider
        self._temperature = float(temperature)

    def run(self, sample: TopicSample) -> SolutionGenerationResult:
        result = SolutionGenerationResult(
            user_id=sample.user_id,
            sample_id=sample.sample_id,
            topic_id=sample.topic_id,
            requirement_ref=sample.requirement,
            memory_context="",
            references=list(sample.positive_solutions),
        )
        try:
            memory_context = self._provider.for_task23(
                sample.user_id,
                query=sample.requirement,
            )
            result.memory_context = memory_context

            prompt = load_prompt(
                "eval/solution_generation/user",
                memory_context=memory_context.strip() or "（无）",
                requirement_ref=sample.requirement.strip() or "(空)",
            )
            text = self._llm.generate_text(
                prompt,
                system_prompt=SOLUTION_GENERATION_SYSTEM,
                temperature=self._temperature,
            )
            result.prediction = (text or "").strip()
            result.bleu = sentence_bleu(result.prediction, sample.positive_solutions)
        except Exception as err:  # noqa: BLE001
            _logger.warning(
                "[task2] %s/%s/%s failed: %s",
                sample.user_id,
                sample.sample_id,
                sample.topic_id,
                err,
            )
            result.error = f"{type(err).__name__}: {err}"
        return result
