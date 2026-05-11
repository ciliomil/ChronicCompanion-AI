"""Task 1 — requirement restatement.

Inputs to the LLM mirror the retrieval planner's input shape (user_query +
dialogue_context + basic_info(4 sections) + recent_status). The LLM's job is
to expand the short ``user_query`` into a full requirement description in the
style of ``data/.../input.json: topics["topic-N"].requirement`` (≈80–250
chars). Reference for BLEU is the gold ``requirement``.
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


REQUIREMENT_RESTATEMENT_SYSTEM: str = load_prompt("eval/requirement_restatement/system")


@dataclass
class RequirementRestatementResult:
    user_id: str
    sample_id: str
    topic_id: str
    user_query: str
    memory_context: str
    prediction: str = ""
    reference: str = ""
    bleu: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RequirementRestatementTask:
    name = "requirement_restatement"

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

    def run(self, sample: TopicSample) -> RequirementRestatementResult:
        result = RequirementRestatementResult(
            user_id=sample.user_id,
            sample_id=sample.sample_id,
            topic_id=sample.topic_id,
            user_query=sample.user_query,
            memory_context="",
            reference=sample.requirement,
        )
        try:
            memory_context = self._provider.for_task1(
                sample.user_id,
                user_query=sample.user_query,
            )
            result.memory_context = memory_context

            prompt = load_prompt(
                "eval/requirement_restatement/user",
                user_query=sample.user_query.strip() or "(空)",
                dialogue_context="(无)",
                memory_context=memory_context.strip() or "（无）",
            )
            text = self._llm.generate_text(
                prompt,
                system_prompt=REQUIREMENT_RESTATEMENT_SYSTEM,
                temperature=self._temperature,
            )
            result.prediction = (text or "").strip()
            result.bleu = sentence_bleu(result.prediction, [sample.requirement])
        except Exception as err:  # noqa: BLE001 — keep one bad sample from killing the run
            _logger.warning(
                "[task1] %s/%s/%s failed: %s",
                sample.user_id,
                sample.sample_id,
                sample.topic_id,
                err,
            )
            result.error = f"{type(err).__name__}: {err}"
        return result
