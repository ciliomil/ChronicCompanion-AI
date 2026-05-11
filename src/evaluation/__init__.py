"""ChronicCompanion eval framework."""

from src.evaluation.bleu import (
    CorpusBleuAccumulator,
    corpus_bleu,
    sentence_bleu,
    tokenize,
)
from src.evaluation.dataset import TopicSample, iter_topic_samples, load_input
from src.evaluation.llm_setup import EvalLLMClient, install_eval_conf
from src.evaluation.memory_context import (
    FullMemoryProvider,
    Mem0MemoryProvider,
    MemoryContextProvider,
    NoMemoryProvider,
    PackMemoryProvider,
    make_provider,
)
from src.evaluation.runner import RunConfig, run_evaluation
from src.evaluation.tasks import (
    RequirementRestatementResult,
    RequirementRestatementTask,
    SolutionGenerationResult,
    SolutionGenerationTask,
    SolutionSelectionResult,
    SolutionSelectionTask,
    TASK_NAMES,
    TaskName,
)

__all__ = [
    # bleu
    "CorpusBleuAccumulator",
    "corpus_bleu",
    "sentence_bleu",
    "tokenize",
    # dataset
    "TopicSample",
    "iter_topic_samples",
    "load_input",
    # llm
    "EvalLLMClient",
    "install_eval_conf",
    # memory
    "MemoryContextProvider",
    "PackMemoryProvider",
    "FullMemoryProvider",
    "NoMemoryProvider",
    "Mem0MemoryProvider",
    "make_provider",
    # runner
    "RunConfig",
    "run_evaluation",
    # tasks
    "TaskName",
    "TASK_NAMES",
    "RequirementRestatementResult",
    "RequirementRestatementTask",
    "SolutionGenerationResult",
    "SolutionGenerationTask",
    "SolutionSelectionResult",
    "SolutionSelectionTask",
]
