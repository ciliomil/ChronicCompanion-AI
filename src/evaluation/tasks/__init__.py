"""Eval tasks: requirement_restatement, solution_generation, solution_selection."""

from src.evaluation.tasks.base import TaskName, TASK_NAMES
from src.evaluation.tasks.requirement_restatement import (
    RequirementRestatementResult,
    RequirementRestatementTask,
)
from src.evaluation.tasks.solution_generation import (
    SolutionGenerationResult,
    SolutionGenerationTask,
)
from src.evaluation.tasks.solution_selection import (
    SolutionSelectionResult,
    SolutionSelectionTask,
)

__all__ = [
    "TaskName",
    "TASK_NAMES",
    "RequirementRestatementResult",
    "RequirementRestatementTask",
    "SolutionGenerationResult",
    "SolutionGenerationTask",
    "SolutionSelectionResult",
    "SolutionSelectionTask",
]
