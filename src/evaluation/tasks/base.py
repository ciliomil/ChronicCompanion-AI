"""Shared task plumbing."""

from __future__ import annotations

from typing import Literal

TaskName = Literal[
    "requirement_restatement",
    "solution_generation",
    "solution_selection",
]

TASK_NAMES: tuple[TaskName, ...] = (
    "requirement_restatement",
    "solution_generation",
    "solution_selection",
)
