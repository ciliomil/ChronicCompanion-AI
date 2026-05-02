"""Lightweight API placeholder for future web serving."""

from __future__ import annotations

from src.experiments.run_layered_memory import run as run_layered


def chat(session_id: str, query: str) -> dict[str, str]:
    response = run_layered(session_id=session_id, query=query)
    return {"session_id": session_id, "response": response}


def health() -> dict[str, str]:
    return {"status": "ok"}
