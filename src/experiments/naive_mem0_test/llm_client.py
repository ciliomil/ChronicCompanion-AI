from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm.llm import chat_completion as _shared_chat_completion
from src.llm.llm import get_basic_model_conf as _shared_get_basic_model_conf


def get_basic_model_conf() -> dict[str, Any]:
    """
    Backward-compatible wrapper for naive_mem0 scripts.
    Unified implementation lives in `src.llm.llm`.
    """
    return _shared_get_basic_model_conf()


def chat_completion(messages: list[dict[str, str]], temperature: float = 0.3) -> str:
    """
    Backward-compatible wrapper for naive_mem0 scripts.
    Delegates to the unified implementation in `src.llm.llm`.
    """
    return _shared_chat_completion(messages=messages, temperature=temperature)
