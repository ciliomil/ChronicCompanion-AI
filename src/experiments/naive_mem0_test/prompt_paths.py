"""
naive_mem0 内 prompt 根目录（见 prompt/），与 Stack-Planner、Mem-PAL 仓库路径解耦。
"""
from pathlib import Path

_NAIVE_MEM0 = Path(__file__).resolve().parent
PROMPT_ROOT = _NAIVE_MEM0 / "prompt"


def memory_extract_md() -> Path:
    return PROMPT_ROOT / "memory_extract.md"


def solution_selection_dir() -> Path:
    return PROMPT_ROOT / "solution_selection"


def solution_qa_dir() -> Path:
    return PROMPT_ROOT / "solution_qa"


def chat_session_dir() -> Path:
    """逐条实时对话专用 prompt（session 上文 + mem0 检索）。"""
    return PROMPT_ROOT / "chat_session"


def prompts_solution_mem_md() -> Path:
    return PROMPT_ROOT / "prompts_solution_mem.md"
