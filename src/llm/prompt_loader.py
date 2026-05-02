"""Load prompt templates from external markdown files.

Conventions
-----------
- Default prompt root: ``<project_root>/src/prompts``.
  Override with the ``LONGMEM_PROMPTS_DIR`` environment variable.
- A template name like ``"memory/event_extract/system"`` resolves to
  ``<prompts_dir>/memory/event_extract/system.md``.
- Variants are sibling files named ``<name>.<variant>.md``. They take
  precedence when explicitly requested (or via the
  ``LONGMEM_PROMPT_VARIANT`` environment variable) and fall back to the
  default file if missing.
- Templates use ``string.Template`` ``$var`` / ``${var}`` placeholders so
  literal ``{}`` (e.g. inside JSON examples) needs no escaping.

Why externalize
---------------
Per SYSTEM_DESIGN §8 we want to swap prompts and strategies for ablation
studies. Storing prompts as markdown keeps diffs reviewable and lets
non-coding collaborators iterate on wording without touching Python.
"""

from __future__ import annotations

import os
from pathlib import Path
from string import Template

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROMPTS_DIR = _PROJECT_ROOT / "src" / "prompts"


def get_prompts_dir() -> Path:
    """Return the active prompts directory (env override aware)."""
    override = os.getenv("LONGMEM_PROMPTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_PROMPTS_DIR


def get_active_variant() -> str | None:
    """Return the global default variant from env, or ``None``."""
    variant = os.getenv("LONGMEM_PROMPT_VARIANT", "").strip()
    return variant or None


def _resolve_path(name: str, variant: str | None) -> Path:
    base_dir = get_prompts_dir()
    if variant:
        variant_path = base_dir / f"{name}.{variant}.md"
        if variant_path.exists():
            return variant_path
    return base_dir / f"{name}.md"


def load_prompt(
    name: str,
    *,
    variant: str | None = None,
    **template_vars: object,
) -> str:
    """Load a prompt template by logical name.

    Args:
        name: Logical template name without extension, using forward slashes
            for subfolders (e.g. ``"memory/event_extract/system"``).
        variant: Optional variant tag. If ``None``, falls back to
            ``LONGMEM_PROMPT_VARIANT`` env var, then to no variant.
        **template_vars: Substitutions for ``$var`` / ``${var}`` placeholders.

    Returns:
        The rendered prompt text, with leading/trailing whitespace stripped.

    Raises:
        FileNotFoundError: if neither the variant nor the default file exists.
        KeyError: if the template references a placeholder not in ``template_vars``.
    """
    if variant is None:
        variant = get_active_variant()

    path = _resolve_path(name, variant)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    text = path.read_text(encoding="utf-8")
    if not template_vars:
        return text.strip()
    return Template(text).substitute(**template_vars).strip()
