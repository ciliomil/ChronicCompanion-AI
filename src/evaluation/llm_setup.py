"""LLM client wiring for the eval CLI.

The eval CLI accepts ``--api_file`` (path to a YAML with ``BASIC_MODEL``
config) and ``--model`` overrides. Rather than refactoring the production
:mod:`src.llm.llm` module, we install a process-wide override of
:func:`src.llm.llm.get_basic_model_conf` so that every existing call site
(``OpenAICompatibleClient``, retrieval planner, response generator, etc.)
picks up the eval-time conf without any changes.

``--temperature`` is *not* installed globally — instead we expose
:class:`EvalLLMClient`, a thin wrapper around the default OpenAI-compatible
client whose ``generate_text`` honors the eval-time temperature while
``generate_json`` keeps the strict ``temperature=0`` behavior we need for
retrieval / structured outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import src.llm.llm as llm_mod
from src.llm.llm import LLMClient


def install_eval_conf(
    api_file: str | Path,
    *,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Install a process-wide BASIC_MODEL override read from ``api_file``.

    Returns the resolved conf dict. Side effect: monkeypatches
    :func:`src.llm.llm.get_basic_model_conf` and resets the cached default
    client so downstream callers pick up the new endpoint.
    """
    path = Path(api_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--api_file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    basic = dict(raw.get("BASIC_MODEL") or {})
    if not basic:
        raise ValueError(f"BASIC_MODEL block missing in {path}")

    if "api_key" not in basic and raw.get("api_key"):
        basic["api_key"] = raw.get("api_key")

    if model_override:
        basic["model"] = model_override

    for key in ("base_url", "model", "api_key"):
        if not str(basic.get(key, "")).strip():
            raise ValueError(f"BASIC_MODEL.{key} is required (api_file={path})")

    snapshot = dict(basic)
    llm_mod.get_basic_model_conf = lambda: dict(snapshot)  # type: ignore[assignment]
    llm_mod.reset_default_llm_client()
    return snapshot


class EvalLLMClient:
    """Thin wrapper that pins ``generate_text`` temperature for eval.

    ``generate_json`` defers to the default OpenAI-compatible client unchanged
    — JSON-mode calls (retrieval planner, ``solution_selection``) are always
    deterministic regardless of ``--temperature``.
    """

    def __init__(self, *, temperature: float = 0.0) -> None:
        self._inner: LLMClient = llm_mod.get_default_llm_client(prefer="basic_model")
        self._temperature = float(temperature)

    @property
    def temperature(self) -> float:
        return self._temperature

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        temp = self._temperature if temperature is None else float(temperature)
        return self._inner.generate_text(prompt, system_prompt=system_prompt, temperature=temp)

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        return self._inner.generate_json(prompt, system_prompt=system_prompt)
