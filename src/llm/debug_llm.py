"""LLM wrapper that echoes prompts and parses to stderr (or another stream).

Used by :func:`src.memory.update.session_ingest.ingest_session` when
``debug=True`` so history-ingest runs can trace every API-backed call without
patching extractors individually.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from src.llm.llm import LLMClient


def _truncate(text: str, max_chars: int) -> tuple[str, int]:
    """Return ``(possibly_truncated_text, omitted_char_count)``."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, 0
    return text[:max_chars], len(text) - max_chars


class DebuggingLLMClient:
    """Delegate to ``inner`` while printing each call."""

    __slots__ = ("_inner", "_stream", "_max_prompt_chars", "_call_no")

    def __init__(
        self,
        inner: LLMClient,
        *,
        stream: TextIO | None = None,
        max_prompt_chars: int = 24_000,
    ) -> None:
        self._inner = inner
        self._stream = stream if stream is not None else sys.stderr
        self._max_prompt_chars = max(0, max_prompt_chars)
        self._call_no = 0

    def _banner(self, kind: str) -> None:
        self._call_no += 1
        self._stream.write(
            f"\n{'=' * 72}\n"
            f"[ingest-debug] LLM #{self._call_no} — {kind}\n"
            f"{'=' * 72}\n"
        )

    def _dump_block(self, label: str, body: str) -> None:
        body_shown, omit = _truncate(body, self._max_prompt_chars)
        omit_note = " (truncated)" if omit else ""
        self._stream.write(f"\n--- {label}{omit_note} ---\n")
        self._stream.write(body_shown)
        if omit:
            self._stream.write(
                f"\n... [{omit} chars omitted; cap={self._max_prompt_chars}]"
            )
        self._stream.write("\n\n")

    def generate_text(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        self._banner("generate_text")
        if system_prompt:
            self._dump_block("system_prompt", system_prompt)
        self._dump_block("user_prompt", prompt)
        out = self._inner.generate_text(prompt, system_prompt=system_prompt, temperature=temperature)
        self._dump_block("assistant_text_response", out)
        return out

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        self._banner("generate_json")
        if system_prompt:
            self._dump_block("system_prompt", system_prompt)
        self._dump_block("user_prompt", prompt)
        data = self._inner.generate_json(prompt, system_prompt=system_prompt)
        try:
            pretty = json.dumps(data, ensure_ascii=False, indent=2)
        except TypeError:
            pretty = repr(data)
        self._dump_block("parsed_json_response", pretty)
        return data
