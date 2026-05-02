"""Mock LLM used for offline prototype runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class MockLLMClient:
    def generate_text(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        del temperature
        header = "[mock-llm]"
        if system_prompt:
            return f"{header} {system_prompt}\n{prompt}"
        return f"{header} {prompt}"

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_preview": prompt[:200],
            "system_prompt_present": system_prompt is not None,
        }
