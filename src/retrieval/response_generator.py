"""Stage 3 of retrieval — turn a :class:`MemoryPack` into the final reply text.

The system prompt (``src/prompts/reply/response_generation/system.md``) defines
the role-aware reply behavior; this module renders the user prompt from the
pack and calls the LLM. On any LLM failure we return a safe fallback message
that respects the frame's ``risk_level`` so the conversation never silently
dies on the user.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.llm.llm import LLMClient, get_default_llm_client
from src.llm.prompt_loader import load_prompt
from src.retrieval.schemas import MemoryPack

_logger = logging.getLogger(__name__)


RESPONSE_GENERATION_SYSTEM: str = load_prompt("reply/response_generation/system")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_response_generation_prompt(pack: MemoryPack) -> str:
    return load_prompt(
        "reply/response_generation/user",
        user_query=pack.user_query,
        current_query_json=_json(pack.current_query.to_dict()),
        selected_basic_info_claims_json=_json(pack.selected_basic_info_claims),
        preference_principles_json=_json(pack.preference_principles),
        relevant_needs_json=_json(pack.relevant_needs),
        relevant_events_json=_json(pack.relevant_events),
        recent_status_slice_json=_json(pack.recent_status_slice),
    )


class ResponseGenerator:
    """LLM-driven reply generator with a safe deterministic fallback."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_default_llm_client()
        return self._client

    def generate(self, pack: MemoryPack) -> str:
        try:
            prompt = build_response_generation_prompt(pack)
            text = self.client.generate_text(
                prompt,
                system_prompt=RESPONSE_GENERATION_SYSTEM,
                temperature=0.5,
            )
            return (text or "").strip() or _safe_fallback_reply(pack)
        except Exception as err:  # noqa: BLE001 — defensive fallback
            _logger.warning(
                "ResponseGenerator.generate failed (%s); using safe fallback.",
                err,
            )
            return _safe_fallback_reply(pack)


def _safe_fallback_reply(pack: MemoryPack) -> str:
    """Last-resort reply that still honors the frame's risk signal."""
    if pack.current_query.risk_level == "urgent":
        return (
            "我有点担心您说的情况，建议您先深呼吸，"
            "尽快联系家人或医生确认一下，必要时可以拨打急救电话。"
        )
    return "我在听呢，您愿意再多说一些吗？"


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


_DEFAULT_GENERATOR: ResponseGenerator | None = None


def _get_default_generator() -> ResponseGenerator:
    global _DEFAULT_GENERATOR
    if _DEFAULT_GENERATOR is None:
        _DEFAULT_GENERATOR = ResponseGenerator()
    return _DEFAULT_GENERATOR


def reset_default_response_generator() -> None:
    """Drop the cached default generator (useful in tests)."""
    global _DEFAULT_GENERATOR
    _DEFAULT_GENERATOR = None


def generate_response(pack: MemoryPack, *, llm: LLMClient | None = None) -> str:
    """Generate the final reply text for a :class:`MemoryPack`."""
    if llm is None:
        return _get_default_generator().generate(pack)
    return ResponseGenerator(client=llm).generate(pack)
