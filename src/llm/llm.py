"""Unified LLM adapter layer with OpenAI-compatible support."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import requests
import yaml

from src.llm.mock_client import MockLLMClient


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONF_PATH = _PROJECT_ROOT / "conf.yaml"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?|\n?```\s*$", re.MULTILINE)


class LLMClient(Protocol):
    def generate_text(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        ...

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        ...


def get_basic_model_conf() -> dict[str, Any]:
    """Load BASIC_MODEL config with optional env key override."""
    if not _CONF_PATH.exists():
        return {}

    with _CONF_PATH.open("r", encoding="utf-8") as file:
        conf = yaml.safe_load(file) or {}

    basic = conf.get("BASIC_MODEL", {}) or {}
    if conf.get("api_key"):
        basic["api_key"] = conf.get("api_key")
    env_key = (
        os.getenv("LONGMEM_API_KEY")
    )
    if env_key:
        basic["api_key"] = env_key
    return basic


def _debug_enabled() -> bool:
    return os.getenv("LONGMEM_LLM_DEBUG", "").strip() not in {"", "0", "false", "False"}


def _debug_log(tag: str, payload: str) -> None:
    if not _debug_enabled():
        return
    preview = payload if len(payload) <= 1200 else f"{payload[:1200]}...<truncated>"
    sys.stderr.write(f"[llm-debug:{tag}] {preview}\n")
    sys.stderr.flush()


def _extract_assistant_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if content is not None and str(content).strip():
        return str(content)

    for key in ("reasoning_content", "reasoning"):
        reasoning = message.get(key)
        if reasoning is not None and str(reasoning).strip():
            return str(reasoning)
    return ""


class _UnsupportedJsonMode(RuntimeError):
    """Raised when the endpoint rejects ``response_format`` (e.g. older self-hosted vLLM).

    ``OpenAICompatibleClient.generate_json`` catches this and retries without
    forcing JSON mode so we keep working on backends that don't yet support it.
    """


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    model: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """
    Call OpenAI-compatible /chat/completions endpoint.

    
    """
    conf = get_basic_model_conf()
    base_url = str(conf.get("base_url", "")).rstrip("/")
    model_name = model or conf.get("model", "")
    api_key = conf.get("api_key")

    extra_body = dict(conf.get("extra_body") or {})
    extra_body.pop("stream", None)

    if not base_url:
        raise ValueError("BASIC_MODEL.base_url is required in conf.yaml")
    if not model_name:
        raise ValueError("BASIC_MODEL.model is required in conf.yaml")
    if not api_key:
        raise ValueError("BASIC_MODEL.api_key is required in conf.yaml")
        
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        **extra_body,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if response_format is not None:
        body["response_format"] = response_format

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout_sec = int(os.getenv("LONGMEM_HTTP_TIMEOUT", "600"))
    max_retries = max(1, int(os.getenv("LONGMEM_HTTP_MAX_RETRIES", "8")))
    retry_base_sec = float(os.getenv("LONGMEM_HTTP_RETRY_BASE_SEC", "3"))

    url = f"{base_url}/chat/completions"
    _debug_log("request", json.dumps({"url": url, "model": model_name, "messages": messages, "response_format": response_format}, ensure_ascii=False))

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
        if response.status_code == 429:
            if attempt + 1 >= max_retries:
                response.raise_for_status()
            wait_sec = min(retry_base_sec * (2**attempt), 120.0)
            time.sleep(wait_sec)
            continue

        if response.status_code == 400 and response_format is not None:
            body_preview = (response.text or "")[:400]
            if "response_format" in body_preview or "json_object" in body_preview:
                raise _UnsupportedJsonMode(body_preview)

        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError(
                "chat/completions returned empty choices."
                f" response_preview={str(data)[:800]!r}"
            )

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else None
        text = _extract_assistant_text(message)
        if text.strip():
            _debug_log("response", text)
            return text.strip()

        raise ValueError(
            "chat/completions message has no content/reasoning text."
            f" response_preview={str(data)[:800]!r}"
        )

    raise RuntimeError("chat_completion exhausted retries")


def _strip_think_blocks(text: str) -> str:
    """Remove Qwen-style <think>...</think> spans that some gateways emit."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _strip_code_fences(text: str) -> str:
    """Strip ``` fences around JSON payloads (with or without language tag)."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return _CODE_FENCE_RE.sub("", stripped).strip()


def _isolate_json_object(text: str) -> str:
    """
    Best-effort isolation of the outermost JSON object/array in `text`.

    Falls back to the original text if no balanced braces/brackets are found.
    """
    candidates: list[tuple[int, int]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append((start, end + 1))
    if not candidates:
        return text
    start, end = min(candidates, key=lambda pair: pair[0])
    return text[start:end].strip()


def parse_json_response(raw: str) -> dict[str, Any]:
    """
    Parse a JSON-only model response with tolerance for common noise.

    Handles:
    - <think>...</think> reasoning blocks
    - Markdown code fences (with or without language tag)
    - Leading/trailing prose around the JSON payload
    """
    text = _strip_think_blocks(raw)
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    isolated = _isolate_json_object(text)
    return json.loads(isolated)


class OpenAICompatibleClient:
    """Client implementing the project LLMClient protocol.

    ``generate_json`` relies on the server-side JSON mode
    (``response_format={"type": "json_object"}``) supported by DeepSeek,
    OpenAI, and recent vLLM builds. 
    """

    def __init__(self, default_temperature: float = 0.2):
        self.default_temperature = default_temperature

    def generate_text(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        temp = temperature if temperature is not None else self.default_temperature
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return chat_completion(messages=messages, temperature=temp)

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        json_directive = (
            "You must return a single valid JSON object that matches the "
            "EXAMPLE OUTPUT shape shown in the system prompt. "
            "Do not wrap the answer in markdown fences, prose, or <think> tags."
        )
        if  not system_prompt:
            system_prompt = json_directive  

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        last_err: Exception | None = None
        # Two attempts; if json_object is unsupported we drop it and retry once.
        use_json_mode = True
        for attempt in range(2):
            try:
                raw = chat_completion(
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"} if use_json_mode else None,
                )
            except _UnsupportedJsonMode as err:
                _debug_log("json-mode-unsupported", str(err))
                use_json_mode = False
                continue
            try:
                return parse_json_response(raw)
            except (json.JSONDecodeError, ValueError) as err:
                last_err = err
                _debug_log("json-parse-fail", f"attempt={attempt} err={err} raw={raw[:600]}")
                continue

        raise ValueError(
            f"generate_json: failed to parse JSON after retries; last_err={last_err}"
        )


def _has_basic_model_endpoint() -> bool:
    conf = get_basic_model_conf()
    return bool(str(conf.get("base_url", "")).strip()) and bool(str(conf.get("model", "")).strip())


def build_llm_client(provider: str) -> LLMClient:
    if provider == "mock":
        return MockLLMClient()
    if provider in {"openai", "openai_compatible", "basic_model"}:
        return OpenAICompatibleClient()
    # Safe fallback for prototype stage.
    return MockLLMClient()


_DEFAULT_CLIENT: LLMClient | None = None


def get_default_llm_client(prefer: str = "auto") -> LLMClient:
    """
    Return a process-wide default LLM client.

    `prefer` can be:
    - "auto":  use OpenAI-compatible client if BASIC_MODEL is configured, else mock
    - "openai" / "basic_model": force OpenAI-compatible
    - "mock":  force mock client
    """
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None and prefer == "auto":
        return _DEFAULT_CLIENT

    if prefer == "mock":
        client: LLMClient = MockLLMClient()
    elif prefer in {"openai", "openai_compatible", "basic_model"}:
        client = OpenAICompatibleClient()
    else:
        client = OpenAICompatibleClient() if _has_basic_model_endpoint() else MockLLMClient()

    if prefer == "auto":
        _DEFAULT_CLIENT = client
    return client


def reset_default_llm_client() -> None:
    """Clear the cached default client; useful for tests that swap providers."""
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = None
