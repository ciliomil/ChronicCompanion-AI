#!/usr/bin/env python3
"""Ping the OpenAI-compatible endpoint configured in conf.yaml (BASIC_MODEL).

Usage (from repo root)::

    python scripts/verify_model_connection.py
    python scripts/verify_model_connection.py --list-models
    LONGMEM_HTTP_TIMEOUT=20 python scripts/verify_model_connection.py

API key is read in this order (same spirit as :func:`src.llm.llm.get_basic_model_conf`)::

    LONGMEM_API_KEY, MODELSCOPE_API_KEY, OPENAI_API_KEY, then conf.yaml BASIC_MODEL.api_key

Never prints the key. Exit code: 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONF = _PROJECT_ROOT / "conf.yaml"


def _load_basic() -> dict:
    if not _CONF.exists():
        print(f"conf not found: {_CONF}", file=sys.stderr)
        sys.exit(1)
    with _CONF.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    basic = data.get("BASIC_MODEL") or {}
    if not basic.get("base_url") or not basic.get("model"):
        print("BASIC_MODEL.base_url and BASIC_MODEL.model are required in conf.yaml", file=sys.stderr)
        sys.exit(1)
    return basic


def _resolve_key(conf_key: str | None) -> str:
    return (
        os.getenv("LONGMEM_API_KEY", "").strip()
        or os.getenv("MODELSCOPE_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or (conf_key or "").strip()
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="GET /v1/models instead of a chat completion (lighter; good for 'can I reach the API?')",
    )
    ap.add_argument(
        "--no-chat",
        action="store_true",
        help="Alias for --list-models",
    )
    args = ap.parse_args()

    basic = _load_basic()
    base = str(basic["base_url"]).rstrip("/")
    model = str(basic["model"])
    key = _resolve_key(basic.get("api_key") if isinstance(basic.get("api_key"), str) else None)

    timeout = int(os.getenv("LONGMEM_HTTP_TIMEOUT", "30"))
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}" if key else ""
    session.headers["Content-Type"] = "application/json"

    list_models = args.list_models or args.no_chat
    if list_models:
        url = f"{base}/v1/models"
        print(f"GET  {url}")
        r = session.get(url, timeout=timeout)
        print("status:", r.status_code)
        if r.status_code == 200:
            try:
                data = r.json()
                items = (data or {}).get("data") or []
                print(f"models returned: {len(items)} (showing up to 8 id lines)")
                for m in items[:8]:
                    mid = m.get("id", m) if isinstance(m, dict) else m
                    print(f"  - {mid}")
            except Exception as e:  # noqa: BLE001
                print("parse json failed:", e)
                print("raw (500 chars):", (r.text or "")[:500])
        else:
            print("body (500 chars):", (r.text or "")[:500])
        return 0 if r.status_code == 200 else 1

    # Minimal chat: proves DNS + TLS + auth + model name in one go
    url = f"{base}/v1/chat/completions"
    print(f"POST {url}")
    print(f"model: {model}")
    if not key:
        print("warning: no API key in env or conf; many providers return 401", file=sys.stderr)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with a single word: ok"}],
        "max_tokens": 16,
        "stream": False,
    }
    r = session.post(url, json=body, timeout=timeout)
    print("status:", r.status_code)
    if r.status_code != 200:
        print("body (800 chars):", (r.text or "")[:800])
        return 1
    try:
        data = r.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        text = (message.get("content") or "").strip() or str(message)[:200]
        print("assistant preview:", text[:200])
    except Exception as e:  # noqa: BLE001
        print("parse error:", e)
        print("raw (500 chars):", (r.text or "")[:500])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
