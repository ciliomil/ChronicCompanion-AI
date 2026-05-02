"""Manual verification script for LLMEventExtractor against a real Qwen3 endpoint.

This is NOT a pytest unit test. Run it explicitly when you can reach the
``BASIC_MODEL.base_url`` configured in ``conf.yaml``::

    python tests/manual_verify_event_extractor.py
    LONGMEM_LLM_DEBUG=1 python tests/manual_verify_event_extractor.py   # verbose

It pulls a small slice from the real dialogue dataset, runs the LLM-backed
event extractor, and prints (a) the raw LLM JSON response and (b) the
schema-validated ``EventItem`` list. A non-zero exit code indicates the
extractor produced no usable events or the LLM call failed entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# When run as ``python tests/manual_verify_event_extractor.py``, Python puts
# ``tests/`` on sys.path, not the project root — so ``import src`` fails unless
# we add the repo root explicitly (same issue as running any top-level script).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm.llm import LLMClient, OpenAICompatibleClient
from src.memory.schemas import RawTurn
from src.memory.update.event_extractor import LLMEventExtractor
from src.memory.update.prompts import (
    EVENT_EXTRACT_SYSTEM,
    build_event_extract_prompt,
)

_DATASET_PATH = (
    _PROJECT_ROOT
    / "data"
    / "ChronicCompanion-set"
    / "dialogue"
    / "history"
    / "history_dialogue.json"
)


def _load_sample_window(user_id: str = "0000", max_turns: int = 12) -> list[RawTurn]:
    """Load up to ``max_turns`` consecutive raw turns from the real dataset."""
    if not _DATASET_PATH.exists():
        raise SystemExit(f"dataset not found: {_DATASET_PATH}")

    with _DATASET_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    user_dialogue = raw.get(user_id)
    if not user_dialogue:
        raise SystemExit(f"user {user_id!r} not found in dataset")

    first_date = next(iter(user_dialogue))
    day_dialogue = user_dialogue[first_date]

    turns: list[RawTurn] = []
    timestamp = f"{first_date}T08:00:00"
    for turn_key, pair in day_dialogue.items():
        turn_idx = turn_key.replace("turn_", "")
        for role in ("user", "assistant"):
            payload = pair.get(role)
            if not payload:
                continue
            text = payload.get("content", "")
            if not text:
                continue
            turns.append(
                RawTurn(
                    session_id=f"{user_id}-{first_date}",
                    turn_id=f"{turn_key}-{role[0]}",  # e.g. turn_1-u, turn_1-a
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    topic_id=f"{user_id}-{first_date}-T{turn_idx}",
                )
            )
            if len(turns) >= max_turns:
                return turns
    return turns


def _print_raw_llm_response(client: LLMClient, turns: list[RawTurn]) -> None:
    """Show the raw model output for the same input the extractor will use."""
    prompt = build_event_extract_prompt([t.to_dict() for t in turns])
    print("\n=== raw LLM JSON response ===")
    try:
        response = client.generate_json(
            prompt,
            system_prompt=EVENT_EXTRACT_SYSTEM,
        )
        print(json.dumps(response, ensure_ascii=False, indent=2))
    except Exception as err:  # noqa: BLE001
        print(f"[error] generate_json failed: {err!r}")
        raise


def main() -> int:
    print(f"loading dialogue window from {_DATASET_PATH.name}...")
    turns = _load_sample_window()
    if not turns:
        print("no turns loaded, aborting.", file=sys.stderr)
        return 2

    print(f"loaded {len(turns)} turns. preview:")
    for t in turns[:6]:
        print(f"  [{t.turn_id}|{t.role}] {t.text[:80]}")
    if len(turns) > 6:
        print("  ...")

    # Force the OpenAI-compatible path; if conf.yaml is misconfigured this
    # will fail fast with a clear error instead of silently using the mock.
    client = OpenAICompatibleClient()
    _print_raw_llm_response(client, turns)

    print("\n=== running LLMEventExtractor.extract_from_window ===")
    extractor = LLMEventExtractor(client=client)
    events = extractor.extract_from_window(turns, window_id=turns[0].session_id)

    if not events:
        print("[warn] no events produced.", file=sys.stderr)
        return 1

    for event in events:
        print(json.dumps(event.to_dict(), ensure_ascii=False, indent=2))

    print(f"\n[ok] {len(events)} event(s) extracted from {len(turns)} turns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
