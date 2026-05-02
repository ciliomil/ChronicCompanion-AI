"""Basic memory read/write smoke test."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.memory.raw.raw_store import RawStore
from src.memory.schemas import RawTurn
from src.storage.json_store import JsonStore


class MemoryWriteReadTest(unittest.TestCase):
    def test_raw_turn_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RawStore(JsonStore(Path(tmp)))
            store.append_turn(RawTurn(session_id="s", turn_id="t1", role="user", text="hello"))
            turns = store.list_turns()
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
