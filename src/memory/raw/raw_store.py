"""Store and read raw dialogue turns."""

from __future__ import annotations

from src.memory.schemas import RawTurn
from src.storage.json_store import JsonStore


class RawStore:
    def __init__(self, json_store: JsonStore):
        self.json_store = json_store
        self.path = "memory/raw_turns.json"

    def append_turn(self, turn: RawTurn) -> None:
        data = self.json_store.read_json(self.path, default=[])
        data.append(turn.to_dict())
        self.json_store.write_json(self.path, data)

    def list_turns(self) -> list[dict]:
        return self.json_store.read_json(self.path, default=[])
