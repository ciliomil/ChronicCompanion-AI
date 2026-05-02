"""Store and query event items."""

from __future__ import annotations

from src.memory.schemas import EventItem
from src.storage.json_store import JsonStore


class EventStore:
    def __init__(self, json_store: JsonStore):
        self.json_store = json_store
        self.path = "memory/events.json"

    def append(self, event: EventItem) -> None:
        items = self.json_store.read_json(self.path, default=[])
        items.append(event.to_dict())
        self.json_store.write_json(self.path, items)

    def list_all(self) -> list[dict]:
        return self.json_store.read_json(self.path, default=[])
