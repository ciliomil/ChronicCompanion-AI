"""Store and query need-solution memory items."""

from __future__ import annotations

from typing import Any

from src.memory.schemas import NeedItem
from src.storage.json_store import JsonStore


class NeedSolutionStore:
    def __init__(self, json_store: JsonStore):
        self.json_store = json_store
        self.path = "memory/need_solutions.json"

    def append(self, item: NeedItem) -> None:
        items = self.json_store.read_json(self.path, default=[])
        items.append(item.to_dict())
        self.json_store.write_json(self.path, items)

    def list_all(self) -> list[dict]:
        return self.json_store.read_json(self.path, default=[])

    def update_item(self, item_id: str, **fields: Any) -> bool:
        """Patch fields on the item with matching ``item_id``.

        Used by :func:`profile_updater.update_profile_from_mid_memory` to
        write back ``cluster_id`` and by the feedback flow to merge
        ``preference`` updates. Returns ``True`` if at least one item was
        modified.
        """
        items = self.json_store.read_json(self.path, default=[])
        changed = False
        for it in items:
            if it.get("item_id") == item_id:
                it.update(fields)
                changed = True
        if changed:
            self.json_store.write_json(self.path, items)
        return changed
