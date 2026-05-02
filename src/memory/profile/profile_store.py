"""Store and load user profile memory."""

from __future__ import annotations

from src.memory.schemas import UserProfile
from src.storage.json_store import JsonStore


class ProfileStore:
    def __init__(self, json_store: JsonStore):
        self.json_store = json_store
        self.path = "memory/user_profile.json"

    def load(self) -> dict:
        return self.json_store.read_json(self.path, default=UserProfile().to_dict())

    def save(self, profile: UserProfile) -> None:
        self.json_store.write_json(self.path, profile.to_dict())

    def save_dict(self, profile: dict) -> None:
        self.json_store.write_json(self.path, profile)
