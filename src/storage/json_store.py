"""Minimal JSON file store rooted at a base directory.

Memory layer stores (:class:`RawStore`, :class:`EventStore`,
:class:`NeedSolutionStore`, :class:`ProfileStore`) all delegate disk I/O to
this class so they share a single base directory and a consistent on-disk
JSON layout.

The base directory is resolved in this order:

1. The :class:`Path` passed to :meth:`__init__`.
2. ``$DATA_DIR`` (used by ``tests/test_smoke_pipeline.py``).
3. ``<repo_root>/data/runtime`` as the production default.

Reads return ``default`` when the file is missing; writes create parent
directories as needed and use ``ensure_ascii=False`` so Chinese content
stays human-readable in the JSON files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNTIME_DIR = _PROJECT_ROOT / "data" / "runtime"


class JsonStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base = self._resolve_base(base_dir)

    @staticmethod
    def _resolve_base(base_dir: str | Path | None) -> Path:
        if base_dir is not None:
            return Path(base_dir).expanduser().resolve()
        env = os.getenv("DATA_DIR", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return _DEFAULT_RUNTIME_DIR

    @property
    def base_dir(self) -> Path:
        return self._base

    def _resolve(self, path: str) -> Path:
        return self._base / path

    def read_json(self, path: str, *, default: Any) -> Any:
        full = self._resolve(path)
        if not full.is_file():
            return default
        with full.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_json(self, path: str, payload: Any) -> None:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        with full.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def reset(self) -> None:
        """Wipe every JSON file under ``base_dir`` (used by tests)."""
        if not self._base.exists():
            return
        for child in self._base.rglob("*.json"):
            try:
                child.unlink()
            except OSError:
                pass
