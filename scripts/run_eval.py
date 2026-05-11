#!/usr/bin/env python3
"""Thin wrapper for ``python -m src.evaluation``.

Usage::

    python scripts/run_eval.py --task all --memory_strategy pack --user 0000 0001

See ``src/evaluation/cli.py`` for all flags.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when this file is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
