"""Bulk-ingest the dataset's ``history`` blocks into the layered memory stores.

Each user → each ``history`` sample becomes one ``ingest_session`` call. The
sample's ``dialogue_timestamp`` is injected as the clock anchor so all
on-disk timestamps line up with the dataset (rather than wall-clock time
at run time).

Usage::

    PYTHONPATH=. python -m src.runner.run_history_ingest \
        --dataset ./data/ChronicCompanion-set/dialogue/history/input.json \
        --data-dir ./tmp/mem-test0505 \
        --use-gold-topics \
        --max-samples-per-user 5 \
        --user 0001 \
        --debug \
        2>ingest-debug5.log


Append ``--debug`` to mirror flattening tables, segmentation JSON, extractor
artifacts, and each LLM system/user prompt plus parsed responses on stderr
(the summary JSON stays on stdout), e.g. redirect with ``2>ingest-debug.log``.

Use ``--debug-prompt-cap 0`` to disable truncation of echoed prompts/blocks.

Pass ``--user`` multiple times to restrict the run; omit it to ingest every
user. ``--data-dir`` is forwarded to the :class:`JsonStore` base directory
(also via ``$DATA_DIR``); the script creates a separate sub-directory per
``user_id`` to keep their memory pools isolated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

from src.memory.mid.event_store import EventStore
from src.memory.mid.need_solution_store import NeedSolutionStore
from src.memory.profile.profile_store import ProfileStore
from src.memory.raw.raw_store import RawStore
from src.memory.update.session_ingest import IngestReport, ingest_session
from src.storage.json_store import JsonStore

_logger = logging.getLogger(__name__)


def _build_stores(data_dir: Path, user_id: str) -> dict[str, Any]:
    user_dir = data_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    js = JsonStore(user_dir)
    return {
        "raw_store": RawStore(js),
        "event_store": EventStore(js),
        "need_solution_store": NeedSolutionStore(js),
        "profile_store": ProfileStore(js),
    }


def _iter_users(
    data: dict[str, Any],
    selected: Iterable[str] | None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    selected_set = set(selected) if selected else None
    for user_id, blob in data.items():
        if selected_set is not None and user_id not in selected_set:
            continue
        if not isinstance(blob, dict):
            continue
        yield user_id, blob


def run(
    *,
    dataset_path: Path,
    data_dir: Path,
    selected_users: Iterable[str] | None = None,
    use_gold_topics: bool = False,
    max_samples_per_user: int | None = None,
    cluster_min_size: int = 3,
    window_days: int = 14,
    min_items_to_cluster: int = 5,
    debug: bool = False,
    debug_prompt_cap: int = 24_000,
) -> list[IngestReport]:
    """Ingest history samples and return a per-session report list."""
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    reports: list[IngestReport] = []
    for user_id, blob in _iter_users(data, selected_users):
        history = blob.get("history") or []
        if not isinstance(history, list):
            continue
        stores = _build_stores(data_dir, user_id)
        for idx, sample in enumerate(history):
            if max_samples_per_user is not None and idx >= max_samples_per_user:
                break
            if not isinstance(sample, dict):
                continue
            dialogue = sample.get("dialogue")
            if not isinstance(dialogue, dict) or not dialogue:
                continue
            sample_id = str(sample.get("sample_id") or f"sample-{idx}")
            session_id = f"{sample_id}"
            gold_topics = sample.get("topics") if use_gold_topics else None

            report = ingest_session(
                user_id=user_id,
                dialogue=dialogue,
                dialogue_timestamp=str(sample.get("dialogue_timestamp", "")),
                **stores,
                session_id=session_id,
                gold_topics=gold_topics if isinstance(gold_topics, dict) else None,
                cluster_min_size=cluster_min_size,
                window_days=window_days,
                min_items_to_cluster=min_items_to_cluster,
                debug=debug,
                debug_stream=sys.stderr,
                debug_max_prompt_chars=debug_prompt_cap,
            )
            _logger.info(
                "ingested user=%s sample=%s windows=%d events=%d need_items=%d",
                user_id, sample_id, report.n_windows, report.n_events, report.n_need_items,
            )
            reports.append(report)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/ChronicCompanion-set/dialogue/history/input.json"),
        help="Path to history input.json.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/runtime"),
        help="Base directory for memory JSON stores (also seeds $DATA_DIR).",
    )
    parser.add_argument(
        "--user",
        action="append",
        default=None,
        help="Restrict to one or more user_ids; pass repeatedly. Default: all users.",
    )
    parser.add_argument(
        "--use-gold-topics",
        action="store_true",
        help="Use the dataset's gold ``topics`` block instead of the LLM segmenter.",
    )
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        default=None,
        help="Cap the number of history samples ingested per user (smoke runs).",
    )
    parser.add_argument(
        "--cluster-min-size",
        type=int,
        default=3,
        metavar="M",
        help=(
            "HDBSCAN min_cluster_size for need-preference bootstrap (smallest "
            "dense group). Not a fixed cluster count K."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=14,
        help="recent_status time window (anchored at sample's dialogue_timestamp).",
    )
    parser.add_argument(
        "--min-items-to-cluster",
        type=int,
        default=5,
        help="Below this many items, every item becomes a singleton cluster.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Stream ingest traces + full LLM prompts/responses on stderr "
            "(summary JSON stays on stdout). See also LONGMEM_LLM_DEBUG in llm.py."
        ),
    )
    parser.add_argument(
        "--debug-prompt-cap",
        type=int,
        default=24_000,
        help=(
            "Max characters per echoed prompt/block in debug mode "
            "(0 = unlimited)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG/INFO/WARNING).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    reports = run(
        dataset_path=args.dataset,
        data_dir=args.data_dir,
        selected_users=args.user,
        use_gold_topics=args.use_gold_topics,
        max_samples_per_user=args.max_samples_per_user,
        cluster_min_size=args.cluster_min_size,
        window_days=args.window_days,
        min_items_to_cluster=args.min_items_to_cluster,
        debug=args.debug,
        debug_prompt_cap=args.debug_prompt_cap,
    )

    summary = {
        "total_sessions": len(reports),
        "total_events": sum(r.n_events for r in reports),
        "total_need_items": sum(r.n_need_items for r in reports),
        "total_windows": sum(r.n_windows for r in reports),
        "users": sorted({r.user_id for r in reports}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
