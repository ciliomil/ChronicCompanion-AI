"""Session-level ingestion orchestrator.

Given one user-session worth of dialogue (one ``sample`` from
``data/ChronicCompanion-set/dialogue/{history,query}/input.json``), this
module drives the whole memory pipeline:

1. Build :class:`RawTurn`s with externally-injected timestamps so dataset
   replays don't depend on wall-clock time.
2. Topic-segment the session into :class:`TopicWindow`s — either via the
   :class:`LLMTopicSegmenter`, or via the dataset's gold ``topics`` block
   when the caller passes one. Topic boundaries become the ``chunk_id`` on
   each persisted turn.
3. Persist turns to :class:`RawStore` window-by-window.
4. Run :class:`LLMEventExtractor` over the **whole session** in a single
   call, persist the resulting events to :class:`EventStore`.
5. Run :meth:`LLMNeedSolutionExtractor.extract_item` **once per topic window**,
   passing **this session’s** events (from step 4) so the model can fill
   ``context`` / ``context_event_ids`` on each :class:`NeedItem`. Persist each
   item
   to :class:`NeedSolutionStore`.
6. Once the session is fully written, call
   :func:`update_profile_from_mid_memory` with a :class:`FixedClock` anchored
   at the session's ``dialogue_timestamp``. Inside that call: ``recent_status``
   is summarised, then need preference clustering runs (driven mainly by **this
   session's** freshly extracted :class:`NeedItem` rows when incremental; if
   ``force_recluster=True``, **all** persisted need rows are passed so
   :meth:`NeedClusterer.initialize` can rebuild globally). **basic_info** is
   updated last from the previous profile plus **this session's** events and
   need items (``session_events`` / ``session_need_items``). Persist the new
   profile and write back ``cluster_id`` onto each affected item via
   :meth:`NeedSolutionStore.update_item`.

The orchestrator returns a small ``IngestReport`` summary (window/event/item
counts and the final cluster sizes) that callers can log or aggregate.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, replace
from typing import Any, TextIO

from src.llm.debug_llm import DebuggingLLMClient

from src.llm.embedder import Embedder, get_default_embedder
from src.llm.llm import LLMClient, get_default_llm_client
from src.memory.mid.event_store import EventStore
from src.memory.mid.need_solution_store import NeedSolutionStore
from src.memory.profile.profile_store import ProfileStore
from src.memory.raw.raw_store import RawStore
from src.memory.schemas import (
    EventItem,
    NeedItem,
    RawTurn,
    TopicWindow,
)
from src.memory.update.event_extractor import LLMEventExtractor
from src.memory.update.need_solution_extractor import LLMNeedSolutionExtractor
from src.memory.update.profile_updater import update_profile_from_mid_memory
from src.memory.update.topic_segmenter import (
    LLMTopicSegmenter,
    windows_from_dataset_topics,
)
from src.utils.clock import FixedClock, parse_external_timestamp

_logger = logging.getLogger(__name__)


def _dbg_out(stream: TextIO | None) -> TextIO:
    return stream if stream is not None else sys.stderr


def _debug_banner(stream: TextIO | None, title: str) -> None:
    s = _dbg_out(stream)
    s.write("\n")
    s.write("=" * 72 + "\n")
    s.write(f"[ingest-debug] {title}\n")
    s.write("=" * 72 + "\n")


def _debug_flat_turn_table(
    stream: TextIO | None,
    title: str,
    turns: list[RawTurn],
    *,
    text_preview_chars: int = 160,
) -> None:
    _debug_banner(stream, title)
    s = _dbg_out(stream)
    for t in turns:
        preview = t.text.strip().replace("\n", " ")
        if len(preview) > text_preview_chars:
            preview = preview[: text_preview_chars] + " …"
        s.write(
            f"{t.turn_id}\t{t.role}\tchunk_id={t.chunk_id}\t{preview}\n"
        )
    s.write(f"(total_turns={len(turns)})\n")


def _debug_windows_json(stream: TextIO | None, windows: list[TopicWindow]) -> None:
    _debug_banner(stream, "topic_windows (segmentation)")
    s = _dbg_out(stream)
    payload = [{"window_id": w.window_id, "turn_ids": w.turn_ids} for w in windows]
    s.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _debug_events_json(stream: TextIO | None, events: list[EventItem]) -> None:
    _debug_banner(stream, "event_extract (stored EventItem dicts)")
    s = _dbg_out(stream)
    rows = [e.to_dict() for e in events]
    s.write(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def _debug_need_items_json(stream: TextIO | None, items: list[NeedItem]) -> None:
    _debug_banner(stream, "need_solution_extract (stored NeedItem dicts)")
    s = _dbg_out(stream)
    rows = [it.to_dict() for it in items]
    s.write(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def _debug_profile_tail(stream: TextIO | None, profile: dict[str, Any], mapping: dict[str, str]) -> None:
    _debug_banner(stream, "profile_update (cluster_sizes + cluster_id assignment count)")
    s = _dbg_out(stream)
    s.write(json.dumps(profile.get("recent_status", {}), ensure_ascii=False, indent=2) + "\n\n")
    s.write(
        f"need_preferences clusters: {len(profile.get('need_preferences', []) or [])}\n"
    )
    s.write(json.dumps(profile.get("need_preferences", []) or [], ensure_ascii=False, indent=2) + "\n\n")
    s.write(f"item_id→cluster_id mapping entries: {len(mapping)}\n")


# ---------------------------------------------------------------------------
# Output report
# ---------------------------------------------------------------------------


@dataclass
class IngestReport:
    user_id: str
    session_id: str
    dialogue_timestamp: str
    n_turns: int
    n_windows: int
    n_events: int
    n_need_items: int
    cluster_sizes: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "dialogue_timestamp": self.dialogue_timestamp,
            "n_turns": self.n_turns,
            "n_windows": self.n_windows,
            "n_events": self.n_events,
            "n_need_items": self.n_need_items,
            "cluster_sizes": self.cluster_sizes,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_session_id(user_id: str, dialogue_timestamp: str) -> str:
    """Compose a stable session id like ``s-0000-20250206-1300``.

    The dataset already ships an authoritative ``sample_id``; production
    callers should pass it through ``session_id=...``. This helper is the
    fallback when only the timestamp is available.
    """
    dt = parse_external_timestamp(dialogue_timestamp)
    return f"s-{user_id}-{dt.strftime('%Y%m%d-%H%M')}"


def _flatten_dialogue(
    dialogue: dict[str, dict[str, dict[str, Any]]],
    *,
    session_id: str,
    timestamp_iso: str,
) -> list[RawTurn]:
    """Flatten the dataset's ``dialogue`` dict into ordered :class:`RawTurn`s.

    Per the agreed convention every turn in a session shares the session's
    ``dialogue_timestamp``; ordering is carried by ``turn_id``
    (``t-{session_id}-{seq}``) and by list position in the store.

    Only ``content`` is taken from the dataset; the ``action`` annotation is
    metadata that does not belong in raw memory.
    """
    turns: list[RawTurn] = []
    seq = 0

    def _ordering_key(name: str) -> tuple[int, str]:
        # "turn_1", "turn_2", ..., "turn_10" — sort numerically when possible.
        if name.startswith("turn_"):
            tail = name[len("turn_"):]
            if tail.isdigit():
                return (int(tail), name)
        return (10**9, name)

    for turn_name in sorted(dialogue.keys(), key=_ordering_key):
        turn_payload = dialogue[turn_name]
        if not isinstance(turn_payload, dict):
            continue
        for role in ("user", "assistant"):
            payload = turn_payload.get(role)
            if not isinstance(payload, dict):
                continue
            content = str(payload.get("content", "")).strip()
            if not content:
                continue
            seq += 1
            turn_id = f"turn-{session_id}-{seq:03d}"
            turns.append(
                RawTurn(
                    session_id=session_id,
                    turn_id=turn_id,
                    role=role,
                    text=content,
                    timestamp=timestamp_iso,
                    chunk_id="chunk-0",  # filled in once segmentation runs
                )
            )
    return turns


def _resolve_windows(
    turns: list[RawTurn],
    *,
    predefined_windows: list[TopicWindow] | None,
    gold_topics: dict[str, dict[str, Any]] | None,
    topic_segmenter: LLMTopicSegmenter,
) -> list[TopicWindow]:
    """Pick the segmentation source: predefined > gold-topics > LLM."""
    if predefined_windows:
        return _coerce_predefined_windows(predefined_windows, turns)
    if gold_topics:
        return windows_from_dataset_topics(turns, gold_topics)
    return topic_segmenter.segment(turns)


def _coerce_predefined_windows(
    windows: list[TopicWindow],
    turns: list[RawTurn],
) -> list[TopicWindow]:
    """Validate caller-supplied windows cover all turn ids exactly once."""
    valid_ids = [t.turn_id for t in turns]
    seen: list[str] = []
    for w in windows:
        for tid in w.turn_ids:
            if tid not in valid_ids or tid in seen:
                raise ValueError(
                    f"predefined_windows reference invalid/duplicate turn_id={tid!r}"
                )
            seen.append(tid)
    if seen != valid_ids:
        raise ValueError(
            "predefined_windows must cover every turn_id exactly once and in order"
        )
    return windows


def _stamp_turns_with_window(
    turns: list[RawTurn],
    windows: list[TopicWindow],
) -> list[RawTurn]:
    """Set each turn's ``chunk_id`` / ``topic_id`` from its enclosing window.

    Both use ``window_id`` so topic chunking scopes raw turns without a separate
    human-readable topic phrase field.
    """
    by_turn: dict[str, TopicWindow] = {}
    for w in windows:
        for tid in w.turn_ids:
            by_turn[tid] = w
    out: list[RawTurn] = []
    for t in turns:
        w = by_turn.get(t.turn_id)
        if w is None:
            out.append(t)
            continue
        out.append(
            RawTurn(
                session_id=t.session_id,
                turn_id=t.turn_id,
                role=t.role,
                text=t.text,
                timestamp=t.timestamp,
                chunk_id=w.window_id,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def ingest_session(
    *,
    user_id: str,
    dialogue: dict[str, dict[str, dict[str, Any]]],
    dialogue_timestamp: str,
    raw_store: RawStore,
    event_store: EventStore,
    need_solution_store: NeedSolutionStore,
    profile_store: ProfileStore,
    event_extractor: LLMEventExtractor | None = None,
    need_extractor: LLMNeedSolutionExtractor | None = None,
    topic_segmenter: LLMTopicSegmenter | None = None,
    embedder: Embedder | None = None,
    llm_client: LLMClient | None = None,
    predefined_windows: list[TopicWindow] | None = None,
    gold_topics: dict[str, dict[str, Any]] | None = None,
    session_id: str | None = None,
    cluster_min_size: int = 3,
    window_days: int = 14,
    min_items_to_cluster: int = 5,
    force_recluster: bool = False,
    run_profile_update: bool = True,
    debug: bool = False,
    debug_stream: TextIO | None = None,
    debug_max_prompt_chars: int = 24_000,
) -> IngestReport:
    """Ingest one session of dialogue end-to-end.

    Args:
        dialogue: The raw ``dialogue`` sub-dict from a dataset sample,
            keyed by ``turn_1``, ``turn_2``, ... and containing
            ``{user|assistant}: {action, content}`` payloads. Only ``content``
            is read; ``action`` is treated as gold annotation metadata and
            ignored.
        dialogue_timestamp: External clock anchor for this session — copied
            verbatim into every :class:`RawTurn`/:class:`EventItem`/
            :class:`NeedItem`, and used as :class:`FixedClock` for
            ``recent_status`` window-cutoff and all ``updated_at`` writes.
        predefined_windows: Caller-supplied :class:`TopicWindow`s; takes
            priority over ``gold_topics``. Useful for unit tests where you
            want to bypass segmentation entirely.
        gold_topics: The dataset's ``topics`` block. When given (and no
            ``predefined_windows``) the ingest uses
            :func:`windows_from_dataset_topics` instead of running
            :class:`LLMTopicSegmenter`. Lets eval runs test downstream
            modules with gold topic boundaries.
        run_profile_update: Set to ``False`` when bulk-loading and you want
            to defer the (expensive) profile update until the very end.
        debug: Echo flattening / segmentation / intermediates plus wrap the
            active :class:`~src.llm.llm.LLMClient` so every ``generate_*`` prints
            system & user prompts and raw responses on ``debug_stream`` (stderr by
            default). Does not intercept extractors instantiated with their own
            client (pass ``None`` defaults only).
        debug_stream: Alternate IO stream for human-readable ingest traces;
            LLM echoes use the same stream.
        debug_max_prompt_chars: Truncate echoed prompts / large JSON blobs to
            this many characters each (0 = unlimited).
    """

    base_llm = llm_client or get_default_llm_client()
    llm: LLMClient = (
        DebuggingLLMClient(
            base_llm,
            stream=_dbg_out(debug_stream),
            max_prompt_chars=max(0, int(debug_max_prompt_chars)),
        )
        if debug
        else base_llm
    )
    emb = embedder or get_default_embedder()
    event_ex = event_extractor or LLMEventExtractor(client=llm)
    need_ex = need_extractor or LLMNeedSolutionExtractor(client=llm)
    segmenter = topic_segmenter or LLMTopicSegmenter(client=llm)

    timestamp_dt = parse_external_timestamp(dialogue_timestamp)
    timestamp_iso = timestamp_dt.isoformat()
    sid = session_id or _default_session_id(user_id, dialogue_timestamp)

    # Step 1: flatten dialogue → ordered RawTurns.
    turns = _flatten_dialogue(dialogue, session_id=sid, timestamp_iso=timestamp_iso)
    if not turns:
        _logger.warning("ingest_session: empty dialogue for user=%s session=%s", user_id, sid)
        return IngestReport(
            user_id=user_id, session_id=sid, dialogue_timestamp=timestamp_iso,
            n_turns=0, n_windows=0, n_events=0, n_need_items=0,
        )

    if debug:
        _debug_flat_turn_table(debug_stream, f"flattened_turns (before topic chunking) [{sid}]", turns)

    # Step 2: topic segmentation.
    windows = _resolve_windows(
        turns,
        predefined_windows=predefined_windows,
        gold_topics=gold_topics,
        topic_segmenter=segmenter,
    )
    stamped_turns = _stamp_turns_with_window(turns, windows)

    if debug:
        _debug_windows_json(debug_stream, windows)
        _debug_flat_turn_table(debug_stream, f"stamped_turns [{sid}]", stamped_turns)

    # Step 3: persist raw turns, walking window-by-window so the on-disk
    # ordering matches the topic structure.
    turn_by_id = {t.turn_id: t for t in stamped_turns}
    n_turns_written = 0
    for w in windows:
        for tid in w.turn_ids:
            t = turn_by_id.get(tid)
            if t is None:
                continue
            raw_store.append_turn(t)
            n_turns_written += 1

    # Step 4: events over the whole session in one call.
    events = event_ex.extract_from_session(stamped_turns, session_id=sid)
    if debug:
        _debug_events_json(debug_stream, events)
    for ev in events:
        event_store.append(ev)

    session_event_dicts = [ev.to_dict() for ev in events]

    # Step 5: need rows per window. 
    need_items: list[NeedItem] = []
    need_session_idx = 0
    for w in windows:
        window_turns = [turn_by_id[tid] for tid in w.turn_ids if tid in turn_by_id]
        for item in need_ex.extract_item(
            window_turns,
            session_events=session_event_dicts,
        ):
            need_session_idx += 1
            item = replace(item, item_id=f"need-{sid}-{need_session_idx}")
            need_solution_store.append(item)
            need_items.append(item)

    if debug:
        _debug_need_items_json(debug_stream, need_items)

    # Step 6: profile update.
    cluster_sizes: list[tuple[str, int]] = []
    mapping: dict[str, str] = {}
    if run_profile_update:
        clock = FixedClock(timestamp_dt)
        profile_dict = profile_store.load()
        all_events = event_store.list_all()
        new_need_dicts = [it.to_dict() for it in need_items]
        existing_clusters = profile_dict.get("need_preferences") or []
        if not isinstance(existing_clusters, list):
            existing_clusters = []
        # Incremental clustering only considers rows without ``cluster_id``, so
        # passing this session's items is enough. Full re-init must see the
        # whole store because :meth:`NeedClusterer.initialize` replaces clusters.
        need_rows_for_profile = (
            need_solution_store.list_all()
            if force_recluster or not existing_clusters
            else new_need_dicts
        )
        profile, mapping = update_profile_from_mid_memory(
            profile_dict,
            all_events,
            need_rows_for_profile,
            embedder=emb,
            llm_client=llm,
            clock=clock,
            window_days=window_days,
            session_events=session_event_dicts,
            session_need_items=new_need_dicts,
        )
        profile_store.save(profile)
        for item_id, cluster_id in mapping.items():
            need_solution_store.update_item(item_id, cluster_id=cluster_id)
        cluster_sizes = [
            (str(c.get("cluster_id", "")), int(c.get("size", 0) or 0))
            for c in profile.need_preferences
        ]
        if debug:
            _debug_profile_tail(debug_stream, profile.to_dict(), mapping)
    elif debug:
        _debug_banner(debug_stream, "profile_update skipped (run_profile_update=False)")

    return IngestReport(
        user_id=user_id,
        session_id=sid,
        dialogue_timestamp=timestamp_iso,
        n_turns=n_turns_written,
        n_windows=len(windows),
        n_events=len(events),
        n_need_items=len(need_items),
        cluster_sizes=cluster_sizes,
    )
