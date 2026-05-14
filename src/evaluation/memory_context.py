"""Memory context providers for the eval CLI.

Four strategies:

- ``pack``: full retrieval pipeline. Task 1 is a "needs understanding" task
  whose input is shaped like the retrieval planner's input, so for task 1 we
  render basic_info (4 sections) + active/uncertain claims + recent_status
  directly. Tasks 2/3 use the gold ``requirement`` as the retrieval query and
  render the resulting :class:`~src.retrieval.MemoryPack`.
- ``full``: no retrieval. Dump the entire profile + needs + events + recent
  status as text. Same content for task 1 and tasks 2/3.
- ``no_memory``: emit a ``"(无记忆)"`` placeholder. Establishes the zero-shot
  lower bound.
- ``mem0``: use the naive_mem0 bucket retrieval baseline and render retrieved
  memories as user profile / semantic memory / SOP sections.

Both task-1 and task-23 entry points are cached per ``(user_id, query)`` so
the planner LLM call (only invoked under ``pack``) is amortized across the
three tasks of a topic and across topics that share a query.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.llm.embedder import Embedder, get_default_embedder
from src.llm.llm import LLMClient
from src.memory.mid.event_store import EventStore
from src.memory.mid.need_solution_store import NeedSolutionStore
from src.memory.ontology import BASIC_INFO_CATEGORIES
from src.memory.profile.profile_store import ProfileStore
from src.retrieval import MemoryPack, build_memory_pack
from src.storage.json_store import JsonStore


# ---------------------------------------------------------------------------
# Provider protocol + factory
# ---------------------------------------------------------------------------


class MemoryContextProvider(Protocol):
    strategy: str

    def for_task1(
        self,
        user_id: str,
        *,
        user_query: str,
        dialogue_context: str = "",
    ) -> str: ...

    def for_task23(
        self,
        user_id: str,
        *,
        query: str,
        dialogue_context: str = "",
    ) -> str: ...


def make_provider(
    strategy: str,
    *,
    memory_root: str | Path,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    mem0_run_id: str = "",
) -> MemoryContextProvider:
    s = strategy.strip().lower()
    if s == "pack":
        return PackMemoryProvider(memory_root=memory_root, llm=llm, embedder=embedder)
    if s == "full":
        return FullMemoryProvider(memory_root=memory_root)
    if s in {"no_memory", "no-memory", "none"}:
        return NoMemoryProvider()
    if s == "mem0":
        return Mem0MemoryProvider(run_id=mem0_run_id)
    raise ValueError(
        f"Unknown memory_strategy={strategy!r}; "
        f"choose one of pack / full / no_memory / mem0."
    )


# ---------------------------------------------------------------------------
# Shared user-data loader
# ---------------------------------------------------------------------------


@dataclass
class _UserData:
    profile: dict[str, Any]
    events: list[dict[str, Any]]
    needs: list[dict[str, Any]]


def _load_user_data(memory_root: Path, user_id: str) -> _UserData:
    base = memory_root / user_id
    if not base.is_dir():
        raise FileNotFoundError(f"No memory dir for user {user_id!r} under {memory_root}")
    store = JsonStore(base)
    profile = ProfileStore(store).load()
    events = EventStore(store).list_all()
    needs = NeedSolutionStore(store).list_all()
    return _UserData(profile=profile, events=events, needs=needs)


# ---------------------------------------------------------------------------
# Renderers shared across strategies
# ---------------------------------------------------------------------------


def _render_basic_info_full(profile: dict[str, Any]) -> str:
    """4 段 summary + 全部 active/uncertain claims, plain-text format."""
    bi = profile.get("basic_info") or {}
    lines: list[str] = []
    for section in BASIC_INFO_CATEGORIES:
        sec = bi.get(section) if isinstance(bi, dict) else None
        if not isinstance(sec, dict):
            continue
        summary = str(sec.get("summary") or "").strip()
        lines.append(f"[{section}] 总结：{summary or '（暂无）'}")
        for c in sec.get("claims") or []:
            if not isinstance(c, dict):
                continue
            status = str(c.get("status") or "").strip()
            if status not in ("active", "uncertain"):
                continue
            content = str(c.get("content") or "").strip()
            ctype = str(c.get("claim_type") or "").strip()
            tags = ", ".join(list(c.get("tags") or []))
            lines.append(
                f"  - {content}"
                f" (claim_type={ctype or 'n/a'}, status={status}"
                + (f", tags={tags}" if tags else "")
                + ")"
            )
    return "\n".join(lines).strip() or "（暂无 basic_info）"

def _render_basic_info_summaries_only(profile: dict[str, Any]) -> str:
    """4 段 summary, plain-text format."""
    bi = profile.get("basic_info") or {}
    lines: list[str] = []
    for section in BASIC_INFO_CATEGORIES:
        summary = str(bi.get(section, {}).get("summary") or "").strip()
        lines.append(f"[{section}] 总结：{summary or '（暂无）'}")
    return "\n".join(lines).strip() or "（暂无 basic_info）"


def _render_recent_status(profile: dict[str, Any]) -> str:
    rs = profile.get("recent_status") or {}
    if not isinstance(rs, dict):
        return "（暂无 recent_status）"
    keys = (
        "health_status",
        "self_management_status",
        "mental_status",
        "family_social_status",
        "interest_changes",
        "risk_flags",
    )
    lines: list[str] = []
    for k in keys:
        v = rs.get(k)
        if isinstance(v, list):
            joined = "、".join(str(x).strip() for x in v if str(x).strip())
            if joined:
                lines.append(f"- {k}: {joined}")
        elif isinstance(v, str) and v.strip():
            lines.append(f"- {k}: {v.strip()}")
    return "\n".join(lines).strip() or "（暂无 recent_status 关键字段）"


def _render_need_preferences(profile: dict[str, Any]) -> str:
    prefs = profile.get("need_preferences") or []
    if not isinstance(prefs, list) or not prefs:
        return "（暂无 need_preferences）"
    lines: list[str] = []
    for row in prefs:
        if not isinstance(row, dict):
            continue
        nd = str(row.get("need_domain") or "").strip()
        principle = str(row.get("preference_principle") or "").strip()
        if not principle:
            continue
        lines.append(f"- (need_domain={nd or 'n/a'}) {principle}")
    return "\n".join(lines).strip() or "（暂无 need_preferences）"


def _render_needs_full(needs: list[dict[str, Any]], limit: int = 14) -> str:
    if not needs:
        return "（暂无历史 NeedItem）"
    lines: list[str] = []
    for it in needs[:limit]:
        if not isinstance(it, dict):
            continue
        head = (
            f"[{it.get('item_id', '')} | {it.get('need_domain', '')}] "
            f"need={it.get('inferred_need', '')}"
        )
        lines.append(head)
        ctx = str(it.get("context") or "").strip()
        if ctx:
            lines.append(f"  上下文：{ctx}")
        for sol in it.get("solutions") or []:
            if not isinstance(sol, dict):
                continue
            summary = str(sol.get("ai_solution_summary") or "").strip()
            fit = sol.get("fit_score")
            rev = str(sol.get("revealed_preference") or "").strip()
            extras = []
            if fit is not None:
                extras.append(f"fit={fit}")
            if rev:
                extras.append(f"revealed_preference={rev}")
            tail = f" ({', '.join(extras)})" if extras else ""
            if summary:
                lines.append(f"  方案：{summary}{tail}")
    return "\n".join(lines).strip()


def _render_events_full(events: list[dict[str, Any]], limit: int = 20) -> str:
    if not events:
        return "（暂无 events）"
    sorted_events = sorted(
        (e for e in events if isinstance(e, dict)),
        key=lambda e: str(e.get("timestamp") or ""),
        reverse=True,
    )[:limit]
    lines: list[str] = []
    for e in sorted_events:
        ts = str(e.get("timestamp") or "")[:10]
        summary = str(e.get("event_summary") or "").strip()
        tags = ", ".join(list(e.get("tags") or []))
        line = f"- [{ts}] {summary}"
        if tags:
            line += f" (tags={tags})"
        lines.append(line)
    return "\n".join(lines).strip()


def _render_pack(pack: MemoryPack) -> str:
    """Render the retrieval pack into reader-friendly Chinese text.

    The 4-section summaries are *not* in the pack itself (pack only carries
    selected claims) — callers add them on top of this rendering.
    """
    parts: list[str] = []

    if pack.selected_basic_info_claims:
        parts.append("== 当前轮选定的画像要点（含应用角色）==")
        for c in pack.selected_basic_info_claims:
            role = c.get("assigned_role", "background_context")
            content = str(c.get("content") or "").strip()
            section = str(c.get("category") or "").strip()
            parts.append(f"- [{role} | {section}] {content}")

    if pack.preference_principles:
        parts.append("\n== 用户偏好原则（匹配 cluster）==")
        for p in pack.preference_principles:
            parts.append(
                f"- (need_domain={p.get('need_domain', 'n/a')}) "
                f"{p.get('preference_principle', '')}"
            )

    if pack.relevant_needs:
        parts.append("\n== 历史相似需求 ==")
        parts.append(_render_needs_full(pack.relevant_needs, limit=len(pack.relevant_needs)))

    if pack.relevant_events:
        parts.append("\n== 相关近期事件 ==")
        parts.append(_render_events_full(pack.relevant_events, limit=len(pack.relevant_events)))

    if pack.recent_status_slice:
        parts.append("\n== 近期状态切片 ==")
        for k, v in pack.recent_status_slice.items():
            if isinstance(v, list):
                joined = "、".join(str(x).strip() for x in v if str(x).strip())
                if joined:
                    parts.append(f"- {k}: {joined}")
            elif isinstance(v, str) and v.strip():
                parts.append(f"- {k}: {v.strip()}")

    cur = pack.current_query
    if cur.risk_level != "normal" or cur.risk_triggers:
        parts.append("\n== 风险提示 ==")
        parts.append(
            f"- risk_level={cur.risk_level}; "
            f"risk_triggers={cur.risk_triggers or '[]'}; "
            f"medical_relevance={cur.medical_relevance}"
        )

    return "\n".join(parts).strip() or "（pack 为空，无可用记忆）"


# ---------------------------------------------------------------------------
# pack strategy
# ---------------------------------------------------------------------------


class PackMemoryProvider:
    """Default strategy: retrieval planner + deterministic retrieval."""

    strategy = "pack"

    def __init__(
        self,
        *,
        memory_root: str | Path,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._memory_root = Path(memory_root).expanduser().resolve()
        self._llm = llm
        self._embedder = embedder or get_default_embedder()
        self._user_cache: dict[str, _UserData] = {}
        self._task1_cache: dict[tuple[str, str], str] = {}
        self._pack_cache: dict[tuple[str, str], MemoryPack] = {}

    def _user(self, user_id: str) -> _UserData:
        if user_id not in self._user_cache:
            self._user_cache[user_id] = _load_user_data(self._memory_root, user_id)
        return self._user_cache[user_id]

    def for_task1(
        self,
        user_id: str,
        *,
        user_query: str,
        dialogue_context: str = "",
    ) -> str:
        # Task 1's input is shaped like the retrieval planner's input — no
        # MemoryPack involved. Renders 4 section summaries + active/uncertain
        # claims + recent_status. Cached per (user_id, user_query) so reruns
        # of the same topic don't re-render.
        key = (user_id, user_query)
        if key not in self._task1_cache:
            data = self._user(user_id)
            sections = ["== basic_info ==", _render_basic_info_full(data.profile)]
            sections.append("\n== recent_status ==")
            sections.append(_render_recent_status(data.profile))
            self._task1_cache[key] = "\n".join(sections).strip()
        return self._task1_cache[key]

    def for_task23(
        self,
        user_id: str,
        *,
        query: str,
        dialogue_context: str = "",
    ) -> str:
        # Tasks 2/3 use the retrieval pipeline with ``query`` (gold requirement)
        # as the retrieval signal. Pack is cached so task 2 and task 3 of the
        # same topic share one plan LLM call.
        key = (user_id, query)
        if key not in self._pack_cache:
            data = self._user(user_id)
            self._pack_cache[key] = build_memory_pack(
                session_id=f"eval-{user_id}",
                turn_id=f"eval-q-{abs(hash(query)) & 0xFFFF:04x}",
                user_query=query,
                dialogue_context=dialogue_context,
                profile=data.profile,
                need_solution_items=data.needs,
                events=data.events,
                llm=self._llm,
                embedder=self._embedder,
            )
        pack = self._pack_cache[key]

        # Tasks 2/3 also get the 4-section summaries on top of the pack so
        # they have the long-term backdrop the planner may have filtered out.
        head = "== 用户长期画像（4 段总结）==\n" + _render_basic_info_summaries_only(
            self._user(user_id).profile
        )
        # Omit historical need/event rows from the LLM prompt; keep pack build
        # (planner + retrieval) unchanged for fair comparison of other pack fields.
        rendered_pack = _render_pack(pack, include_need_and_event_items=False)
        prefs = "\n\n== 用户偏好原则（全量）==\n" + _render_need_preferences(
            self._user(user_id).profile
        )
        return head + "\n\n" + rendered_pack + prefs


def _render_basic_info_summaries_only(profile: dict[str, Any]) -> str:
    bi = profile.get("basic_info") or {}
    lines: list[str] = []
    for section in BASIC_INFO_CATEGORIES:
        sec = bi.get(section) if isinstance(bi, dict) else None
        summary = ""
        if isinstance(sec, dict):
            summary = str(sec.get("summary") or "").strip()
        lines.append(f"[{section}] {summary or '（暂无）'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# full strategy
# ---------------------------------------------------------------------------


class FullMemoryProvider:
    """Dump the whole profile + needs + events without retrieval."""

    strategy = "full"

    def __init__(self, *, memory_root: str | Path) -> None:
        self._memory_root = Path(memory_root).expanduser().resolve()
        self._cache: dict[str, _UserData] = {}
        self._render_cache: dict[str, str] = {}

    def _user(self, user_id: str) -> _UserData:
        if user_id not in self._cache:
            self._cache[user_id] = _load_user_data(self._memory_root, user_id)
        return self._cache[user_id]

    def _render(self, user_id: str) -> str:
        if user_id not in self._render_cache:
            data = self._user(user_id)
            blocks = [
                "== basic_info ==",
                _render_basic_info_full(data.profile),
                "\n== recent_status ==",
                _render_recent_status(data.profile),
                "\n== need_preferences ==",
                _render_need_preferences(data.profile),
                "\n== 历史 NeedItem（全量）==",
                _render_needs_full(data.needs, limit=len(data.needs)),
                "\n== events（最近）==",
                _render_events_full(data.events, limit=20),
            ]
            self._render_cache[user_id] = "\n".join(blocks).strip()
        return self._render_cache[user_id]

    def for_task1(
        self,
        user_id: str,
        *,
        user_query: str,
        dialogue_context: str = "",
    ) -> str:
        return self._render(user_id)

    def for_task23(
        self,
        user_id: str,
        *,
        query: str,
        dialogue_context: str = "",
    ) -> str:
        return self._render(user_id)


# ---------------------------------------------------------------------------
# no_memory strategy
# ---------------------------------------------------------------------------


class NoMemoryProvider:
    """Zero-shot lower bound — no memory exposed to the LLM."""

    strategy = "no_memory"
    _PLACEHOLDER = "（无记忆，纯零样本基线）"

    def for_task1(
        self,
        user_id: str,
        *,
        user_query: str,
        dialogue_context: str = "",
    ) -> str:
        return self._PLACEHOLDER

    def for_task23(
        self,
        user_id: str,
        *,
        query: str,
        dialogue_context: str = "",
    ) -> str:
        return self._PLACEHOLDER


# ---------------------------------------------------------------------------
# mem0 strategy
# ---------------------------------------------------------------------------


def _mem0_safe_json_loads(text: str) -> Any:
    import json

    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001 - malformed memory text should not break eval
        return None


def _extract_mem0_memories_for_prompt(
    memory_results: list[dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "user_profile": [],
        "semantic_memory": [],
        "sop": [],
        "other": [],
    }

    for item in memory_results:
        result_obj = item.get("result", {})
        for mem in result_obj.get("results", []):
            memory_text = (mem.get("memory") or "").strip()
            if not memory_text:
                continue
            metadata = mem.get("metadata") or {}
            category = str(metadata.get("catagory") or metadata.get("category") or "").lower()
            parsed_json = _mem0_safe_json_loads(memory_text)

            if category == "user_profile":
                grouped["user_profile"].append(memory_text)
            elif category == "semantic_memory":
                grouped["semantic_memory"].append(memory_text)
            elif category == "sop":
                if isinstance(parsed_json, dict) and {"scenario", "steps", "rationale"} <= set(parsed_json.keys()):
                    steps = parsed_json.get("steps") or []
                    steps_text = "；".join([str(step).strip() for step in steps if str(step).strip()])
                    grouped["sop"].append(
                        f"场景：{parsed_json.get('scenario', '')}；步骤：{steps_text}；原理：{parsed_json.get('rationale', '')}"
                    )
                else:
                    grouped["sop"].append(memory_text)
            else:
                grouped["other"].append(memory_text)
    return grouped


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _to_bullet_block(items: list[str], fallback: str) -> str:
    final_items = _dedupe_keep_order([item for item in items if item.strip()])
    if not final_items:
        return "- " + fallback
    return "\n".join([f'- "{item}"' for item in final_items[:12]])


def _build_mem0_memory_block(memory_results: list[dict[str, Any]]) -> str:
    grouped = _extract_mem0_memories_for_prompt(memory_results)
    profiles = _to_bullet_block(grouped["user_profile"], "暂无明显用户画像记忆。")
    semantics = _to_bullet_block(grouped["semantic_memory"], "暂无明显事实记忆。")
    sops = _to_bullet_block(grouped["sop"], "暂无明显SOP记忆。")
    return (
        "## 用户画像记忆\n"
        f"{profiles}\n\n"
        "## 事实记忆\n"
        f"{semantics}\n\n"
        "## SOP记忆\n"
        f"{sops}"
    )


class Mem0MemoryProvider:
    """naive_mem0-compatible retrieval provider."""

    strategy = "mem0"

    def __init__(self, *, run_id: str = "") -> None:
        self._run_id = run_id.strip()
        self._cache: dict[tuple[str, str], str] = {}

    def _memory_context(self, user_id: str, query: str) -> str:
        key = (user_id, query)
        if key not in self._cache:
            from src.experiments.naive_mem0.mem_client import mem0_search_sync

            memory_results = mem0_search_sync(
                [query],
                user_id=user_id,
                run_id=self._run_id,
            )
            self._cache[key] = _build_mem0_memory_block(memory_results)
        return self._cache[key]

    def for_task1(
        self,
        user_id: str,
        *,
        user_query: str,
        dialogue_context: str = "",
    ) -> str:
        return self._memory_context(user_id, user_query)

    def for_task23(
        self,
        user_id: str,
        *,
        query: str,
        dialogue_context: str = "",
    ) -> str:
        return self._memory_context(user_id, query)


__all__ = [
    "MemoryContextProvider",
    "PackMemoryProvider",
    "FullMemoryProvider",
    "NoMemoryProvider",
    "Mem0MemoryProvider",
    "make_provider",
]
