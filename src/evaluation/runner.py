"""Eval runner: orchestrates task execution, persistence, resume, aggregation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.bleu import CorpusBleuAccumulator
from src.evaluation.dataset import TopicSample, iter_topic_samples, load_input
from src.evaluation.memory_context import MemoryContextProvider, make_provider
from src.evaluation.tasks import (
    RequirementRestatementResult,
    RequirementRestatementTask,
    SolutionGenerationResult,
    SolutionGenerationTask,
    SolutionSelectionResult,
    SolutionSelectionTask,
    TASK_NAMES,
    TaskName,
)
from src.llm.llm import LLMClient

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RunConfig + paths
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    input_path: Path
    memory_path: Path
    output_dir: Path
    api_file: Path
    tasks: tuple[TaskName, ...]
    user_ids: list[str] | None
    memory_strategy: str
    model: str | None
    temperature: float
    max_samples: int | None
    resume: bool
    mem0_run_id: str = ""

    def predictions_path(self, task: TaskName, user_id: str) -> Path:
        return self.output_dir / "predictions" / task / f"{user_id}.jsonl"

    def metrics_path(self, task: TaskName) -> Path:
        return self.output_dir / "metrics" / f"{task}.json"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def config_path(self) -> Path:
        return self.output_dir / "config.json"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    keys: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("user_id") or "")
            sid = str(row.get("sample_id") or "")
            tid = str(row.get("topic_id") or "")
            if uid and sid and tid:
                keys.add((uid, sid, tid))
    return keys


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_all_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class _BleuAgg:
    acc: CorpusBleuAccumulator = field(default_factory=CorpusBleuAccumulator)
    n_total: int = 0
    n_errors: int = 0


@dataclass
class _SelectionAgg:
    n_total: int = 0
    n_errors: int = 0
    n_invalid_response: int = 0
    sum_score: float = 0.0
    sum_hits: int = 0
    n_exact: int = 0
    feedback_dist: dict[str, int] = field(
        default_factory=lambda: {"pos": 0, "neu": 0, "neg": 0, "other": 0}
    )

    def update(self, row: dict[str, Any]) -> None:
        self.n_total += 1
        if row.get("error"):
            self.n_errors += 1
            self.n_invalid_response += 1
            return
        if not row.get("selected_indices"):
            self.n_invalid_response += 1
            return
        self.sum_score += float(row.get("selection_score") or 0.0)
        self.sum_hits += int(row.get("hit_pos_count") or 0)
        if row.get("exact_match"):
            self.n_exact += 1
        for fb in row.get("selected_feedback") or []:
            key = fb if fb in ("pos", "neu", "neg") else "other"
            self.feedback_dist[key] = self.feedback_dist.get(key, 0) + 1

    def result(self) -> dict[str, Any]:
        valid = max(self.n_total - self.n_invalid_response, 0)
        return {
            "n_total": self.n_total,
            "n_errors": self.n_errors,
            "n_invalid_response": self.n_invalid_response,
            "n_valid": valid,
            "mean_selection_score": (self.sum_score / valid) if valid else 0.0,
            "mean_hit_pos_count": (self.sum_hits / valid) if valid else 0.0,
            "exact_match_rate": (self.n_exact / valid) if valid else 0.0,
            "selected_feedback_distribution": self.feedback_dist,
        }


def _aggregate_bleu_task(task: TaskName, cfg: RunConfig) -> dict[str, Any]:
    acc = CorpusBleuAccumulator()
    n_total = 0
    n_errors = 0
    for uid in _enumerate_users_with_predictions(cfg, task):
        for row in _read_all_predictions(cfg.predictions_path(task, uid)):
            n_total += 1
            if row.get("error"):
                n_errors += 1
                continue
            cand = str(row.get("prediction") or "")
            if task == "requirement_restatement":
                refs = [str(row.get("reference") or "")]
                refs = [r for r in refs if r]
            else:
                refs = [str(r) for r in (row.get("references") or []) if r]
            acc.add(cand, refs)
    out: dict[str, Any] = {"n_total": n_total, "n_errors": n_errors}
    out.update(acc.result())
    return out


def _aggregate_selection(cfg: RunConfig) -> dict[str, Any]:
    agg = _SelectionAgg()
    for uid in _enumerate_users_with_predictions(cfg, "solution_selection"):
        for row in _read_all_predictions(cfg.predictions_path("solution_selection", uid)):
            agg.update(row)
    return agg.result()


def _enumerate_users_with_predictions(cfg: RunConfig, task: TaskName) -> list[str]:
    base = cfg.output_dir / "predictions" / task
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.jsonl"))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class _Counters:
    n_topics: int = 0
    n_topics_skipped: int = 0
    n_calls: dict[str, int] = field(default_factory=lambda: {t: 0 for t in TASK_NAMES})
    n_errors: dict[str, int] = field(default_factory=lambda: {t: 0 for t in TASK_NAMES})


def run_evaluation(
    cfg: RunConfig,
    *,
    llm: LLMClient,
    provider: MemoryContextProvider | None = None,
) -> dict[str, Any]:
    """Run the configured eval. Returns the summary dict."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(cfg)

    if provider is None:
        provider = make_provider(
            cfg.memory_strategy,
            memory_root=cfg.memory_path,
            llm=llm,
            mem0_run_id=cfg.mem0_run_id,
        )

    task1 = RequirementRestatementTask(llm, provider, temperature=cfg.temperature)
    task2 = SolutionGenerationTask(llm, provider, temperature=cfg.temperature)
    task3 = SolutionSelectionTask(llm, provider)

    data = load_input(cfg.input_path)
    counters = _Counters()
    started = time.time()

    existing_keys: dict[TaskName, set[tuple[str, str, str]]] = {}
    if cfg.resume:
        for t in cfg.tasks:
            existing_keys[t] = set()
    samples_iter = iter_topic_samples(
        data,
        user_ids=cfg.user_ids,
        max_samples=cfg.max_samples,
    )

    for sample in samples_iter:
        counters.n_topics += 1
        any_done = False

        if "requirement_restatement" in cfg.tasks:
            if _should_skip(cfg, "requirement_restatement", sample, existing_keys):
                counters.n_topics_skipped += 1
            else:
                _run_one(cfg, task1, sample, counters)
                any_done = True

        if "solution_generation" in cfg.tasks:
            if _should_skip(cfg, "solution_generation", sample, existing_keys):
                pass
            else:
                _run_one(cfg, task2, sample, counters)
                any_done = True

        if "solution_selection" in cfg.tasks:
            if _should_skip(cfg, "solution_selection", sample, existing_keys):
                pass
            else:
                _run_one(cfg, task3, sample, counters)
                any_done = True

        if any_done:
            _logger.info(
                "topic done: %s / %s / %s",
                sample.user_id,
                sample.sample_id,
                sample.topic_id,
            )

    metrics = _finalize_metrics(cfg)
    summary = _build_summary(cfg, counters, metrics, started)
    _write_summary(cfg, summary)
    return summary


def _should_skip(
    cfg: RunConfig,
    task: TaskName,
    sample: TopicSample,
    existing_keys: dict[TaskName, set[tuple[str, str, str]]],
) -> bool:
    if not cfg.resume:
        return False
    if task not in existing_keys:
        existing_keys[task] = _read_existing_keys(cfg.predictions_path(task, sample.user_id))
        # cache per-user; reset when user changes
    keys = existing_keys.setdefault(task, set())
    # Lazy refresh per-user — easier to recompute on user change to avoid mixing.
    # We rebuild from disk whenever a different user's key is missing.
    if (sample.user_id, sample.sample_id, sample.topic_id) in keys:
        return True
    # Refresh on first encounter of a new user_id
    fresh = _read_existing_keys(cfg.predictions_path(task, sample.user_id))
    if fresh != keys:
        keys.update(fresh)
        existing_keys[task] = keys
    return (sample.user_id, sample.sample_id, sample.topic_id) in keys


def _run_one(
    cfg: RunConfig,
    task: Any,
    sample: TopicSample,
    counters: _Counters,
) -> None:
    result = task.run(sample)
    counters.n_calls[task.name] += 1
    if result.error:
        counters.n_errors[task.name] += 1
    _append_jsonl(cfg.predictions_path(task.name, sample.user_id), result.to_dict())


# ---------------------------------------------------------------------------
# Snapshots / summaries
# ---------------------------------------------------------------------------


def _write_config_snapshot(cfg: RunConfig) -> None:
    snap = {
        "input_path": str(cfg.input_path),
        "memory_path": str(cfg.memory_path),
        "output_dir": str(cfg.output_dir),
        "api_file": str(cfg.api_file),
        "tasks": list(cfg.tasks),
        "user_ids": cfg.user_ids,
        "memory_strategy": cfg.memory_strategy,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_samples": cfg.max_samples,
        "resume": cfg.resume,
        "mem0_run_id": cfg.mem0_run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    cfg.config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.config_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")


def _finalize_metrics(cfg: RunConfig) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    if "requirement_restatement" in cfg.tasks:
        metrics["requirement_restatement"] = _aggregate_bleu_task(
            "requirement_restatement", cfg
        )
    if "solution_generation" in cfg.tasks:
        metrics["solution_generation"] = _aggregate_bleu_task("solution_generation", cfg)
    if "solution_selection" in cfg.tasks:
        metrics["solution_selection"] = _aggregate_selection(cfg)

    for task, payload in metrics.items():
        path = cfg.metrics_path(task)  # type: ignore[arg-type]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _build_summary(
    cfg: RunConfig,
    counters: _Counters,
    metrics: dict[str, dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    return {
        "memory_strategy": cfg.memory_strategy,
        "tasks": list(cfg.tasks),
        "n_topics": counters.n_topics,
        "n_topics_skipped": counters.n_topics_skipped,
        "n_calls_per_task": dict(counters.n_calls),
        "n_errors_per_task": dict(counters.n_errors),
        "elapsed_sec": round(time.time() - started, 2),
        "metrics": metrics,
    }


def _write_summary(cfg: RunConfig, summary: dict[str, Any]) -> None:
    cfg.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = ["RunConfig", "run_evaluation"]
