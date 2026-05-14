"""CLI for the eval runner."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.llm_setup import EvalLLMClient, install_eval_conf
from src.evaluation.memory_context import make_provider
from src.evaluation.runner import RunConfig, run_evaluation
from src.evaluation.tasks import TASK_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORK_ROOT = _REPO_ROOT.parent
_DEFAULT_INPUT = _WORK_ROOT / "data" / "ChronicCompanion-Set" / "input.json"
_DEFAULT_MEMORY = _WORK_ROOT / "data" / "memory"
_DEFAULT_API_FILE = _REPO_ROOT / "conf.yaml"
_DEFAULT_OUTPUT = _REPO_ROOT / "outputs" / "eval"

_TASK_CHOICES = ("all",) + TASK_NAMES
_STRATEGY_CHOICES = ("pack", "full", "no_memory", "mem0")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description="ChronicCompanion 三任务评测：requirement_restatement / "
        "solution_generation / solution_selection。",
    )
    p.add_argument("--input_path", type=Path, default=_DEFAULT_INPUT)
    p.add_argument("--memory_path", type=Path, default=_DEFAULT_MEMORY)
    p.add_argument("--api_file", type=Path, default=_DEFAULT_API_FILE)
    p.add_argument(
        "--output_dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "输出根目录。实际写入路径会自动按 strategy 分区为 "
            "<output_dir>/<memory_strategy>/（若 output_dir 已以 strategy 结尾则不再追加）。"
        ),
    )

    p.add_argument(
        "--task",
        choices=_TASK_CHOICES,
        default="all",
        help="all（默认）一次跑三项，或指定单项任务。",
    )
    p.add_argument(
        "--user",
        nargs="+",
        default=None,
        metavar="USER_ID",
        help="只跑给定 user_id（如 0000 0001）；缺省时全部跑。",
    )
    p.add_argument(
        "--memory_strategy",
        choices=_STRATEGY_CHOICES,
        default="pack",
        help=(
            "pack（默认，走 frame+plan 检索 pipeline）；full（不检索，直接全量给 LLM）；"
            "no_memory（零样本基线）；mem0（naive_mem0 检索基线）。"
        ),
    )
    p.add_argument(
        "--mem0_run_id",
        default="",
        help=(
            "memory_strategy=mem0 时使用的 run_id；空字符串表示 default，"
            "读取 work/data/mem0/<run_id>/<user_id>/。"
        ),
    )

    p.add_argument("--model", default=None, help="覆盖 BASIC_MODEL.model")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=None, help="每用户最多取多少 sample（topic 不截断）")
    p.add_argument("--resume", action="store_true", help="按 (sample_id, topic_id) 跳过已写入预测的 topic")

    p.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def _resolve_tasks(arg: str) -> tuple[str, ...]:
    if arg == "all":
        return TASK_NAMES
    return (arg,)


def _resolve_output_dir(base_output_dir: Path, memory_strategy: str) -> Path:
    """Partition outputs by memory strategy to avoid metric contamination."""
    base = base_output_dir.resolve()
    if base.name == memory_strategy:
        return base
    return base / memory_strategy


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    install_eval_conf(args.api_file, model_override=args.model)
    llm = EvalLLMClient(temperature=args.temperature)
    provider = make_provider(
        args.memory_strategy,
        memory_root=args.memory_path,
        llm=llm,
        mem0_run_id=args.mem0_run_id,
    )
    strategy_output_dir = _resolve_output_dir(args.output_dir, args.memory_strategy)
    logging.getLogger(__name__).info(
        "Using strategy-partitioned output_dir: %s",
        strategy_output_dir,
    )

    cfg = RunConfig(
        input_path=args.input_path.resolve(),
        memory_path=args.memory_path.resolve(),
        output_dir=strategy_output_dir,
        api_file=args.api_file.resolve(),
        tasks=_resolve_tasks(args.task),  # type: ignore[arg-type]
        user_ids=list(args.user) if args.user else None,
        memory_strategy=args.memory_strategy,
        model=args.model,
        temperature=args.temperature,
        max_samples=args.max_samples,
        resume=bool(args.resume),
        mem0_run_id=args.mem0_run_id,
    )

    summary = run_evaluation(cfg, llm=llm, provider=provider)
    _print_summary(summary)
    return 0


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
