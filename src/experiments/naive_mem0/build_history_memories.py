#!/usr/bin/env python3
"""
从 history 对话数据做 memory_extract 并写入 mem0。

仅支持 ChronicCompanion-Set ``input.json`` 形态：每用户含 ``history`` 列表
（每条 history 含 ``dialogue`` / ``dialogue_timestamp`` 等）。

向量存储路径：work/data/mem0/<run_id 或 default>/<user_id>/（仓库在 work/ChronicCompanion-AI 时）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_NAIVE_MEM0 = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORK_ROOT = _REPO_ROOT.parent
for _p in (_NAIVE_MEM0,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_DEFAULT_INPUT = _WORK_ROOT / "data" / "ChronicCompanion-Set" / "input.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="预建 mem0：从 ChronicCompanion-Set input.json 的 history 抽取并入库")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=_DEFAULT_INPUT,
        help="ChronicCompanion-Set input.json 路径；默认 work/data/ChronicCompanion-Set/input.json",
    )
    parser.add_argument("--user_start_idx", type=int, default=0)
    parser.add_argument("--user_end_idx", type=int, default=4)
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="与 inference/evaluation 的 --mem0_run_id 共用；空则使用子目录 default/",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印每条 history 的 tag、h_text、extract 后的 payload（入库前）",
    )
    args = parser.parse_args()

    run_id = (args.run_id or "").strip()
    input_path = args.input_file.expanduser().resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"input_file 必须是按 user_id 分组的 JSON object: {input_path}")

    from dialogue_memory_extract import ingest_user_history_dialogue

    user_ids = list(data.keys())[args.user_start_idx : args.user_end_idx + 1]

    if args.debug:
        print(f"start. input={input_path} run_id={run_id!r}")
    for uid in user_ids:
        user_payload = data.get(uid)
        if not isinstance(user_payload, dict):
            if args.debug:
                print(f"[SKIP user={uid}] user payload is not an object", flush=True)
            continue
        ingest_user_history_dialogue(
            user_payload,
            user_id=uid,
            run_id=run_id,
            debug=args.debug,
        )

    print(
        f"完成：input={str(input_path)!r} "
        f"run_id={run_id!r}，用户数={len(user_ids)}，向量目录 work/data/mem0/…"
    )


if __name__ == "__main__":
    main()
