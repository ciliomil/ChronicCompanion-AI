#!/usr/bin/env python3
"""
从 history 对话数据做 memory_extract 并写入 mem0。

支持两种数据格式（见 ``--format``）：

- **v1**：Mem-PAL ``input.json`` 形态，每用户含 ``history`` 列表（含 ``dialogue`` / ``dialogue_timestamp`` 等）。
- **v2**：ChronicCompanion ``history_dialogue.json`` 形态，每用户为 ``{ 日期: { turn_*: ... } }``。

向量存储路径：naive_mem0/vector_stores/<run_id 或 default>/<user_id>/ 。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_NAIVE_MEM0 = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LONGMEM_ROOT = _NAIVE_MEM0.parent
for _p in (_NAIVE_MEM0, _LONGMEM_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_DEFAULT_INPUT_V1 = str(_LONGMEM_ROOT.parent / "Mem-PAL" / "data_synthesis_v2" / "data" / "input.json")
_DEFAULT_INPUT_V2 = str(
    _REPO_ROOT / "data" / "ChronicCompanion-set" / "dialogue" / "history" / "history_dialogue.json"
)

from dialogue_memory_extract import ingest_user_history_dialogue, ingest_user_history_dialogue_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="预建 mem0：仅 history dialogue 抽取并入库")
    parser.add_argument(
        "--format",
        type=str,
        choices=("v1", "v2"),
        default="v2",
        help="v1=Mem-PAL input.json（history 列表）；v2=ChronicCompanion history_dialogue.json（按日期分块）",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=_DEFAULT_INPUT_V2,
        help="JSON 数据路径；省略时 v1 用 Mem-PAL input.json，v2 用仓库内 history_dialogue.json",
    )
    parser.add_argument("--user_start_idx", type=int, default=0)
    parser.add_argument("--user_end_idx", type=int, default=4)
    parser.add_argument(
        "--run_id",
        type=str,
        default="chronic_test",
        help="与 inference 共用；空则使用子目录 default/",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印每条 history 的 tag、h_text、extract 后的 payload（入库前）",
    )
    args = parser.parse_args()

    input_path = args.input_file
    if not input_path:
        input_path = _DEFAULT_INPUT_V1 if args.format == "v1" else _DEFAULT_INPUT_V2

    run_id = (args.run_id or "").strip()
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    user_ids = list(data.keys())[args.user_start_idx : args.user_end_idx + 1]
    
    if args.debug:
        print(f"start.")
    for uid in user_ids:
        if args.format == "v1":
            ingest_user_history_dialogue(
                data[uid], user_id=uid, run_id=run_id, debug=args.debug
            )
        else:
            ingest_user_history_dialogue_v2(
                data[uid], user_id=uid, run_id=run_id, debug=args.debug
            )

    print(
        f"完成：format={args.format!r} input={input_path!r} "
        f"run_id={run_id!r}，用户数={len(user_ids)}，向量目录 naive_mem0/vector_stores/…"
    )


if __name__ == "__main__":
    main()
