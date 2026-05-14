#!/usr/bin/env python3
"""
naive_mem0 非实验实时对话入口。

特点：
- 每轮先用 mem0 检索长期记忆，再生成回复
- 仅做demo展示，不更新记忆库
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .llm_client import chat_completion
    from .mem_client import mem0_search_sync
    from .prompt_paths import chat_session_dir
except ImportError:
    from llm_client import chat_completion
    from mem_client import mem0_search_sync
    from prompt_paths import chat_session_dir

_NAIVE_MEM0_ROOT = Path(__file__).resolve().parent
_LOG_DIR = _NAIVE_MEM0_ROOT / "log"


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_memories_for_prompt(memory_results: list[dict[str, Any]]) -> dict[str, list[str]]:
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
            parsed_json = _safe_json_loads(memory_text)

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


def build_memory_block(memory_results: list[dict[str, Any]]) -> str:
    grouped = _extract_memories_for_prompt(memory_results)
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


def render_chat_session_prompt(
    *,
    session_context: str,
    memory_results: list[dict[str, Any]],
    current_query: str,
) -> list[dict[str, str]]:
    """逐条 chat 专用：仅注入本会话上文 + mem0 检索结果 + 当前用户句。"""
    prompt_root = chat_session_dir()
    system_prompt = (prompt_root / "system_prompt.txt").read_text(encoding="utf-8")
    user_prompt_tpl = (prompt_root / "user_prompt.txt").read_text(encoding="utf-8")

    memory_block = build_memory_block(memory_results)
    user_prompt = (
        user_prompt_tpl.replace("<session_context>", session_context.strip())
        .replace("<memory>", memory_block)
        .replace("<current_query>", current_query.strip())
    )
    user_prompt = re.sub(r"\$\$\{(.*?)\$\$\}", r"\1", user_prompt, flags=re.DOTALL)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



class NaiveMem0ChatSession:
    def __init__(
        self,
        *,
        user_id: str,
        run_id: str,
        debug: bool = False,
        log_path: Path | None = None,
    ):
        self.user_id = user_id
        self.run_id = run_id
        self.debug = debug
        self._log_path = log_path
        self._log_turn = 0
        # 本会话内已完成的 (用户句, 助手句)，不含当前正在处理的一句
        self._turns: list[tuple[str, str]] = []

    def _session_context_text(self) -> str:
        if not self._turns:
            return "（本 session 尚无更早对话轮次。）"
        blocks: list[str] = []
        for i, (user_line, assistant_line) in enumerate(self._turns, start=1):
            blocks.append(f"第{i}轮\n用户：{user_line}\n助手：{assistant_line}")
        return "\n\n".join(blocks)

    def _append_debug_log(
        self,
        *,
        user_query: str,
        memory_results: list[dict[str, Any]],
        messages: list[dict[str, str]],
        reply: str,
    ) -> None:
        if self._log_path is None:
            return
        self._log_turn += 1
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "turn": self._log_turn,
            "user_id": self.user_id,
            "run_id": self.run_id,
            "user_query": user_query,
            "memory_results": memory_results,
            "messages": messages,
            "reply": reply,
        }
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def chat_once(self, user_query: str) -> str:
        memory_results = mem0_search_sync(
            [user_query],
            user_id=self.user_id,
            run_id=self.run_id,
        )
        session_context = self._session_context_text()
        messages = render_chat_session_prompt(
            session_context=session_context,
            memory_results=memory_results,
            current_query=user_query,
        )
        assistant_text = chat_completion(messages, temperature=0.2)
        self._append_debug_log(
            user_query=user_query,
            memory_results=memory_results,
            messages=messages,
            reply=assistant_text,
        )
        self._turns.append((user_query, assistant_text))
        return assistant_text


def main() -> None:
    parser = argparse.ArgumentParser(description="naive_mem0 实时对话 CLI")
    parser.add_argument("--user-id", type=str, default="0000")
    parser.add_argument(
        "--run-id",
        type=str,
        default="chat",
        help="向量库隔离目录：work/data/mem0/<run_id>/<user_id>/（仓库需在 work/ChronicCompanion-AI）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="将每轮的 user_query / memory_results / messages(完整 prompt) / reply 写入 naive_mem0/log/",
    )
    args = parser.parse_args()

    log_path: Path | None = None
    if args.debug:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = _LOG_DIR / f"run_chat_{stamp}.log"
        print(f"[debug] 日志文件: {log_path}")

    session = NaiveMem0ChatSession(
        user_id=args.user_id,
        run_id=args.run_id,
        debug=args.debug,
        log_path=log_path,
    )
    print("naive_mem0 chat ready. 输入 exit 或 quit 退出。")

    while True:
        user_query = input("User> ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_query:
            continue

        reply = session.chat_once(user_query)
        print(f"Assistant> {reply}")


if __name__ == "__main__":
    main()
