"""
memory_extract 工具与入库逻辑。

- `ingest_user_memories_from_dataset`：仅对 **history** dialogue 抽取并写入 mem0（供 build_dialogue_memories / --build-memory）。
- `add_memories_from_query_dialogue_item`：对单条 **query** dialogue 抽取并写入，由各 inference 在遍历 query 时调用。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .llm_client import chat_completion
    from .mem_client import mem0_add
    from .prompt_paths import memory_extract_md
except ImportError:
    from llm_client import chat_completion
    from mem_client import mem0_add
    from prompt_paths import memory_extract_md


def stringify_dialogue(dialogue: Dict[str, Any]) -> str:
    lines: List[str] = []
    turn_keys = sorted(dialogue.keys(), key=lambda x: int(str(x).split("_")[-1]))
    for turn in turn_keys:
        turn_obj = dialogue.get(turn, {})
        user_text = turn_obj.get("user", {}).get("content", "")
        assistant_text = turn_obj.get("assistant", {}).get("content", "")
        if user_text:
            lines.append(f"user: {user_text}")
        if assistant_text:
            lines.append(f"assistant: {assistant_text}")
    return "\n".join(lines)


def extract_memory_from_dialogue(dialogue_text: str, timestamp: str, memory_extract_prompt_path: Optional[Path] = None) -> Dict[str, Any]:
    path = memory_extract_prompt_path or memory_extract_md()
    prompt_tpl = path.read_text(encoding="utf-8")
    task_memory = {"dialogue_text": dialogue_text}
    prompt = (
        prompt_tpl.replace("{{text}}", json.dumps(task_memory, ensure_ascii=False)).replace(
            "{{timestamp}}", json.dumps(timestamp, ensure_ascii=False)
        )
    )
    messages = [{"role": "system", "content": prompt}]
    raw = chat_completion(messages, temperature=0.1)
    content = raw.strip() if isinstance(raw, str) else str(raw).strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def ingest_user_history_dialogue(
    user_dict: Dict[str, Any],
    *,
    user_id: str,
    run_id: str,
    debug: bool = False,
) -> None:
    """
    仅对该 user 的 **history** dialogue 依次 extract + mem0 add。
    向量库路径由 mem_client 按 (run_id, user_id) 隔离。
    """
    mem_path = memory_extract_md()
    history = user_dict.get("history", [])

    for h_item in history:
        timestamp = h_item.get("dialogue_timestamp", "")
        h_text = stringify_dialogue(h_item.get("dialogue", {}))
        if not h_text.strip():
            continue
        tag = f"MEM_ADD user={user_id} sample={h_item.get('sample_id', '')}"
        try:
            payload = extract_memory_from_dialogue(h_text, timestamp, mem_path)
            if debug:
                print(
                    f"[{tag}]\nh_text:\n{h_text}\npayload:\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n---",
                    flush=True,
                )
            mem0_add(payload, user_id=user_id, run_id=run_id)
        except Exception as e:
            if debug:
                print(f"[{tag}] ERROR before/after extract\nh_text:\n{h_text}\n{e!r}\n---", flush=True)
