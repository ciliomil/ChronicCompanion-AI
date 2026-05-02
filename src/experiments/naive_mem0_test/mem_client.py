import asyncio
import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# mem0 默认 MEM0_TELEMETRY=True 时，每个 Memory 会在 ~/.mem0/migrations_qdrant 再开一个嵌入式 Qdrant。
# 同一进程内多 (run_id, user_id) 会并存多个 Memory，争用该固定路径会触发
# "already accessed by another instance"。未显式设置时关闭遥测；需要统计可 export MEM0_TELEMETRY=true。
if "MEM0_TELEMETRY" not in os.environ:
    os.environ["MEM0_TELEMETRY"] = "false"

from mem0 import Memory

_NAIVE_MEM0_ROOT = Path(__file__).resolve().parent
# mem_client.py 位于 .../src/experiments/naive_mem0/；parents[3] 为 ChronicCompanion-AI 仓库根
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VECTOR_ROOT = _NAIVE_MEM0_ROOT / "vector_stores"

# 按 (run_id, user_id) 复用同步 Memory，避免重复加载 embedder
_memory_by_scope: Dict[Tuple[str, str], Memory] = {}


def _conf_path() -> Path:
    """
    配置查找顺序（与项目统一使用仓库根目录 conf.yaml 一致）：
    1) naive_mem0/conf.yaml（可选，仅用于局部覆盖）
    2) <ChronicCompanion-AI>/conf.yaml（主配置，含 MEM_MODEL / BASIC_MODEL）
    """
    local = _NAIVE_MEM0_ROOT / "conf.yaml"
    if local.is_file():
        return local
    root_conf = _REPO_ROOT / "conf.yaml"
    if root_conf.is_file():
        return root_conf
    raise FileNotFoundError(
        f"未找到 mem0 配置：请在仓库根目录放置 conf.yaml：{root_conf} "
        f"或在 naive_mem0 目录放置 conf.yaml：{local}"
    )


def _load_yaml() -> Dict[str, Any]:
    with _conf_path().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_mem0_config() -> Dict[str, Any]:
    """
    读取 conf.yaml 中的 MEM_MODEL（embedder / vector_store / llm 等）。
    配置路径见 `_conf_path()`：默认同仓库根目录 `conf.yaml`。

    """
    conf = _load_yaml()
    mem = copy.deepcopy(conf.get("MEM_MODEL", {}) or {})
    llm_cfg = mem.get("llm", {}).get("config")
    if isinstance(llm_cfg, dict) and not str(llm_cfg.get("api_key", "") or "").strip():
        llm_cfg["api_key"] = (
            "local"
        )
    return mem


def _sanitize_segment(s: str) -> str:
    s = (s or "").strip() or "default"
    return re.sub(r"[^\w.\-]+", "_", s)[:200]


def vector_store_dir_for_run_user(run_id: str, user_id: str) -> Path:
    """每个 run_id × user_id 独立目录，避免共用 qdrant_memory 路径冲突。"""
    run_slug = _sanitize_segment(run_id)
    user_slug = _sanitize_segment(user_id)
    return _VECTOR_ROOT / run_slug / user_slug


def get_mem0_config_for_run_user(run_id: str, user_id: str) -> Dict[str, Any]:
    """
    在 MEM_MODEL 上覆盖 vector_store.path 为 vector_stores/<run>/<user>/ 。
    run_id 为空时使用子目录 default。
    """
    raw = get_mem0_config()
    cfg = copy.deepcopy(raw)
    vs = cfg.setdefault("vector_store", {}).setdefault("config", {})
    vs["path"] = str(vector_store_dir_for_run_user(run_id, user_id))
    return cfg


def _scope_key(run_id: str, user_id: str) -> Tuple[str, str]:
    return (_sanitize_segment(run_id), _sanitize_segment(user_id))


def _prepare_mem0_config_dict(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    cfg = copy.deepcopy(raw)
    llm_cfg = cfg.get("llm", {}).get("config")
    extra: Optional[Dict[str, Any]] = None
    if isinstance(llm_cfg, dict) and "extra_body" in llm_cfg:
        extra = copy.deepcopy(llm_cfg["extra_body"])
        del llm_cfg["extra_body"]
    return cfg, extra


def _patch_llm_extra_body(obj: Any, extra_body: Optional[Dict[str, Any]]) -> Any:
    if not extra_body:
        return obj
    llm = getattr(obj, "llm", None)
    if llm is None:
        return obj
    _orig = llm.generate_response

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("extra_body", extra_body)
        return _orig(*args, **kwargs)

    llm.generate_response = _wrapped  # type: ignore[method-assign]
    return obj


def _get_memory_sync(run_id: str, user_id: str) -> Any:
    key = _scope_key(run_id, user_id)
    if key not in _memory_by_scope:
        raw = get_mem0_config_for_run_user(run_id, user_id)
        config_dict, extra_body = _prepare_mem0_config_dict(raw)
        m = Memory.from_config(config_dict=config_dict)
        _patch_llm_extra_body(m, extra_body)
        _memory_by_scope[key] = m
    return _memory_by_scope[key]


async def mem0_search(
    queries: List[str],
    user_id: str = "dev",
    *,
    run_id: str = "",
) -> List[Dict[str, Any]]:
    """
    调用 mem0 做批量检索。向量库由 (run_id, user_id) 决定；mem0 内 user_id 仍用传入的 user_id。

    必须与 mem0_add 共用同一套同步 Memory（_get_memory_sync），否则会再开一个
    AsyncMemory/嵌入式 Qdrant，与已打开的 vector_store.path 冲突：
    "Storage folder ... is already accessed by another instance"。
    """
    m = _get_memory_sync(run_id, user_id)
    paired: List[Dict[str, Any]] = []
    for q in queries:
        try:
            r = m.search(q, user_id=user_id)
        except Exception:
            continue
        paired.append({"query": q, "result": r})
    return paired


def mem0_search_sync(
    queries: List[str],
    user_id: str = "dev",
    *,
    run_id: str = "",
) -> List[Dict[str, Any]]:
    """
    同步封装。默认 run_id 为空串，对应 vector_stores/default/<user>/ 。
    """
    return asyncio.run(mem0_search(queries, user_id=user_id, run_id=run_id))


def mem0_add(
    memory_payload: Union[str, Dict[str, Any]],
    user_id: str,
    *,
    run_id: str = "",
) -> List[Any]:
    """
    将 memory_extract 产出的结构化 JSON 写入 mem0。按 (run_id, user_id) 使用独立向量目录。
    """
    m = _get_memory_sync(run_id, user_id)

    if isinstance(memory_payload, str):
        import json

        data = json.loads(memory_payload)
    else:
        data = memory_payload

    results: List[Any] = []

    for item in data.get("user_profiles", []):
        try:
            results.append(
                m.add(messages=item, user_id=user_id, metadata={"catagory": "user_profile"}, infer=True)
            )
        except Exception as e:
            results.append(e)

    for item in data.get("semantic_memory", []):
        try:
            results.append(
                m.add(messages=item, user_id=user_id, metadata={"catagory": "semantic_memory"}, infer=True)
            )
        except Exception as e:
            results.append(e)

    import json

    for sop in data.get("SOP", []):
        try:
            sop_text = json.dumps(sop, ensure_ascii=False)
            results.append(
                m.add(messages=sop_text, user_id=user_id, metadata={"catagory": "SOP"}, infer=True)
            )
        except Exception as e:
            results.append(e)

    return results
