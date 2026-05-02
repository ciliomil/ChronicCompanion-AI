"""Environment-backed settings for the prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _to_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str
    app_host: str
    app_port: int
    log_level: str
    data_dir: Path
    output_dir: Path
    dataset_path: Path
    storage_type: str
    sqlite_path: Path
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    enable_safety: bool
    enable_need_inference: bool


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / os.getenv("DATA_DIR", "data")
    output_dir = project_root / os.getenv("OUTPUT_DIR", "outputs")
    dataset_path = project_root / os.getenv("DATASET_PATH", "data/sample_sessions.jsonl")
    sqlite_path = project_root / os.getenv("SQLITE_PATH", "data/chronic_companion.db")

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_env=os.getenv("APP_ENV", "dev"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        data_dir=data_dir,
        output_dir=output_dir,
        dataset_path=dataset_path,
        storage_type=os.getenv("STORAGE_TYPE", "json"),
        sqlite_path=sqlite_path,
        llm_provider=os.getenv("LLM_PROVIDER", "mock"),
        llm_model=os.getenv("LLM_MODEL", "mock-chat"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        enable_safety=_to_bool(os.getenv("ENABLE_SAFETY"), default=True),
        enable_need_inference=_to_bool(os.getenv("ENABLE_NEED_INFERENCE"), default=True),
    )
