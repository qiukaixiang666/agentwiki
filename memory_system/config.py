from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_ROOT = PROJECT_ROOT


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _root_from_env() -> Path:
    return Path(os.environ.get("MEMORY_ROOT", DEFAULT_MEMORY_ROOT)).resolve()


_load_env_file(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    memory_root: Path = _root_from_env()
    data_dir: Path = memory_root / "data"
    runtime_dir: Path = memory_root / "runtime"
    bge_model_dir: Path = Path(
        os.environ.get("BGE_MODEL_DIR", str(memory_root / "bge-m3"))
    ).resolve()

    embedding_service_url: str = os.environ.get(
        "EMBEDDING_SERVICE_URL", "http://127.0.0.1:18081"
    ).rstrip("/")
    embedding_provider: str = os.environ.get("EMBEDDING_PROVIDER", "legacy").lower()
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", str(memory_root / "bge-m3")
    )
    embedding_api_key: str = os.environ.get("EMBEDDING_API_KEY", "EMPTY")
    embedding_timeout_seconds: float = float(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "60"))
    embedding_api_host: str = os.environ.get("EMBEDDING_API_HOST", "127.0.0.1")
    embedding_api_port: int = int(os.environ.get("EMBEDDING_API_PORT", "18081"))
    bge_device: str = os.environ.get("BGE_DEVICE", "cuda:1")
    bge_batch_size: int = int(os.environ.get("BGE_BATCH_SIZE", "16"))

    memory_api_host: str = os.environ.get("MEMORY_API_HOST", "127.0.0.1")
    memory_api_port: int = int(os.environ.get("MEMORY_API_PORT", "18080"))

    llm_provider: str = os.environ.get("LLM_PROVIDER", "deepseek").lower()
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    llm_api_key: str = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    llm_model: str = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
    llm_temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
    llm_top_p: float | None = (
        float(os.environ["LLM_TOP_P"]) if os.environ.get("LLM_TOP_P") else None
    )
    llm_timeout_seconds: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
    llm_extra_body_json: str = os.environ.get("LLM_EXTRA_BODY_JSON", "")

    memory_top_k: int = int(os.environ.get("MEMORY_TOP_K", "5"))
    recent_turns: int = int(os.environ.get("RECENT_TURNS", "8"))
    privacy_reject_sensitive: bool = _parse_bool(
        os.environ.get("PRIVACY_REJECT_SENSITIVE"), True
    )


settings = Settings()


def ensure_directories() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "sessions").mkdir(parents=True, exist_ok=True)
