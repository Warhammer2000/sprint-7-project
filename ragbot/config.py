"""Централизованная конфигурация бота.

Все параметры читаются из переменных окружения (.env). Значения по умолчанию
подобраны так, чтобы бот запускался полностью офлайн, без ключей и без скачивания
тяжёлых моделей (EMBEDDINGS_BACKEND=st можно переключить на tfidf).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv может быть не установлен — не критично
    pass


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Корень проекта (на уровень выше пакета ragbot/)
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # LLM
    llm_backend: str = os.getenv("LLM_BACKEND", "offline").strip().lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "").strip()
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1").strip()

    # Embeddings
    embeddings_backend: str = os.getenv("EMBEDDINGS_BACKEND", "st").strip().lower()
    st_model: str = os.getenv(
        "ST_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ).strip()

    # Retrieval
    top_k: int = _int("TOP_K", 5)
    recall_k: int = _int("RECALL_K", 20)
    min_score: float = _float("MIN_SCORE", 0.25)
    chunk_size: int = _int("CHUNK_SIZE", 600)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 120)

    # Prompting
    use_few_shot: bool = _bool("USE_FEW_SHOT", True)
    use_cot: bool = _bool("USE_COT", True)

    # Security
    security_enabled: bool = _bool("SECURITY_ENABLED", True)

    # Paths
    kb_dir: Path = ROOT / os.getenv("KB_DIR", "knowledge_base")
    index_dir: Path = ROOT / os.getenv("INDEX_DIR", "index")
    log_dir: Path = ROOT / os.getenv("LOG_DIR", "logs")

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = _int("API_PORT", 8000)

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
