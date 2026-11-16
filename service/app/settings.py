"""Konfigurasi layanan. Semua nilai datang dari environment dan divalidasi saat start."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEMO_KEYS = {"kunci-publik-demo", "kunci-internal-demo"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    openrouter_api_key: str = Field(min_length=1)  # wajib: layanan menolak start tanpa key
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/free"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    request_timeout_s: float = 90.0  # batas waktu total satu permintaan

    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = 384
    table_name: str = "rag_nodes"
    chunk_tokens: int = 180
    chunk_overlap_tokens: int = 25
    top_k: int = Field(default=5, ge=1, le=20)

    # Tingkat pengaman ditentukan server, bukan oleh klien.
    guard_level: Literal["off", "rules", "guardrails"] = "rules"
    anomaly_threshold_default: float = 0.35

    # "kunci:peran" dipisah koma. Peran pengguna diturunkan dari kunci, bukan dari isi permintaan.
    # Wajib diisi lewat environment; tidak ada nilai bawaan di kode.
    service_api_keys: str = Field(min_length=1)

    log_level: str = "INFO"

    @property
    def sync_url(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    @property
    def async_url(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    @property
    def uses_demo_keys(self) -> bool:
        """True bila kunci contoh dari .env.example masih dipakai."""
        return any(key in DEMO_KEYS for key in self.api_keys)

    @property
    def api_keys(self) -> dict[str, str]:
        pairs = (item.split(":", 1) for item in self.service_api_keys.split(",") if ":" in item)
        return {key.strip(): role.strip() for key, role in pairs}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
