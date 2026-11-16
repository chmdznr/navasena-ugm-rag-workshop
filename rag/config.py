"""Konfigurasi workshop. Semua nilai bisa diganti lewat environment variable atau file .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# PostgreSQL + pgvector. Di dalam docker compose nilainya diisi otomatis (host "db").
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")

# LLM lewat OpenRouter (API-nya kompatibel dengan OpenAI SDK).
# "openrouter/free" memilih salah satu model gratis untuk setiap panggilan.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/free")

# Model embedding berjalan lokal (CPU), multibahasa, 384 dimensi.
# Batasnya 128 token per teks: bagian chunk yang lebih panjang dari itu tidak ikut terbaca.
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBED_DIM = 384

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
