"""Memanggil LLM lewat OpenRouter dengan OpenAI SDK."""
from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from .config import LLM_BASE_URL, LLM_MODEL, OPENROUTER_API_KEY


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str  # model yang benar-benar menjawab (router gratis memilihnya sendiri)
    latency_s: float


def has_key() -> bool:
    return bool(OPENROUTER_API_KEY)


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2,
         max_tokens: int = 1200, retries: int = 3) -> LLMResult:
    if not OPENROUTER_API_KEY:
        raise LLMError("OPENROUTER_API_KEY belum diisi. Salin .env.example menjadi .env lalu isi key-nya.")
    client = OpenAI(base_url=LLM_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=90)
    last_error = "tidak ada jawaban"
    for attempt in range(retries):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model or LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (response.choices[0].message.content or "").strip() if response.choices else ""
            if text:
                return LLMResult(text, response.model or (model or LLM_MODEL), time.time() - start)
            last_error = "model mengembalikan jawaban kosong"
        except Exception as error:  # batas kuota, model sedang penuh, jaringan
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(2 * (attempt + 1))
    raise LLMError(f"LLM gagal menjawab setelah {retries} percobaan ({last_error})")
