"""Pemeriksaan masukan dan keluaran dengan NVIDIA NeMo Guardrails.

Berbeda dari rag/safety.py yang memakai pola tetap, di sini sebuah LLM menilai pesan
terhadap kebijakan yang ditulis di guardrails/prompts.yml. Lebih luwes terhadap variasi
kalimat, tetapi setiap pemeriksaan menambah satu panggilan LLM.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from .config import LLM_BASE_URL, LLM_MODEL, ROOT

CONFIG_DIR = ROOT / "guardrails"
GUARD_MODEL = os.getenv("GUARD_MODEL") or LLM_MODEL

# NeMo Guardrails berjalan asinkron dan menolak dipanggil secara sinkron dari dalam event loop
# yang sedang aktif (misalnya Jupyter). Satu thread khusus membuat pemanggilannya aman dari mana saja.
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemo")


@lru_cache(maxsize=1)
def _rails():
    from nemoguardrails import LLMRails, RailsConfig

    config_yaml = (
        (CONFIG_DIR / "config.yml").read_text()
        .replace("${GUARD_MODEL}", GUARD_MODEL)
        .replace("${LLM_BASE_URL}", LLM_BASE_URL)
    )
    prompts_yaml = (CONFIG_DIR / "prompts.yml").read_text()
    return LLMRails(RailsConfig.from_content(yaml_content=f"{config_yaml}\n{prompts_yaml}"))


def _blocked(messages: list[dict], rail_type: str) -> bool:
    from nemoguardrails.rails.llm.options import RailStatus, RailType

    def run():
        return _rails().check(messages, rail_types=[RailType(rail_type)])

    return _WORKER.submit(run).result(timeout=180).status == RailStatus.BLOCKED


def input_blocked(question: str) -> bool:
    """True bila rail masukan menilai pertanyaan melanggar kebijakan."""
    return _blocked([{"role": "user", "content": question}], "input")


def output_blocked(question: str, answer: str) -> bool:
    """True bila rail keluaran menilai jawaban melanggar kebijakan."""
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return _blocked(messages, "output")
