"""Inti layanan: guardrail, retrieval dengan filter hak akses, lalu jawaban dari LLM."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from llama_index.core.llms import ChatMessage
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from rag import db, safety
from rag.pipeline import (BLOCKED_MESSAGE, OUT_OF_SCOPE_MESSAGE, OUTPUT_BLOCKED_MESSAGE,
                          SYSTEM_RAG_GUARDED)

from .components import build_index, build_llm
from .settings import Settings


@dataclass
class Source:
    source_id: str
    title: str
    score: float
    text: str


@dataclass
class Result:
    answer: str
    model: str = ""
    blocked: bool = False
    flags: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    top_score: float | None = None
    latency_ms: int = 0


class RagEngine:
    """Dibuat sekali saat layanan start; aman dipakai bersama oleh banyak request."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index = build_index(settings)
        self.llm = build_llm(settings)
        self.threshold = self._load_threshold()

    def _load_threshold(self) -> float:
        with db.connect() as conn:
            value = db.get_setting(conn, "service_anomaly_threshold")
        return float(value) if value else self.settings.anomaly_threshold_default

    async def ready(self) -> bool:
        """Siap bila database terjangkau dan knowledge base tidak kosong."""
        nodes = await self._retrieve("uji kesiapan", "public", 1)
        return bool(nodes)

    async def _retrieve(self, question: str, role: str, k: int):
        levels = ["public", "internal"] if role == "internal" else ["public"]
        filters = MetadataFilters(filters=[
            MetadataFilter(key="access_level", value=levels, operator=FilterOperator.IN)
        ])
        retriever = self.index.as_retriever(similarity_top_k=k, filters=filters)
        return await retriever.aretrieve(question)

    async def query(self, question: str, role: str) -> Result:
        start = time.perf_counter()
        result = await self._query(question, role)
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        return result

    async def _query(self, question: str, role: str) -> Result:
        level = self.settings.guard_level
        guarded = level != "off"

        flags: list[str] = []
        if guarded:  # mask PII sebelum retrieval dan sebelum teks dikirim ke provider LLM
            from rag import guards

            question, pii = await asyncio.to_thread(guards.mask_pii, question)
            if pii:
                flags.append("pii_pada_pertanyaan")
        flags += safety.check_question(question) if guarded else []
        if "injeksi_pada_pertanyaan" in flags:
            return Result(BLOCKED_MESSAGE, blocked=True, flags=flags)
        if level == "guardrails":
            failed = await self._guard("check_input", question, flags)
            if failed:
                return Result(BLOCKED_MESSAGE, blocked=True, flags=flags + ["guardrails_masukan"])

        nodes = await self._retrieve(question, role, self.settings.top_k)
        top = float(nodes[0].score) if nodes else None
        if guarded:
            if safety.is_out_of_scope(top, self.threshold):
                return Result(OUT_OF_SCOPE_MESSAGE, blocked=True, flags=flags + ["di_luar_cakupan"],
                              top_score=top)
            clean = [n for n in nodes if not safety.scan_chunk(n.node.get_content())]
            if len(clean) < len(nodes):
                flags.append("injeksi_pada_dokumen")
                nodes = clean

        sources = [
            Source(n.node.metadata.get("source_id", ""), n.node.metadata.get("title", ""),
                   float(n.score or 0), n.node.get_content())
            for n in nodes
        ]
        context = "\n\n".join(f"[{i}] {s.title}\n{s.text}" for i, s in enumerate(sources, start=1))
        messages = [
            ChatMessage(role="system", content=SYSTEM_RAG_GUARDED),
            ChatMessage(role="user", content=f"<konteks>\n{context}\n</konteks>\n\nPertanyaan: {question}"),
        ]
        text, model = await self._generate(messages)
        result = Result(text, model=model, flags=flags, sources=sources, top_score=top)
        if guarded:
            result.answer, pii = await asyncio.to_thread(guards.mask_pii, text)
            if pii:
                result.flags.append("pii_pada_jawaban")

        if level == "guardrails":
            failed = await self._guard("check_output", result.answer, result.flags)
            if "HasCitation" in failed:
                result.flags.append("jawaban_tanpa_sumber")
            if set(failed) - {"HasCitation"}:
                result.answer, result.blocked = OUTPUT_BLOCKED_MESSAGE, True
                result.flags.append("guardrails_keluaran")
        return result

    async def _generate(self, messages: list[ChatMessage], attempts: int = 3) -> tuple[str, str]:
        """Panggil LLM. Jawaban kosong dianggap gagal dan diulang; model gratis kadang begitu."""
        for attempt in range(attempts):
            response = await self.llm.achat(messages)
            text = (response.message.content or "").strip()
            if text:
                return text, getattr(response.raw, "model", None) or self.settings.llm_model
            await asyncio.sleep(1 + attempt)
        raise RuntimeError("LLM mengembalikan jawaban kosong")

    @staticmethod
    async def _guard(check: str, text: str, flags: list[str]) -> dict[str, str]:
        """Validator Guardrails AI bersifat sinkron, jadi dijalankan di thread terpisah."""
        from rag import guards

        try:
            return await asyncio.to_thread(getattr(guards, check), text)
        except Exception:
            flags.append("guardrails_tidak_tersedia")
            return {}
