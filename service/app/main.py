"""API layanan RAG.

Jalankan:  uvicorn service.app.main:app --port 8080
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .settings import Settings, get_settings

log = logging.getLogger("service")

REQUESTS = Counter("rag_requests_total", "Jumlah permintaan /v1/query", ["status"])
BLOCKED = Counter("rag_blocked_total", "Permintaan yang ditahan pengaman", ["flag"])
LATENCY = Histogram("rag_request_seconds", "Lama permintaan /v1/query",
                    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80))
TOP_SCORE = Histogram("rag_top_similarity", "Kemiripan tertinggi hasil retrieval",
                      buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))


class JsonFormatter(logging.Formatter):
    """Satu baris JSON per kejadian, supaya mudah dikumpulkan dan dicari."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname,
                 "logger": record.name, "message": record.getMessage()}
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    source_id: str
    title: str
    score: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    model: str
    blocked: bool
    flags: list[str]
    sources: list[SourceOut]
    latency_ms: int


def create_app(engine=None, settings: Settings | None = None) -> FastAPI:
    """`engine` bisa diisi tiruan saat pengujian; bila kosong dibuat dari konfigurasi."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or get_settings()  # gagal di sini bila konfigurasi tidak lengkap
        configure_logging(app.state.settings.log_level)
        if engine is None:
            from .engine import RagEngine

            app.state.engine = RagEngine(app.state.settings)
        else:
            app.state.engine = engine
        if app.state.settings.uses_demo_keys:
            log.warning("SERVICE_API_KEYS masih memakai kunci contoh; ganti sebelum layanan dibuka ke jaringan")
        log.info("layanan siap", extra={"fields": {"guard_level": app.state.settings.guard_level}})
        yield

    app = FastAPI(title="Layanan RAG", version="1.0.0", lifespan=lifespan)

    def role_from_key(request: Request, x_api_key: str | None = Header(default=None)) -> str:
        given = (x_api_key or "").encode()
        # Bandingkan dalam waktu tetap supaya kunci tidak bisa ditebak dari lama respons.
        for key, role in request.app.state.settings.api_keys.items():
            if hmac.compare_digest(given, key.encode()):
                return role
        raise HTTPException(status_code=401, detail="X-API-Key tidak dikenal")

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        """Liveness: proses hidup."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> dict:
        """Readiness: basis data terjangkau dan pustaka terisi."""
        try:
            ready = await request.app.state.engine.ready()
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"belum siap: {type(error).__name__}") from error
        if not ready:
            raise HTTPException(status_code=503, detail="pustaka kosong; jalankan job ingest")
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request, role: str = Depends(role_from_key)) -> QueryResponse:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                request.app.state.engine.query(body.question, role),
                timeout=request.app.state.settings.request_timeout_s,
            )
        except asyncio.TimeoutError as error:
            REQUESTS.labels("timeout").inc()
            log.error("permintaan melewati batas waktu", extra={"fields": {"request_id": request_id, "role": role}})
            raise HTTPException(status_code=504, detail="melewati batas waktu") from error
        except Exception as error:
            REQUESTS.labels("error").inc()
            log.error("permintaan gagal", extra={"fields": {
                "request_id": request_id, "role": role, "error": f"{type(error).__name__}: {error}"[:300]}})
            raise HTTPException(status_code=502, detail="LLM atau basis data tidak menjawab") from error

        LATENCY.observe(time.perf_counter() - start)
        REQUESTS.labels("blocked" if result.blocked else "ok").inc()
        for flag in result.flags:
            BLOCKED.labels(flag).inc()
        if result.top_score is not None:
            TOP_SCORE.observe(result.top_score)
        # Isi pertanyaan tidak dicatat di log; cukup panjangnya.
        log.info("query", extra={"fields": {
            "request_id": request_id, "role": role, "blocked": result.blocked, "flags": result.flags,
            "top_score": result.top_score, "model": result.model, "latency_ms": result.latency_ms,
            "question_chars": len(body.question), "sources": [s.source_id for s in result.sources]}})
        return QueryResponse(
            request_id=request_id, answer=result.answer, model=result.model, blocked=result.blocked,
            flags=result.flags, latency_ms=result.latency_ms,
            sources=[SourceOut(source_id=s.source_id, title=s.title, score=round(s.score, 3)) for s in result.sources],
        )

    return app


app = create_app()
