"""API kecil di atas pipeline RAG, dipakai promptfoo untuk pengujian otomatis.

Jalankan:  uvicorn app.api:app --port 8000   lalu buka http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import db, llm, pipeline

app = FastAPI(title="Workshop RAG API")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    guard: Literal["off", "rules", "nemo"] = "rules"
    role: Literal["public", "internal"] = "public"
    use_rag: bool = True
    k: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    output: str
    model: str
    blocked: bool
    flags: list[str]
    top_similarity: float | None
    sources: list[str]
    dropped: list[str]


@app.get("/health")
def health() -> dict:
    conn = db.connect()
    return {"status": "ok", **db.stats(conn)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    conn = db.connect()
    try:
        answer = pipeline.ask(conn, request.question, use_rag=request.use_rag, guard=request.guard,
                              role=request.role, k=request.k)
    except llm.LLMError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        conn.close()
    return AskResponse(
        output=answer.text,
        model=answer.model,
        blocked=answer.blocked,
        flags=answer.flags,
        top_similarity=answer.top_similarity,
        sources=[hit.source_id for hit in answer.hits],
        dropped=[hit.source_id for hit in answer.dropped],
    )
