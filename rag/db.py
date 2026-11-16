"""Akses PostgreSQL + pgvector: simpan dokumen, simpan chunk, dan cari yang paling mirip."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL, ROOT


@dataclass
class Hit:
    chunk_id: int
    source_id: str
    title: str
    section: str
    content: str
    access_level: str
    url: str
    similarity: float


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # supaya numpy array bisa langsung dikirim sebagai tipe vector
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute((ROOT / "db" / "init.sql").read_text())


def upsert_document(conn, source_id, title, authors="", url="", license="", access_level="public") -> int:
    """Simpan dokumen. Bila source_id sudah ada, chunk lamanya dibuang agar tidak dobel."""
    conn.execute("DELETE FROM documents WHERE source_id = %s", (source_id,))
    row = conn.execute(
        """INSERT INTO documents (source_id, title, authors, url, license, access_level)
           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
        (source_id, title, authors, url, license, access_level),
    ).fetchone()
    return row[0]


def insert_chunks(conn, document_id: int, chunks, vectors: np.ndarray) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (document_id, chunk_index, section, content, embedding)
               VALUES (%s, %s, %s, %s, %s)""",
            [(document_id, i, c.section, c.content, v) for i, (c, v) in enumerate(zip(chunks, vectors))],
        )


def search(conn, query_vector: np.ndarray, k: int = 5, role: str = "public") -> list[Hit]:
    """Ambil k chunk terdekat. Hak akses disaring di SQL yang sama dengan pencarian vektornya.

    Operator <=> adalah cosine distance milik pgvector; kemiripan = 1 - jarak.
    """
    levels = ["public", "internal"] if role == "internal" else ["public"]
    rows = conn.execute(
        """SELECT c.id, d.source_id, d.title, c.section, c.content, d.access_level, d.url,
                  1 - (c.embedding <=> %(q)s) AS similarity
           FROM chunks c
           JOIN documents d ON d.id = c.document_id
           WHERE d.access_level = ANY(%(levels)s)
           ORDER BY c.embedding <=> %(q)s
           LIMIT %(k)s""",
        {"q": query_vector, "levels": levels, "k": k},
    ).fetchall()
    return [Hit(*row[:7], float(row[7])) for row in rows]


def get_setting(conn, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key = %s", (key,)).fetchone()
    return row[0] if row else default


def set_setting(conn, key: str, value: str) -> None:
    conn.execute(
        """INSERT INTO settings (key, value) VALUES (%s, %s)
           ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
        (key, value),
    )


def log_query(conn, question, role, top_similarity, flags, blocked, model, latency_ms) -> None:
    conn.execute(
        """INSERT INTO query_log (question, role, top_similarity, flags, blocked, model, latency_ms)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (question, role, top_similarity, flags, blocked, model, latency_ms),
    )


def stats(conn) -> dict:
    docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"documents": docs, "chunks": chunks}
