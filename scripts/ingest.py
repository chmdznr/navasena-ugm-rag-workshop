"""Bangun pustaka RAG: Markdown -> chunk -> embedding -> PostgreSQL (pgvector).

Jalankan:  python scripts/ingest.py
Aman dijalankan ulang; dokumen yang sama akan ditimpa, bukan digandakan.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import db, safety  # noqa: E402
from rag.chunk import chunk_markdown  # noqa: E402
from rag.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP, ROOT  # noqa: E402
from rag.embed import embed_texts  # noqa: E402

DATA = ROOT / "data"


def text_for_embedding(title: str, chunk) -> str:
    """Judul dokumen dan judul bagian ikut di-embed agar chunk pendek tetap punya konteks."""
    return f"{title}. {chunk.section}. {chunk.content}" if chunk.section else f"{title}. {chunk.content}"


def ingest_document(conn, meta: dict, markdown: str) -> int:
    chunks = chunk_markdown(markdown, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP)
    vectors = embed_texts([text_for_embedding(meta["title"], c) for c in chunks])
    document_id = db.upsert_document(
        conn,
        source_id=meta["source_id"],
        title=meta["title"],
        authors=meta.get("authors", ""),
        url=meta.get("url", ""),
        license=meta.get("license", ""),
        access_level=meta.get("access_level", "public"),
    )
    db.insert_chunks(conn, document_id, chunks, vectors)
    return len(chunks)


def load_sources() -> list[tuple[dict, Path]]:
    sources = []
    for paper in json.loads((DATA / "papers.json").read_text()):
        meta = {
            "source_id": f"arxiv:{paper['id']}",
            "title": paper["title"],
            "authors": f"{paper['authors']} ({paper['year']})",
            "url": f"https://arxiv.org/abs/{paper['id']}",
            "license": paper["license"],
        }
        sources.append((meta, DATA / "parsed" / f"{paper['id']}.md"))
    for extra in json.loads((DATA / "extra" / "extras.json").read_text()):
        sources.append((extra, DATA / "extra" / extra["file"]))
    return sources


def calibrate(conn) -> float:
    """Hitung ambang deteksi anomali dari pertanyaan contoh yang memang relevan."""
    samples = json.loads((DATA / "sample_questions.json").read_text())
    questions = [sample["question"] for sample in samples]
    vectors = embed_texts(questions)
    tops = [db.search(conn, vector, k=1)[0].similarity for vector in vectors]
    value = safety.calibrate_threshold(tops)
    db.set_setting(conn, "anomaly_threshold", str(value))
    return value


def main() -> None:
    conn = db.connect()
    db.ensure_schema(conn)
    start = time.time()
    for meta, path in load_sources():
        if not path.exists():
            print(f"lewati  {meta['source_id']}  (berkas {path.name} tidak ada)")
            continue
        count = ingest_document(conn, meta, path.read_text(encoding="utf-8"))
        print(f"masuk   {meta['source_id']:<24} {count:>4} chunk  {meta['title'][:50]}")
    threshold = calibrate(conn)
    totals = db.stats(conn)
    print(f"\nSelesai dalam {time.time() - start:.0f} detik: "
          f"{totals['documents']} dokumen, {totals['chunks']} chunk.")
    print(f"Ambang deteksi anomali: {threshold}")


if __name__ == "__main__":
    main()
