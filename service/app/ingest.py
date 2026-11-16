"""Job ingest: Markdown -> node -> embedding -> pgvector, memakai IngestionPipeline LlamaIndex.

Dijalankan terpisah dari layanan API (lihat service-ingest di docker-compose.yml).
Aman diulang: node lama milik dokumen yang sama dihapus sebelum yang baru ditulis.

Jalankan:  python -m service.app.ingest
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent

from rag import db, safety

from .components import build_embed_model, build_index, build_vector_store
from .settings import get_settings

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SKIP = ("references", "bibliography", "acknowledgment", "acknowledgement", "daftar pustaka")
log = logging.getLogger("service.ingest")


class SectionFilter(TransformComponent):
    """Catat judul bagian sebagai metadata dan buang bagian yang mengotori pencarian.

    MarkdownNodeParser menghasilkan satu node per bagian. Judul bagian disimpan di metadata
    supaya tetap menempel ketika node itu dipecah lagi menjadi potongan yang lebih kecil.
    """

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        kept = []
        for node in nodes:
            text = node.get_content().strip()
            first_line = text.split("\n", 1)[0]
            section = first_line.lstrip("# ").strip() if first_line.startswith("#") else ""
            if section.lower().lstrip("0123456789. ").startswith(SKIP) or len(text) < 40:
                continue
            node.metadata["section"] = section
            node.metadata.pop("header_path", None)
            kept.append(node)
        return kept


def load_documents() -> list[Document]:
    documents = []
    papers = json.loads((DATA / "papers.json").read_text())
    extras = json.loads((DATA / "extra" / "extras.json").read_text())
    sources = [
        ({"source_id": f"arxiv:{p['id']}", "title": p["title"], "access_level": "public",
          "url": f"https://arxiv.org/abs/{p['id']}"}, DATA / "parsed" / f"{p['id']}.md")
        for p in papers
    ] + [
        ({"source_id": e["source_id"], "title": e["title"], "access_level": e["access_level"], "url": ""},
         DATA / "extra" / e["file"])
        for e in extras
    ]
    for metadata, path in sources:
        if not path.exists():
            log.warning("lewati %s: file %s tidak ada", metadata["source_id"], path.name)
            continue
        documents.append(Document(
            text=path.read_text(encoding="utf-8"),
            doc_id=metadata["source_id"],
            metadata=metadata,
            # Hanya judul yang ikut di-embed dan dikirim ke LLM; metadata lain untuk filter.
            excluded_embed_metadata_keys=["source_id", "access_level", "url"],
            excluded_llm_metadata_keys=["source_id", "access_level", "url"],
        ))
    return documents


def calibrate(index) -> float:
    """Threshold deteksi anomali dari pertanyaan contoh: rata-rata dikurangi tiga standar deviasi."""
    samples = json.loads((DATA / "sample_questions.json").read_text())
    retriever = index.as_retriever(similarity_top_k=1)
    tops = [retriever.retrieve(sample["question"])[0].score for sample in samples]
    log.info("kemiripan pertanyaan contoh: rata-rata %.3f, standar deviasi %.3f",
             statistics.mean(tops), statistics.stdev(tops))
    return safety.calibrate_threshold(tops)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    start = time.time()
    vector_store = build_vector_store(settings)
    documents = load_documents()
    for document in documents:
        vector_store.delete(ref_doc_id=document.doc_id)

    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),  # satu node per bagian dokumen
            SectionFilter(),
            SentenceSplitter(chunk_size=settings.chunk_tokens, chunk_overlap=settings.chunk_overlap_tokens),
            build_embed_model(settings),
        ],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents, show_progress=False)

    conn = db.connect()
    db.ensure_schema(conn)
    threshold = calibrate(build_index(settings))
    db.set_setting(conn, "service_anomaly_threshold", str(threshold))
    log.info("selesai: %d dokumen, %d node, %.0f detik, threshold anomali %.3f",
             len(documents), len(nodes), time.time() - start, threshold)


if __name__ == "__main__":
    main()
