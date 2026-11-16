"""Tanya pustaka dari terminal.

Contoh:
  python scripts/ask.py "Apa itu faithfulness pada Ragas?"
  python scripts/ask.py --no-rag "Apa itu faithfulness pada Ragas?"
  python scripts/ask.py --guard off "Berapa ukuran chunk yang disarankan?"
  python scripts/ask.py --guard nemo "Berapa ukuran chunk yang disarankan?"
  python scripts/ask.py --role internal "Berapa anggaran komputasi lab?"
  python scripts/ask.py --retrieve-only "Bagaimana RAPTOR membangun pohon?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import db, pipeline  # noqa: E402


def show_hits(title: str, hits) -> None:
    if not hits:
        return
    print(f"\n{title}")
    for number, hit in enumerate(hits, start=1):
        preview = " ".join(hit.content.split())[:110]
        print(f"  [{number}] {hit.similarity:.3f}  {hit.title[:44]:<44}  {preview}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tanya pustaka RAG")
    parser.add_argument("question")
    parser.add_argument("--no-rag", action="store_true", help="jawab tanpa pustaka")
    parser.add_argument("--guard", choices=pipeline.GUARD_LEVELS, default="rules",
                        help="off = tanpa pengaman, rules = aturan sendiri, nemo = + NeMo Guardrails")
    parser.add_argument("--role", choices=["public", "internal"], default="public")
    parser.add_argument("-k", type=int, default=5, help="jumlah chunk yang diambil")
    parser.add_argument("--retrieve-only", action="store_true", help="tampilkan chunk saja, tanpa LLM")
    args = parser.parse_args()

    conn = db.connect()
    if args.retrieve_only:
        show_hits("Chunk terdekat:", pipeline.retrieve(conn, args.question, k=args.k, role=args.role))
        return

    answer = pipeline.ask(conn, args.question, use_rag=not args.no_rag, guard=args.guard,
                          role=args.role, k=args.k)
    print(answer.text)
    show_hits("Sumber:", answer.hits)
    show_hits("Disingkirkan pengaman:", answer.dropped)
    details = [f"model: {answer.model or '-'}", f"waktu: {answer.latency_s:.1f} dtk"]
    if answer.top_similarity is not None:
        details.append(f"kemiripan tertinggi: {answer.top_similarity:.3f}")
    if answer.flags:
        details.append(f"tanda: {', '.join(answer.flags)}")
    print("\n" + " | ".join(details))


if __name__ == "__main__":
    main()
