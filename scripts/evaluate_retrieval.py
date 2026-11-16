"""Ukur kualitas retrieval tanpa memanggil LLM.

Untuk setiap pertanyaan di data/sample_questions.json sudah dicatat dokumen yang
seharusnya menjawabnya. Skrip ini menghitung seberapa sering dokumen itu muncul
di peringkat 1 (hit@1) dan di lima besar (hit@5).

Jalankan:  python scripts/evaluate_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import db, pipeline  # noqa: E402
from rag.config import ROOT  # noqa: E402


def evaluate(conn, k: int = 5) -> list[dict]:
    samples = json.loads((ROOT / "data" / "sample_questions.json").read_text())
    rows = []
    for sample in samples:
        sources = [hit.source_id for hit in pipeline.retrieve(conn, sample["question"], k=k)]
        rank = sources.index(sample["source"]) + 1 if sample["source"] in sources else None
        rows.append({"question": sample["question"], "expected": sample["source"], "rank": rank})
    return rows


def main() -> None:
    rows = evaluate(db.connect())
    for row in rows:
        rank = f"peringkat {row['rank']}" if row["rank"] else "tidak ditemukan"
        print(f"{rank:<16} {row['question']}")
    total = len(rows)
    hit1 = sum(row["rank"] == 1 for row in rows)
    hit5 = sum(row["rank"] is not None for row in rows)
    print(f"\nhit@1 = {hit1}/{total} ({hit1 / total:.0%})   hit@5 = {hit5}/{total} ({hit5 / total:.0%})")


if __name__ == "__main__":
    main()
