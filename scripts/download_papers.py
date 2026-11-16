"""Unduh PDF paper dari arXiv sesuai daftar di data/papers.json.

Jalankan:  python scripts/download_papers.py
PDF disimpan ke data/pdf/ (tidak ikut di-commit).
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdf"


def main() -> None:
    papers = json.loads((ROOT / "data" / "papers.json").read_text())
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        target = PDF_DIR / f"{paper['id']}.pdf"
        if target.exists() and target.stat().st_size > 10_000:
            print(f"lewati  {paper['id']}  (sudah ada)")
            continue
        url = f"https://arxiv.org/pdf/{paper['id']}"
        request = urllib.request.Request(url, headers={"User-Agent": "rag-workshop/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
        print(f"unduh   {paper['id']}  {target.stat().st_size / 1e6:.1f} MB  {paper['title'][:60]}")
        time.sleep(3)  # arXiv meminta jeda antarpermintaan


if __name__ == "__main__":
    main()
