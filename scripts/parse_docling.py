"""Ubah PDF menjadi Markdown dengan Docling.

Docling membaca tata letak halaman (judul, paragraf, tabel, rumus) lalu menyusunnya
kembali sebagai teks terstruktur. Hasilnya jauh lebih bersih daripada menyalin teks
mentah dari PDF, terutama untuk paper dua kolom dan tabel.

Jalankan:  python scripts/parse_docling.py            # semua PDF di data/pdf/
           python scripts/parse_docling.py 2309.15217 # satu paper saja
Hasil disimpan ke data/parsed/<id>.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdf"
OUT_DIR = ROOT / "data" / "parsed"


def build_converter() -> DocumentConverter:
    options = PdfPipelineOptions()
    options.do_ocr = False  # paper arXiv sudah berupa teks digital, OCR tidak perlu
    options.do_table_structure = True  # kenali baris dan kolom tabel
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def main() -> None:
    wanted = set(sys.argv[1:])
    pdfs = sorted(p for p in PDF_DIR.glob("*.pdf") if not wanted or p.stem in wanted)
    if not pdfs:
        sys.exit("Tidak ada PDF. Jalankan dulu: python scripts/download_papers.py")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = build_converter()
    for pdf in pdfs:
        start = time.time()
        result = converter.convert(pdf)
        markdown = result.document.export_to_markdown()
        (OUT_DIR / f"{pdf.stem}.md").write_text(markdown, encoding="utf-8")
        pages = len(result.document.pages)
        print(f"{pdf.stem}  {pages} halaman  {len(markdown):,} karakter  {time.time() - start:.0f} dtk")


if __name__ == "__main__":
    main()
