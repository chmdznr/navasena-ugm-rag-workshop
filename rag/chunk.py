"""Memotong dokumen menjadi chunk.

Dua cara disediakan supaya bisa dibandingkan:
- chunk_fixed     : potong setiap N karakter tanpa melihat isi (cara naif).
- chunk_markdown  : ikuti struktur dokumen (judul bagian, paragraf), baru batasi ukurannya.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Bagian yang tidak membantu menjawab pertanyaan dan hanya mengotori hasil pencarian.
SKIP_SECTIONS = re.compile(
    r"^(\d+[.\s]*)?(references|bibliography|acknowledge?ments?|daftar pustaka)\b", re.I
)
IMAGE_PLACEHOLDER = re.compile(r"^<!--\s*image\s*-->$", re.I)


@dataclass
class Chunk:
    section: str
    content: str


def chunk_fixed(text: str, size: int = 700) -> list[Chunk]:
    """Cara naif: potong rata setiap `size` karakter. Kalimat dan tabel bisa terbelah."""
    return [Chunk("", text[i : i + size]) for i in range(0, len(text), size)]


def split_sections(markdown: str) -> list[tuple[str, list[str]]]:
    """Pecah Markdown menjadi daftar (judul bagian, daftar paragraf)."""
    sections: list[tuple[str, list[str]]] = []
    title, buffer, paragraphs = "", [], []

    def flush_paragraph() -> None:
        if buffer:
            paragraphs.append("\n".join(buffer).strip())
            buffer.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            sections.append((title, paragraphs.copy()))
            paragraphs.clear()

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            flush_section()
            title = match.group(2)
        elif not line.strip():
            flush_paragraph()
        elif not IMAGE_PLACEHOLDER.match(line.strip()):
            buffer.append(line.rstrip())
    flush_section()
    return sections


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Paragraf yang melebihi batas dipecah per kalimat (tabel dipecah per baris)."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    is_table = paragraph.lstrip().startswith("|")
    units = paragraph.split("\n") if is_table else re.split(r"(?<=[.!?])\s+", paragraph)
    joiner = "\n" if is_table else " "
    pieces, current = [], ""
    for unit in units:
        while len(unit) > max_chars:  # satu kalimat/baris yang sangat panjang
            if current:
                pieces.append(current)
                current = ""
            pieces.append(unit[:max_chars])
            unit = unit[max_chars:]
        if current and len(current) + len(joiner) + len(unit) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}{joiner}{unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(markdown: str, max_chars: int = 700, overlap: int = 100) -> list[Chunk]:
    """Potong mengikuti struktur: tidak melewati batas bagian, paragraf dijaga utuh bila muat.

    `overlap` menyalin ekor chunk sebelumnya ke awal chunk berikutnya, supaya kalimat
    yang berada di perbatasan tetap punya konteks.
    """
    chunks: list[Chunk] = []
    for title, paragraphs in split_sections(markdown):
        if SKIP_SECTIONS.match(title):
            continue
        current = ""
        for paragraph in paragraphs:
            for piece in _split_long(paragraph, max_chars):
                if current and len(current) + 2 + len(piece) > max_chars:
                    chunks.append(Chunk(title, current))
                    tail = current[-overlap:] if overlap else ""
                    # mulai ekor dari batas kata agar tidak memotong di tengah kata
                    tail = tail[tail.find(" ") + 1 :] if " " in tail else ""
                    current = f"{tail} {piece}".strip() if tail else piece
                else:
                    current = f"{current}\n\n{piece}" if current else piece
        if current.strip():
            chunks.append(Chunk(title, current))
    return [c for c in chunks if len(c.content.strip()) >= 40]
