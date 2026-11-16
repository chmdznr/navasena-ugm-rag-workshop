from rag.chunk import chunk_fixed, chunk_markdown, split_sections

MARKDOWN = """# Judul

Paragraf pembuka yang cukup panjang untuk menjadi satu chunk sendiri di dalam uji ini.

## Metode

Kalimat pertama metode. Kalimat kedua metode yang menjelaskan langkah berikutnya.

| kolom a | kolom b |
|---|---|
| 1 | 2 |

## References

[1] Sebuah rujukan yang tidak perlu ikut masuk ke knowledge base sama sekali, panjangnya cukup.
"""


def test_split_sections_keeps_titles():
    titles = [title for title, _ in split_sections(MARKDOWN)]
    assert titles == ["Judul", "Metode", "References"]


def test_references_are_skipped():
    chunks = chunk_markdown(MARKDOWN)
    assert all("rujukan" not in chunk.content for chunk in chunks)
    assert {chunk.section for chunk in chunks} == {"Judul", "Metode"}


def test_chunks_respect_max_chars():
    text = "## Bagian\n\n" + " ".join(f"Kalimat nomor {i} cukup panjang." for i in range(200))
    chunks = chunk_markdown(text, max_chars=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(chunk.content) <= 300 + 50 + 1 for chunk in chunks)


def test_overlap_repeats_tail():
    text = "## Bagian\n\n" + " ".join(f"Kalimat nomor {i} cukup panjang." for i in range(60))
    chunks = chunk_markdown(text, max_chars=300, overlap=80)
    tail_word = chunks[0].content.split()[-1]
    assert tail_word in chunks[1].content[:120]


def test_fixed_chunks_cover_whole_text():
    text = "a" * 1650
    chunks = chunk_fixed(text, 700)
    assert [len(chunk.content) for chunk in chunks] == [700, 700, 250]
