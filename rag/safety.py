"""Lapisan guardrail sistem RAG.

Tiga pemeriksaan, semuanya murah dan tidak memanggil LLM:
1. check_question : pertanyaan pengguna yang mencoba mengubah perilaku asisten.
2. scan_chunk     : dokumen yang menyisipkan perintah untuk asisten (indirect prompt injection).
3. is_out_of_scope: pertanyaan yang jauh dari isi knowledge base (deteksi anomali berbasis kemiripan).

Pola di bawah adalah contoh untuk belajar. Sistem produksi menambah lapisan lain,
misalnya model klasifikasi khusus dan peninjauan manusia.
"""
from __future__ import annotations

import re
import statistics

# Pola kalimat yang mencoba mengambil alih instruksi asisten (Indonesia dan Inggris).
INJECTION_PATTERNS = [
    r"abaikan\s+(semua|seluruh|setiap)?\s*(instruksi|perintah|aturan|petunjuk)",
    r"lupakan\s+(semua|seluruh)?\s*(instruksi|perintah|aturan)",
    r"ignore\s+(all|any|the|your)?\s*(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)",
    r"disregard\s+(all|any|the|your)?\s*(previous|prior|above)\s+(instructions?|rules?)",
    r"(untuk|kepada|bagi)\s+asisten\s+ai",
    r"(kirim|kirimkan|berikan|masukkan)\s+.{0,40}(kata\s+sandi|password|kode\s+otp)",
    r"(akhiri|tutup|awali)\s+(setiap|semua)\s+jawaban",
]
# Pola tambahan yang hanya diperiksa pada pertanyaan pengguna. Pola ini tidak dipakai
# untuk dokumen karena paper tentang LLM wajar menyebut "system prompt".
QUESTION_ONLY_PATTERNS = [
    r"(system|developer)\s+prompt",
    r"prompt\s+sistem",
    r"(instruksi|aturan)\s+(awal|rahasia|tersembunyi)",
]
_COMPILED = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
_COMPILED_QUESTION = _COMPILED + [re.compile(p, re.I) for p in QUESTION_ONLY_PATTERNS]

MAX_QUESTION_CHARS = 600


def find_injection(text: str, question: bool = False) -> list[str]:
    """Kembalikan potongan teks yang cocok dengan pola injeksi (kosong berarti bersih)."""
    patterns = _COMPILED_QUESTION if question else _COMPILED
    return [m.group(0) for pattern in patterns if (m := pattern.search(text))]


def check_question(question: str) -> list[str]:
    flags = []
    if find_injection(question, question=True):
        flags.append("injeksi_pada_pertanyaan")
    if len(question) > MAX_QUESTION_CHARS:
        flags.append("pertanyaan_terlalu_panjang")
    return flags


def scan_chunk(content: str) -> bool:
    """True bila chunk memuat perintah untuk asisten dan harus dibuang dari konteks."""
    return bool(find_injection(content))


def calibrate_threshold(similarities: list[float], k: float = 3.0, floor: float = 0.25) -> float:
    """Tentukan threshold "di luar cakupan" dari pertanyaan contoh yang memang relevan.

    Threshold = rata-rata - k x standar deviasi. Pertanyaan yang kemiripan tertingginya
    berada di bawah threshold dianggap anomali, karena knowledge base tidak memuat hal serupa.
    """
    if len(similarities) < 3:
        return floor
    threshold = statistics.mean(similarities) - k * statistics.stdev(similarities)
    return round(max(threshold, floor), 3)


def is_out_of_scope(top_similarity: float | None, threshold: float) -> bool:
    return top_similarity is None or top_similarity < threshold
