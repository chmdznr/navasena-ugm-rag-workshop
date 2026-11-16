# Workshop RAG Navasena × UGM

Materi workshop "Dari PDF ke asisten riset: membangun RAG yang aman" (Navasena × UGM, 21 November 2026; pembicara Achmad Zaenuri). Isinya slide, tiga notebook latihan, aplikasi web, dan semua kode pendukungnya. Seluruhnya berjalan di laptop lewat Docker Compose.

Studi kasusnya satu dari awal sampai akhir: asisten riset yang menjawab pertanyaan dari delapan paper arXiv tentang Retrieval-Augmented Generation.

## Quick start

Butuh Docker Desktop (atau Docker Engine dengan Compose v2) dan sekitar 3 GB ruang disk.

```bash
git clone https://github.com/chmdznr/navasena-ugm-rag-workshop.git
cd navasena-ugm-rag-workshop
cp .env.example .env          # lalu isi OPENROUTER_API_KEY
docker compose up --build
```

Build pertama memakan beberapa menit karena mengunduh paket Python dan model embedding. Setelah itu, knowledge base dibangun otomatis (sekitar dua menit) dan tiga layanan siap dibuka:

| URL | Isi |
|---|---|
| http://localhost:8501 | Aplikasi web asisten riset |
| http://localhost:8888 | JupyterLab berisi notebook latihan |
| http://localhost:8000/docs | API yang dipakai untuk pengujian |

Key OpenRouter gratis dibuat di https://openrouter.ai/settings/keys. Tanpa key, tab Retrieval, Chunking, dan Peta embedding serta notebook Sesi 2 tetap bisa dipakai, karena bagian itu tidak memanggil LLM.

## Materi per sesi

| Sesi | Notebook | Yang dikerjakan |
|---|---|---|
| 1. Pengenalan ML dan RAG | `notebooks/01_llm_tanpa_rag.ipynb` | LLM ditanya hal yang tidak ia ketahui, lalu RAG dikerjakan secara manual |
| 2. Setting up RAG | `notebooks/02_setting_up_rag.ipynb` | Docling, chunking, embedding, pgvector, peta embedding, evaluasi retrieval |
| 3. Implementation and safety | `notebooks/03_rag_dan_safety.ipynb` | Tanya-jawab end-to-end, hak akses, dokumen yang disusupi, deteksi anomali, Guardrails AI, perlindungan PII, promptfoo |

Slide ada di `slides/dist/` (PPTX dan PDF).

Setelah ketiga sesi, folder `service/` menunjukkan pipeline yang sama disusun dengan framework (LlamaIndex) sebagai layanan yang mendekati production. Lihat `service/README.md`.

## Arsitektur

```
PDF ── Docling ──> Markdown ── chunking ──> embedding ──> PostgreSQL + pgvector
                                                              │
pertanyaan ── guardrail ── retrieve ───────────────────────────┘
                              │
                          augment ── generate (LLM lewat OpenRouter) ── guardrail ── jawaban + sumber
```

| Bagian | Pilihan | File |
|---|---|---|
| Parsing PDF | Docling | `scripts/parse_docling.py` |
| Chunking | mengikuti struktur Markdown, 700 karakter, overlap 100 | `rag/chunk.py` |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` lewat fastembed, lokal di CPU | `rag/embed.py` |
| Vector database | PostgreSQL 17 + pgvector, indeks HNSW | `db/init.sql`, `rag/db.py` |
| LLM | OpenRouter, default `openrouter/free` | `rag/llm.py` |
| Alur tanya-jawab | retrieve, augment, generate | `rag/pipeline.py` |
| Guardrail | rule-based, lalu Guardrails AI (validator Hub dan validator custom) | `rag/safety.py`, `rag/guards.py` |
| Pengujian | promptfoo lewat API | `uji/promptfooconfig.yaml`, `app/api.py` |
| Versi framework | LlamaIndex + FastAPI sebagai layanan | `service/` |

## Perintah yang sering dipakai

Semua perintah dijalankan dari folder repo.

```bash
# Tanya dari terminal
docker compose exec app python scripts/ask.py "Bagaimana RAPTOR membangun pohon ringkasan dari dokumen?"
docker compose exec app python scripts/ask.py --retrieve-only "Bagaimana Ragas mengukur faithfulness?"
docker compose exec app python scripts/ask.py --guard off "Berapa ukuran chunk yang disarankan?"
docker compose exec app python scripts/ask.py --role internal "Berapa anggaran komputasi lab?"

# Ukur kualitas retrieval (tanpa LLM)
docker compose exec app python scripts/evaluate_retrieval.py

# Bangun ulang knowledge base setelah mengubah dokumen atau ukuran chunk
docker compose run --rm ingest

# Parsing ulang PDF dengan Docling (image terpisah, beberapa GB)
docker compose --profile docling run --rm docling

# Uji keamanan dengan promptfoo
docker compose --profile uji run --rm promptfoo

# Layanan versi framework (LlamaIndex) di http://localhost:8080/docs
docker compose --profile service up --build

# Berhenti; tambahkan -v untuk menghapus isi database
docker compose down
```

## Level guardrail

Aplikasi, API, dan `scripts/ask.py` menerima tiga level guardrail:

| Tingkat | Isi | Panggilan LLM tambahan |
|---|---|---|
| `off` | tanpa guardrail | 0 |
| `rules` | masking PII pada pertanyaan dan jawaban, pola injeksi pada pertanyaan dan dokumen, threshold kemiripan untuk pertanyaan di luar cakupan, prompt yang memperlakukan konteks sebagai data | 0 |
| `guardrails` | semua di atas, ditambah guard input dan guard output Guardrails AI | 2 per pertanyaan |

Hak akses dokumen (`public` atau `internal`) selalu disaring di SQL, apa pun level guardrail-nya.

Dua dokumen di `data/extra/` adalah contoh buatan untuk demo keamanan: satu catatan internal fiktif (termasuk kontak dan NIK karangan) dan satu unggahan forum yang disisipi perintah untuk asisten. Pola di `rag/safety.py` adalah contoh untuk belajar, bukan daftar yang lengkap.

Guardrails AI dipakai dengan dua jenis validator: dari Guardrails Hub (`ValidLength`, `SecretsPresent`, dipasang lewat pip tanpa akun atau API key) dan buatan sendiri di `rag/guards.py`. Validator `LLMPolicy` memakai LLM sebagai penilai, sehingga hasilnya bisa berbeda antarpanggilan.

### Perlindungan PII

Pada level `rules` dan `guardrails`, email, nomor HP, NIK, NPWP, dan nomor kartu di-mask di pertanyaan (sebelum retrieval, sebelum dikirim ke LLM, dan sebelum dicatat di log) serta di jawaban. Tiga batasnya:

- Nama orang tidak dikenali regex. Isi `PII_ENGINE=presidio` setelah `pip install guardrails-ai-detect-pii` untuk menambahkan `DetectPII` dari Guardrails Hub. Validator itu mengunduh model sekitar 400 MB dan tidak mengenal NIK.
- Chunk dokumen yang memuat PII tetap terkirim ke provider LLM sebagai konteks. Untuk data sungguhan, mask PII saat ingest.
- Pola bisa meleset: deretan 16 digit yang bukan NIK ikut di-mask.

## Kuota OpenRouter

Model gratis OpenRouter dibatasi per akun, baik per menit maupun per hari. Batas hariannya kecil untuk akun tanpa saldo, jadi satu key yang dipakai bersama satu kelas akan cepat habis. Minta setiap peserta membuat key sendiri, dan periksa batas terbaru di https://openrouter.ai/docs/api-reference/limits sebelum acara.

`openrouter/free` memilih model secara otomatis, sehingga gaya dan kepatuhan jawaban bisa berbeda antarpanggilan. Untuk demo yang lebih stabil, isi `LLM_MODEL` di `.env` dengan satu id model tertentu.

## Menjalankan tanpa Docker untuk aplikasinya

Database tetap dari Compose; Python dijalankan langsung.

```bash
docker compose up -d db
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python scripts/ingest.py
.venv/bin/streamlit run app/app.py
.venv/bin/python -m pytest tests -q
```

Uji layanan (`service/tests`) butuh dependensi tambahan:

```bash
.venv/bin/pip install -r service/requirements.txt httpx
.venv/bin/python -m pytest tests service/tests -q
```

## Sumber dokumen

Paper di `data/parsed/` berasal dari arXiv dan berlisensi CC BY 4.0. Teksnya adalah hasil konversi otomatis dengan Docling, jadi bisa memuat salah baca pada rumus dan tabel. Rujuk PDF aslinya untuk kutipan.

| arXiv | Judul | Penulis |
|---|---|---|
| [2407.01219](https://arxiv.org/abs/2407.01219) | Searching for Best Practices in Retrieval-Augmented Generation | Wang et al., 2024 |
| [2401.05856](https://arxiv.org/abs/2401.05856) | Seven Failure Points When Engineering a Retrieval Augmented Generation System | Barnett et al., 2024 |
| [2309.15217](https://arxiv.org/abs/2309.15217) | Ragas: Automated Evaluation of Retrieval Augmented Generation | Es et al., 2023 |
| [2310.11511](https://arxiv.org/abs/2310.11511) | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Asai et al., 2023 |
| [2405.20485](https://arxiv.org/abs/2405.20485) | Phantom: General Backdoor Attacks on Retrieval Augmented Language Generation | Chaudhari et al., 2024 |
| [2404.16130](https://arxiv.org/abs/2404.16130) | From Local to Global: A Graph RAG Approach to Query-Focused Summarization | Edge et al., 2024 |
| [2401.18059](https://arxiv.org/abs/2401.18059) | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | Sarthi et al., 2024 |
| [2407.16833](https://arxiv.org/abs/2407.16833) | Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach | Li et al., 2024 |
