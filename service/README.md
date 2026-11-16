# Layanan RAG (versi framework)

Pipeline yang sama dengan folder `rag/`, tetapi disusun dengan LlamaIndex dan dibentuk sebagai layanan yang bisa dijalankan terus-menerus, bukan notebook. Tujuannya menunjukkan apa saja yang perlu ada di sekitar sebuah pipeline RAG sebelum layak dipakai orang lain.

## Menjalankan

```bash
docker compose --profile service up --build
```

Job `service-ingest` mengisi pustaka lebih dulu, lalu API menyala di http://localhost:8080 (dokumentasi interaktif di `/docs`).

```bash
curl -s localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: kunci-publik-demo' \
  -d '{"question": "Bagaimana RAPTOR membangun pohon ringkasan dari dokumen?"}'
```

Ganti kuncinya dengan `kunci-internal-demo` untuk peran internal. Kedua kunci itu hanya contoh dari `.env.example`. `SERVICE_API_KEYS` wajib diisi lewat environment (tidak ada nilai bawaan di kode maupun di compose). Bila kosong, layanan menolak start; bila masih memakai kunci contoh, layanan mencatat peringatan saat start.

## Yang digantikan framework

| Ditulis sendiri di `rag/` | Komponen LlamaIndex di `service/` |
|---|---|
| `rag/chunk.py` | `MarkdownNodeParser` + `SentenceSplitter` |
| `rag/embed.py` | `FastEmbedEmbedding` |
| `rag/db.py` | `PGVectorStore` + `MetadataFilters` |
| `scripts/ingest.py` | `IngestionPipeline` |
| `rag/llm.py` | `OpenAILike` |

Pengaman tetap memakai `rag/safety.py` dan `rag/guards.py` (Guardrails AI). Framework tidak menggantikan bagian itu.

## Yang membuatnya mendekati production

| Hal | Di mana | Keterangan |
|---|---|---|
| Konfigurasi dari environment, divalidasi saat start | `app/settings.py` | Layanan menolak start bila `OPENROUTER_API_KEY` kosong |
| Ingest terpisah dari penyajian | `app/ingest.py`, `service-ingest` | Job sekali jalan, aman diulang; API tidak pernah menulis pustaka |
| Peran dari kunci API | `app/main.py` | Hak akses diturunkan dari `X-API-Key`, bukan dari isi permintaan; kunci dibandingkan dalam waktu tetap |
| Tingkat pengaman ditentukan server | `GUARD_LEVEL` | Klien tidak bisa mematikan pengaman |
| Liveness dan readiness | `/healthz`, `/readyz` | Readiness gagal bila basis data tidak terjangkau atau pustaka kosong |
| Batas waktu dan pengulangan | `app/engine.py`, `app/main.py` | Batas waktu LLM, pengulangan untuk jawaban kosong, batas total per permintaan (504) |
| Galat tidak bocor ke klien | `app/main.py` | Klien menerima 502 atau 504; rinciannya hanya masuk log |
| Log JSON ber-`request_id` | `app/main.py` | Isi pertanyaan tidak dicatat, hanya panjangnya |
| Penyamaran PII | `app/engine.py`, `rag/guards.py` | Pertanyaan disamarkan sebelum retrieval dan sebelum dikirim ke LLM; jawaban disamarkan sebelum dikembalikan |
| Metrik Prometheus | `/metrics` | Jumlah permintaan, yang ditahan per tanda, latensi, kemiripan tertinggi |
| Container diperketat | `Dockerfile`, `docker-compose.yml` | Pengguna non-root, sistem berkas hanya-baca, tanpa capability, batas CPU dan memori |
| Tanpa unduhan saat start | `Dockerfile` | Model embedding, tokenizer, dan data NLTK diambil saat build |
| Uji otomatis | `tests/test_api.py`, `.github/workflows/ci.yml` | API diuji dengan mesin tiruan, tanpa basis data dan tanpa LLM |

## Yang masih kurang untuk production sungguhan

Daftar ini sengaja ditulis supaya batasnya jelas.

- PII di dalam dokumen tetap terkirim ke penyedia LLM sebagai konteks. Samarkan saat ingest, atau pakai model yang dijalankan sendiri.
- TLS dan pembatasan laju permintaan. Biasanya ditangani reverse proxy atau API gateway di depan layanan.
- Kunci API disimpan sebagai teks di environment. Pakai secret manager, simpan hash-nya, dan sediakan rotasi.
- Skema basis data dibuat otomatis oleh LlamaIndex saat start. Untuk perubahan skema yang terkendali, pakai alat migrasi.
- Satu replika dan satu PostgreSQL tanpa cadangan. Belum ada replikasi, backup, atau pemulihan.
- Belum ada tracing terdistribusi dan belum ada alert atas metrik.
- Evaluasi kualitas jawaban belum menjadi gerbang rilis. Saat ini hanya uji unit yang berjalan di CI.
- Model gratis dari router tidak punya jaminan ketersediaan. Untuk layanan sungguhan, pilih satu model dengan SLA.

## Menjalankan tanpa Docker

```bash
docker compose up -d db
pip install -r service/requirements.txt pytest httpx
python -m service.app.ingest
uvicorn service.app.main:app --port 8080
python -m pytest service/tests -q
```
