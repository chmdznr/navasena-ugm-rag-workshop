# Layanan RAG (versi framework)

Pipeline yang sama dengan folder `rag/`, tetapi disusun dengan LlamaIndex dan dibentuk sebagai layanan yang bisa dijalankan terus-menerus, bukan notebook. Tujuannya menunjukkan apa saja yang perlu ada di sekitar sebuah pipeline RAG sebelum layak dipakai orang lain.

## Menjalankan

```bash
docker compose --profile service up --build
```

Job `service-ingest` mengisi knowledge base lebih dulu, lalu API menyala di http://localhost:8080 (dokumentasi interaktif di `/docs`).

```bash
curl -s localhost:8080/v1/query \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: kunci-publik-demo' \
  -d '{"question": "Bagaimana RAPTOR membangun pohon ringkasan dari dokumen?"}'
```

Ganti key-nya dengan `kunci-internal-demo` untuk role internal. Kedua key itu hanya contoh dari `.env.example`. `SERVICE_API_KEYS` wajib diisi lewat environment (tidak ada nilai default di kode maupun di compose). Bila kosong, layanan menolak start; bila masih memakai key contoh, layanan mencatat peringatan saat start.

## Yang digantikan framework

| Ditulis sendiri di `rag/` | Komponen LlamaIndex di `service/` |
|---|---|
| `rag/chunk.py` | `MarkdownNodeParser` + `SentenceSplitter` |
| `rag/embed.py` | `FastEmbedEmbedding` |
| `rag/db.py` | `PGVectorStore` + `MetadataFilters` |
| `scripts/ingest.py` | `IngestionPipeline` |
| `rag/llm.py` | `OpenAILike` |

Guardrail tetap memakai `rag/safety.py` dan `rag/guards.py` (Guardrails AI). Framework tidak menggantikan bagian itu.

## Yang membuatnya mendekati production

| Hal | Di mana | Keterangan |
|---|---|---|
| Konfigurasi dari environment, divalidasi saat start | `app/settings.py` | Layanan menolak start bila `OPENROUTER_API_KEY` kosong |
| Ingest terpisah dari serving | `app/ingest.py`, `service-ingest` | Job sekali jalan, aman diulang; API tidak pernah menulis knowledge base |
| Role dari API key | `app/main.py` | Hak akses ditentukan dari `X-API-Key`, bukan dari isi request; key dibandingkan secara constant-time |
| Level guardrail ditentukan server | `GUARD_LEVEL` | Client tidak bisa mematikan guardrail |
| Liveness dan readiness | `/healthz`, `/readyz` | Readiness gagal bila database tidak terjangkau atau knowledge base kosong |
| Timeout dan retry | `app/engine.py`, `app/main.py` | Timeout LLM, retry untuk jawaban kosong, batas total per request (504) |
| Error tidak bocor ke client | `app/main.py` | Client menerima 502 atau 504; rinciannya hanya masuk log |
| Log JSON ber-`request_id` | `app/main.py` | Isi pertanyaan tidak dicatat, hanya panjangnya |
| Masking PII | `app/engine.py`, `rag/guards.py` | Pertanyaan di-mask sebelum retrieval dan sebelum dikirim ke LLM; jawaban di-mask sebelum dikembalikan |
| Metrik Prometheus | `/metrics` | Jumlah request, yang diblokir per flag, latensi, kemiripan tertinggi |
| Hardening container | `Dockerfile`, `docker-compose.yml` | Pengguna non-root, sistem file read-only, tanpa capability, batas CPU dan memori |
| Tanpa unduhan saat start | `Dockerfile` | Model embedding, tokenizer, dan data NLTK diambil saat build |
| Uji otomatis | `tests/test_api.py`, `.github/workflows/ci.yml` | API diuji dengan mock engine, tanpa database dan tanpa LLM |

## Yang masih kurang untuk production sungguhan

Daftar ini sengaja ditulis supaya batasnya jelas.

- PII di dalam dokumen tetap terkirim ke provider LLM sebagai konteks. Mask saat ingest, atau pakai model self-hosted.
- TLS dan rate limiting. Biasanya ditangani reverse proxy atau API gateway di depan layanan.
- API key disimpan sebagai teks di environment. Pakai secret manager, simpan hash-nya, dan sediakan rotasi.
- Skema database dibuat otomatis oleh LlamaIndex saat start. Untuk perubahan skema yang terkendali, pakai migration tool.
- Satu replika dan satu PostgreSQL tanpa backup. Belum ada replikasi atau disaster recovery.
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
