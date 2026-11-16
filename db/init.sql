-- Skema database workshop RAG. Aman dijalankan berulang kali.

CREATE EXTENSION IF NOT EXISTS vector;

-- Satu baris per dokumen sumber.
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    source_id     TEXT UNIQUE NOT NULL,      -- mis. nomor arXiv
    title         TEXT NOT NULL,
    authors       TEXT,
    url           TEXT,
    license       TEXT,
    access_level  TEXT NOT NULL DEFAULT 'public'   -- 'public' atau 'internal'
);

-- Satu baris per potongan teks, lengkap dengan vektornya.
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    section       TEXT,
    content       TEXT NOT NULL,
    embedding     vector(384) NOT NULL
);

-- Indeks HNSW untuk pencarian tetangga terdekat berdasarkan cosine distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Catatan setiap pertanyaan yang masuk, dipakai untuk monitoring dan deteksi anomali.
CREATE TABLE IF NOT EXISTS query_log (
    id              BIGSERIAL PRIMARY KEY,
    asked_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    question        TEXT NOT NULL,
    role            TEXT NOT NULL,
    top_similarity  REAL,
    flags           TEXT[] NOT NULL DEFAULT '{}',
    blocked         BOOLEAN NOT NULL DEFAULT FALSE,
    model           TEXT,
    latency_ms      INT
);

-- Nilai kecil yang perlu diingat antarproses, mis. threshold deteksi anomali.
CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
