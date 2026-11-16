"""Paket kecil untuk workshop RAG: chunking, embedding, pgvector, LLM, dan guardrail."""
import os

# Guardrails AI secara default mengirim metrik pemakaian anonim lewat OpenTelemetry.
# Untuk workshop ini pengirimannya dimatikan, kecuali pengguna sengaja mengaturnya sendiri.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
