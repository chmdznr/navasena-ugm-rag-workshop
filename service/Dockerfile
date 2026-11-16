# Image layanan RAG. Konteks build adalah akar repo:  docker build -f service/Dockerfile .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    PYTHONPATH=/srv

# Proses berjalan sebagai pengguna biasa, bukan root.
RUN useradd --create-home --uid 10001 app
WORKDIR /srv

COPY service/requirements.txt service/requirements.txt
RUN pip install -r service/requirements.txt

USER app
# Semua yang biasanya diunduh saat pertama dipakai diambil saat build, supaya container
# tidak butuh internet ketika start: model embedding, tokenizer, dan data NLTK.
RUN printf 'enable_metrics=false\nuse_remote_inferencing=false\n' > /home/app/.guardrailsrc \
    && python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.encoding_for_model('gpt-3.5-turbo')" \
    && python -c "import guardrails_ai.valid_length, guardrails_ai.secrets_present"

COPY --chown=app:app rag/ rag/
COPY --chown=app:app db/ db/
COPY --chown=app:app data/ data/
COPY --chown=app:app service/ service/

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz', timeout=3)"

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
