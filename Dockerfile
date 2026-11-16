# Image utama workshop: aplikasi web, API, JupyterLab, dan skrip ingest.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    PYTHONPATH=/workshop

WORKDIR /workshop

COPY requirements.txt .
RUN pip install -r requirements.txt

# Unduh model embedding saat build, supaya tidak perlu internet ketika workshop berjalan.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Impor sekali saat build: validator Hub mengunduh data tokenizer NLTK pada impor pertama.
# Metrik pemakaian anonim Guardrails AI dimatikan lewat berkas konfigurasinya.
RUN printf 'enable_metrics=false\nuse_remote_inferencing=false\n' > /root/.guardrailsrc \
    && python -c "import guardrails_ai.valid_length, guardrails_ai.secrets_present"

COPY . .
