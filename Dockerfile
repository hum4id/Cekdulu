# syntax=docker/dockerfile:1.7
# ---------- builder: buat wheel dependensi ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheels .

# ---------- runtime ----------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CEKDULU_MODEL_DIR=/app/models/baseline \
    CEKDULU_DATABASE_URL=sqlite:////app/var/cekdulu.db
# Tesseract (+ data Bahasa Indonesia) sebagai OCR offline; libgl tidak perlu (opencv headless)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ind curl \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install /wheels/* pytesseract && rm -rf /wheels
COPY --chown=app:app models/baseline ./models/baseline
RUN mkdir -p /app/var && chown app:app /app/var
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1
CMD ["uvicorn", "cekdulu.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--no-access-log"]
# 1 worker per container: rate limiter & metrik Prometheus bersifat per-proses.
# Skala horizontal lewat replika container di belakang load balancer.
