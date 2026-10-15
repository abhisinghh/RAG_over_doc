# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAGDOC_CONFIG=/app/configs/default.yaml \
    RAGDOC__INDEX_DIR=/data

WORKDIR /app

# ffmpeg: audio/video ingestion (transcription pieces, keyframes)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# EXTRAS=all adds Docling (layout-aware PDF parsing) and local models; much larger image.
ARG EXTRAS=api,ui,mcp,tracing,scale,neo4j

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[${EXTRAS}]"

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "--factory", "ragdoc.api.app:app_factory", "--host", "0.0.0.0", "--port", "8000"]
