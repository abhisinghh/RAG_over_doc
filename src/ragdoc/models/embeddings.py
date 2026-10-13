"""Batched, disk-cached text embeddings (OpenAI, OpenAI-compatible, or local)."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from ragdoc.config import EmbeddingConfig, Secrets
from ragdoc.observability.tracing import add_usage


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class OpenAIEmbedder:
    def __init__(self, config: EmbeddingConfig, secrets: Secrets) -> None:
        from openai import OpenAI

        compatible = config.provider == "openai_compatible"
        key = secrets.get("llm_api_key") if compatible else secrets.get("openai_api_key")
        self._client = OpenAI(
            api_key=key or ("not-needed" if compatible else None), base_url=config.base_url
        )
        self.config = config
        self.model_name = (
            config.model if not config.dimensions else f"{config.model}@{config.dimensions}"
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = [
                t.replace("\n", " ").strip() or " "
                for t in texts[start : start + self.config.batch_size]
            ]
            kwargs = {"model": self.config.model, "input": batch}
            if self.config.dimensions:
                kwargs["dimensions"] = self.config.dimensions
            resp = self._client.embeddings.create(**kwargs)
            usage = getattr(resp, "usage", None)
            add_usage(getattr(usage, "prompt_tokens", 0) or 0, 0)
            out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return _normalise(np.array(out))


class LocalEmbedder:
    """sentence-transformers model, e.g. BAAI/bge-small-en-v1.5 (pip install ragdoc[local])."""

    def __init__(self, config: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.model)
        self.config = config
        self.model_name = config.model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), batch_size=self.config.batch_size, normalize_embeddings=True
        )
        return _normalise(np.asarray(vectors))


class CachedEmbedder:
    """Wraps an embedder with a SQLite cache keyed by (model, text) hash."""

    def __init__(self, inner: Embedder, cache_path: Path) -> None:
        self.inner = inner
        self.model_name = inner.model_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\x00{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                part = keys[start : start + 500]
                rows = self._db.execute(
                    f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part
                ).fetchall()
                found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            vectors = self.inner.embed([texts[i] for i in missing])
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
                    [
                        (keys[i], vectors[j].astype(np.float32).tobytes())
                        for j, i in enumerate(missing)
                    ],
                )
                self._db.commit()
            for j, i in enumerate(missing):
                found[keys[i]] = vectors[j]
        return np.stack([found[k] for k in keys]).astype(np.float32)


def build_embedder(
    config: EmbeddingConfig, secrets: Secrets | None = None, cache_path: Path | None = None
) -> Embedder:
    secrets = secrets or Secrets()
    inner: Embedder = (
        LocalEmbedder(config) if config.provider == "local" else OpenAIEmbedder(config, secrets)
    )
    return CachedEmbedder(inner, cache_path) if cache_path else inner
