"""Storage: chunk/document records (SQLite) and vectors (Qdrant or in-memory numpy).

SQLite is the source of truth for chunk text and metadata; the vector store only holds
vectors keyed by chunk id. That keeps vector backends swappable (Qdrant, pgvector, ...)
behind the small ``VectorStore`` protocol.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ragdoc.types import Chunk

_UUID_NS = uuid.UUID("6f0c3c55-7a3e-4b8e-9d0f-3c1a2b4d5e6f")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_UUID_NS, chunk_id))


# --------------------------------------------------------------------------- documents + chunks


class DocStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY, source TEXT, title TEXT, sha256 TEXT UNIQUE,
                num_pages INTEGER, num_chunks INTEGER, config_fp TEXT, ingested_at TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT, position INTEGER, data TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
        self._version = 0

    @property
    def version(self) -> int:
        """Bumped on every write in this process; used to rebuild the BM25 index lazily."""
        return self._version

    def fingerprint(self) -> str:
        """Hash of the indexed documents; part of the answer-cache key."""
        with self._lock:
            rows = self._db.execute(
                "SELECT doc_id, sha256, config_fp, ingested_at FROM documents ORDER BY doc_id"
            ).fetchall()
        return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]

    def find_by_sha(self, sha256: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT doc_id, source, title, num_pages, num_chunks, config_fp FROM documents WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        if not row:
            return None
        keys = ["doc_id", "source", "title", "num_pages", "num_chunks", "config_fp"]
        return dict(zip(keys, row, strict=True))

    def add_document(
        self,
        *,
        doc_id: str,
        source: str,
        title: str,
        sha256: str,
        num_pages: int | None,
        chunks: Sequence[Chunk],
        config_fp: str,
    ) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._db.execute(
                "DELETE FROM documents WHERE doc_id = ? OR sha256 = ?", (doc_id, sha256)
            )
            self._db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    source,
                    title,
                    sha256,
                    num_pages,
                    len(chunks),
                    config_fp,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            self._db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                [(c.chunk_id, c.doc_id, c.position, c.model_dump_json()) for c in chunks],
            )
            self._version += 1

    def delete_document(self, doc_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._version += 1

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT doc_id, source, title, num_pages, num_chunks, ingested_at FROM documents ORDER BY ingested_at"
            ).fetchall()
        keys = ["doc_id", "source", "title", "num_pages", "num_chunks", "ingested_at"]
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
        ids = list(dict.fromkeys(chunk_ids))
        out: dict[str, Chunk] = {}
        with self._lock:
            for start in range(0, len(ids), 500):
                part = ids[start : start + 500]
                rows = self._db.execute(
                    f"SELECT chunk_id, data FROM chunks WHERE chunk_id IN ({','.join('?' * len(part))})",
                    part,
                ).fetchall()
                out.update({cid: Chunk.model_validate_json(data) for cid, data in rows})
        return out

    def all_chunks(self, doc_ids: Sequence[str] | None = None) -> list[Chunk]:
        with self._lock:
            if doc_ids:
                rows = self._db.execute(
                    f"SELECT data FROM chunks WHERE doc_id IN ({','.join('?' * len(doc_ids))}) ORDER BY doc_id, position",
                    list(doc_ids),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT data FROM chunks ORDER BY doc_id, position"
                ).fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows]

    def pages(self, doc_id: str, pages: Sequence[int]) -> list[Chunk]:
        wanted = set(pages)
        return [
            c
            for c in self.all_chunks([doc_id])
            if c.kind == "chunk" and wanted.intersection(c.pages)
        ]


# --------------------------------------------------------------------------- vectors


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self, vector: np.ndarray, k: int, doc_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]: ...

    def delete_document(self, doc_id: str) -> None: ...


class InMemoryVectorStore:
    """Exact cosine search over a numpy matrix, persisted to .npz. Good for tests and small corpora."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        if path and path.exists():
            data = np.load(path, allow_pickle=False)
            self._matrix = data["matrix"]
            self._ids = [str(x) for x in data["ids"]]
            self._docs = [str(x) for x in data["docs"]]

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.path, matrix=self._matrix, ids=np.array(self._ids), docs=np.array(self._docs)
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        incoming = {c.chunk_id for c in chunks}
        keep = [i for i, cid in enumerate(self._ids) if cid not in incoming]
        old = (
            self._matrix[keep] if self._matrix.size else np.zeros((0, vectors.shape[1]), np.float32)
        )
        self._matrix = np.vstack([old, vectors.astype(np.float32)])
        self._ids = [self._ids[i] for i in keep] + [c.chunk_id for c in chunks]
        self._docs = [self._docs[i] for i in keep] + [c.doc_id for c in chunks]
        self._save()

    def search(
        self, vector: np.ndarray, k: int, doc_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        scores = self._matrix @ vector.astype(np.float32)
        if doc_ids:
            allowed = set(doc_ids)
            scores = np.where([d in allowed for d in self._docs], scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def delete_document(self, doc_id: str) -> None:
        keep = [i for i, d in enumerate(self._docs) if d != doc_id]
        self._matrix = self._matrix[keep] if self._matrix.size else self._matrix
        self._ids = [self._ids[i] for i in keep]
        self._docs = [self._docs[i] for i in keep]
        self._save()


class QdrantVectorStore:
    """Qdrant: embedded (local path) for development, server URL for production."""

    def __init__(self, collection: str, path: Path | None = None, url: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self.collection = collection
        if url:
            self._client = QdrantClient(url=url)
        else:
            assert path is not None
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))

    def _ensure(self, dim: int) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client import models

        if not chunks:
            return
        self._ensure(vectors.shape[1])
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector=vectors[i].tolist(),
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "kind": c.kind},
            )
            for i, c in enumerate(chunks)
        ]
        for start in range(0, len(points), 256):
            self._client.upsert(self.collection, points=points[start : start + 256])

    def search(
        self, vector: np.ndarray, k: int, doc_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        from qdrant_client import models

        if not self._client.collection_exists(self.collection):
            return []
        query_filter = (
            models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
            )
            if doc_ids
            else None
        )
        result = self._client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in result.points if p.payload]

    def delete_document(self, doc_id: str) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(self.collection):
            return
        self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
        )
