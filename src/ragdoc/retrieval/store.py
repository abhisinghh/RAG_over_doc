"""Storage: chunk/document records (SQLite) and vectors (Qdrant or in-memory numpy).

SQLite is the source of truth for chunk text and metadata; the vector store only holds
vectors keyed by chunk id. That keeps vector backends swappable (Qdrant, pgvector, ...)
behind the small ``VectorStore`` protocol.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT doc_id, source, title, num_pages, num_chunks, config_fp, sha256 FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if not row:
            return None
        keys = ["doc_id", "source", "title", "num_pages", "num_chunks", "config_fp", "sha256"]
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

    def list_documents(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        sql = "SELECT doc_id, source, title, num_pages, num_chunks, ingested_at FROM documents ORDER BY ingested_at"
        if limit is not None:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        with self._lock:
            rows = self._db.execute(sql).fetchall()
        keys = ["doc_id", "source", "title", "num_pages", "num_chunks", "ingested_at"]
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def has_modality(self, modality: str) -> bool:
        pattern = f'%"modality":"{modality}"%'
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM chunks WHERE data LIKE ? LIMIT 1", (pattern,)
            ).fetchone()
        return row is not None

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


@dataclass(frozen=True)
class SearchFilter:
    """Metadata pre-filter applied inside the vector store (payload indexes at scale)."""

    doc_ids: tuple[str, ...] | None = None
    modalities: tuple[str, ...] | None = None
    kinds: tuple[str, ...] | None = None  # include only these chunk kinds
    exclude_kinds: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        doc_ids: Sequence[str] | None = None,
        modalities: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
        exclude_kinds: Sequence[str] = (),
    ) -> SearchFilter:
        return cls(
            tuple(doc_ids) if doc_ids else None,
            tuple(modalities) if modalities else None,
            tuple(kinds) if kinds else None,
            tuple(exclude_kinds),
        )

    def accepts(self, doc_id: str, kind: str, modality: str) -> bool:
        return (
            (self.doc_ids is None or doc_id in self.doc_ids)
            and (self.modalities is None or modality in self.modalities)
            and (self.kinds is None or kind in self.kinds)
            and kind not in self.exclude_kinds
        )


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self, vector: np.ndarray, k: int, flt: SearchFilter | None = None
    ) -> list[tuple[str, float]]: ...

    def delete_document(self, doc_id: str) -> None: ...


class InMemoryVectorStore:
    """Exact cosine search over a numpy matrix, persisted to .npz. Good for tests and small corpora."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._kinds: list[str] = []
        self._modalities: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        if path and path.exists():
            data = np.load(path, allow_pickle=False)
            self._matrix = data["matrix"]
            self._ids = [str(x) for x in data["ids"]]
            self._docs = [str(x) for x in data["docs"]]
            self._kinds = (
                [str(x) for x in data["kinds"]] if "kinds" in data else ["chunk"] * len(self._ids)
            )
            self._modalities = (
                [str(x) for x in data["modalities"]]
                if "modalities" in data
                else ["text"] * len(self._ids)
            )

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.path,
                matrix=self._matrix,
                ids=np.array(self._ids),
                docs=np.array(self._docs),
                kinds=np.array(self._kinds),
                modalities=np.array(self._modalities),
            )

    def _keep(self, keep: list[int]) -> None:
        self._matrix = self._matrix[keep] if self._matrix.size else self._matrix
        self._ids = [self._ids[i] for i in keep]
        self._docs = [self._docs[i] for i in keep]
        self._kinds = [self._kinds[i] for i in keep]
        self._modalities = [self._modalities[i] for i in keep]

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        with self._lock:
            self._upsert(chunks, vectors)

    def _upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        incoming = {c.chunk_id for c in chunks}
        if self._matrix.size and self._matrix.shape[1] != vectors.shape[1]:
            raise ValueError("Vector dimension changed; re-index into a new collection")
        self._keep([i for i, cid in enumerate(self._ids) if cid not in incoming])
        old = self._matrix if self._matrix.size else np.zeros((0, vectors.shape[1]), np.float32)
        self._matrix = np.vstack([old, vectors.astype(np.float32)])
        self._ids += [c.chunk_id for c in chunks]
        self._docs += [c.doc_id for c in chunks]
        self._kinds += [c.kind for c in chunks]
        self._modalities += [c.modality for c in chunks]
        self._save()

    def search(
        self, vector: np.ndarray, k: int, flt: SearchFilter | None = None
    ) -> list[tuple[str, float]]:
        with self._lock:
            if not self._ids:
                return []
            scores = self._matrix @ vector.astype(np.float32)
            if flt is not None:
                mask = [
                    flt.accepts(d, kd, m)
                    for d, kd, m in zip(self._docs, self._kinds, self._modalities, strict=True)
                ]
                scores = np.where(mask, scores, -np.inf)
            order = np.argsort(-scores)[:k]
            return [(self._ids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._keep([i for i, d in enumerate(self._docs) if d != doc_id])
            self._save()


class _SerializedClient:
    """Wraps an embedded Qdrant client so concurrent ingestion threads take turns
    (the local-mode engine is not designed for concurrent writers). Server clients are
    used directly."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return call


def make_qdrant_client(path: Path | None = None, url: str | None = None) -> Any:
    """One client per process: embedded Qdrant locks its folder, so stores share a client."""
    from qdrant_client import QdrantClient

    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"), prefer_grpc=True)
    assert path is not None
    path.mkdir(parents=True, exist_ok=True)
    return _SerializedClient(QdrantClient(path=str(path)))


class SparseEncoder:
    """BM25 sparse vectors computed client-side (fastembed), IDF applied by Qdrant."""

    def __init__(self, model: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model)

    def encode(self, texts: Sequence[str]) -> list[Any]:
        from qdrant_client import models

        return [
            models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.embed(list(texts))
        ]

    def encode_query(self, text: str) -> Any:
        from qdrant_client import models

        e = next(iter(self._model.query_embed(text)))
        return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


class QdrantVectorStore:
    """Qdrant collection. With ``sparse=True`` the same collection also holds BM25 sparse
    vectors and ``hybrid_search`` fuses dense + lexical server-side (RRF), so lexical search
    scales with the cluster instead of process memory."""

    DENSE = "dense"
    SPARSE = "bm25"

    def __init__(
        self,
        collection: str,
        client: Any = None,
        *,
        path: Path | None = None,
        url: str | None = None,
        sparse: bool = False,
        quantization: str = "none",
        on_disk: bool = False,
    ) -> None:
        self.collection = collection
        self._client = client or make_qdrant_client(path, url)
        self.sparse = sparse
        self.quantization = quantization
        self.on_disk = on_disk
        self._encoder: SparseEncoder | None = SparseEncoder() if sparse else None
        self._named = sparse  # named vectors are needed when a sparse vector is present
        if self._client.collection_exists(collection):
            info = self._client.get_collection(collection)
            vectors = info.config.params.vectors
            self._named = isinstance(vectors, dict)

    def _ensure(self, dim: int) -> None:
        from qdrant_client import models

        if self._client.collection_exists(self.collection):
            return
        params = models.VectorParams(
            size=dim, distance=models.Distance.COSINE, on_disk=self.on_disk
        )
        quant = (
            models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            )
            if self.quantization == "int8"
            else None
        )
        self._client.create_collection(
            self.collection,
            vectors_config={self.DENSE: params} if self._named else params,
            sparse_vectors_config=(
                {self.SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                if self.sparse
                else None
            ),
            quantization_config=quant,
        )
        for field in ("doc_id", "kind", "modality"):
            with contextlib.suppress(Exception):  # embedded mode has no payload indexes
                self._client.create_payload_index(
                    self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                )

    def _filter(self, flt: SearchFilter | None) -> Any:
        from qdrant_client import models

        if flt is None:
            return None
        must, must_not = [], []
        if flt.doc_ids:
            must.append(
                models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(flt.doc_ids)))
            )
        if flt.modalities:
            must.append(
                models.FieldCondition(
                    key="modality", match=models.MatchAny(any=list(flt.modalities))
                )
            )
        if flt.kinds:
            must.append(
                models.FieldCondition(key="kind", match=models.MatchAny(any=list(flt.kinds)))
            )
        if flt.exclude_kinds:
            must_not.append(
                models.FieldCondition(
                    key="kind", match=models.MatchAny(any=list(flt.exclude_kinds))
                )
            )
        return models.Filter(must=must or None, must_not=must_not or None)

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client import models

        if not chunks:
            return
        self._ensure(vectors.shape[1])
        sparse = self._encoder.encode([c.index_text for c in chunks]) if self._encoder else None
        points = []
        for i, c in enumerate(chunks):
            dense = vectors[i].tolist()
            if self._named:
                vector: Any = {self.DENSE: dense}
                if sparse is not None:
                    vector[self.SPARSE] = sparse[i]
            else:
                vector = dense
            points.append(
                models.PointStruct(
                    id=point_id(c.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "kind": c.kind,
                        "modality": c.modality,
                    },
                )
            )
        for start in range(0, len(points), 256):
            self._client.upsert(self.collection, points=points[start : start + 256])

    def search(
        self, vector: np.ndarray, k: int, flt: SearchFilter | None = None
    ) -> list[tuple[str, float]]:
        if not self._client.collection_exists(self.collection):
            return []
        result = self._client.query_points(
            self.collection,
            query=vector.tolist(),
            using=self.DENSE if self._named else None,
            limit=k,
            query_filter=self._filter(flt),
            with_payload=True,
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in result.points if p.payload]

    def hybrid_search(
        self, vector: np.ndarray, text: str, k: int, flt: SearchFilter | None = None
    ) -> list[tuple[str, float]]:
        """Dense + BM25 in one round trip, fused with RRF inside Qdrant."""
        from qdrant_client import models

        if not self._encoder or not self._client.collection_exists(self.collection):
            return self.search(vector, k, flt)
        qf = self._filter(flt)
        result = self._client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=vector.tolist(), using=self.DENSE, limit=k, filter=qf),
                models.Prefetch(
                    query=self._encoder.encode_query(text), using=self.SPARSE, limit=k, filter=qf
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
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
