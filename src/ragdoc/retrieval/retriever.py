"""Two-stage retriever.

Stage 1 (recall): dense + lexical (in-process BM25, or Qdrant sparse BM25 fused server-side),
optional HyDE query, optional CLIP text→image search, fused with Reciprocal Rank Fusion.
Metadata filters (documents, modalities, chunk kinds) are pushed into the stores.
At very large scale, ``doc_routing`` first picks the most relevant documents from one
summary vector per document, then searches only their chunks.

Stage 2 (precision): prompt-injection screen, then reranking down to ``final_k``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

from ragdoc.config import RetrievalConfig
from ragdoc.guardrails import injection_score
from ragdoc.models.embeddings import Embedder
from ragdoc.retrieval.bm25 import BM25Index
from ragdoc.retrieval.hybrid import reciprocal_rank_fusion
from ragdoc.retrieval.rerank import Reranker
from ragdoc.retrieval.store import DocStore, SearchFilter, VectorStore
from ragdoc.types import ScoredChunk

log = logging.getLogger(__name__)
DEFAULT_EXCLUDED_KINDS = ("community", "doc_summary", "entity")


class HybridRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        embedder: Embedder,
        vectors: VectorStore,
        docs: DocStore,
        reranker: Reranker,
        image_vectors: VectorStore | None = None,
        clip: Any = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.vectors = vectors
        self.docs = docs
        self.reranker = reranker
        self.image_vectors = image_vectors
        self.clip = clip
        self._bm25: BM25Index | None = None
        self._bm25_version = -1
        self._lock = threading.Lock()

    def _bm25_index(self) -> BM25Index:
        with self._lock:
            if self._bm25 is None or self._bm25_version != self.docs.version:
                self._bm25 = BM25Index(self.docs.all_chunks())
                self._bm25_version = self.docs.version
            return self._bm25

    def route_documents(self, query: str, top_docs: int | None = None) -> list[str]:
        """Stage 0 for huge corpora: most relevant documents by their summary vectors."""
        hits = self.vectors.search(
            self.embedder.embed([query])[0],
            top_docs or self.config.doc_routing_top_docs,
            SearchFilter.build(kinds=["doc_summary"]),
        )
        chunks = self.docs.get_chunks(cid for cid, _ in hits)
        return list(dict.fromkeys(chunks[cid].doc_id for cid, _ in hits if cid in chunks))

    def candidates(
        self,
        query: str,
        *,
        dense_text: str | None = None,
        doc_ids: Sequence[str] | None = None,
        modalities: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Stage 1: recall-oriented candidate set (no reranking)."""
        cfg = self.config
        if cfg.doc_routing and not doc_ids and not kinds:
            routed = self.route_documents(query)
            doc_ids = routed or None
        flt = SearchFilter.build(
            doc_ids=doc_ids,
            modalities=modalities,
            kinds=kinds,
            exclude_kinds=() if kinds else DEFAULT_EXCLUDED_KINDS,
        )
        rankings: list[list[tuple[str, float]]] = []

        server_hybrid = cfg.lexical == "qdrant_sparse" and hasattr(self.vectors, "hybrid_search")
        if cfg.mode in ("dense", "hybrid"):
            vec = self.embedder.embed([dense_text or query])[0]
            if server_hybrid and cfg.mode == "hybrid":
                rankings.append(self.vectors.hybrid_search(vec, query, cfg.dense_k, flt))  # type: ignore[attr-defined]
            else:
                rankings.append(self.vectors.search(vec, cfg.dense_k, flt))
            if dense_text and dense_text != query:
                # HyDE: also search with the raw query so a bad hypothesis cannot sink recall.
                rankings.append(
                    self.vectors.search(self.embedder.embed([query])[0], cfg.dense_k, flt)
                )
        if cfg.mode == "bm25" or (cfg.mode == "hybrid" and not server_hybrid):
            rankings.append(self._bm25_index().search(query, cfg.bm25_k, flt))
        if (
            self.image_vectors is not None
            and self.clip is not None
            and (not modalities or {"image", "chart"} & set(modalities))
        ):
            try:
                img_vec = self.clip.embed_texts([query])[0]
                rankings.append(self.image_vectors.search(img_vec, 10, flt))
            except Exception as exc:  # image search is best-effort
                log.debug("Image search skipped: %s", exc)

        fused = reciprocal_rank_fusion([r for r in rankings if r], k=cfg.rrf_k)[: cfg.fused_k]
        chunks = self.docs.get_chunks(cid for cid, _ in fused)
        return [
            ScoredChunk(chunk=chunks[cid], score=score, source="rrf")
            for cid, score in fused
            if cid in chunks
        ]

    def search(
        self,
        query: str,
        *,
        dense_text: str | None = None,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        modalities: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Full two-stage retrieval: candidates → injection screen → rerank → top_k."""
        top_k = top_k or self.config.final_k
        candidates = self.candidates(
            query, dense_text=dense_text, doc_ids=doc_ids, modalities=modalities, kinds=kinds
        )
        if not candidates:
            return []
        # Demote passages that look like prompt-injection attempts (kept, but ranked last).
        clean = [c for c in candidates if injection_score(c.chunk.text) == 0]
        flagged = [
            c.model_copy(
                update={
                    "chunk": c.chunk.model_copy(
                        update={"metadata": {**c.chunk.metadata, "injection_flag": True}}
                    )
                }
            )
            for c in candidates
            if injection_score(c.chunk.text) > 0
        ]
        if flagged:
            log.warning("Demoted %d passage(s) with instruction-like text", len(flagged))
        return self.reranker.rerank(query, clean + flagged, top_k)
