"""Two-stage retriever: hybrid candidate generation (BM25 + dense, RRF) → rerank."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

import numpy as np

from ragdoc.config import RetrievalConfig
from ragdoc.guardrails import injection_score
from ragdoc.models.embeddings import Embedder
from ragdoc.retrieval.bm25 import BM25Index
from ragdoc.retrieval.hybrid import reciprocal_rank_fusion
from ragdoc.retrieval.rerank import Reranker
from ragdoc.retrieval.store import DocStore, VectorStore
from ragdoc.types import ScoredChunk

log = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        embedder: Embedder,
        vectors: VectorStore,
        docs: DocStore,
        reranker: Reranker,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.vectors = vectors
        self.docs = docs
        self.reranker = reranker
        self._bm25: BM25Index | None = None
        self._bm25_version = -1
        self._lock = threading.Lock()

    def _bm25_index(self) -> BM25Index:
        with self._lock:
            if self._bm25 is None or self._bm25_version != self.docs.version:
                self._bm25 = BM25Index(self.docs.all_chunks())
                self._bm25_version = self.docs.version
            return self._bm25

    def candidates(
        self,
        query: str,
        *,
        dense_text: str | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Stage 1: recall-oriented candidate set (no reranking)."""
        cfg = self.config
        rankings: list[list[tuple[str, float]]] = []
        detail: dict[str, int] = {}

        if cfg.mode in ("dense", "hybrid"):
            vec: np.ndarray = self.embedder.embed([dense_text or query])[0]
            dense = self.vectors.search(vec, cfg.dense_k, doc_ids)
            rankings.append(dense)
            detail["dense"] = len(dense)
            if dense_text and dense_text != query:
                # HyDE: also search with the raw query so a bad hypothesis cannot sink recall.
                plain = self.vectors.search(self.embedder.embed([query])[0], cfg.dense_k, doc_ids)
                rankings.append(plain)
        if cfg.mode in ("bm25", "hybrid"):
            lexical = self._bm25_index().search(query, cfg.bm25_k, doc_ids)
            rankings.append(lexical)
            detail["bm25"] = len(lexical)

        fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k)[: cfg.fused_k]
        chunks = self.docs.get_chunks(cid for cid, _ in fused)
        out = [
            ScoredChunk(chunk=chunks[cid], score=score, source="rrf")
            for cid, score in fused
            if cid in chunks
        ]
        log.debug("candidates for %r: %s", query, detail)
        return out

    def search(
        self,
        query: str,
        *,
        dense_text: str | None = None,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Full two-stage retrieval: candidates → injection screen → rerank → top_k."""
        top_k = top_k or self.config.final_k
        candidates = self.candidates(query, dense_text=dense_text, doc_ids=doc_ids)
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
