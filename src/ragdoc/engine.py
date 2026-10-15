"""RagEngine: the single entry point used by the CLI, API, UI, MCP server and evals."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ragdoc.agent.graph import build_graph, initial_state
from ragdoc.agent.nodes import AgentDeps
from ragdoc.agent.state import AgentState
from ragdoc.cache import AnswerCache
from ragdoc.config import Secrets, Settings, load_settings
from ragdoc.ingestion.parsers import SUPPORTED_SUFFIXES
from ragdoc.ingestion.pipeline import IngestionPipeline, IngestReport
from ragdoc.models.embeddings import Embedder, build_embedder
from ragdoc.models.llm import LLM, build_llm
from ragdoc.observability.tracing import TraceRecorder, init_tracing
from ragdoc.retrieval.rerank import Reranker, build_reranker
from ragdoc.retrieval.retriever import HybridRetriever
from ragdoc.retrieval.store import DocStore, InMemoryVectorStore, QdrantVectorStore, VectorStore
from ragdoc.types import AnswerResult, Chunk, ScoredChunk

log = logging.getLogger(__name__)
RECURSION_LIMIT = 50


class RagEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        secrets: Secrets | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        secrets = secrets or Secrets()
        s = self.settings
        root = s.collection_dir
        root.mkdir(parents=True, exist_ok=True)
        init_tracing(s.tracing.otlp_endpoint, s.tracing.service_name)

        self.llm = llm or build_llm(s.llm, secrets)
        self.embedder = embedder or build_embedder(
            s.embeddings,
            secrets,
            cache_path=root / "embeddings.sqlite" if s.cache.embeddings else None,
        )
        if vector_store is None:
            if s.retrieval.vector_store == "memory":
                vector_store = InMemoryVectorStore(root / "vectors.npz")
            else:
                vector_store = QdrantVectorStore(
                    s.collection, path=root / "qdrant", url=s.retrieval.qdrant_url
                )
        self.vectors = vector_store
        self.docs = DocStore(root / "docs.sqlite")
        self.reranker = reranker or build_reranker(s.retrieval, self.llm, secrets)
        self.retriever = HybridRetriever(
            s.retrieval, self.embedder, self.vectors, self.docs, self.reranker
        )
        self.pipeline = IngestionPipeline(s, self.llm, self.embedder, self.docs, self.vectors)
        self.cache = AnswerCache(root / "answers.sqlite") if s.cache.answers else None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        overlays: list[str | Path] | None = None,
        **kwargs: Any,
    ) -> RagEngine:
        return cls(load_settings(config_path, overlays), **kwargs)

    # ------------------------------------------------------------------ documents

    def ingest(self, path: str | Path, force: bool = False) -> IngestReport:
        return self.pipeline.ingest(path, force=force)

    def ingest_path(self, path: str | Path, force: bool = False) -> list[IngestReport]:
        """Ingest a file, or every supported file under a directory."""
        path = Path(path)
        files = (
            sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
            if path.is_dir()
            else [path]
        )
        return [self.ingest(f, force=force) for f in files]

    def list_documents(self) -> list[dict[str, Any]]:
        return self.docs.list_documents()

    def delete_document(self, doc_id: str) -> None:
        self.vectors.delete_document(doc_id)
        self.docs.delete_document(doc_id)

    def search(
        self, query: str, top_k: int | None = None, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        return self.retriever.search(query, doc_ids=doc_ids, top_k=top_k)

    def read_pages(self, doc_id: str, pages: Sequence[int]) -> list[Chunk]:
        return self.docs.pages(doc_id, pages)

    # ------------------------------------------------------------------ answering

    def _deps(self, recorder: TraceRecorder) -> AgentDeps:
        titles = [d["title"] or d["source"] for d in self.docs.list_documents()][:20]
        return AgentDeps(
            settings=self.settings,
            llm=self.llm,
            retriever=self.retriever,
            recorder=recorder,
            documents=titles,
        )

    def _cache_key(self, question: str, doc_ids: list[str] | None) -> str:
        return AnswerCache.key(
            question,
            doc_ids,
            self.docs.fingerprint(),
            self.settings.fingerprint("llm", "retrieval", "agent", "embeddings"),
        )

    def _result(self, state: AgentState, recorder: TraceRecorder, started: float) -> AnswerResult:
        return AnswerResult(
            question=state["question"],
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            abstained=state.get("abstained", False),
            route=state.get("route", "simple"),
            sub_questions=state.get("sub_questions", []),
            queries_tried=state.get("queries_tried", []),
            evidence=state.get("evidence", []),
            verification=state.get("verification"),
            trace=recorder.events,
            usage=recorder.usage,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def ask(
        self, question: str, doc_ids: list[str] | None = None, use_cache: bool = True
    ) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")
        key = self._cache_key(question, doc_ids)
        if use_cache and self.cache and (hit := self.cache.get(key)):
            return hit.model_copy(update={"cached": True})

        started = time.perf_counter()
        recorder = TraceRecorder("ask")
        graph = build_graph(self._deps(recorder))
        final: AgentState = graph.invoke(
            initial_state(question, doc_ids), {"recursion_limit": RECURSION_LIMIT}
        )
        result = self._result(final, recorder, started)
        if self.cache and result.route != "out_of_scope":
            self.cache.put(key, result)
        return result

    def stream(self, question: str, doc_ids: list[str] | None = None) -> Iterator[dict[str, Any]]:
        """Yield one event per agent step, then the final result (for SSE / live UIs)."""
        started = time.perf_counter()
        recorder = TraceRecorder("ask")
        graph = build_graph(self._deps(recorder))
        state: dict[str, Any] = dict(initial_state(question.strip(), doc_ids))
        for update in graph.stream(
            dict(state), {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"
        ):
            for node, delta in update.items():
                if not delta:
                    continue
                state.update(delta)
                event = recorder.events[-1] if recorder.events else None
                yield {
                    "type": "step",
                    "node": node,
                    "latency_ms": event.latency_ms if event else None,
                    "detail": _summarise_delta(delta),
                }
        result = self._result(state, recorder, started)  # type: ignore[arg-type]
        yield {"type": "result", "result": result.to_public(include_trace=True)}

    def batch(
        self, questions: Sequence[str], doc_ids: list[str] | None = None
    ) -> list[AnswerResult]:
        return [self.ask(q, doc_ids) for q in questions if q.strip()]


def _summarise_delta(delta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in delta.items():
        if key == "evidence":
            out["evidence"] = [
                {"chunk_id": e.chunk.chunk_id, "pages": e.chunk.pages, "relevance": e.relevance}
                for e in value
            ]
        elif key == "citations":
            out["citations"] = len(value)
        elif key in {
            "answer",
            "route",
            "sub_questions",
            "current_query",
            "rewrites",
            "abstained",
            "verification",
        }:
            out[key] = value
    return out
