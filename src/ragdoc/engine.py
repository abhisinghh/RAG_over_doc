"""RagEngine: the single entry point used by the CLI, API, UI, MCP server and evals.

It owns every store (chunks, vectors, assets, tables, graph) and exposes ingestion for
documents, images, audio, video, code repositories and batch corpora, plus the agentic
question-answering loop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ragdoc.agent.graph import build_graph, initial_state
from ragdoc.agent.nodes import AgentDeps
from ragdoc.agent.state import AgentState
from ragdoc.cache import AnswerCache
from ragdoc.config import Secrets, Settings, load_settings
from ragdoc.ingestion.assets import AssetStore
from ragdoc.ingestion.parsers import SUPPORTED_SUFFIXES
from ragdoc.ingestion.pipeline import IngestionPipeline, IngestReport
from ragdoc.ingestion.tables import TableStore
from ragdoc.models.embeddings import Embedder, build_embedder
from ragdoc.models.llm import LLM, build_llm
from ragdoc.observability.tracing import TraceRecorder, init_tracing
from ragdoc.retrieval.rerank import Reranker, build_reranker
from ragdoc.retrieval.retriever import HybridRetriever
from ragdoc.retrieval.store import (
    DocStore,
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStore,
    make_qdrant_client,
)
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
        transcriber_factory: Callable[[], Any] | None = None,
        clip: Any = None,
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

        # --- vector stores: chunks, graph entities, figure images (one Qdrant client)
        # Reuse the client of a shared Qdrant store (embedded Qdrant locks its folder).
        self._qdrant = getattr(vector_store, "_client", None)
        if self._qdrant is None and s.retrieval.vector_store == "qdrant" and vector_store is None:
            self._qdrant = make_qdrant_client(root / "qdrant", s.retrieval.qdrant_url)

        def store(name: str, sparse: bool = False) -> VectorStore:
            if self._qdrant is not None:
                return QdrantVectorStore(
                    name,
                    self._qdrant,
                    sparse=sparse,
                    quantization=s.retrieval.quantization,
                    on_disk=s.retrieval.on_disk,
                )
            return InMemoryVectorStore(root / f"{name}.npz")

        self.vectors = vector_store or store(
            s.collection, sparse=s.retrieval.lexical == "qdrant_sparse"
        )
        self.entity_vectors = store(f"{s.collection}__entities")
        self.clip = clip
        if self.clip is None and s.multimodal.image_embeddings:
            from ragdoc.models.image_embeddings import ClipEmbedder

            self.clip = ClipEmbedder(s.multimodal.image_embedding_model)
        self.image_vectors = store(f"{s.collection}__images") if self.clip is not None else None

        # --- record stores
        self.docs = DocStore(root / "docs.sqlite")
        self.assets = AssetStore(root / "assets")
        self.tables = TableStore(root / "tables.sqlite") if s.tables.sql else None
        self.graph = self._make_graph_store(secrets)

        self.reranker = reranker or build_reranker(s.retrieval, self.llm, secrets)
        self.retriever = HybridRetriever(
            s.retrieval,
            self.embedder,
            self.vectors,
            self.docs,
            self.reranker,
            image_vectors=self.image_vectors,
            clip=self.clip,
        )
        from ragdoc.graph.search import GraphRetriever

        self.graph_retriever = GraphRetriever(
            self.graph,
            self.docs,
            self.embedder,
            self.entity_vectors,
            self.vectors,
            s.graph,
            s.retrieval.final_k,
        )
        # HippoRAG 2 (Personalized PageRank) shares the graph store and adds two small indexes.
        from ragdoc.graph.hipporag import HippoIndexer, HippoRetriever

        phrase_vectors = store(f"{s.collection}__phrases")
        fact_vectors = store(f"{s.collection}__facts")
        self.ppr_indexer = HippoIndexer(
            self.graph,
            self.embedder,
            phrase_vectors,
            fact_vectors,
            s.graph.ppr,
            root / "ppr_index.sqlite",
        )
        self.ppr_retriever = HippoRetriever(
            self.graph,
            self.docs,
            self.embedder,
            self.llm,
            phrase_vectors,
            fact_vectors,
            self.vectors,
            s.graph.ppr,
            s.retrieval.final_k,
        )
        self.pipeline = IngestionPipeline(
            s,
            self.llm,
            self.embedder,
            self.docs,
            self.vectors,
            assets=self.assets,
            tables=self.tables,
            graph=self.graph,
            graph_retriever=self.graph_retriever,
            ppr_indexer=self.ppr_indexer,  # always indexed, so experiments can switch PPR on/off
            image_vectors=self.image_vectors,
            clip=self.clip,
            transcriber_factory=transcriber_factory,
        )
        self.cache = AnswerCache(root / "answers.sqlite") if s.cache.answers else None
        self._caps: tuple[int, list[str], bool] | None = None
        self._caps_lock = threading.Lock()

    def _make_graph_store(self, secrets: Secrets) -> Any:
        from ragdoc.graph.store import Neo4jGraphStore, SQLiteGraphStore

        g = self.settings.graph
        if g.store == "neo4j":
            if not g.neo4j_uri:
                raise ValueError("graph.store is neo4j but graph.neo4j_uri is not set")
            return Neo4jGraphStore(g.neo4j_uri, g.neo4j_user, secrets.get("neo4j_password"))
        return SQLiteGraphStore(self.settings.collection_dir / "graph.sqlite")

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        overlays: list[str | Path] | None = None,
        **kwargs: Any,
    ) -> RagEngine:
        return cls(load_settings(config_path, overlays), **kwargs)

    # ------------------------------------------------------------------ ingestion

    def ingest(self, path: str | Path, force: bool = False) -> IngestReport:
        """Any single file: document, image, audio, video or source file."""
        return self.pipeline.ingest(path, force=force)

    def ingest_path(self, path: str | Path, force: bool = False) -> list[IngestReport]:
        """A file, or every supported document/media file under a directory (sequential).
        For code repositories use ``ingest_repo``; for large corpora ``ingest_batch``."""
        path = Path(path)
        if not path.is_dir():
            return [self.ingest(path, force=force)]
        return [self.ingest(f, force=force) for f in self.discover_files(path)]

    @staticmethod
    def discover_files(root: Path) -> list[Path]:
        from ragdoc.ingestion.media import AUDIO_SUFFIXES, VIDEO_SUFFIXES

        suffixes = SUPPORTED_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES
        return sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in suffixes
            and not any(part.startswith(".") for part in p.parts)
        )

    def ingest_batch(
        self,
        paths: Sequence[Path],
        workers: int | None = None,
        shard: tuple[int, int] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, int]:
        from ragdoc.ingestion.batch import BatchIngestor

        return BatchIngestor(self).run(paths, workers=workers, shard=shard, on_progress=on_progress)

    def ingest_repo(
        self,
        path: str | Path,
        name: str | None = None,
        force: bool = False,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any:
        """A source repository (or a monorepo of microservices): code, configs, docs, code
        graph and service graph."""
        from ragdoc.code.ingest import ingest_repository

        return ingest_repository(self, Path(path), name, force=force, on_progress=on_progress)

    def build_graph_index(self) -> dict[str, Any]:
        """Detect communities, write their reports and index them for global search."""
        from ragdoc.graph.communities import (
            GRAPH_DOC_ID,
            community_chunks,
            detect_communities,
            write_reports,
        )

        ppr_stats = self.ppr_indexer.index()
        communities = detect_communities(self.graph, self.settings.graph)
        reports = write_reports(self.llm, self.graph, communities, self.settings.ingestion.workers)
        self.graph.set_communities(reports)
        chunks = community_chunks(reports)
        self.vectors.delete_document(GRAPH_DOC_ID)
        if chunks:
            self.vectors.upsert(chunks, self.embedder.embed([c.index_text for c in chunks]))
        self.docs.add_document(
            doc_id=GRAPH_DOC_ID,
            source="knowledge graph",
            title="Knowledge graph communities",
            sha256=f"graph-{time.time()}",
            num_pages=None,
            chunks=chunks,
            config_fp="graph",
        )
        self._caps = None
        return {"communities": len(reports), "hipporag": ppr_stats.__dict__, **self.graph.stats()}

    def list_documents(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        return self.docs.list_documents(limit, offset)

    def delete_document(self, doc_id: str) -> None:
        self.vectors.delete_document(doc_id)
        if self.image_vectors is not None:
            self.image_vectors.delete_document(doc_id)
        if self.tables is not None:
            self.tables.delete_document(doc_id)
        self.graph.delete_document(doc_id)
        self.assets.delete_document(doc_id)
        self.docs.delete_document(doc_id)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        modalities: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        return self.retriever.search(query, doc_ids=doc_ids, top_k=top_k, modalities=modalities)

    def graph_search(
        self, query: str, mode: str = "ppr", doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        """Graph retrieval only, for inspection: ``ppr`` (HippoRAG 2), ``local`` or ``global``."""
        if mode == "ppr":
            return self.ppr_retriever.search(query, doc_ids)
        if mode == "local":
            return self.graph_retriever.local(query, doc_ids)
        if mode == "global":
            return self.graph_retriever.global_(query)
        raise ValueError(f"Unknown graph search mode: {mode!r} (use ppr, local or global)")

    def read_pages(self, doc_id: str, pages: Sequence[int]) -> list[Chunk]:
        return self.docs.pages(doc_id, pages)

    # ------------------------------------------------------------------ answering

    def _capabilities(self) -> tuple[list[str], bool]:
        """Routes the index can serve, and whether the graph has entities (cached per corpus version)."""
        with self._caps_lock:
            if self._caps is not None and self._caps[0] == self.docs.version:
                return self._caps[1], self._caps[2]
            stats = self.graph.stats()
            structural = {"file", "symbol", "module", "endpoint", "service", "topic", "datastore"}
            has_entities = any(t not in structural for t in stats.get("nodes", {}))
            checks = {
                "simple": True,
                "multi_hop": True,
                "global": stats.get("communities", 0) > 0,
                "relational": has_entities,
                "structured": self.tables is not None and self.tables.count() > 0,
                "code": self.docs.has_modality("code") or self.docs.has_modality("service"),
            }
            routes = [r for r in self.settings.agent.routes if checks.get(r, False)]
            self._caps = (self.docs.version, routes, has_entities)
            return routes, has_entities

    def available_routes(self) -> list[str]:
        """Routes the router may pick, given what is actually indexed."""
        return self._capabilities()[0]

    def _deps(self, recorder: TraceRecorder) -> AgentDeps:
        routes, has_entities = self._capabilities()
        titles = [d["title"] or d["source"] for d in self.docs.list_documents(limit=20)]
        return AgentDeps(
            settings=self.settings,
            llm=self.llm,
            retriever=self.retriever,
            recorder=recorder,
            documents=titles,
            graph=self.graph_retriever,
            ppr=self.ppr_retriever if self.settings.graph.ppr.enabled else None,
            tables=self.tables,
            assets=self.assets,
            available_routes=routes,
            has_entities=has_entities,
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
