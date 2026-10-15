"""HippoRAG 2 retrieval: Personalized PageRank seeded by recognised facts and dense passages.

Reference: Gutiérrez et al., "From RAG to Memory: Non-Parametric Continual Learning for
Large Language Models" (2025), arXiv:2502.14802. This is an independent implementation on
top of ragdoc's knowledge graph; default settings follow the authors' code
(github.com/OSU-NLP-Group/HippoRAG, MIT licence).

Indexing (:class:`HippoIndexer`, incremental, no LLM calls beyond graph extraction):

1. Every extracted entity ("phrase") is embedded by its name.
2. New phrases are linked to existing phrases by **synonym edges** when their name
   embeddings are at least ``synonym_threshold`` similar.
3. Every fact (subject, relation, object) is embedded as a sentence.

Retrieval (:class:`HippoRetriever`):

1. **Query-to-fact matching**: the ``linking_top_k`` facts closest to the question.
2. **Recognition memory**: an LLM keeps only the facts that help answer the question.
   If none survive, the caller falls back to ordinary retrieval.
3. **Seeding**: each phrase in a kept fact gets the fact's score, divided by the number of
   passages the phrase appears in (so very common phrases do not dominate), averaged over
   facts; the top ``linking_top_k`` phrases are kept. Passages are seeded with their
   min-max-normalised dense similarity times ``passage_node_weight``.
4. **Personalized PageRank** spreads relevance through facts, synonyms and context edges.
5. Passages are ranked by their PageRank score.

Deviation from the reference, for scale: only the ``passage_seed_k`` most similar passages
are seeded (the reference seeds every passage), and fact scores are raw cosine similarities
from the vector index rather than normalised over all facts.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ragdoc.config import PPRConfig
from ragdoc.graph.communities import GRAPH_DOC_ID
from ragdoc.graph.ppr import (
    PASSAGE_PREFIX,
    SYNONYM_RELATION,
    PPRGraph,
    build_ppr_graph,
    is_fact,
    is_phrase,
    passage_node_id,
    personalized_pagerank,
)
from ragdoc.graph.store import GEdge, GraphStore, Node
from ragdoc.models.embeddings import Embedder
from ragdoc.models.llm import LLM, structured
from ragdoc.prompts import render
from ragdoc.retrieval.retriever import DEFAULT_EXCLUDED_KINDS
from ragdoc.retrieval.store import DocStore, SearchFilter, VectorStore
from ragdoc.types import Chunk, ScoredChunk

log = logging.getLogger(__name__)

FACTS_SOURCE = "graph_facts"  # pinned evidence: bypasses the relevance grader
PASSAGE_SOURCE = "ppr"  # ordinary evidence: graded like any retrieved passage
_MIN_NAME_CHARS = 3  # names shorter than this (after removing symbols) get no synonym edges


# --------------------------------------------------------------------------- helpers


def fact_text(edge: GEdge, names: dict[str, str]) -> str:
    """A fact as a plain sentence, e.g. ``"Acme Corp acquired Widget Inc"``."""
    relation = edge.relation.replace("_", " ")
    return f"{names.get(edge.source, edge.source)} {relation} {names.get(edge.target, edge.target)}"


def _vector_chunk(chunk_id: str, text: str) -> Chunk:
    """Vector stores index Chunks; phrases and facts are stored as minimal ones."""
    return Chunk(
        chunk_id=chunk_id, doc_id=GRAPH_DOC_ID, source="knowledge graph", text=text, kind="entity"
    )


def _min_max(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.ones_like(values)
    return (values - low) / (high - low)


class _IndexedState:
    """Remembers which phrases and facts are already embedded, so indexing is incremental."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS phrases (id TEXT PRIMARY KEY);"
            "CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY);"
        )

    def missing(self, table: str, keys: Iterable[str]) -> list[str]:
        column = "id" if table == "phrases" else "key"
        keys = list(dict.fromkeys(keys))
        found: set[str] = set()
        with self._lock:
            for start in range(0, len(keys), 500):
                part = keys[start : start + 500]
                rows = self._db.execute(
                    f"SELECT {column} FROM {table} WHERE {column} IN ({','.join('?' * len(part))})",
                    part,
                ).fetchall()
                found.update(r[0] for r in rows)
        return [k for k in keys if k not in found]

    def mark(self, table: str, keys: Iterable[str]) -> None:
        with self._lock, self._db:
            self._db.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?)", [(k,) for k in keys])


# --------------------------------------------------------------------------- indexing


@dataclass
class IndexStats:
    phrases: int = 0
    facts: int = 0
    synonym_edges: int = 0


class HippoIndexer:
    """Embeds phrases and facts and adds synonym edges. Safe to call repeatedly."""

    def __init__(
        self,
        graph: GraphStore,
        embedder: Embedder,
        phrase_vectors: VectorStore,
        fact_vectors: VectorStore,
        config: PPRConfig,
        state_path: Path,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.phrase_vectors = phrase_vectors
        self.fact_vectors = fact_vectors
        self.config = config
        self._state = _IndexedState(state_path)

    def index(self, node_ids: Sequence[str] | None = None) -> IndexStats:
        """Index the given phrase nodes and their facts, or the whole graph if ``None``."""
        nodes = self.graph.get_nodes(node_ids) if node_ids is not None else self.graph.all_nodes()
        phrases = {n.id: n for n in nodes if is_phrase(n.id)}
        stats = IndexStats()

        new_ids = self._state.missing("phrases", phrases)
        if new_ids:
            new_phrases = [phrases[i] for i in new_ids]
            vectors = self.embedder.embed([n.name for n in new_phrases])
            self.phrase_vectors.upsert([_vector_chunk(n.id, n.name) for n in new_phrases], vectors)
            stats.synonym_edges = self._add_synonym_edges(new_phrases, vectors)
            self._state.mark("phrases", new_ids)
            stats.phrases = len(new_ids)

        edges = self._edges_of(list(phrases)) if node_ids is not None else self.graph.all_edges()
        facts = {e.key: e for e in edges if is_fact(e)}
        new_keys = self._state.missing("facts", facts)
        if new_keys:
            new_facts = [facts[k] for k in new_keys]
            endpoints = {i for e in new_facts for i in (e.source, e.target)}
            names = {n.id: n.name for n in self.graph.get_nodes(endpoints)}
            texts = [fact_text(e, names) for e in new_facts]
            self.fact_vectors.upsert(
                [_vector_chunk(e.key, t) for e, t in zip(new_facts, texts, strict=True)],
                self.embedder.embed(texts),
            )
            self._state.mark("facts", new_keys)
            stats.facts = len(new_keys)

        log.info(
            "HippoRAG index: %d phrases, %d facts, %d synonym edges added",
            stats.phrases,
            stats.facts,
            stats.synonym_edges,
        )
        return stats

    def _edges_of(self, node_ids: list[str], batch: int = 400) -> list[GEdge]:
        """All edges touching ``node_ids``, queried in batches (SQL parameter limits)."""
        edges: dict[str, GEdge] = {}
        for start in range(0, len(node_ids), batch):
            for edge in self.graph.edges_for(node_ids[start : start + batch], limit=1_000_000):
                edges[edge.key] = edge
        return list(edges.values())

    def _add_synonym_edges(self, phrases: Sequence[Node], vectors: np.ndarray) -> int:
        cfg = self.config
        best: dict[tuple[str, str], float] = {}
        for phrase, vector in zip(phrases, vectors, strict=True):
            if len(re.sub(r"[^A-Za-z0-9]", "", phrase.name)) < _MIN_NAME_CHARS:
                continue
            hits = self.phrase_vectors.search(vector, cfg.synonym_top_k + 1)
            for other_id, score in hits:
                if score < cfg.synonym_threshold:
                    break  # hits are sorted by similarity
                if other_id == phrase.id:
                    continue
                pair = tuple(sorted((phrase.id, other_id)))
                best[pair] = max(best.get(pair, 0.0), float(score))  # type: ignore[index]
        edges = [
            GEdge(
                source=a,
                target=b,
                relation=SYNONYM_RELATION,
                description="similar names",
                weight=score,
            )
            for (a, b), score in best.items()
        ]
        if edges:
            self.graph.upsert_edges(edges)
        return len(edges)


# --------------------------------------------------------------------------- retrieval


class _FactSelection(BaseModel):
    ids: list[int] = Field(
        default_factory=list, description="Ids of the relevant facts, most useful first"
    )


@dataclass
class _Fact:
    edge: GEdge
    text: str
    score: float


class HippoRetriever:
    """Query-time HippoRAG 2 search. Returns ``[]`` when no fact survives recognition, so
    the caller can fall back to ordinary retrieval."""

    def __init__(
        self,
        graph: GraphStore,
        docs: DocStore,
        embedder: Embedder,
        llm: LLM,
        phrase_vectors: VectorStore,
        fact_vectors: VectorStore,
        chunk_vectors: VectorStore,
        config: PPRConfig,
        final_k: int = 8,
    ) -> None:
        self.graph = graph
        self.docs = docs
        self.embedder = embedder
        self.llm = llm
        self.phrase_vectors = phrase_vectors
        self.fact_vectors = fact_vectors
        self.chunk_vectors = chunk_vectors
        self.config = config
        self.final_k = final_k
        self._cache: tuple[str, PPRGraph, dict[str, Node]] | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- public

    def search(self, query: str, doc_ids: Sequence[str] | None = None) -> list[ScoredChunk]:
        query_vector = self.embedder.embed([query])[0]
        facts = self._recognise(query, self._match_facts(query_vector))
        if not facts:
            return []
        graph, phrases = self._graph()
        phrase_seeds = self._phrase_seeds(facts, phrases)
        passage_seeds = self._passage_seeds(query_vector, doc_ids)
        if not phrase_seeds and not passage_seeds:
            return []

        graph = graph.with_extra_nodes(passage_seeds)
        reset = np.zeros(graph.num_nodes)
        for node_id, weight in {**phrase_seeds, **passage_seeds}.items():
            if node_id in graph.index:
                reset[graph.index[node_id]] += weight
        if reset.sum() <= 0:
            return []

        scores = personalized_pagerank(
            graph.num_nodes,
            graph.sources,
            graph.targets,
            graph.weights,
            reset,
            damping=self.config.damping,
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
        )
        return [self._facts_chunk(facts), *self._rank_passages(graph, scores, doc_ids)]

    # ---------------------------------------------------------------- steps

    def _match_facts(self, query_vector: np.ndarray) -> list[_Fact]:
        """Step 1: the facts whose sentence embedding is closest to the question."""
        hits = self.fact_vectors.search(query_vector, self.config.linking_top_k)
        edges = {e.key: e for e in self.graph.get_edges(key for key, _ in hits)}
        endpoints = {i for e in edges.values() for i in (e.source, e.target)}
        names = {n.id: n.name for n in self.graph.get_nodes(endpoints)}
        return [
            _Fact(edge=edges[key], text=fact_text(edges[key], names), score=float(score))
            for key, score in hits
            if key in edges  # facts of deleted documents are skipped
        ]

    def _recognise(self, query: str, facts: list[_Fact]) -> list[_Fact]:
        """Step 2 (recognition memory): keep only the facts that help answer the question."""
        if not facts or not self.config.recognition_memory:
            return facts
        candidates = "\n".join(f"{i}. {f.text}" for i, f in enumerate(facts))
        try:
            selection = structured(
                self.llm,
                system=render("recognize_facts", max_facts=self.config.linking_top_k),
                user=f"Question: {query}\n\nCandidate facts:\n{candidates}",
                schema=_FactSelection,
                role="fast",
                task="recognize_facts",
            )
        except Exception as exc:  # keep retrieval working if the filter fails
            log.warning("Recognition memory failed (%s); keeping all matched facts", exc)
            return facts
        kept = [facts[i] for i in dict.fromkeys(selection.ids) if 0 <= i < len(facts)]
        return kept[: self.config.linking_top_k]

    def _phrase_seeds(self, facts: list[_Fact], phrases: dict[str, Node]) -> dict[str, float]:
        """Step 3a: phrase weights from fact scores, damped by how common each phrase is."""
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for fact in facts:
            for node_id in (fact.edge.source, fact.edge.target):
                node = phrases.get(node_id)
                if node is None:
                    continue
                totals[node_id] += fact.score / max(1, len(node.source_ids))
                counts[node_id] += 1
        averaged = {node_id: totals[node_id] / counts[node_id] for node_id in totals}
        top = sorted(averaged.items(), key=lambda kv: -kv[1])[: self.config.linking_top_k]
        return {node_id: weight for node_id, weight in top if weight > 0}

    def _passage_seeds(
        self, query_vector: np.ndarray, doc_ids: Sequence[str] | None
    ) -> dict[str, float]:
        """Step 3b: passage weights from dense similarity, scaled by ``passage_node_weight``."""
        hits = self.chunk_vectors.search(
            query_vector,
            self.config.passage_seed_k,
            SearchFilter.build(doc_ids=doc_ids, exclude_kinds=DEFAULT_EXCLUDED_KINDS),
        )
        if not hits:
            return {}
        normalised = _min_max(np.asarray([score for _, score in hits], dtype=np.float64))
        weight = self.config.passage_node_weight
        return {
            passage_node_id(chunk_id): float(score) * weight
            for (chunk_id, _), score in zip(hits, normalised, strict=True)
            if score > 0
        }

    def _rank_passages(
        self, graph: PPRGraph, scores: np.ndarray, doc_ids: Sequence[str] | None
    ) -> list[ScoredChunk]:
        """Step 5: passages ordered by PageRank score."""
        passages = [
            (node_id.removeprefix(PASSAGE_PREFIX), float(scores[i]))
            for i, node_id in enumerate(graph.node_ids)
            if node_id.startswith(PASSAGE_PREFIX)
        ]
        passages.sort(key=lambda item: -item[1])
        allowed = set(doc_ids) if doc_ids else None
        # With a document filter, many top passages may be excluded, so look further down.
        candidates = passages if allowed else passages[: self.final_k * 4]
        chunks = self.docs.get_chunks(cid for cid, _ in candidates)
        ranked: list[ScoredChunk] = []
        for chunk_id, score in candidates:
            chunk = chunks.get(chunk_id)
            if chunk is None or (allowed and chunk.doc_id not in allowed):
                continue
            ranked.append(ScoredChunk(chunk=chunk, score=score, source=PASSAGE_SOURCE))
            if len(ranked) >= self.final_k:
                break
        return ranked

    def _facts_chunk(self, facts: list[_Fact]) -> ScoredChunk:
        lines = "\n".join(f"- {f.text}" for f in facts)
        sources = list(dict.fromkeys(s for f in facts for s in f.edge.source_ids))
        chunk = Chunk(
            chunk_id=f"{GRAPH_DOC_ID}:facts:{hashlib.sha256(lines.encode()).hexdigest()[:10]}",
            doc_id=GRAPH_DOC_ID,
            source="knowledge graph",
            text=f"Knowledge-graph facts linked to the question:\n{lines}",
            heading_path=["Graph facts"],
            metadata={"graph": "hipporag", "source_ids": sources[:30]},
        )
        return ScoredChunk(chunk=chunk, score=1.0, source=FACTS_SOURCE, relevance=3)

    # ---------------------------------------------------------------- graph cache

    def _graph(self) -> tuple[PPRGraph, dict[str, Node]]:
        """The PageRank graph, rebuilt only when the knowledge graph has changed."""
        with self._lock:
            revision = self.graph.revision()
            if self._cache is None or self._cache[0] != revision:
                nodes = [n for n in self.graph.all_nodes() if is_phrase(n.id)]
                edges = self.graph.all_edges()
                self._cache = (revision, build_ppr_graph(nodes, edges), {n.id: n for n in nodes})
            return self._cache[1], self._cache[2]
