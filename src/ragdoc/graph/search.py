"""Graph retrieval strategies, each returning evidence chunks for the shared generator.

* local:    entities matched by vector + name → neighbourhood relations → the text chunks
            that mention them (GraphRAG local search, LightRAG low-level keywords).
* global:   community reports ranked against the question (corpus-level themes).
* services: service dependency facts: calls in/out, async topic flows, datastores and
            transitive impact ("what breaks if X is down").
* code:     symbols named in the question → callers and callees → their code chunks.

Graph "fact" chunks cite the knowledge graph; the text units behind them cite pages, lines
or timestamps as usual.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence

from ragdoc.config import GraphConfig
from ragdoc.graph.communities import GRAPH_DOC_ID
from ragdoc.graph.store import GEdge, GraphStore, Node
from ragdoc.models.embeddings import Embedder
from ragdoc.retrieval.store import DocStore, SearchFilter, VectorStore
from ragdoc.types import Chunk, ScoredChunk

STRUCTURAL = {"file", "symbol", "module", "endpoint", "service", "topic", "datastore"}
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")


def _ngrams(text: str, max_n: int = 4) -> list[str]:
    words = re.findall(r"[\w.\-/]+", text)
    out = []
    for n in range(1, max_n + 1):
        out += [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return out


def _fact_chunk(kind: str, text: str, sources: list[str]) -> ScoredChunk:
    cid = f"{GRAPH_DOC_ID}:{kind}:{hashlib.sha256(text.encode()).hexdigest()[:10]}"
    chunk = Chunk(
        chunk_id=cid,
        doc_id=GRAPH_DOC_ID,
        source="knowledge graph",
        text=text,
        heading_path=[f"Graph {kind}"],
        metadata={"graph": kind, "source_ids": sources[:30]},
    )
    return ScoredChunk(chunk=chunk, score=1.0, source=f"graph_{kind}")


def entity_chunk(node: Node) -> Chunk:
    """Pseudo-chunk used to embed an entity for vector matching."""
    return Chunk(
        chunk_id=node.id,
        doc_id=GRAPH_DOC_ID,
        source="knowledge graph",
        text=f"{node.name} ({node.type}): {node.description}"[:2000],
        kind="entity",
    )


class GraphRetriever:
    def __init__(
        self,
        store: GraphStore,
        docs: DocStore,
        embedder: Embedder,
        entity_vectors: VectorStore,
        chunk_vectors: VectorStore,
        cfg: GraphConfig,
        final_k: int = 8,
    ) -> None:
        self.store = store
        self.docs = docs
        self.embedder = embedder
        self.entity_vectors = entity_vectors
        self.chunk_vectors = chunk_vectors
        self.cfg = cfg
        self.final_k = final_k

    # ---------------------------------------------------------------- indexing

    def index_entities(self, node_ids: Sequence[str]) -> None:
        nodes = [n for n in self.store.get_nodes(node_ids) if n.type not in STRUCTURAL]
        if not nodes:
            return
        chunks = [entity_chunk(n) for n in nodes]
        self.entity_vectors.upsert(chunks, self.embedder.embed([c.text for c in chunks]))

    # ---------------------------------------------------------------- helpers

    def match_entities(self, query: str, top: int | None = None) -> list[Node]:
        top = top or self.cfg.local_top_entities
        by_name = [n for n in self.store.find_nodes(_ngrams(query)) if n.type not in STRUCTURAL]
        hits = self.entity_vectors.search(self.embedder.embed([query])[0], top * 2)
        by_vector = self.store.get_nodes(i for i, _ in hits)
        merged = {n.id: n for n in [*by_name, *by_vector] if n.type not in STRUCTURAL}
        return list(merged.values())[:top]

    def _text_units(
        self, weighted_sources: Counter[str], doc_ids: Sequence[str] | None, limit: int
    ) -> list[ScoredChunk]:
        ranked = [cid for cid, _ in weighted_sources.most_common(limit * 3)]
        chunks = self.docs.get_chunks(ranked)
        out = []
        for cid in ranked:
            c = chunks.get(cid)
            if c is None or (doc_ids and c.doc_id not in doc_ids):
                continue
            out.append(
                ScoredChunk(chunk=c, score=float(weighted_sources[cid]), source="graph_text_unit")
            )
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _edge_line(e: GEdge, names: dict[str, str]) -> str:
        line = f"- {names.get(e.source, e.source)} —{e.relation}→ {names.get(e.target, e.target)}"
        if e.description:
            line += f": {e.description[:200]}"
        if e.data.get("evidence"):
            line += f" [{e.data['evidence']}]"
        return line

    # ---------------------------------------------------------------- strategies

    def local(self, query: str, doc_ids: Sequence[str] | None = None) -> list[ScoredChunk]:
        seeds = self.match_entities(query)
        if not seeds:
            return []
        seed_ids = [n.id for n in seeds]
        edges = self.store.edges_for(seed_ids, limit=self.cfg.max_relations)
        if self.cfg.max_hops > 1:
            frontier = {e.source for e in edges} | {e.target for e in edges}
            edges += self.store.edges_for(
                frontier - set(seed_ids), limit=self.cfg.max_relations // 2
            )
        neighbour_ids = {e.source for e in edges} | {e.target for e in edges}
        nodes = {n.id: n for n in self.store.get_nodes([*seed_ids, *neighbour_ids])}
        names = {nid: n.name for nid, n in nodes.items()}

        weights: Counter[str] = Counter()
        for n in seeds:
            weights.update({s: 2 for s in n.source_ids})
        for e in edges:
            weights.update(e.source_ids)

        entity_lines = "\n".join(f"- {n.name} [{n.type}]: {n.description[:300]}" for n in seeds)
        relation_lines = "\n".join(
            self._edge_line(e, names) for e in edges[: self.cfg.max_relations]
        )
        facts = _fact_chunk(
            "local",
            f"Knowledge-graph entities:\n{entity_lines}\n\nRelationships:\n{relation_lines or '(none)'}",
            [cid for cid, _ in weights.most_common(30)],
        )
        return [facts, *self._text_units(weights, doc_ids, self.final_k)]

    def global_(self, query: str) -> list[ScoredChunk]:
        hits = self.chunk_vectors.search(
            self.embedder.embed([query])[0],
            self.cfg.global_top_communities,
            SearchFilter.build(kinds=["community"]),
        )
        chunks = self.docs.get_chunks(cid for cid, _ in hits)
        return [
            ScoredChunk(chunk=chunks[cid], score=score, source="graph_global")
            for cid, score in hits
            if cid in chunks
        ]

    def services(self, query: str, max_depth: int = 3) -> list[ScoredChunk]:
        from ragdoc.code.services import _norm

        grams = _ngrams(query)
        candidates = grams + [_norm(g) for g in grams] + [g.replace(" ", "-") for g in grams]
        matched = self.store.find_nodes(candidates, types=["service", "topic", "datastore"])
        if not matched:
            # Fall back to the closest service summaries.
            hits = self.chunk_vectors.search(
                self.embedder.embed([query])[0], 4, SearchFilter.build(modalities=["service"])
            )
            ids = [cid for cid, _ in hits]
            chunks = self.docs.get_chunks(ids)
            return [
                ScoredChunk(chunk=chunks[c], score=s, source="service")
                for c, s in hits
                if c in chunks
            ]

        ids = [n.id for n in matched]
        relations = ["CALLS", "PUBLISHES", "SUBSCRIBES", "DEPENDS_ON", "EXPOSES"]
        edges = [
            e for e in self.store.edges_for(ids, relations, limit=300) if e.relation != "EXPOSES"
        ]
        # topic flows: publishers → topic → subscribers
        topics = {e.target for e in edges if e.relation in ("PUBLISHES", "SUBSCRIBES")} | {
            n.id for n in matched if n.type == "topic"
        }
        if topics:
            edges += self.store.edges_for(topics, ["PUBLISHES", "SUBSCRIBES"], limit=200)
        # transitive impact: who (directly or indirectly) calls the matched services
        impact: dict[str, int] = {}
        frontier = {n.id for n in matched if n.type in ("service", "datastore")}
        for depth in range(1, max_depth + 1):
            callers = {
                e.source
                for e in self.store.edges_for(frontier, ["CALLS", "DEPENDS_ON"], limit=500)
                if e.target in frontier
            }
            new = callers - set(impact) - {n.id for n in matched}
            for c in new:
                impact[c] = depth
            frontier = new
            if not frontier:
                break

        all_ids = {e.source for e in edges} | {e.target for e in edges} | set(impact)
        nodes = {n.id: n for n in self.store.get_nodes(all_ids | set(ids))}
        names = {nid: n.name for nid, n in nodes.items()}
        unique = list({e.key: e for e in edges}.values())
        lines = [f"Matched: {', '.join(f'{n.name} [{n.type}]' for n in matched)}", "Dependencies:"]
        lines += [self._edge_line(e, names) for e in unique[:80]]
        if impact:
            lines.append("Transitively affected if the matched component fails (hops):")
            lines += [
                f"- {names.get(i, i)} ({d})"
                for i, d in sorted(impact.items(), key=lambda kv: kv[1])
            ]
        sources = [s for n in matched for s in n.source_ids]
        facts = _fact_chunk("services", "\n".join(lines), sources)
        weights: Counter[str] = Counter(sources)
        return [facts, *self._text_units(weights, None, 4)]

    def code(self, query: str, doc_ids: Sequence[str] | None = None) -> list[ScoredChunk]:
        idents = [
            t.split(".")[-1]
            for t in _IDENT_RE.findall(query)
            if "_" in t
            or "." in t
            or (t[:1].islower() and any(ch.isupper() for ch in t))
            or t[:1].isupper()
        ]
        symbols = self.store.find_nodes(idents, types=["symbol"]) if idents else []
        if not symbols:
            return []
        ids = [n.id for n in symbols][:10]
        edges = self.store.edges_for(ids, ["CALLS"], limit=120)
        related = {e.source for e in edges} | {e.target for e in edges}
        nodes = {n.id: n for n in self.store.get_nodes(related | set(ids))}
        names = {
            nid: f"{n.data.get('qualname', n.name)} ({n.data.get('path', '')})"
            for nid, n in nodes.items()
        }
        lines = []
        for s in symbols[:10]:
            callers = sorted(
                {names[e.source] for e in edges if e.target == s.id and e.source in names}
            )
            callees = sorted(
                {names[e.target] for e in edges if e.source == s.id and e.target in names}
            )
            lines.append(
                f"- {names.get(s.id, s.name)}: called by {', '.join(callers[:15]) or 'nothing found'}; "
                f"calls {', '.join(callees[:15]) or 'nothing found'}"
            )
        weights: Counter[str] = Counter()
        for s in symbols:
            weights.update({c: 3 for c in s.source_ids})
        for nid in related:
            if nid in nodes:
                weights.update(nodes[nid].source_ids)
        facts = _fact_chunk("code", "Call graph:\n" + "\n".join(lines), list(weights))
        return [facts, *self._text_units(weights, doc_ids, self.final_k)]
