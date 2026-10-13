"""Hierarchical communities and their LLM-written reports (GraphRAG "global" index).

Louvain (networkx) finds level-0 communities; communities larger than
``community_max_size`` are split again into level-1 sub-communities. Each community gets a
report (title, summary, key findings) that is indexed as a ``community`` chunk, so thematic
questions ("what are the main risks discussed across these filings?") retrieve corpus-level
summaries instead of scattered passages. For very large graphs, run Leiden in Neo4j GDS
(``CALL gds.leiden.write``) and load the result with ``set_communities``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ragdoc.config import GraphConfig
from ragdoc.graph.store import Community, GraphStore
from ragdoc.ingestion.contextualize import run_parallel
from ragdoc.models.llm import LLM, structured
from ragdoc.prompts import render
from ragdoc.types import Chunk

log = logging.getLogger(__name__)
GRAPH_DOC_ID = "__graph__"
STRUCTURAL_TYPES = {"file", "symbol", "module", "endpoint"}


def detect_communities(store: GraphStore, cfg: GraphConfig) -> list[Community]:
    import networkx as nx

    nodes = [n for n in store.all_nodes() if n.type not in STRUCTURAL_TYPES]
    keep = {n.id for n in nodes}
    graph = nx.Graph()
    graph.add_nodes_from(keep)
    for e in store.all_edges():
        if e.source in keep and e.target in keep and e.source != e.target:
            w = (
                graph[e.source][e.target]["weight"] + e.weight
                if graph.has_edge(e.source, e.target)
                else e.weight
            )
            graph.add_edge(e.source, e.target, weight=w)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        return []

    communities: list[Community] = []
    top = nx.community.louvain_communities(graph, weight="weight", resolution=1.0, seed=42)
    for i, members in enumerate(sorted(top, key=len, reverse=True)):
        if len(members) < cfg.community_min_size:
            continue
        cid = f"c0-{i:04d}"
        communities.append(Community(id=cid, level=0, node_ids=sorted(members)))
        if len(members) > cfg.community_max_size:
            sub = nx.community.louvain_communities(
                graph.subgraph(members), weight="weight", resolution=2.0, seed=42
            )
            for j, sub_members in enumerate(sorted(sub, key=len, reverse=True)):
                if len(sub_members) >= cfg.community_min_size and len(sub_members) < len(members):
                    communities.append(
                        Community(
                            id=f"c1-{i:04d}-{j:03d}",
                            level=1,
                            parent=cid,
                            node_ids=sorted(sub_members),
                        )
                    )
    log.info("Detected %d communities", len(communities))
    return communities


class CommunityReport(BaseModel):
    title: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    rating: float = Field(default=5.0, ge=0, le=10, description="Importance 0-10")


def write_reports(
    llm: LLM,
    store: GraphStore,
    communities: Sequence[Community],
    workers: int = 8,
    max_entities: int = 40,
) -> list[Community]:
    def one(community: Community) -> Community:
        members = set(community.node_ids)
        nodes = sorted(store.get_nodes(community.node_ids), key=lambda n: -len(n.source_ids))[
            :max_entities
        ]
        edges = [
            e
            for e in store.edges_for(community.node_ids, limit=400)
            if e.source in members and e.target in members
        ][:60]
        names = {n.id: n.name for n in nodes}
        entity_lines = "\n".join(f"- {n.name} [{n.type}]: {n.description[:300]}" for n in nodes)
        relation_lines = "\n".join(
            f"- {names.get(e.source, e.source)} —{e.relation}→ {names.get(e.target, e.target)}: {e.description[:200]}"
            for e in edges
        )
        try:
            report = structured(
                llm,
                system=render("community_report"),
                user=f"Entities:\n{entity_lines}\n\nRelationships:\n{relation_lines or '(none)'}",
                schema=CommunityReport,
                role="fast",
                task="community_report",
            )
        except Exception as exc:
            log.warning("Community report failed for %s: %s", community.id, exc)
            report = CommunityReport(
                title=", ".join(n.name for n in nodes[:3]), summary=entity_lines[:800]
            )
        sources = list(dict.fromkeys(s for n in nodes for s in n.source_ids))[:60]
        return community.model_copy(
            update={
                "title": report.title,
                "summary": report.summary,
                "findings": report.findings,
                "rating": report.rating,
                "source_ids": sources,
            }
        )

    return run_parallel(one, list(communities), workers)


def community_chunks(communities: Sequence[Community]) -> list[Chunk]:
    chunks = []
    for c in communities:
        findings = "\n".join(f"- {f}" for f in c.findings)
        chunks.append(
            Chunk(
                chunk_id=f"{GRAPH_DOC_ID}:{c.id}",
                doc_id=GRAPH_DOC_ID,
                source="knowledge graph",
                text=f"{c.title}\n{c.summary}"
                + (f"\nKey findings:\n{findings}" if findings else ""),
                heading_path=["Community report", f"level {c.level}"],
                kind="community",
                level=c.level,
                metadata={"community": c.id, "rating": c.rating, "source_ids": c.source_ids[:30]},
            )
        )
    return chunks
