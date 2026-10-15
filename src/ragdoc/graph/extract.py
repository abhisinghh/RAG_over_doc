"""LLM entity/relation extraction from text chunks (GraphRAG indexing step)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ragdoc.graph.store import GEdge, Node, node_id
from ragdoc.ingestion.contextualize import run_parallel
from ragdoc.models.llm import LLM, structured
from ragdoc.prompts import render
from ragdoc.types import Chunk

log = logging.getLogger(__name__)
SKIP_MODALITIES = {"code", "service"}  # these graphs are built deterministically
SKIP_KINDS = {"community", "doc_summary", "entity"}


class ExtractedEntity(BaseModel):
    name: str
    type: str = "concept"
    description: str = ""


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation: str = Field(default="related_to", description="short verb phrase, snake_case")
    description: str = ""
    strength: int = Field(default=5, ge=1, le=10)


class Extraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


def extract_chunk(llm: LLM, chunk: Chunk, entity_types: Sequence[str]) -> Extraction:
    return structured(
        llm,
        system=render("extract_graph", entity_types=", ".join(entity_types)),
        user=f"Source: {chunk.source} ({chunk.locator})\n\n{chunk.index_text}",
        schema=Extraction,
        role="fast",
        task="extract_graph",
    )


def to_graph(
    chunk: Chunk, extraction: Extraction, entity_types: Sequence[str]
) -> tuple[list[Node], list[GEdge]]:
    allowed = {t.lower() for t in entity_types}
    nodes: dict[str, Node] = {}
    for ent in extraction.entities:
        if not ent.name.strip():
            continue
        etype = ent.type.lower().strip() if ent.type.lower().strip() in allowed else "concept"
        nid = node_id(etype, ent.name)
        if nid in nodes:
            continue
        nodes[nid] = Node(
            id=nid,
            name=ent.name.strip(),
            type=etype,
            description=ent.description.strip(),
            source_ids=[chunk.chunk_id],
            doc_ids=[chunk.doc_id],
        )
    edges: list[GEdge] = []
    for rel in extraction.relations:
        src, tgt = node_id("concept", rel.source), node_id("concept", rel.target)
        for nid, name in ((src, rel.source), (tgt, rel.target)):
            if nid not in nodes and name.strip():  # relation mentions an entity not listed
                nodes[nid] = Node(
                    id=nid,
                    name=name.strip(),
                    type="concept",
                    source_ids=[chunk.chunk_id],
                    doc_ids=[chunk.doc_id],
                )
        if src == tgt:
            continue
        edges.append(
            GEdge(
                source=src,
                target=tgt,
                relation=(rel.relation or "related_to").strip().lower().replace(" ", "_")[:60],
                description=rel.description.strip(),
                weight=float(rel.strength),
                source_ids=[chunk.chunk_id],
            )
        )
    return list(nodes.values()), edges


def extract_graph(
    llm: LLM, chunks: Sequence[Chunk], entity_types: Sequence[str], workers: int = 8
) -> tuple[list[Node], list[GEdge]]:
    targets = [c for c in chunks if c.modality not in SKIP_MODALITIES and c.kind not in SKIP_KINDS]

    def one(chunk: Chunk) -> tuple[list[Node], list[GEdge]]:
        try:
            return to_graph(chunk, extract_chunk(llm, chunk, entity_types), entity_types)
        except Exception as exc:
            log.warning("Graph extraction failed for %s: %s", chunk.chunk_id, exc)
            return [], []

    nodes: list[Node] = []
    edges: list[GEdge] = []
    for n, e in run_parallel(one, targets, workers):
        nodes += n
        edges += e
    return nodes, edges
