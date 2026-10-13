"""Ingestion pipeline: parse → (redact) → chunk → contextualize → (summarise) → embed → index.

Idempotent: a file whose content hash and ingestion config are unchanged is skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ragdoc.config import Settings
from ragdoc.guardrails import redact
from ragdoc.ingestion.chunking import chunk_document
from ragdoc.ingestion.contextualize import contextualize_chunks
from ragdoc.ingestion.parsers import file_sha256, parse_document
from ragdoc.ingestion.raptor import build_summary_tree
from ragdoc.models.embeddings import Embedder
from ragdoc.models.llm import LLM
from ragdoc.observability.tracing import TraceRecorder
from ragdoc.retrieval.store import DocStore, VectorStore
from ragdoc.types import Block

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    doc_id: str
    source: str
    num_chunks: int
    num_summaries: int
    num_pages: int | None
    skipped: bool
    seconds: float = 0.0
    redactions: dict[str, int] | None = None


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        embedder: Embedder,
        docs: DocStore,
        vectors: VectorStore,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.docs = docs
        self.vectors = vectors

    @property
    def config_fp(self) -> str:
        return self.settings.fingerprint("ingestion", "embeddings")

    def ingest(self, path: str | Path, force: bool = False) -> IngestReport:
        path = Path(path)
        cfg = self.settings.ingestion
        recorder = TraceRecorder("ingest")
        with recorder:
            existing = self.docs.find_by_sha(file_sha256(path))
            if existing and existing["config_fp"] == self.config_fp and not force:
                log.info("Skipping %s: already indexed as %s", path.name, existing["doc_id"])
                return IngestReport(
                    doc_id=existing["doc_id"],
                    source=existing["source"],
                    num_chunks=existing["num_chunks"],
                    num_summaries=0,
                    num_pages=existing["num_pages"],
                    skipped=True,
                )

            with recorder.span("parse", file=path.name) as d:
                doc = parse_document(path, parser=cfg.parser)
                d.update(blocks=len(doc.blocks), pages=doc.num_pages)

            redactions: dict[str, int] | None = None
            if cfg.redact_pii:
                with recorder.span("redact") as d:
                    redactions = {}
                    new_blocks: list[Block] = []
                    for block in doc.blocks:
                        text, counts = redact(block.text)
                        for k, v in counts.items():
                            redactions[k] = redactions.get(k, 0) + v
                        new_blocks.append(block.model_copy(update={"text": text}))
                    doc = doc.model_copy(update={"blocks": new_blocks})
                    d.update(redactions)

            with recorder.span("chunk") as d:
                chunks = chunk_document(
                    doc,
                    chunk_tokens=cfg.chunk_tokens,
                    overlap_tokens=cfg.chunk_overlap_tokens,
                    min_chunk_tokens=cfg.min_chunk_tokens,
                )
                d["chunks"] = len(chunks)

            if cfg.contextualize:
                with recorder.span("contextualize", chunks=len(chunks)):
                    chunks = contextualize_chunks(
                        self.llm, doc, chunks, max_doc_tokens=cfg.context_max_doc_tokens
                    )

            summaries = []
            if cfg.raptor and len(chunks) > 1:
                with recorder.span("raptor") as d:
                    summaries = build_summary_tree(self.llm, doc, chunks, cfg.raptor_group_size)
                    d["summaries"] = len(summaries)

            all_chunks = chunks + summaries
            with recorder.span("embed", texts=len(all_chunks)):
                vectors = self.embedder.embed([c.index_text for c in all_chunks])

            with recorder.span("index"):
                # Remove any previous version of this document (re-ingest with new config).
                if existing:
                    self.vectors.delete_document(existing["doc_id"])
                self.vectors.delete_document(doc.doc_id)
                self.vectors.upsert(all_chunks, vectors)
                self.docs.add_document(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    sha256=doc.sha256,
                    num_pages=doc.num_pages,
                    chunks=all_chunks,
                    config_fp=self.config_fp,
                )

        seconds = sum(e.latency_ms for e in recorder.events) / 1000
        log.info(
            "Ingested %s: %d chunks, %d summaries, %d LLM/embedding calls, %.1fs",
            doc.source,
            len(chunks),
            len(summaries),
            recorder.usage.calls,
            seconds,
        )
        return IngestReport(
            doc_id=doc.doc_id,
            source=doc.source,
            num_chunks=len(chunks),
            num_summaries=len(summaries),
            num_pages=doc.num_pages,
            skipped=False,
            seconds=round(seconds, 2),
            redactions=redactions,
        )
