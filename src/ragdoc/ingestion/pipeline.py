"""Ingestion pipeline for every modality.

    parse (document | image | audio | video | code)
      → figures: store images, VLM description + OCR, chart → data table
      → tables: stitch multi-page tables, load into SQLite for text-to-SQL
      → (redact PII) → chunk (modality-aware)
      → context headers (LLM for prose/tables/transcripts; deterministic for code)
      → (summary tree) → (document summary for doc routing)
      → embed + index (dense, optional sparse/BM25, optional CLIP image vectors)
      → (GraphRAG entity/relation extraction)

Idempotent: a file whose content hash and ingestion config are unchanged is skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragdoc.config import Settings
from ragdoc.guardrails import redact
from ragdoc.ingestion.assets import AssetStore
from ragdoc.ingestion.chunking import chunk_document, count_tokens
from ragdoc.ingestion.contextualize import contextualize_chunks
from ragdoc.ingestion.figures import IMAGE_KEY, describe_figures, store_figures
from ragdoc.ingestion.parsers import IMAGE_SUFFIXES, file_sha256, parse_document
from ragdoc.ingestion.raptor import build_summary_tree
from ragdoc.ingestion.tables import TableStore, merge_multipage_tables
from ragdoc.models.embeddings import Embedder
from ragdoc.models.llm import LLM, Message
from ragdoc.models.vision import FigureDescription
from ragdoc.observability.tracing import TraceRecorder
from ragdoc.prompts import render
from ragdoc.retrieval.store import DocStore, VectorStore
from ragdoc.types import Block, Chunk, ParsedDocument

if TYPE_CHECKING:
    from ragdoc.graph.hipporag import HippoIndexer
    from ragdoc.graph.search import GraphRetriever
    from ragdoc.graph.store import GraphStore
    from ragdoc.models.transcribe import Transcriber

log = logging.getLogger(__name__)
LLM_CONTEXT_MODALITIES = {"text", "table", "audio", "video"}


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
    kind: str = "document"
    figures: int = 0
    tables: int = 0
    entities: int = 0
    duration_s: float | None = None
    warnings: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        embedder: Embedder,
        docs: DocStore,
        vectors: VectorStore,
        *,
        assets: AssetStore | None = None,
        tables: TableStore | None = None,
        graph: GraphStore | None = None,
        graph_retriever: GraphRetriever | None = None,
        ppr_indexer: HippoIndexer | None = None,
        image_vectors: VectorStore | None = None,
        clip: Any = None,
        transcriber_factory: Callable[[], Transcriber] | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.docs = docs
        self.vectors = vectors
        self.assets = assets or AssetStore(settings.collection_dir / "assets")
        self.tables = tables
        self.graph = graph
        self.graph_retriever = graph_retriever
        self.ppr_indexer = ppr_indexer
        self.image_vectors = image_vectors
        self.clip = clip
        self._transcriber_factory = transcriber_factory
        self._transcriber: Transcriber | None = None

    @property
    def config_fp(self) -> str:
        s = self.settings
        data = s.model_dump(
            mode="json",
            include={"ingestion", "embeddings", "multimodal", "tables", "media", "code"},
        )
        data["graph"] = {"enabled": s.graph.enabled, "entity_types": s.graph.entity_types}
        data["ingestion"].pop("workers", None)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            if self._transcriber_factory is None:
                from ragdoc.models.transcribe import build_transcriber

                self._transcriber = build_transcriber(self.settings.media)
            else:
                self._transcriber = self._transcriber_factory()
        return self._transcriber

    # ------------------------------------------------------------------ parsing

    def parse(self, path: Path) -> ParsedDocument:
        from ragdoc.code.parser import detect_language
        from ragdoc.ingestion.media import AUDIO_SUFFIXES, VIDEO_SUFFIXES, parse_audio, parse_video

        suffix = path.suffix.lower()
        s = self.settings
        if suffix in AUDIO_SUFFIXES:
            return parse_audio(path, self.transcriber())
        if suffix in VIDEO_SUFFIXES:
            return parse_video(path, self.transcriber(), s.media)
        if suffix in {".pdf", ".docx", ".md", ".markdown", ".txt"} | IMAGE_SUFFIXES:
            return parse_document(
                path,
                parser=s.ingestion.parser,
                images=s.multimodal.describe_images or s.multimodal.image_embeddings,
                max_images=s.multimodal.max_images_per_doc,
            )
        if detect_language(path):
            from ragdoc.code.repo import code_file_to_document

            doc, _ = code_file_to_document(path, path.parent, s.code)
            return doc
        raise ValueError(f"Unsupported file type: {path.name}")

    def ingest(self, path: str | Path, force: bool = False) -> IngestReport:
        path = Path(path)
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
        recorder = TraceRecorder("ingest")
        with recorder, recorder.span("parse", file=path.name) as d:
            doc = self.parse(path)
            d.update(blocks=len(doc.blocks), kind=doc.kind)
        return self.ingest_parsed(
            doc, recorder=recorder, previous_doc_id=existing["doc_id"] if existing else None
        )

    # ------------------------------------------------------------------ stages

    def _figures(self, doc: ParsedDocument, recorder: TraceRecorder) -> tuple[ParsedDocument, int]:
        if not any(IMAGE_KEY in b.meta for b in doc.blocks):
            return doc, 0
        with recorder.span("figures") as d:
            doc = store_figures(doc, self.assets, self.settings.multimodal)
            n = sum(1 for b in doc.blocks if b.asset)
            d["stored"] = n

            def register_chart(block: Block, desc: FigureDescription) -> None:
                if self.tables is None or not self.settings.tables.sql:
                    return
                table_id = self.tables.add_table(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    page=block.page,
                    title=desc.title or "chart data",
                    markdown=desc.data_table,
                    origin="chart" if desc.is_chart else "document",
                )
                if table_id:
                    block.meta["table_id"] = table_id

            doc = describe_figures(
                self.llm,
                doc,
                self.assets,
                self.settings.multimodal,
                workers=self.settings.ingestion.workers,
                on_table=register_chart,
            )
        return doc, n

    def _tables(self, doc: ParsedDocument, recorder: TraceRecorder) -> tuple[ParsedDocument, int]:
        if not any(b.modality == "table" for b in doc.blocks):
            return doc, 0
        cfg = self.settings.tables
        with recorder.span("tables") as d:
            blocks = merge_multipage_tables(doc.blocks) if cfg.merge_multipage else doc.blocks
            count = 0
            if self.tables is not None and cfg.sql:
                updated = []
                for b in blocks:
                    if b.modality == "table":
                        title = " > ".join(b.heading_path[-2:]) or doc.title
                        table_id = self.tables.add_table(
                            doc_id=doc.doc_id,
                            source=doc.source,
                            page=b.page,
                            title=title,
                            markdown=b.text,
                        )
                        if table_id:
                            b = b.model_copy(update={"meta": {**b.meta, "table_id": table_id}})
                            count += 1
                    updated.append(b)
                blocks = updated
            d["tables"] = count
        return doc.model_copy(update={"blocks": blocks}), count

    def _context(
        self, doc: ParsedDocument, chunks: list[Chunk], recorder: TraceRecorder
    ) -> list[Chunk]:
        if doc.kind in ("code", "service"):
            from ragdoc.code.repo import code_context

            return [
                c.model_copy(update={"context": code_context(doc, c.metadata)})
                if c.modality == "code"
                else c
                for c in chunks
            ]
        if not self.settings.ingestion.contextualize:
            return chunks
        eligible = [c for c in chunks if c.modality in LLM_CONTEXT_MODALITIES]
        if not eligible:
            return chunks
        with recorder.span("contextualize", chunks=len(eligible)):
            done = contextualize_chunks(
                self.llm,
                doc,
                eligible,
                max_doc_tokens=self.settings.ingestion.context_max_doc_tokens,
                workers=self.settings.ingestion.workers,
            )
        by_id = {c.chunk_id: c for c in done}
        return [by_id.get(c.chunk_id, c) for c in chunks]

    def _doc_summary(self, doc: ParsedDocument, chunks: list[Chunk]) -> Chunk | None:
        text = "\n\n".join(c.text for c in chunks if c.kind == "chunk")
        words = text.split()
        excerpt = " ".join(words[:5000])
        response = self.llm.generate(
            [Message(role="user", content=render("doc_summary", title=doc.title, text=excerpt))],
            role="fast",
            task="doc_summary",
        )
        return Chunk(
            chunk_id=f"{doc.doc_id}:summary",
            doc_id=doc.doc_id,
            source=doc.source,
            text=f"{doc.title}\n{response.text.strip()}",
            kind="doc_summary",
            modality="text",
            metadata={"tokens": count_tokens(response.text)},
        )

    def ingest_parsed(
        self,
        doc: ParsedDocument,
        recorder: TraceRecorder | None = None,
        previous_doc_id: str | None = None,
    ) -> IngestReport:
        s = self.settings
        recorder = recorder or TraceRecorder("ingest")
        with recorder:
            # Drop the previous version first (vectors, tables, graph contributions); the new
            # version's tables are registered by the stages below.
            for old in sorted({previous_doc_id, doc.doc_id} - {None}):
                self._delete_vectors(old)  # type: ignore[arg-type]
            doc, n_figures = self._figures(doc, recorder)
            doc, n_tables = self._tables(doc, recorder)

            redactions: dict[str, int] | None = None
            if s.ingestion.redact_pii:
                with recorder.span("redact") as d:
                    redactions = {}
                    new_blocks: list[Block] = []
                    for block in doc.blocks:
                        if block.modality == "code":
                            new_blocks.append(block)
                            continue
                        text, counts = redact(block.text)
                        for k, v in counts.items():
                            redactions[k] = redactions.get(k, 0) + v
                        new_blocks.append(block.model_copy(update={"text": text}))
                    doc = doc.model_copy(update={"blocks": new_blocks})
                    d.update(redactions)

            with recorder.span("chunk") as d:
                chunks = chunk_document(
                    doc,
                    chunk_tokens=s.ingestion.chunk_tokens,
                    overlap_tokens=s.ingestion.chunk_overlap_tokens,
                    min_chunk_tokens=s.ingestion.min_chunk_tokens,
                    rows_per_chunk=s.tables.rows_per_chunk,
                    window_seconds=s.media.window_seconds,
                )
                d["chunks"] = len(chunks)
            if not chunks:
                raise ValueError(f"No indexable content in {doc.source}")

            chunks = self._context(doc, chunks, recorder)

            summaries: list[Chunk] = []
            if (
                s.ingestion.raptor
                and doc.kind in ("document", "audio", "video")
                and len(chunks) > 1
            ):
                with recorder.span("raptor") as d:
                    summaries = build_summary_tree(
                        self.llm,
                        doc,
                        [c for c in chunks if c.modality != "image"],
                        s.ingestion.raptor_group_size,
                    )
                    d["summaries"] = len(summaries)
            if s.ingestion.doc_summary and doc.kind != "code":
                with recorder.span("doc_summary"):
                    summary = self._doc_summary(doc, chunks)
                    if summary:
                        summaries.append(summary)

            all_chunks = chunks + summaries
            with recorder.span("embed", texts=len(all_chunks)):
                vectors = self.embedder.embed([c.index_text for c in all_chunks])

            with recorder.span("index"):
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

            image_chunks = [c for c in chunks if c.asset and c.modality in ("image", "chart")]
            if image_chunks and self.image_vectors is not None and self.clip is not None:
                with recorder.span("image_vectors", images=len(image_chunks)):
                    images = [self.assets.read(c.asset or "") for c in image_chunks]
                    self.image_vectors.upsert(image_chunks, self.clip.embed_images(images))

            entities = 0
            if s.graph.enabled and self.graph is not None and doc.kind not in ("code", "service"):
                from ragdoc.graph.extract import extract_graph

                with recorder.span("graph_extract") as d:
                    nodes, edges = extract_graph(
                        self.llm, chunks, s.graph.entity_types, s.ingestion.workers
                    )
                    self.graph.upsert_nodes(nodes)
                    self.graph.upsert_edges(edges)
                    node_ids = list(dict.fromkeys(n.id for n in nodes))
                    if self.graph_retriever is not None:
                        self.graph_retriever.index_entities(node_ids)
                    if self.ppr_indexer is not None:
                        stats = self.ppr_indexer.index(node_ids)
                        d.update(synonym_edges=stats.synonym_edges, facts=stats.facts)
                    entities = len(node_ids)
                    d.update(entities=entities, relations=len(edges))

        seconds = sum(e.latency_ms for e in recorder.events) / 1000
        log.info(
            "Ingested %s (%s): %d chunks, %d figures, %d tables, %d entities, %.1fs",
            doc.source,
            doc.kind,
            len(chunks),
            n_figures,
            n_tables,
            entities,
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
            kind=doc.kind,
            figures=n_figures,
            tables=n_tables,
            entities=entities,
            duration_s=doc.duration_s,
        )

    def _delete_vectors(self, doc_id: str) -> None:
        self.vectors.delete_document(doc_id)
        if self.image_vectors is not None:
            self.image_vectors.delete_document(doc_id)
        if self.tables is not None:
            self.tables.delete_document(doc_id)
        if self.graph is not None:
            self.graph.delete_document(doc_id)
