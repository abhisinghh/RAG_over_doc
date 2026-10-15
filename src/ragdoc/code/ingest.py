"""Repository ingestion: code, configs, docs, the code graph and the service graph.

Designed for estates of hundreds of services and thousands of files:
* unchanged files (same hash + config) are skipped but still parsed for the graphs;
* files are processed in parallel; code needs no LLM calls (deterministic context);
* the service map is rebuilt from the whole repo each run (it is cheap and global);
* every service gets a summary chunk, plus paged system-overview chunks.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragdoc.code.graph import code_graph, service_graph
from ragdoc.code.parser import CodeFile
from ragdoc.code.repo import DOC_SUFFIXES, code_file_to_document, iter_repo_files
from ragdoc.code.services import ServiceMap, discover_services
from ragdoc.ingestion.contextualize import run_parallel
from ragdoc.models.llm import Message
from ragdoc.prompts import render
from ragdoc.types import Block, ParsedDocument

if TYPE_CHECKING:
    from ragdoc.engine import RagEngine

log = logging.getLogger(__name__)


@dataclass
class RepoReport:
    repo: str
    files: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    symbols: int = 0
    services: int = 0
    service_edges: int = 0
    topics: int = 0


def _summarise_services(engine: RagEngine, smap: ServiceMap) -> None:
    def one(name: str) -> tuple[str, str]:
        info = smap.services[name]
        endpoints = "; ".join(e.label for e in info.endpoints[:30]) or "none found"
        files = ", ".join(info.files[:25])
        try:
            response = engine.llm.generate(
                [
                    Message(
                        role="user",
                        content=render(
                            "service_summary",
                            name=name,
                            readme=info.readme[:2500] or "(none)",
                            endpoints=endpoints,
                            files=files or "(none)",
                        ),
                    )
                ],
                role="fast",
                task="service_summary",
            )
            return name, response.text.strip()
        except Exception as exc:
            log.warning("Service summary failed for %s: %s", name, exc)
            return name, ""

    for name, summary in run_parallel(
        one, sorted(smap.services), engine.settings.ingestion.workers
    ):
        smap.services[name].summary = summary


def service_document(smap: ServiceMap) -> ParsedDocument:
    blocks = [
        Block(
            text=page,
            kind="text",
            modality="service",
            heading_path=[smap.repo, "System overview"],
            meta={"service": "__overview__"},
        )
        for page in smap.overview()
    ]
    for name in sorted(smap.services):
        info = smap.services[name]
        blocks.append(
            Block(
                text=info.describe(smap.edges),
                kind="text",
                modality="service",
                heading_path=[smap.repo, "Services", name],
                meta={
                    "service": name,
                    "endpoints": [e.label for e in info.endpoints[:50]],
                    "path": info.root,
                },
            )
        )
    content = "\n".join(b.text for b in blocks)
    sha = hashlib.sha256(content.encode()).hexdigest()
    return ParsedDocument(
        doc_id=f"{smap.repo.lower()}-service-map",
        source=f"{smap.repo} service map",
        title=f"{smap.repo} service map",
        blocks=blocks,
        sha256=sha,
        kind="service",
    )


def ingest_repository(
    engine: RagEngine,
    root: Path,
    repo_name: str | None = None,
    force: bool = False,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> RepoReport:
    s = engine.settings
    root = root.resolve()
    repo = repo_name or root.name
    report = RepoReport(repo=repo)
    files = list(iter_repo_files(root, s.code))
    report.files = len(files)
    smap = discover_services(root, files, repo) if s.code.services else None

    parsed_files: list[tuple[CodeFile, str, str | None]] = []
    lock_progress = {"done": 0}

    def one(path: Path) -> None:
        rel = path.relative_to(root).as_posix()
        try:
            if path.suffix.lower() in DOC_SUFFIXES:
                r = engine.ingest(path, force=force)
                report.skipped += r.skipped
                report.indexed += not r.skipped
                return
            service = smap.service_for(rel) if smap else None
            doc, parsed = code_file_to_document(path, root, s.code, repo, service)
            existing = engine.docs.get_document(doc.doc_id)
            unchanged = (
                existing is not None
                and existing["sha256"] == doc.sha256
                and existing["config_fp"] == engine.pipeline.config_fp
            )
            if unchanged and not force:
                report.skipped += 1
            elif doc.blocks:
                engine.pipeline.ingest_parsed(doc, previous_doc_id=doc.doc_id)
                report.indexed += 1
            parsed_files.append((parsed, doc.doc_id, service))
        except Exception as exc:
            log.warning("Failed to ingest %s: %s", rel, exc)
            report.failed.append((rel, f"{type(exc).__name__}: {exc}"))
        finally:
            lock_progress["done"] += 1
            if on_progress:
                on_progress({"done": lock_progress["done"], "total": len(files), "path": rel})

    with ThreadPoolExecutor(max_workers=max(1, s.ingestion.workers)) as pool:
        list(pool.map(one, files))

    graph = engine.graph
    if graph is not None and parsed_files:
        chunks_by_doc: dict[str, list] = defaultdict(list)
        for c in engine.docs.all_chunks([doc_id for _, doc_id, _ in parsed_files]):
            chunks_by_doc[c.doc_id].append(c)
        nodes, edges = code_graph(
            [(p, doc_id, chunks_by_doc.get(doc_id, []), svc) for p, doc_id, svc in parsed_files]
        )
        graph.upsert_nodes(nodes)
        graph.upsert_edges(edges)
        report.symbols = sum(1 for n in nodes if n.type == "symbol")

    if smap is not None:
        if s.code.summarize_services:
            _summarise_services(engine, smap)
        doc = service_document(smap)
        existing = engine.docs.get_document(doc.doc_id)
        if existing is None or existing["sha256"] != doc.sha256 or force:
            engine.pipeline.ingest_parsed(doc, previous_doc_id=doc.doc_id)
        service_chunks: dict[str, list[str]] = defaultdict(list)
        for c in engine.docs.all_chunks([doc.doc_id]):
            service_chunks[str(c.metadata.get("service", ""))].append(c.chunk_id)
        if graph is not None:
            nodes, edges = service_graph(smap, service_chunks, doc.doc_id)
            graph.upsert_nodes(nodes)
            graph.upsert_edges(edges)
        report.services = len(smap.services)
        report.service_edges = len(smap.edges)
        report.topics = len(smap.topics)
    return report
