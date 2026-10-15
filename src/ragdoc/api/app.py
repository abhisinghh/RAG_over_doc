"""REST API (FastAPI).

Endpoints
  GET    /health                 liveness + index stats
  GET    /documents              indexed documents
  POST   /documents              upload and ingest a file (multipart)
  DELETE /documents/{doc_id}     remove a document
  POST   /search                 hybrid search + rerank (no generation)
  POST   /ask                    cited answer (JSON)
  POST   /ask/stream             agent steps + final answer as Server-Sent Events
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ragdoc import __version__
from ragdoc.engine import RagEngine
from ragdoc.ingestion.parsers import SUPPORTED_SUFFIXES


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    doc_ids: list[str] | None = None
    debug: bool = False
    use_cache: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=8, ge=1, le=50)
    doc_ids: list[str] | None = None


def create_app(engine: RagEngine | None = None) -> FastAPI:
    engine = engine or RagEngine()
    ingest_lock = threading.Lock()
    app = FastAPI(title="ragdoc", version=__version__, description="Agentic RAG over documents")

    @app.get("/health")
    def health() -> dict[str, Any]:
        documents = engine.list_documents()
        return {
            "status": "ok",
            "version": __version__,
            "documents": len(documents),
            "agent": engine.settings.agent.enabled,
            "retrieval": engine.settings.retrieval.mode,
        }

    @app.get("/documents")
    def documents() -> list[dict[str, Any]]:
        return engine.list_documents()

    @app.post("/documents", status_code=201)
    def upload(file: Annotated[UploadFile, File()], force: bool = False) -> dict[str, Any]:
        name = Path(file.filename or "upload").name
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                415, f"Unsupported file type; supported: {sorted(SUPPORTED_SUFFIXES)}"
            )
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / name
            with target.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            try:
                with ingest_lock:
                    report = engine.ingest(target, force=force)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        return report.__dict__

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete(doc_id: str) -> None:
        engine.delete_document(doc_id)

    @app.post("/search")
    def search(req: SearchRequest) -> list[dict[str, Any]]:
        results = engine.search(req.query, top_k=req.top_k, doc_ids=req.doc_ids)
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "source": r.chunk.source,
                "pages": r.chunk.pages,
                "section": r.chunk.heading_path,
                "text": r.chunk.text,
                "score": r.score,
            }
            for r in results
        ]

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        result = engine.ask(req.question, doc_ids=req.doc_ids, use_cache=req.use_cache)
        return result.to_public(include_trace=req.debug)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        def events() -> Iterator[str]:
            try:
                for event in engine.stream(req.question, doc_ids=req.doc_ids):
                    yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
            except Exception as exc:  # surface errors to the client instead of a dropped stream
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def app_factory() -> FastAPI:
    """For ``uvicorn --factory ragdoc.api.app:app_factory``."""
    return create_app()
