"""Retrieval exposed as plain tool functions (used by the MCP server and any external agent).

Each function returns JSON-serialisable data with page numbers so the calling agent can cite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragdoc.engine import RagEngine


def search_document(
    engine: RagEngine, query: str, top_k: int = 8, doc_id: str | None = None
) -> list[dict[str, Any]]:
    """Hybrid search + rerank over the indexed documents."""
    results = engine.search(query, top_k=top_k, doc_ids=[doc_id] if doc_id else None)
    return [
        {
            "chunk_id": r.chunk.chunk_id,
            "doc_id": r.chunk.doc_id,
            "source": r.chunk.source,
            "pages": r.chunk.pages,
            "section": " > ".join(r.chunk.heading_path),
            "text": r.chunk.text,
            "score": round(r.score, 4),
        }
        for r in results
    ]


def read_pages(engine: RagEngine, doc_id: str, pages: list[int]) -> list[dict[str, Any]]:
    """Full text of the chunks that cover the given pages, in reading order."""
    return [
        {"chunk_id": c.chunk_id, "pages": c.pages, "text": c.text}
        for c in engine.read_pages(doc_id, pages)
    ]


def list_documents(engine: RagEngine) -> list[dict[str, Any]]:
    return engine.list_documents()


def ask(engine: RagEngine, question: str, doc_id: str | None = None) -> dict[str, Any]:
    """Run the full agentic pipeline and return the cited answer."""
    result = engine.ask(question, doc_ids=[doc_id] if doc_id else None)
    return result.to_public()
