"""MCP server: exposes the document index as tools for other agents (Claude Desktop,
Claude Code, IDE agents...). Runs over stdio.

Example client config:
    {"mcpServers": {"ragdoc": {"command": "ragdoc", "args": ["mcp"]}}}
"""

from __future__ import annotations

from typing import Any

from ragdoc.agent import tools
from ragdoc.engine import RagEngine


def build_server(engine: RagEngine) -> Any:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("ragdoc")

    @server.tool()
    def list_documents() -> list[dict[str, Any]]:
        """List the indexed documents with their doc_id, title and page count."""
        return tools.list_documents(engine)

    @server.tool()
    def search_document(
        query: str, top_k: int = 8, doc_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Search the indexed documents (hybrid BM25 + vector search, reranked).

        Returns passages with source file, page numbers and section, so you can cite them.
        Optionally restrict to one doc_id from list_documents.
        """
        return tools.search_document(engine, query, top_k=top_k, doc_id=doc_id)

    @server.tool()
    def read_pages(doc_id: str, pages: list[int]) -> list[dict[str, Any]]:
        """Read the full text covering specific pages of a document, in reading order."""
        return tools.read_pages(engine, doc_id, pages)

    @server.tool()
    def ask_documents(question: str, doc_id: str | None = None) -> dict[str, Any]:
        """Answer a question with the full agentic RAG pipeline; returns a cited answer."""
        return tools.ask(engine, question, doc_id=doc_id)

    return server


def run(engine: RagEngine | None = None) -> None:
    build_server(engine or RagEngine()).run()


if __name__ == "__main__":
    run()
