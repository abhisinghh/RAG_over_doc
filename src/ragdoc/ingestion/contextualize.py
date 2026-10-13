"""Contextual retrieval (Anthropic, 2024): prepend LLM-written context to each chunk.

The whole document goes in the (cached) system prompt and each chunk is sent as the user
message, so the long prefix is paid for once per document with prompt caching.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from ragdoc.ingestion.chunking import count_tokens
from ragdoc.models.llm import LLM, Message
from ragdoc.prompts import render
from ragdoc.types import Chunk, ParsedDocument

log = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")


def run_parallel(fn: Callable[[T], R], items: Iterable[T], workers: int = 8) -> list[R]:
    """Thread-pool map that preserves contextvars (so token usage reaches the active trace)."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, fn, item) for item in items]
        return [f.result() for f in futures]


def _document_window(doc: ParsedDocument, chunk: Chunk, max_tokens: int) -> str:
    """The full document, or a window around the chunk when the document is too long."""
    full = doc.full_text
    if count_tokens(full) <= max_tokens:
        return full
    max_chars = max_tokens * 4
    anchor = full.find(chunk.text[:120])
    anchor = anchor if anchor >= 0 else 0
    start = max(0, anchor - max_chars // 2)
    return ("[...]\n" if start else "") + full[start : start + max_chars] + "\n[...]"


def contextualize_chunks(
    llm: LLM,
    doc: ParsedDocument,
    chunks: list[Chunk],
    max_doc_tokens: int = 60_000,
    workers: int = 8,
) -> list[Chunk]:
    whole_doc_fits = count_tokens(doc.full_text) <= max_doc_tokens
    shared_system = (
        render("contextualize_system", title=doc.title, document=doc.full_text)
        if whole_doc_fits
        else None
    )

    def one(chunk: Chunk) -> Chunk:
        system = shared_system or render(
            "contextualize_system",
            title=doc.title,
            document=_document_window(doc, chunk, max_doc_tokens),
        )
        try:
            response = llm.generate(
                [Message(role="user", content=f"<passage>\n{chunk.text}\n</passage>")],
                system=system,
                role="fast",
                task="contextualize",
                cache_system=True,
            )
            context = response.text.strip()
        except Exception as exc:  # degrade gracefully: the chunk is still indexed
            log.warning("Contextualization failed for %s: %s", chunk.chunk_id, exc)
            context = ""
        return chunk.model_copy(update={"context": context})

    # Warm the prompt cache with one call before fanning out.
    if not chunks:
        return []
    first = one(chunks[0])
    return [first, *run_parallel(one, chunks[1:], workers)]
