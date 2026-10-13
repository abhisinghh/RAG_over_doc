"""Structure-aware, token-sized chunking.

Rules:
* chunks never cross a section boundary unless the running chunk is tiny;
* tables stay whole (oversized tables are split by rows, header repeated);
* oversized paragraphs are split on sentence boundaries;
* consecutive chunks inside one section share a small sentence-level overlap;
* every chunk keeps its page numbers and heading path for citations and filtering.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

from ragdoc.types import Block, Chunk, ParsedDocument

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@lru_cache(maxsize=1)
def _token_counter() -> Callable[[str], int]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text, disallowed_special=()))
    except Exception:  # offline or not installed: ~0.75 words per token
        return lambda text: int(len(text.split()) * 1.33) + 1


def count_tokens(text: str) -> int:
    return _token_counter()(text)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _split_oversized_text(text: str, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in split_sentences(text) or [text]:
        n = count_tokens(sentence)
        if n > max_tokens:  # pathological sentence: hard split on words
            words = sentence.split()
            step = max(1, int(len(words) * max_tokens / n))
            for i in range(0, len(words), step):
                pieces.append(" ".join(words[i : i + step]))
            continue
        if current and size + n > max_tokens:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_table(text: str, max_tokens: int) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 3:
        return _split_oversized_text(text, max_tokens)
    header, rows = lines[:2], lines[2:]
    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join(header + current + [row])
        if current and count_tokens(candidate) > max_tokens:
            pieces.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join(header + current))
    return pieces


def _tail_overlap(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    tail: list[str] = []
    size = 0
    for sentence in reversed(split_sentences(text)):
        n = count_tokens(sentence)
        if size + n > overlap_tokens:
            break
        tail.insert(0, sentence)
        size += n
    return " ".join(tail)


def _expand_blocks(blocks: list[Block], max_tokens: int) -> list[Block]:
    """Split any block larger than ``max_tokens`` into several blocks."""
    out: list[Block] = []
    for block in blocks:
        if count_tokens(block.text) <= max_tokens:
            out.append(block)
            continue
        splitter = _split_table if block.kind == "table" else _split_oversized_text
        out.extend(
            block.model_copy(update={"text": part}) for part in splitter(block.text, max_tokens)
        )
    return out


def chunk_document(
    doc: ParsedDocument,
    chunk_tokens: int = 600,
    overlap_tokens: int = 80,
    min_chunk_tokens: int = 40,
) -> list[Chunk]:
    blocks = _expand_blocks([b for b in doc.blocks if b.kind != "heading"], chunk_tokens)

    groups: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0
    current_section: tuple[str, ...] | None = None

    for block in blocks:
        section = tuple(block.heading_path[:2])  # top two heading levels define a section
        n = count_tokens(block.text)
        section_changed = current_section is not None and section != current_section
        too_big = current_tokens + n > chunk_tokens

        if current and ((section_changed and current_tokens >= min_chunk_tokens) or too_big):
            groups.append(current)
            prev_text = "\n\n".join(b.text for b in current if b.kind != "table")
            overlap = "" if section_changed else _tail_overlap(prev_text, overlap_tokens)
            current, current_tokens = [], 0
            if overlap and count_tokens(overlap) + n <= chunk_tokens:
                # Overlap inherits the previous block's page so citations stay accurate.
                current.append(
                    Block(text=overlap, page=groups[-1][-1].page, heading_path=block.heading_path)
                )
                current_tokens = count_tokens(overlap)
        current.append(block)
        current_tokens += n
        current_section = section
    if current:
        groups.append(current)

    # Merge a tiny trailing group into its predecessor.
    if len(groups) > 1 and sum(count_tokens(b.text) for b in groups[-1]) < min_chunk_tokens:
        last = groups.pop()
        groups[-1] = groups[-1] + last

    chunks: list[Chunk] = []
    for position, group in enumerate(groups):
        text = "\n\n".join(b.text for b in group).strip()
        if not text:
            continue
        pages = sorted({b.page for b in group if b.page is not None})
        heading_path = next((b.heading_path for b in reversed(group) if b.heading_path), [])
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{position:04d}",
                doc_id=doc.doc_id,
                source=doc.source,
                text=text,
                pages=pages,
                heading_path=list(heading_path),
                position=position,
                metadata={
                    "tokens": count_tokens(text),
                    "has_table": any(b.kind == "table" for b in group),
                },
            )
        )
    return chunks
