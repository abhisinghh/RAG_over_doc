"""Structure- and modality-aware chunking.

Rules:
* prose chunks never cross a section boundary unless the running chunk is tiny;
* tables are their own chunks, split by row groups with the header repeated;
* figures, charts and video keyframes are their own chunks (one per image);
* code arrives pre-split by symbol and is kept as-is;
* audio/video transcripts are grouped into time windows (citations show timestamps);
* oversized paragraphs are split on sentence boundaries;
* consecutive prose chunks inside one section share a small sentence-level overlap.
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


def _split_table(text: str, max_tokens: int, max_rows: int = 10_000) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 3:
        return _split_oversized_text(text, max_tokens)
    header, rows = lines[:2], lines[2:]
    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join(header + current + [row])
        if current and (len(current) >= max_rows or count_tokens(candidate) > max_tokens):
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


ATOMIC_MODALITIES = {"image", "chart", "table", "code", "service"}


def _expand_blocks(blocks: list[Block], max_tokens: int, max_rows: int = 40) -> list[Block]:
    """Split any block larger than ``max_tokens`` (or a table longer than ``max_rows``)."""
    out: list[Block] = []
    for block in blocks:
        if block.kind == "table" or block.modality == "table":
            parts = _split_table(block.text, max_tokens, max_rows)
        elif block.modality in ("code", "image", "chart", "service"):
            parts = [block.text]  # already sized by their parsers
        elif count_tokens(block.text) <= max_tokens:
            parts = [block.text]
        else:
            parts = _split_oversized_text(block.text, max_tokens)
        out.extend(block.model_copy(update={"text": p}) for p in parts)
    return out


def _prose_groups(
    blocks: list[Block], chunk_tokens: int, overlap_tokens: int, min_chunk_tokens: int
) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0
    current_section: tuple[str, ...] | None = None

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            groups.append(current)
        current, current_tokens = [], 0

    for block in blocks:
        if block.modality in ATOMIC_MODALITIES or block.kind in ("figure", "symbol"):
            flush()
            groups.append([block])
            current_section = None
            continue
        section = tuple(block.heading_path[:2])  # top two heading levels define a section
        n = count_tokens(block.text)
        section_changed = current_section is not None and section != current_section
        too_big = current_tokens + n > chunk_tokens

        if current and ((section_changed and current_tokens >= min_chunk_tokens) or too_big):
            previous = current
            prev_text = "\n\n".join(b.text for b in previous)
            overlap = "" if section_changed else _tail_overlap(prev_text, overlap_tokens)
            flush()
            if overlap and count_tokens(overlap) + n <= chunk_tokens:
                # Overlap inherits the previous block's page so citations stay accurate.
                current.append(
                    Block(text=overlap, page=previous[-1].page, heading_path=block.heading_path)
                )
                current_tokens = count_tokens(overlap)
        current.append(block)
        current_tokens += n
        current_section = section
    flush()

    # Merge a tiny trailing prose group into its (prose) predecessor.
    if (
        len(groups) > 1
        and all(b.modality == "text" for b in groups[-1] + groups[-2])
        and sum(count_tokens(b.text) for b in groups[-1]) < min_chunk_tokens
    ):
        last = groups.pop()
        groups[-1] = groups[-1] + last
    return groups


def _time_groups(blocks: list[Block], window_s: float, chunk_tokens: int) -> list[list[Block]]:
    """Group transcript segments into windows of ~window_s seconds; frames stay separate."""
    groups: list[list[Block]] = []
    current: list[Block] = []
    tokens = 0
    for block in sorted(blocks, key=lambda b: (b.start_s or 0.0, b.kind != "segment")):
        if block.modality in ("image", "chart"):
            groups.append([block])
            continue
        n = count_tokens(block.text)
        start = (current[0].start_s or 0.0) if current else 0.0
        if current and ((block.end_s or 0.0) - start > window_s or tokens + n > chunk_tokens):
            groups.append(current)
            current, tokens = [], 0
        current.append(block)
        tokens += n
    if current:
        groups.append(current)
    return groups


def _chunk_from_group(doc: ParsedDocument, group: list[Block], position: int) -> Chunk:
    text = "\n\n".join(b.text for b in group).strip()
    pages = sorted(
        {p for b in group for p in (b.meta.get("pages") or ([b.page] if b.page else []))}
    )
    heading_path = next((b.heading_path for b in reversed(group) if b.heading_path), [])
    first = group[0]
    single = len(group) == 1
    if single:
        modality = first.modality
    elif doc.kind in ("audio", "video"):
        modality = doc.kind
    else:
        modality = "table" if any(b.modality == "table" for b in group) else "text"
    starts = [b.start_s for b in group if b.start_s is not None]
    ends = [b.end_s for b in group if b.end_s is not None]
    table_ids = [b.meta["table_id"] for b in group if b.meta.get("table_id")]
    metadata: dict[str, object] = {
        "tokens": count_tokens(text),
        "has_table": any(b.modality == "table" for b in group),
    }
    if table_ids:
        metadata["table_ids"] = table_ids
    if single:
        for key in (
            "symbol",
            "language",
            "service",
            "figure_kind",
            "title",
            "caption",
            "path",
            "kind",
            "calls",
            "docstring",
            "outline",
            "endpoints",
        ):
            if first.meta.get(key):
                metadata[key] = first.meta[key]
    return Chunk(
        chunk_id=f"{doc.doc_id}:{position:04d}",
        doc_id=doc.doc_id,
        source=doc.source,
        text=text,
        pages=pages,
        heading_path=list(heading_path),
        modality=modality,
        position=position,
        asset=first.asset if single else None,
        start_s=min(starts) if starts else None,
        end_s=max(ends) if ends else None,
        line_start=first.line_start if single else None,
        line_end=first.line_end if single else None,
        metadata=metadata,
    )


def chunk_document(
    doc: ParsedDocument,
    chunk_tokens: int = 600,
    overlap_tokens: int = 80,
    min_chunk_tokens: int = 40,
    rows_per_chunk: int = 40,
    window_seconds: float = 90,
) -> list[Chunk]:
    blocks = [b for b in doc.blocks if b.kind != "heading" and (b.text.strip() or b.asset)]
    if doc.kind in ("audio", "video"):
        groups = _time_groups(
            _expand_blocks(blocks, chunk_tokens, rows_per_chunk), window_seconds, chunk_tokens
        )
    else:
        groups = _prose_groups(
            _expand_blocks(blocks, chunk_tokens, rows_per_chunk),
            chunk_tokens,
            overlap_tokens,
            min_chunk_tokens,
        )
    chunks = [
        _chunk_from_group(doc, group, position)
        for position, group in enumerate(g for g in groups if g)
    ]
    return [c for c in chunks if c.text]
