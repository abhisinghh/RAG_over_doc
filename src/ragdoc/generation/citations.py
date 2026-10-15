"""Numbered sources in, ``[n]`` citations out, mapped back to document pages."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragdoc.guardrails import neutralise
from ragdoc.types import Citation, ScoredChunk

ABSTAIN_TEXT = "The document does not contain enough information to answer this question."
_MARKER_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def format_sources(evidence: Sequence[ScoredChunk], max_chars: int = 4000) -> str:
    """Render evidence as numbered <source> blocks (1-based) for the generator prompt."""
    blocks = []
    for i, item in enumerate(evidence, start=1):
        c = item.chunk
        # No ">" or quotes inside attribute values, so the tag stays well-formed.
        section = " / ".join(c.heading_path).replace('"', "'").replace(">", " ")
        source = c.source.replace('"', "'")
        locator = c.locator.replace('"', "'")
        attrs = f'id="{i}" file="{source}" location="{locator}" type="{c.modality}"'
        if section:
            attrs += f' section="{section}"'
        if c.kind != "chunk":
            attrs += f' role="{c.kind}"'
        blocks.append(f"<source {attrs}>\n{neutralise(c.text[:max_chars])}\n</source>")
    return "\n\n".join(blocks)


def extract_markers(answer: str) -> list[int]:
    markers: list[int] = []
    for match in _MARKER_RE.finditer(answer):
        for part in re.split(r"[,;]", match.group(1)):
            n = int(part.strip())
            if n not in markers:
                markers.append(n)
    return markers


def strip_invalid_markers(answer: str, num_sources: int) -> str:
    """Remove citation markers that point at sources that do not exist."""

    def fix(match: re.Match[str]) -> str:
        valid = [p.strip() for p in re.split(r"[,;]", match.group(1)) if 1 <= int(p) <= num_sources]
        return f"[{', '.join(valid)}]" if valid else ""

    return _MARKER_RE.sub(fix, answer)


def build_citations(
    answer: str, evidence: Sequence[ScoredChunk], snippet_chars: int = 280
) -> list[Citation]:
    citations = []
    for n in extract_markers(answer):
        if 1 <= n <= len(evidence):
            c = evidence[n - 1].chunk
            snippet = (
                c.text[:snippet_chars].rsplit(" ", 1)[0] + "…"
                if len(c.text) > snippet_chars
                else c.text
            )
            citations.append(
                Citation(
                    marker=n,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    pages=c.pages,
                    snippet=snippet,
                    modality=c.modality,
                    locator=c.locator,
                    asset=c.asset,
                )
            )
    return sorted(citations, key=lambda x: x.marker)


def is_abstention(answer: str) -> bool:
    normalised = answer.strip().lower()
    return normalised.startswith(ABSTAIN_TEXT.lower()[:40]) or normalised in {
        "",
        "data not available",
    }
