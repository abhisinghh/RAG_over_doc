"""Core data types shared across the pipeline.

Every piece of content, whatever its source, becomes a ``Chunk`` with a ``modality`` and a
*locator* (page, figure, timestamp range or file:line range), so retrieval, citation,
grading and evaluation work the same way for PDFs, slides, audio, video and code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

BlockKind = Literal["text", "heading", "table", "list", "caption", "figure", "segment", "symbol"]
Modality = Literal["text", "table", "image", "chart", "audio", "video", "code", "service"]
DocKind = Literal["document", "image", "audio", "video", "code", "service"]
# Structural role of an indexed unit (modality says what it *is*, kind says what it is *for*).
ChunkKind = Literal["chunk", "summary", "doc_summary", "community", "entity"]


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Block(BaseModel):
    """A structural unit produced by a parser (paragraph, table, figure, transcript segment...)."""

    text: str
    page: int | None = None
    kind: BlockKind = "text"
    modality: Modality = "text"
    heading_path: list[str] = Field(default_factory=list)
    asset: str | None = None  # relative path of an extracted image / keyframe
    start_s: float | None = None  # audio / video time range
    end_s: float | None = None
    line_start: int | None = None  # code line range (1-based, inclusive)
    line_end: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    doc_id: str
    source: str
    title: str
    blocks: list[Block]
    num_pages: int | None = None
    sha256: str
    kind: DocKind = "document"
    duration_s: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    text: str
    context: str = ""  # contextual-retrieval header, prepended for embedding and BM25
    pages: list[int] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    kind: ChunkKind = "chunk"
    modality: Modality = "text"
    level: int = 0  # 0 = leaf, >0 = summary / community level
    position: int = 0
    asset: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Text used for embeddings and BM25: context header + section path + body."""
        parts = []
        if self.context:
            parts.append(self.context)
        if self.heading_path:
            parts.append(" > ".join(self.heading_path))
        parts.append(self.text)
        return "\n".join(parts)

    @property
    def page_label(self) -> str:
        if not self.pages:
            return "n/a"
        lo, hi = min(self.pages), max(self.pages)
        return f"p. {lo}" if lo == hi else f"pp. {lo}-{hi}"

    @property
    def locator(self) -> str:
        """Human-readable position used in citations."""
        if self.line_start is not None:
            end = f"-{self.line_end}" if self.line_end and self.line_end != self.line_start else ""
            symbol = self.metadata.get("symbol")
            return f"lines {self.line_start}{end}" + (f" ({symbol})" if symbol else "")
        if self.start_s is not None:
            span = format_timestamp(self.start_s)
            if self.end_s is not None and self.end_s > self.start_s:
                span += f"-{format_timestamp(self.end_s)}"
            return ("frame at " if self.modality in ("image", "chart") else "") + span
        if self.modality in ("image", "chart"):
            label = "chart" if self.modality == "chart" else "figure"
            return f"{label}, {self.page_label}" if self.pages else label
        if self.kind == "community":
            return "community report"
        return self.page_label


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""  # which retriever / stage produced the score
    relevance: int | None = None  # 0-3 grade from the evidence grader


class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    source: str
    pages: list[int]
    snippet: str
    modality: Modality = "text"
    locator: str = ""
    asset: str | None = None


class TraceEvent(BaseModel):
    step: str
    latency_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1


class AnswerResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    route: str = "simple"
    sub_questions: list[str] = Field(default_factory=list)
    queries_tried: list[str] = Field(default_factory=list)
    evidence: list[ScoredChunk] = Field(default_factory=list)
    verification: dict[str, Any] | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    cached: bool = False

    def to_public(self, include_trace: bool = False) -> dict[str, Any]:
        data = self.model_dump(
            mode="json",
            exclude={"evidence"} if not include_trace else set(),
        )
        if not include_trace:
            data.pop("trace", None)
        return data
