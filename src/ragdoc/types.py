"""Core data types shared across the pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

BlockKind = Literal["text", "heading", "table", "list", "caption"]
ChunkKind = Literal["chunk", "summary"]


class Block(BaseModel):
    """A structural unit produced by a parser (paragraph, heading, table...)."""

    text: str
    page: int | None = None
    kind: BlockKind = "text"
    heading_path: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    doc_id: str
    source: str
    title: str
    blocks: list[Block]
    num_pages: int | None = None
    sha256: str

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
    level: int = 0  # 0 = leaf chunk, >0 = RAPTOR summary level
    position: int = 0
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
