"""Agent state and the structured outputs each node asks the model for."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from ragdoc.types import Citation, ScoredChunk

RouteName = Literal[
    "simple", "multi_hop", "global", "relational", "structured", "code", "out_of_scope"
]
ROUTE_DESCRIPTIONS: dict[str, str] = {
    "simple": "answerable from one passage or lookup (a definition, a figure, a single fact)",
    "multi_hop": "needs several facts combined, a comparison or a multi-part answer",
    "global": "asks about themes, trends or a summary across the whole collection",
    "relational": "asks how named entities (people, organisations, products, systems) relate or connect",
    "structured": "needs numbers computed from tables or charts (totals, averages, rankings, growth)",
    "code": "about source code, APIs, services, their dependencies or architecture",
    "out_of_scope": "greetings, chit-chat, or requests unrelated to the indexed content",
}


class AgentState(TypedDict, total=False):
    question: str
    doc_ids: list[str] | None
    route: RouteName
    sub_questions: list[str]
    current_query: str
    hyde_text: str | None
    queries_tried: list[str]
    evidence: list[ScoredChunk]
    rewrites: int
    answer: str
    citations: list[Citation]
    verification: dict[str, Any] | None
    revisions: int
    feedback: str | None
    abstained: bool


# --------------------------------------------------------------------------- structured outputs


class RouteDecision(BaseModel):
    route: str
    reason: str = ""


class Grade(BaseModel):
    id: int
    score: int = Field(ge=0, le=3)


class Grades(BaseModel):
    grades: list[Grade]


class Claim(BaseModel):
    claim: str
    supported: bool
    sources: list[int] = Field(default_factory=list)


class Verification(BaseModel):
    claims: list[Claim] = Field(default_factory=list)

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]

    @property
    def passed(self) -> bool:
        return not self.unsupported
