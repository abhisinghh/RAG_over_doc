"""Agent state and the structured outputs each node asks the model for."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from ragdoc.types import Citation, ScoredChunk

RouteName = Literal["simple", "multi_hop", "out_of_scope"]


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
    route: RouteName
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
