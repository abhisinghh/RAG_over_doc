"""Query transformations used by the agent: decomposition, rewriting and HyDE."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ragdoc.models.llm import LLM, structured
from ragdoc.prompts import render


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(description="Self-contained search questions")


class Rewrite(BaseModel):
    query: str = Field(description="Improved search query")
    hypothetical_passage: str = Field(
        default="", description="A short passage that would answer the question (HyDE)"
    )


def decompose(llm: LLM, question: str, max_sub_questions: int = 4) -> list[str]:
    """Split a multi-hop question into independent search questions (deduplicated)."""
    result = structured(
        llm,
        system=render("plan", max_sub_questions=max_sub_questions),
        user=question,
        schema=SubQuestions,
        role="fast",
        task="plan",
    )
    seen: set[str] = set()
    out: list[str] = []
    for q in result.sub_questions:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:max_sub_questions] or [question]


def rewrite_query(llm: LLM, question: str, tried: list[str]) -> Rewrite:
    """Reformulate a query that found weak evidence, plus a HyDE passage for dense search."""
    tried_text = "\n".join(f"- {q}" for q in tried) or "- (none)"
    return structured(
        llm,
        system=render("rewrite"),
        user=f"Question: {question}\n\nQueries already tried:\n{tried_text}",
        schema=Rewrite,
        role="fast",
        task="rewrite",
    )
