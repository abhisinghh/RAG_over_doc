"""LLM-as-a-judge: answer correctness against a reference, and faithfulness to sources."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ragdoc.agent.state import Verification
from ragdoc.generation.citations import format_sources, is_abstention
from ragdoc.models.llm import LLM, structured
from ragdoc.prompts import render
from ragdoc.types import ScoredChunk


class CorrectnessJudgement(BaseModel):
    score: float = Field(ge=0, le=1)
    reason: str = ""


def judge_correctness(llm: LLM, question: str, reference: str, answer: str) -> CorrectnessJudgement:
    return structured(
        llm,
        system=render("judge_correctness"),
        user=f"Question: {question}\n\nReference answer: {reference}\n\nSystem answer: {answer}",
        schema=CorrectnessJudgement,
        role="strong",
        task="judge_correctness",
    )


def judge_faithfulness(llm: LLM, answer: str, evidence: Sequence[ScoredChunk]) -> float | None:
    """Share of the answer's claims supported by the evidence it was given (RAGAS-style)."""
    if is_abstention(answer) or not evidence:
        return None
    result = structured(
        llm,
        system=render("verify"),
        user=f"<sources>\n{format_sources(evidence)}\n</sources>\n\n<answer>\n{answer}\n</answer>",
        schema=Verification,
        role="strong",
        task="judge_faithfulness",
    )
    if not result.claims:
        return None
    return sum(c.supported for c in result.claims) / len(result.claims)
