"""Second-stage rerankers: retrieve wide, rerank to a narrow, high-precision top-k."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from ragdoc.config import RetrievalConfig, Secrets
from ragdoc.guardrails import neutralise
from ragdoc.models.llm import LLM, structured
from ragdoc.prompts import render
from ragdoc.types import ScoredChunk

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class NoReranker:
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(candidates[:top_k])


class _Ranking(BaseModel):
    ranked_ids: list[int] = Field(description="Passage ids, most useful first")


class LLMReranker:
    """Listwise reranking with the fast model. No extra dependencies or API keys."""

    def __init__(self, llm: LLM, max_chars: int = 900) -> None:
        self.llm = llm
        self.max_chars = max_chars

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if len(candidates) <= 1:
            return list(candidates)
        passages = "\n\n".join(
            f'<passage id="{i}">\n{neutralise(c.chunk.index_text[: self.max_chars])}\n</passage>'
            for i, c in enumerate(candidates)
        )
        try:
            ranking = structured(
                self.llm,
                system=render("rerank", top_k=top_k),
                user=f"Question: {query}\n\n{passages}",
                schema=_Ranking,
                role="fast",
                task="rerank",
            )
        except Exception as exc:
            log.warning("LLM rerank failed (%s); keeping fused order", exc)
            return list(candidates[:top_k])
        seen: set[int] = set()
        ordered: list[ScoredChunk] = []
        for rank, idx in enumerate(ranking.ranked_ids):
            if 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                ordered.append(
                    candidates[idx].model_copy(
                        update={"score": 1.0 / (rank + 1), "source": "llm_rerank"}
                    )
                )
        # Keep fused order for anything the model left out, so recall never drops below top_k.
        ordered += [c for i, c in enumerate(candidates) if i not in seen]
        return ordered[:top_k]


class CrossEncoderReranker:
    """Local cross-encoder, e.g. BAAI/bge-reranker-v2-m3 (pip install ragdoc[local])."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.index_text) for c in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -float(pair[1]))
        return [
            c.model_copy(update={"score": float(s), "source": "cross_encoder"})
            for c, s in ranked[:top_k]
        ]


class CohereReranker:
    def __init__(self, model: str, api_key: str | None) -> None:
        import cohere

        self._client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        resp = self._client.rerank(
            model=self.model,
            query=query,
            documents=[c.chunk.index_text for c in candidates],
            top_n=min(top_k, len(candidates)),
        )
        return [
            candidates[r.index].model_copy(
                update={"score": float(r.relevance_score), "source": "cohere"}
            )
            for r in resp.results
        ]


def build_reranker(config: RetrievalConfig, llm: LLM, secrets: Secrets | None = None) -> Reranker:
    secrets = secrets or Secrets()
    if config.reranker == "llm":
        return LLMReranker(llm)
    if config.reranker == "cross_encoder":
        return CrossEncoderReranker(config.cross_encoder_model)
    if config.reranker == "cohere":
        return CohereReranker(config.cohere_model, secrets.get("cohere_api_key"))
    return NoReranker()
