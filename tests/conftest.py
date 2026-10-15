"""Offline test doubles: a deterministic embedder and a scripted LLM.

The scripted LLM answers each task (route, plan, grade, generate, verify...) with simple
rules, so the whole pipeline, including the LangGraph loop, runs without network access.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ragdoc.config import load_settings
from ragdoc.engine import RagEngine
from ragdoc.models.llm import LLMResponse, Message
from ragdoc.retrieval.bm25 import tokenize

FIXTURES = Path(__file__).parent / "fixtures"


class HashEmbedder:
    """Bag-of-words hashing embedder: similar wording → similar vectors."""

    model_name = "hash-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


_PASSAGE_RE = re.compile(r'<passage id="(\d+)">\n(.*?)\n</passage>', re.DOTALL)
_SOURCE_RE = re.compile(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)


def _overlap(a: str, b: str) -> int:
    return len(set(tokenize(a)) & set(tokenize(b)))


class ScriptedLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        role: str = "strong",
        task: str = "generate",
        json_mode: bool = False,
        cache_system: bool = False,
    ) -> LLMResponse:
        self.calls.append(task)
        user = messages[0].content
        text = getattr(self, f"_{task}", self._default)(user, messages)
        return LLMResponse(text=text, model="scripted", input_tokens=10, output_tokens=5)

    # ---- tasks
    def _default(self, user: str, messages: Sequence[Message]) -> str:
        return "ok"

    def _contextualize(self, user: str, messages: Sequence[Message]) -> str:
        return "From the Acme Remote Work Handbook."

    def _summarize(self, user: str, messages: Sequence[Message]) -> str:
        return "Summary: " + user[-200:]

    def _route(self, user: str, messages: Sequence[Message]) -> str:
        q = user.lower()
        if "joke" in q:
            route = "out_of_scope"
        elif " and " in q or "compare" in q:
            route = "multi_hop"
        else:
            route = "simple"
        return json.dumps({"route": route, "reason": "scripted"})

    def _plan(self, user: str, messages: Sequence[Message]) -> str:
        parts = [p.strip(" ?") + "?" for p in re.split(r"\band\b", user) if p.strip(" ?")]
        return json.dumps({"sub_questions": parts})

    def _question(self, user: str) -> str:
        match = re.search(r"Question: (.*)", user)
        return match.group(1) if match else user

    def _grade(self, user: str, messages: Sequence[Message]) -> str:
        q = self._question(user)
        grades = [
            {"id": int(pid), "score": 3 if _overlap(q, text) >= 2 else 0}
            for pid, text in _PASSAGE_RE.findall(user)
        ]
        return json.dumps({"grades": grades})

    def _rerank(self, user: str, messages: Sequence[Message]) -> str:
        q = self._question(user)
        scored = sorted(
            ((int(pid), _overlap(q, text)) for pid, text in _PASSAGE_RE.findall(user)),
            key=lambda pair: -pair[1],
        )
        return json.dumps({"ranked_ids": [pid for pid, _ in scored]})

    def _rewrite(self, user: str, messages: Sequence[Message]) -> str:
        return json.dumps({"query": self._question(user), "hypothetical_passage": ""})

    def _generate(self, user: str, messages: Sequence[Message]) -> str:
        sources = _SOURCE_RE.findall(user)
        if not sources:
            return "The document does not contain enough information to answer this question."
        sid, text = sources[0]
        first_sentence = re.split(r"(?<=[.!?])\s", text.strip())[0]
        return f"{first_sentence} [{sid}]"

    def _verify(self, user: str, messages: Sequence[Message]) -> str:
        answer = re.search(r"<answer>\n(.*?)\n</answer>", user, re.DOTALL)
        claim = answer.group(1) if answer else ""
        return json.dumps({"claims": [{"claim": claim, "supported": True, "sources": [1]}]})

    def _judge_correctness(self, user: str, messages: Sequence[Message]) -> str:
        return json.dumps({"score": 1.0, "reason": "scripted"})

    def _judge_faithfulness(self, user: str, messages: Sequence[Message]) -> str:
        return self._verify(user, messages)


def make_settings(tmp_path: Path, **overrides: object):
    base = {
        "index_dir": str(tmp_path / "index"),
        "ingestion": {"chunk_tokens": 120, "chunk_overlap_tokens": 0, "min_chunk_tokens": 5},
        "retrieval": {"vector_store": "memory", "reranker": "llm", "final_k": 4},
        "cache": {"answers": False, "embeddings": False},
    }
    for key, value in overrides.items():
        if isinstance(value, dict):
            base.setdefault(key, {}).update(value)  # type: ignore[union-attr]
        else:
            base[key] = value
    return load_settings(overrides=base, environ={})


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def engine(tmp_path: Path, llm: ScriptedLLM) -> RagEngine:
    eng = RagEngine(make_settings(tmp_path), llm=llm, embedder=HashEmbedder())
    eng.ingest(FIXTURES / "handbook.md")
    return eng
