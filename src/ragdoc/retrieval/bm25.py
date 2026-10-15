"""Lexical BM25 index over contextualized chunk text.

Catches exact terms dense vectors miss: IDs, codes, names, numbers and rare jargon.
Rebuilt in memory from the chunk store (fast for up to ~100k chunks).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from ragdoc.types import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "too",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "shall",
        "about",
        "over",
        "under",
        "between",
    ]
)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    out = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        # Light plural stemming keeps "policies"/"policy" and "documents"/"document" together.
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._ids = [c.chunk_id for c in chunks]
        self._docs = [c.doc_id for c in chunks]
        corpus = [tokenize(c.index_text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None

    def __len__(self) -> int:
        return len(self._ids)

    def search(
        self, query: str, k: int, doc_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        allowed = set(doc_ids) if doc_ids else None
        ranked = sorted(
            (
                (self._ids[i], float(s))
                for i, s in enumerate(scores)
                if s > 0 and (allowed is None or self._docs[i] in allowed)
            ),
            key=lambda pair: -pair[1],
        )
        return ranked[:k]
