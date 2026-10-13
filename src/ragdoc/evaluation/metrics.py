"""Retrieval and answer metrics (pure functions, no I/O)."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def _hits(retrieved_pages: Sequence[Sequence[int]], gold: set[int]) -> list[bool]:
    return [bool(gold.intersection(pages)) for pages in retrieved_pages]


def hit_at_k(retrieved_pages: Sequence[Sequence[int]], gold_pages: Sequence[int], k: int) -> float:
    gold = set(gold_pages)
    return float(any(_hits(retrieved_pages[:k], gold))) if gold else float("nan")


def page_recall_at_k(
    retrieved_pages: Sequence[Sequence[int]], gold_pages: Sequence[int], k: int
) -> float:
    gold = set(gold_pages)
    if not gold:
        return float("nan")
    covered = set().union(*[set(p) for p in retrieved_pages[:k]]) if retrieved_pages else set()
    return len(gold & covered) / len(gold)


def mrr(retrieved_pages: Sequence[Sequence[int]], gold_pages: Sequence[int]) -> float:
    gold = set(gold_pages)
    if not gold:
        return float("nan")
    for rank, hit in enumerate(_hits(retrieved_pages, gold), start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_pages: Sequence[Sequence[int]], gold_pages: Sequence[int], k: int) -> float:
    gold = set(gold_pages)
    if not gold:
        return float("nan")
    hits = _hits(retrieved_pages[:k], gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.fmean(clean) if clean else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return float("nan")
    idx = min(len(clean) - 1, max(0, math.ceil(q / 100 * len(clean)) - 1))
    return clean[idx]
