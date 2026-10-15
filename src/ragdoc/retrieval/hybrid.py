"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

RRF merges ranked lists using ranks only, so BM25 and cosine scores never need to be
calibrated against each other: score(d) = sum over lists of 1 / (k + rank(d)).
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    for weight, ranking in zip(weights, ranked_lists, strict=True):
        for rank, (item_id, _score) in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda pair: -pair[1])
