"""Personalized PageRank over a phrase-and-passage graph.

This module is pure computation: no models, no I/O. It provides

* :func:`personalized_pagerank`, a weighted power-iteration solver for undirected graphs;
* :class:`PPRGraph` and :func:`build_ppr_graph`, which turn knowledge-graph nodes and edges
  into the graph HippoRAG 2 runs PageRank on.

Graph layout (HippoRAG 2):

* **phrase nodes**: extracted entities;
* **passage nodes**: the chunks those entities were extracted from;
* **fact edges** between phrases, weighted by how many chunks state the fact;
* **synonym edges** between phrases with near-identical embeddings, weighted by similarity;
* **context edges** between each phrase and every passage it appears in, weight 1.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ragdoc.graph.store import GEdge, Node

PHRASE_PREFIX = "entity:"
PASSAGE_PREFIX = "passage:"
SYNONYM_RELATION = "SYNONYM"


def is_phrase(node_id: str) -> bool:
    return node_id.startswith(PHRASE_PREFIX)


def passage_node_id(chunk_id: str) -> str:
    return PASSAGE_PREFIX + chunk_id


def is_fact(edge: GEdge) -> bool:
    """A fact edge links two extracted entities (not code/service structure, not synonymy)."""
    return is_phrase(edge.source) and is_phrase(edge.target) and edge.relation != SYNONYM_RELATION


def personalized_pagerank(
    num_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    reset: np.ndarray,
    *,
    damping: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Personalized PageRank on an undirected, weighted graph.

    At each step the walker follows an edge (chosen in proportion to its weight) with
    probability ``damping``, or jumps back to a node drawn from ``reset``. Nodes with no
    edges send their mass back to ``reset``. The result is a probability distribution.

    Args:
        num_nodes: number of nodes; node ids are ``0 .. num_nodes - 1``.
        sources, targets: edge endpoints (each undirected edge listed once).
        weights: positive edge weights, same length as ``sources``.
        reset: non-negative restart weights per node; normalised internally.
        damping: probability of following an edge rather than restarting.
        max_iterations, tolerance: power-iteration stopping rules (L1 change).

    Raises:
        ValueError: if ``reset`` has no positive entry or the inputs are inconsistent.
    """
    if not 0 < damping < 1:
        raise ValueError("damping must be strictly between 0 and 1")
    if not (len(sources) == len(targets) == len(weights)):
        raise ValueError("sources, targets and weights must have the same length")
    if len(reset) != num_nodes:
        raise ValueError("reset must have one entry per node")

    restart = np.clip(np.nan_to_num(np.asarray(reset, dtype=np.float64)), 0.0, None)
    total = restart.sum()
    if total <= 0:
        raise ValueError("reset must contain at least one positive weight")
    restart /= total

    # Undirected: every edge carries mass both ways.
    src = np.concatenate([sources, targets]).astype(np.int64)
    dst = np.concatenate([targets, sources]).astype(np.int64)
    w = np.concatenate([weights, weights]).astype(np.float64)

    strength = np.bincount(src, weights=w, minlength=num_nodes)
    dangling = strength == 0
    transition = np.zeros_like(w)
    connected = strength[src] > 0
    transition[connected] = w[connected] / strength[src][connected]

    scores = restart.copy()
    for _ in range(max_iterations):
        flow = np.bincount(dst, weights=scores[src] * transition, minlength=num_nodes)
        updated = damping * (flow + scores[dangling].sum() * restart) + (1 - damping) * restart
        converged = np.abs(updated - scores).sum() < tolerance
        scores = updated
        if converged:
            break
    return scores


@dataclass(frozen=True)
class PPRGraph:
    """Node index plus edge arrays, ready for :func:`personalized_pagerank`."""

    node_ids: list[str]
    index: dict[str, int]
    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    def with_extra_nodes(self, node_ids: Iterable[str]) -> PPRGraph:
        """Copy with additional isolated nodes (e.g. seeded passages that mention no entity)."""
        extra = [n for n in dict.fromkeys(node_ids) if n not in self.index]
        if not extra:
            return self
        ids = self.node_ids + extra
        return PPRGraph(
            ids, {n: i for i, n in enumerate(ids)}, self.sources, self.targets, self.weights
        )


def build_ppr_graph(nodes: Sequence[Node], edges: Sequence[GEdge]) -> PPRGraph:
    """Build the HippoRAG 2 graph from phrase nodes and their fact/synonym edges."""
    phrases = [n for n in nodes if is_phrase(n.id)]
    ids: list[str] = [n.id for n in phrases]
    index: dict[str, int] = {n: i for i, n in enumerate(ids)}
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    def node(node_id: str) -> int:
        if node_id not in index:
            index[node_id] = len(ids)
            ids.append(node_id)
        return index[node_id]

    def add(a: str, b: str, weight: float) -> None:
        if a != b and weight > 0:
            sources.append(node(a))
            targets.append(node(b))
            weights.append(weight)

    # Context edges: phrase ↔ every passage (chunk) it was extracted from.
    for phrase in phrases:
        for chunk_id in phrase.source_ids:
            add(phrase.id, passage_node_id(chunk_id), 1.0)

    known = set(index)
    for edge in edges:
        if edge.source not in known or edge.target not in known:
            continue
        if edge.relation == SYNONYM_RELATION:
            add(edge.source, edge.target, float(edge.weight))
        elif is_fact(edge):
            add(edge.source, edge.target, float(max(1, len(edge.source_ids))))

    return PPRGraph(
        node_ids=ids,
        index=index,
        sources=np.asarray(sources, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
    )
