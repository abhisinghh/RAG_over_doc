"""HippoRAG 2: PageRank solver, graph construction, indexing and retrieval (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES, HashEmbedder, ScriptedLLM, make_settings
from ragdoc.config import PPRConfig
from ragdoc.engine import RagEngine
from ragdoc.graph.hipporag import HippoIndexer
from ragdoc.graph.ppr import (
    SYNONYM_RELATION,
    build_ppr_graph,
    passage_node_id,
    personalized_pagerank,
)
from ragdoc.graph.store import GEdge, Node, SQLiteGraphStore, node_id
from ragdoc.retrieval.store import InMemoryVectorStore

# --------------------------------------------------------------------------- PageRank solver


def _ppr(n: int, edges: list[tuple[int, int, float]], reset: list[float], damping: float = 0.5):
    src = np.array([a for a, _, _ in edges], dtype=np.int64)
    dst = np.array([b for _, b, _ in edges], dtype=np.int64)
    w = np.array([x for _, _, x in edges], dtype=np.float64)
    return personalized_pagerank(n, src, dst, w, np.array(reset), damping=damping)


def test_two_node_closed_form() -> None:
    # p0 = (1 - d) + d * p1 and p1 = d * p0  =>  p0 = 1 / (1 + d), p1 = d / (1 + d)
    scores = _ppr(2, [(0, 1, 1.0)], [1.0, 0.0], damping=0.5)
    assert scores == pytest.approx([2 / 3, 1 / 3], abs=1e-6)


def test_scores_form_a_distribution_and_decay_with_distance() -> None:
    chain = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    scores = _ppr(4, chain, [1.0, 0, 0, 0])
    assert scores.sum() == pytest.approx(1.0)
    assert (scores >= 0).all()
    assert scores[0] > scores[1] > scores[2] > scores[3]


def test_heavier_edges_carry_more_mass() -> None:
    scores = _ppr(3, [(0, 1, 5.0), (0, 2, 1.0)], [1.0, 0, 0])
    assert scores[1] > scores[2]


def test_isolated_seed_keeps_mass() -> None:
    scores = _ppr(3, [(0, 1, 1.0)], [0.5, 0.0, 0.5])
    assert scores.sum() == pytest.approx(1.0)
    assert scores[2] > 0


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        _ppr(2, [(0, 1, 1.0)], [0.0, 0.0])
    with pytest.raises(ValueError):
        _ppr(2, [(0, 1, 1.0)], [1.0, 0.0], damping=1.0)


# --------------------------------------------------------------------------- graph construction


def _entity(name: str, chunks: list[str]) -> Node:
    return Node(
        id=node_id("concept", name), name=name, type="concept", source_ids=chunks, doc_ids=["d"]
    )


def test_build_graph_layout() -> None:
    acme, vpn = _entity("Acme", ["d:1", "d:2"]), _entity("VPN", ["d:2"])
    edges = [
        GEdge(source=acme.id, target=vpn.id, relation="requires", source_ids=["d:1", "d:2"]),
        GEdge(source=acme.id, target=vpn.id, relation=SYNONYM_RELATION, weight=0.9),
        GEdge(source="symbol:a.py#f", target="symbol:b.py#g", relation="CALLS"),  # code: ignored
    ]
    graph = build_ppr_graph([acme, vpn], edges)
    assert passage_node_id("d:1") in graph.index and passage_node_id("d:2") in graph.index
    assert "symbol:a.py#f" not in graph.index
    # 3 context edges + 1 fact edge (weight = 2 stating chunks) + 1 synonym edge
    assert len(graph.weights) == 5
    assert sorted(graph.weights.tolist()) == [0.9, 1.0, 1.0, 1.0, 2.0]


# --------------------------------------------------------------------------- indexing


def test_indexer_adds_synonyms_and_facts_once(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "g.sqlite")
    a, b = _entity("Acme Corp", ["d:1"]), _entity("Acme Corp Ltd", ["d:2"])
    short, vpn = _entity("IT", ["d:1"]), _entity("VPN", ["d:2"])
    store.upsert_nodes([a, b, short, vpn])
    store.upsert_edges([GEdge(source=a.id, target=vpn.id, relation="uses", source_ids=["d:1"])])
    indexer = HippoIndexer(
        store,
        HashEmbedder(),
        InMemoryVectorStore(),
        InMemoryVectorStore(),
        PPRConfig(),  # synonym threshold 0.8: cos({acme, corp}, {acme, corp, ltd}) = 0.816
        tmp_path / "state.sqlite",
    )
    first = indexer.index()
    assert (first.phrases, first.facts, first.synonym_edges) == (4, 1, 1)
    synonyms = store.all_edges([SYNONYM_RELATION])
    assert {synonyms[0].source, synonyms[0].target} == {a.id, b.id}

    again = indexer.index()
    assert (again.phrases, again.facts, again.synonym_edges) == (0, 0, 0)
    assert len(store.all_edges([SYNONYM_RELATION])) == 1


# --------------------------------------------------------------------------- retrieval


def _graph_engine(tmp_path: Path) -> RagEngine:
    eng = RagEngine(
        make_settings(tmp_path, graph={"enabled": True}),
        llm=ScriptedLLM(),
        embedder=HashEmbedder(),
    )
    eng.ingest(FIXTURES / "handbook.md")
    return eng


def test_ppr_search_returns_facts_then_ranked_passages(tmp_path: Path) -> None:
    eng = _graph_engine(tmp_path)
    results = eng.graph_search("How does Acme relate to the VPN?", mode="ppr")
    assert results[0].source == "graph_facts" and "Acme" in results[0].chunk.text
    passages = results[1:]
    assert passages and all(r.source == "ppr" for r in passages)
    assert [r.score for r in passages] == sorted((r.score for r in passages), reverse=True)


def test_relational_question_uses_ppr(tmp_path: Path) -> None:
    eng = _graph_engine(tmp_path)
    result = eng.ask("How does Acme relate to the VPN?")
    retrieve = next(e for e in result.trace if e.step == "retrieve")
    assert result.route == "relational" and retrieve.detail["graph_mode"] == "ppr"
    assert any(e.source == "graph_facts" for e in result.evidence)
    assert result.citations


def test_multi_hop_question_adds_ppr_evidence(tmp_path: Path) -> None:
    eng = _graph_engine(tmp_path)
    result = eng.ask("Is the VPN required and what does Acme provide?")
    retrieve = next(e for e in result.trace if e.step == "retrieve")
    assert result.route == "multi_hop" and retrieve.detail["graph_mode"] == "ppr"


def test_no_relevant_fact_falls_back_to_local_search(tmp_path: Path) -> None:
    eng = _graph_engine(tmp_path)
    assert eng.graph_search("How is the cafeteria connected to parking?", mode="ppr") == []
    result = eng.ask("How is the cafeteria connected to parking?")
    retrieve = next(e for e in result.trace if e.step == "retrieve")
    assert retrieve.detail["graph_mode"].startswith("local")


def test_unknown_graph_mode_is_rejected(tmp_path: Path) -> None:
    eng = _graph_engine(tmp_path)
    with pytest.raises(ValueError):
        eng.graph_search("anything", mode="random-walk")
