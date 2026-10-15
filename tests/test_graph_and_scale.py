"""GraphRAG (extraction, communities, local/global search) and scale features, offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import FIXTURES, HashEmbedder, ScriptedLLM, make_settings
from ragdoc.engine import RagEngine
from ragdoc.graph.store import GEdge, Node, SQLiteGraphStore, node_id, normalise_name
from ragdoc.ingestion.batch import in_shard
from ragdoc.retrieval.store import InMemoryVectorStore, SearchFilter
from ragdoc.types import Chunk


def _graph_engine(tmp_path: Path, ppr: bool = True) -> RagEngine:
    eng = RagEngine(
        make_settings(
            tmp_path,
            graph={"enabled": True, "community_min_size": 2, "ppr": {"enabled": ppr}},
        ),
        llm=ScriptedLLM(),
        embedder=HashEmbedder(),
    )
    report = eng.ingest(FIXTURES / "handbook.md")
    assert report.entities >= 3
    return eng


def test_graph_store_merges_and_deletes(tmp_path: Path) -> None:
    g = SQLiteGraphStore(tmp_path / "g.sqlite")
    a, b = node_id("organization", "The Acme Corp"), node_id("product", "Widget")
    assert a == node_id("concept", "acme corp")  # entities share one namespace
    g.upsert_nodes(
        [
            Node(
                id=a,
                name="Acme Corp",
                type="organization",
                description="x",
                source_ids=["d1:0001"],
                doc_ids=["d1"],
            )
        ]
    )
    g.upsert_nodes(
        [
            Node(
                id=a,
                name="Acme Corp",
                type="organization",
                description="y",
                source_ids=["d2:0001"],
                doc_ids=["d2"],
            )
        ]
    )
    g.upsert_nodes(
        [Node(id=b, name="Widget", type="product", source_ids=["d1:0002"], doc_ids=["d1"])]
    )
    g.upsert_edges([GEdge(source=a, target=b, relation="makes", weight=2, source_ids=["d1:0002"])])
    g.upsert_edges([GEdge(source=a, target=b, relation="makes", weight=3)])
    node = g.get_nodes([a])[0]
    assert "x" in node.description and "y" in node.description and node.doc_ids == ["d1", "d2"]
    assert g.edges_for([a])[0].weight == 5
    assert g.find_nodes(["acme corp"])[0].id == a
    g.delete_document("d1")
    assert g.get_nodes([b]) == []  # only mentioned by d1
    assert g.get_nodes([a])[0].doc_ids == ["d2"]
    assert g.edges_for([a]) == []
    assert normalise_name("  The  Big-Co. ") == "big-co."


def test_relational_question_uses_local_graph_search(tmp_path: Path) -> None:
    eng = _graph_engine(tmp_path, ppr=False)
    assert "relational" in eng.available_routes()
    result = eng.ask("How does Acme relate to the VPN?")
    assert result.route == "relational"
    facts = next(e for e in result.evidence if e.source == "graph_local")
    assert "Acme" in facts.chunk.text and "VPN" in facts.chunk.text


def test_global_question_uses_community_reports(tmp_path: Path) -> None:
    eng = _graph_engine(tmp_path)
    assert "global" not in eng.available_routes()
    stats = eng.build_graph_index()
    assert stats["communities"] >= 1
    assert "global" in eng.available_routes()
    result = eng.ask("What are the overall themes of the handbook?")
    assert result.route == "global"
    assert any(e.chunk.kind == "community" for e in result.evidence)
    # Community reports are excluded from ordinary retrieval.
    assert all(r.chunk.kind != "community" for r in eng.search("remote work VPN"))


def test_search_filters_in_memory_store() -> None:
    store = InMemoryVectorStore()
    chunks = [
        Chunk(chunk_id="a", doc_id="d1", source="s", text="t", modality="code"),
        Chunk(chunk_id="b", doc_id="d1", source="s", text="t", modality="text"),
        Chunk(chunk_id="c", doc_id="d2", source="s", text="t", modality="text", kind="community"),
    ]
    store.upsert(chunks, np.eye(3, dtype=np.float32))
    q = np.ones(3, dtype=np.float32)
    assert {i for i, _ in store.search(q, 5, SearchFilter.build(modalities=["code"]))} == {"a"}
    assert {i for i, _ in store.search(q, 5, SearchFilter.build(exclude_kinds=["community"]))} == {
        "a",
        "b",
    }
    assert {i for i, _ in store.search(q, 5, SearchFilter.build(doc_ids=["d2"]))} == {"c"}


def test_doc_routing_restricts_to_top_documents(tmp_path: Path) -> None:
    eng = RagEngine(
        make_settings(
            tmp_path,
            ingestion={"doc_summary": True},
            retrieval={"doc_routing": True, "doc_routing_top_docs": 1},
        ),
        llm=ScriptedLLM(),
        embedder=HashEmbedder(),
    )
    eng.ingest(FIXTURES / "handbook.md")
    assert eng.retriever.route_documents("remote work handbook") == [
        eng.list_documents()[0]["doc_id"]
    ]
    assert eng.search("Who is eligible for remote work?")


def test_batch_ingestion_is_resumable(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(3):
        (corpus / f"note{i}.md").write_text(
            f"# Note {i}\n\nRemote work note number {i} about the VPN policy.\n"
        )
    (corpus / "broken.pdf").write_bytes(b"not a pdf")
    eng = RagEngine(make_settings(tmp_path), llm=ScriptedLLM(), embedder=HashEmbedder())
    files = eng.discover_files(corpus)
    summary = eng.ingest_batch(files, workers=2)
    assert summary.get("done") == 3 and summary.get("failed") == 1
    again = eng.ingest_batch(files, workers=2)  # only the failure is retried
    assert again.get("done") == 3 and again.get("failed") == 1
    assert len(eng.list_documents()) == 3


def test_sharding_is_disjoint_and_complete() -> None:
    paths = [f"/data/doc{i}.pdf" for i in range(200)]
    shards = [{p for p in paths if in_shard(p, (i, 4))} for i in range(4)]
    assert set().union(*shards) == set(paths)
    assert sum(len(s) for s in shards) == len(paths)
