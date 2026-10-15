"""End-to-end tests of ingestion and the agent loop with scripted models (no network)."""

from __future__ import annotations

from pathlib import Path

from conftest import FIXTURES, HashEmbedder, ScriptedLLM, make_settings
from ragdoc.engine import RagEngine
from ragdoc.generation.citations import ABSTAIN_TEXT


def test_ingest_creates_contextualized_chunks(engine: RagEngine) -> None:
    docs = engine.list_documents()
    assert len(docs) == 1
    chunks = engine.docs.all_chunks()
    assert len(chunks) >= 4
    assert all(c.context.startswith("From the Acme") for c in chunks)
    assert any(c.metadata.get("has_table") for c in chunks)
    assert any("Security" in c.heading_path for c in chunks)


def test_reingest_unchanged_file_is_skipped(engine: RagEngine) -> None:
    report = engine.ingest(FIXTURES / "handbook.md")
    assert report.skipped


def test_simple_question_gets_cited_answer(engine: RagEngine) -> None:
    result = engine.ask("Who is eligible for remote work?")
    assert result.route == "simple"
    assert not result.abstained
    assert result.citations, result.answer
    assert "eligible" in result.answer.lower()
    assert result.verification and result.verification["passed"]
    steps = [e.step for e in result.trace]
    assert steps[:4] == ["route", "retrieve", "grade", "generate"]
    assert "verify" in steps


def test_unanswerable_question_abstains_after_rewrites(engine: RagEngine, llm: ScriptedLLM) -> None:
    result = engine.ask("What is the capital of Mars?")
    assert result.abstained
    assert result.answer == ABSTAIN_TEXT
    assert llm.calls.count("rewrite") == engine.settings.agent.max_rewrites


def test_multi_hop_question_is_decomposed(engine: RagEngine) -> None:
    result = engine.ask("What is the internet stipend and when must lost devices be reported?")
    assert result.route == "multi_hop"
    assert len(result.sub_questions) == 2
    assert not result.abstained


def test_out_of_scope_question(engine: RagEngine) -> None:
    result = engine.ask("Tell me a joke")
    assert result.route == "out_of_scope"
    assert result.abstained
    assert [e.step for e in result.trace] == ["route"]


def test_baseline_mode_is_single_pass(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    settings = make_settings(tmp_path, agent={"enabled": False}, retrieval={"reranker": "none"})
    eng = RagEngine(settings, llm=llm, embedder=HashEmbedder())
    eng.ingest(FIXTURES / "handbook.md")
    llm.calls.clear()
    result = eng.ask("Who is eligible for remote work?")
    assert [e.step for e in result.trace] == ["retrieve", "generate"]
    assert llm.calls == ["generate"]


def test_stream_yields_steps_then_result(engine: RagEngine) -> None:
    events = list(engine.stream("Who is eligible for remote work?"))
    assert events[0]["type"] == "step" and events[0]["node"] == "route"
    assert events[-1]["type"] == "result"
    assert events[-1]["result"]["citations"]


def test_answer_cache(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    eng = RagEngine(
        make_settings(tmp_path, cache={"answers": True}), llm=llm, embedder=HashEmbedder()
    )
    eng.ingest(FIXTURES / "handbook.md")
    first = eng.ask("Who is eligible for remote work?")
    second = eng.ask("who is ELIGIBLE for remote work?")
    assert not first.cached and second.cached
    assert second.answer == first.answer


def test_delete_document(engine: RagEngine) -> None:
    doc_id = engine.list_documents()[0]["doc_id"]
    engine.delete_document(doc_id)
    assert engine.list_documents() == []
    assert engine.search("remote work") == []


def test_raptor_summaries_are_indexed(tmp_path: Path) -> None:
    eng = RagEngine(
        make_settings(tmp_path, ingestion={"raptor": True, "raptor_group_size": 2}),
        llm=ScriptedLLM(),
        embedder=HashEmbedder(),
    )
    report = eng.ingest(FIXTURES / "handbook.md")
    assert report.num_summaries >= 1
    kinds = {c.kind for c in eng.docs.all_chunks()}
    assert kinds == {"chunk", "summary"}
