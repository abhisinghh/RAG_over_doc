"""Tables (SQL), figures/charts (VLM), audio and video, all offline."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from conftest import FIXTURES, HashEmbedder, ScriptedLLM, make_settings
from ragdoc.engine import RagEngine
from ragdoc.ingestion import media
from ragdoc.ingestion.chunking import chunk_document
from ragdoc.ingestion.tables import TableStore, _parse_number, merge_multipage_tables
from ragdoc.models.transcribe import Segment
from ragdoc.types import Block, ParsedDocument


def _png(path: Path, size: tuple[int, int] = (320, 200)) -> Path:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (30, 120, 200)).save(buf, format="PNG")
    path.write_bytes(buf.getvalue())
    return path


# --------------------------------------------------------------------------- tables


def test_number_parsing() -> None:
    assert _parse_number("1,234.5") == 1234.5
    assert _parse_number("(12)") == -12
    assert _parse_number("$4.2m") == 4_200_000
    assert _parse_number("15%") == 15
    assert _parse_number("Q1") is None


def test_table_store_types_and_read_only_sql(tmp_path: Path) -> None:
    store = TableStore(tmp_path / "t.sqlite")
    md = "| Region | Revenue (EUR m) | Note |\n|---|---|---|\n| EU | 1,200 | a |\n| US | 900 | b |"
    tid = store.add_table(doc_id="d", source="r.pdf", page=3, title="Revenue", markdown=md)
    assert tid
    _cols, rows = store.run_sql(f'SELECT SUM(revenue_eur_m) FROM "t_{tid}"')
    assert rows[0][0] == 2100
    for bad in (
        f'DELETE FROM "t_{tid}"',
        f'SELECT 1; DROP TABLE "t_{tid}"',
        "PRAGMA table_info(_registry)",
        "ATTACH DATABASE 'x.db' AS x",
    ):
        with pytest.raises(ValueError):
            store.run_sql(bad)
    assert "page 3" in store.schema_text(store.get([tid])[0])


def test_multipage_tables_are_merged() -> None:
    a = Block(text="| A | B |\n|---|---|\n| 1 | 2 |", page=4, kind="table", modality="table")
    b = Block(text="| A | B |\n|---|---|\n| 3 | 4 |", page=5, kind="table", modality="table")
    merged = merge_multipage_tables([a, b])
    assert len(merged) == 1 and "| 3 | 4 |" in merged[0].text
    assert merged[0].meta["pages"] == [4, 5]


def test_structured_question_runs_sql(engine: RagEngine) -> None:
    assert "structured" in engine.available_routes()
    result = engine.ask("What is the total of the limits in the equipment table?")
    assert result.route == "structured"
    assert any(e.source == "sql" for e in result.evidence)
    retrieve = next(e for e in result.trace if e.step == "retrieve")
    assert retrieve.detail.get("sql", "").startswith("SELECT")


# --------------------------------------------------------------------------- figures


def test_image_is_described_as_chart_and_queryable(tmp_path: Path, llm: ScriptedLLM) -> None:
    eng = RagEngine(make_settings(tmp_path), llm=llm, embedder=HashEmbedder())
    report = eng.ingest(_png(tmp_path / "revenue_chart.png"))
    assert report.figures == 1
    chunks = eng.docs.all_chunks([report.doc_id])
    chart = next(c for c in chunks if c.modality == "chart")
    assert chart.asset and eng.assets.exists(chart.asset)
    assert "Q2" in chart.text and chart.metadata.get("table_ids")
    assert eng.tables is not None and eng.tables.count() == 1

    result = eng.ask("What was the revenue in Q2 in the quarterly revenue chart?")
    assert result.citations and result.citations[0].modality == "chart"
    assert any(
        task == "generate" and n == 1 and role == "vision" for task, n, role in llm.image_calls
    )


def test_image_files_are_parsed_as_images(tmp_path: Path) -> None:
    """Regression: image files once fell through to the Markdown parser and were indexed as text."""
    from ragdoc.ingestion.parsers import parse_document

    doc = parse_document(_png(tmp_path / "diagram.png"))
    assert doc.kind == "image"
    assert len(doc.blocks) == 1 and doc.blocks[0].modality == "image"
    assert doc.blocks[0].meta["_image"].startswith(b"\x89PNG")


def test_tiny_images_are_skipped(tmp_path: Path) -> None:
    eng = RagEngine(make_settings(tmp_path), llm=ScriptedLLM(), embedder=HashEmbedder())
    with pytest.raises(ValueError):
        eng.ingest(_png(tmp_path / "icon.png", (32, 32)))


# --------------------------------------------------------------------------- audio / video


class FakeTranscriber:
    def transcribe(self, audio_path: Path, offset_s: float = 0.0) -> list[Segment]:
        return [
            Segment(
                start=offset_s + 0,
                end=offset_s + 40,
                text="Welcome to the quarterly earnings call.",
            ),
            Segment(
                start=offset_s + 40,
                end=offset_s + 95,
                text="Revenue grew twelve percent in the third quarter.",
            ),
            Segment(
                start=offset_s + 95, end=offset_s + 130, text="Our new product launches in March."
            ),
        ]


def test_time_window_chunking() -> None:
    segments = FakeTranscriber().transcribe(Path("x"))
    doc = ParsedDocument(
        doc_id="a",
        source="call.mp3",
        title="call",
        sha256="0" * 64,
        kind="audio",
        blocks=[
            Block(text=s.text, kind="segment", modality="audio", start_s=s.start, end_s=s.end)
            for s in segments
        ],
    )
    chunks = chunk_document(doc, window_seconds=90)
    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 40), (40, 130)]
    assert chunks[1].locator == "00:40-02:10"
    assert chunks[1].modality == "audio"


def test_audio_ingestion_cites_timestamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "earnings-call.mp3"
    audio.write_bytes(b"ID3 not really audio")
    monkeypatch.setattr(media, "split_audio", lambda src, out, segment_s=600: [(src, 0.0)])
    monkeypatch.setattr(media, "probe_duration", lambda p: 130.0)
    eng = RagEngine(
        make_settings(tmp_path),
        llm=ScriptedLLM(),
        embedder=HashEmbedder(),
        transcriber_factory=FakeTranscriber,
    )
    report = eng.ingest(audio)
    assert report.kind == "audio" and report.duration_s == 130.0
    result = eng.ask("How much did revenue grow in the third quarter?")
    assert result.citations
    assert result.citations[0].locator == "00:40-02:10"


def test_video_keyframes_become_described_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"not really video")
    frame = _png(tmp_path / "frame.png")
    monkeypatch.setattr(media, "has_audio_stream", lambda p: False)
    monkeypatch.setattr(media, "probe_duration", lambda p: 60.0)
    monkeypatch.setattr(
        media, "extract_keyframes", lambda src, out, interval, scene, max_frames: [(frame, 30.0)]
    )
    eng = RagEngine(
        make_settings(tmp_path),
        llm=ScriptedLLM(),
        embedder=HashEmbedder(),
        transcriber_factory=FakeTranscriber,
    )
    report = eng.ingest(video)
    assert report.kind == "video" and report.figures == 1
    chunk = next(c for c in eng.docs.all_chunks([report.doc_id]) if c.asset)
    assert chunk.start_s == 30.0 and chunk.locator.startswith("frame at 00:30")


def test_media_tools_missing_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(media.MediaToolMissing):
        media.probe_duration(FIXTURES / "handbook.md")
