"""Gradio UI: upload documents, ask questions with citations and a live agent trace,
and run the original batch workflow (questions file → output.json)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from ragdoc.engine import RagEngine


def _path(file: Any) -> str:
    return str(getattr(file, "name", file))


def _doc_choices(engine: RagEngine) -> list[tuple[str, str]]:
    return [
        (f"{d['source']} ({d['doc_id']})", d["doc_id"]) for d in engine.list_documents(limit=500)
    ]


def _sources_md(result: dict[str, Any]) -> str:
    lines = []
    for c in result.get("citations", []):
        where = c.get("locator") or (", ".join(map(str, c["pages"])) or "n/a")
        snippet = c["snippet"].replace("\n", " ")[:300]
        lines.append(f"**[{c['marker']}]** {c['source']}, {where}  \n> {snippet}")
    verification = result.get("verification") or {}
    if verification.get("passed") is False:
        lines.append("\n⚠️ Some statements could not be verified against the sources.")
    return "\n\n".join(lines) or "_No sources cited._"


def build_ui(engine: RagEngine) -> gr.Blocks:
    with gr.Blocks(title="ragdoc · Agentic RAG") as demo:
        gr.Markdown("# ragdoc\nAgentic RAG over your documents, with cited answers.")

        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.File(
                        label="Add documents",
                        file_types=[
                            ".pdf",
                            ".docx",
                            ".md",
                            ".txt",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".mp3",
                            ".wav",
                            ".m4a",
                            ".mp4",
                            ".mov",
                            ".mkv",
                            ".webm",
                        ],
                        file_count="multiple",
                    )
                    ingest_btn = gr.Button("Index documents")
                    ingest_status = gr.Markdown()
                    docs = gr.Dropdown(
                        label="Search in (empty = all)",
                        choices=_doc_choices(engine),
                        multiselect=True,
                    )
                with gr.Column(scale=2):
                    question = gr.Textbox(
                        label="Question", placeholder="Ask about your documents..."
                    )
                    ask_btn = gr.Button("Ask", variant="primary")
                    steps = gr.Markdown()
                    answer = gr.Markdown()
                    sources = gr.Markdown()
                    figures = gr.Gallery(label="Cited figures and frames", columns=3, height="auto")
                    with gr.Accordion("Trace", open=False):
                        trace = gr.JSON()

            def do_ingest(files: list[Any] | None) -> tuple[str, Any]:
                if not files:
                    return "Choose at least one file.", gr.update()
                lines = []
                for f in files:
                    try:
                        r = engine.ingest(_path(f))
                        status = "unchanged" if r.skipped else f"{r.num_chunks} chunks"
                        lines.append(f"- **{r.source}**: {status}")
                    except Exception as exc:
                        lines.append(f"- **{Path(_path(f)).name}**: failed ({exc})")
                return "\n".join(lines), gr.update(choices=_doc_choices(engine))

            def do_ask(q: str, doc_ids: list[str] | None):  # generator → live updates
                if not q.strip():
                    yield "", "Please enter a question.", "", [], None
                    return
                log: list[str] = []
                for event in engine.stream(q, doc_ids=doc_ids or None):
                    if event["type"] == "step":
                        ms = event.get("latency_ms")
                        log.append(f"`{event['node']}`" + (f" · {ms:.0f} ms" if ms else ""))
                        yield " → ".join(log), "_Working..._", "", [], None
                    else:
                        result = event["result"]
                        images = [
                            (str(engine.assets.path(c["asset"])), f"[{c['marker']}] {c['locator']}")
                            for c in result.get("citations", [])
                            if c.get("asset") and engine.assets.exists(c["asset"])
                        ]
                        yield (
                            " → ".join(log),
                            result["answer"],
                            _sources_md(result),
                            images,
                            result.get("trace"),
                        )

            ingest_btn.click(do_ingest, inputs=upload, outputs=[ingest_status, docs])
            outputs = [steps, answer, sources, figures, trace]
            ask_btn.click(do_ask, inputs=[question, docs], outputs=outputs)
            question.submit(do_ask, inputs=[question, docs], outputs=outputs)

        with gr.Tab("Batch"):
            gr.Markdown(
                "Upload a document and a `.txt` file with one question per line; get `output.json`."
            )
            with gr.Row():
                batch_doc = gr.File(label="Document", file_types=[".pdf", ".docx", ".md", ".txt"])
                batch_q = gr.File(label="Questions (.txt)", file_types=[".txt"])
            batch_btn = gr.Button("Run", variant="primary")
            batch_status = gr.Markdown()
            batch_out = gr.File(label="output.json")

            def do_batch(doc_file: Any, q_file: Any) -> tuple[str, str | None]:
                if doc_file is None or q_file is None:
                    return "Upload both a document and a questions file.", None
                report = engine.ingest(_path(doc_file))
                questions = [
                    line.strip()
                    for line in Path(_path(q_file)).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                rows = []
                for q in questions:
                    r = engine.ask(q, doc_ids=[report.doc_id])
                    rows.append(
                        {
                            "query": q,
                            "answer": r.answer,
                            "abstained": r.abstained,
                            "citations": [c.model_dump() for c in r.citations],
                        }
                    )
                out = Path(tempfile.mkdtemp()) / "output.json"
                out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                return f"Answered {len(rows)} questions.", str(out)

            batch_btn.click(
                do_batch, inputs=[batch_doc, batch_q], outputs=[batch_status, batch_out]
            )

    return demo


def main() -> None:
    build_ui(RagEngine()).launch()


if __name__ == "__main__":
    main()
