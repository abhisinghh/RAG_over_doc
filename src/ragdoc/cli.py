"""Command-line interface: ``ragdoc ingest | ask | batch | docs | delete | eval | serve | ui | mcp``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

app = typer.Typer(
    help="Agentic RAG over your documents.", no_args_is_help=True, add_completion=False
)
console = Console()


def _engine(ctx: typer.Context) -> Any:
    from ragdoc.engine import RagEngine

    if ctx.obj.get("engine") is None:
        ctx.obj["engine"] = RagEngine.from_config(ctx.obj["config"], ctx.obj["overlays"])
    return ctx.obj["engine"]


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Base YAML config")] = None,
    overlay: Annotated[
        list[Path] | None,
        typer.Option(
            "--overlay", "-o", help="Overlay YAML(s), e.g. configs/experiments/dense.yaml"
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ctx.obj = {"config": config, "overlays": overlay or [], "engine": None}


@app.command()
def ingest(
    ctx: typer.Context,
    path: Annotated[
        Path, typer.Argument(exists=True, help="A file or folder: documents, images, audio, video")
    ],
    force: Annotated[bool, typer.Option(help="Re-index even if unchanged")] = False,
) -> None:
    """Index documents and media (PDF, Word, Markdown, images, audio, video)."""
    engine = _engine(ctx)
    with console.status("Ingesting..."):
        reports = engine.ingest_path(path, force=force)
    table = Table("doc_id", "source", "kind", "chunks", "figures", "tables", "entities", "status")
    for r in reports:
        table.add_row(
            r.doc_id,
            r.source,
            r.kind,
            str(r.num_chunks),
            str(r.figures),
            str(r.tables),
            str(r.entities),
            "unchanged" if r.skipped else f"indexed in {r.seconds}s",
        )
    console.print(table)


@app.command("ingest-repo")
def ingest_repo(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Repository root")],
    name: Annotated[str | None, typer.Option(help="Repository name (default: folder name)")] = None,
    force: bool = False,
) -> None:
    """Index a code repository or microservice monorepo: code, configs, docs, call graph and
    service dependency graph."""
    engine = _engine(ctx)
    from rich.progress import Progress

    with Progress(console=console) as progress:
        task = progress.add_task("Indexing repository", total=None)

        def on_progress(info: dict[str, object]) -> None:
            progress.update(task, total=info["total"], completed=info["done"])

        report = engine.ingest_repo(path, name, force=force, on_progress=on_progress)
    console.print(
        f"[bold]{report.repo}[/bold]: {report.files} files ({report.indexed} indexed, "
        f"{report.skipped} unchanged, {len(report.failed)} failed) · {report.symbols} symbols · "
        f"{report.services} services · {report.service_edges} dependencies · {report.topics} topics"
    )
    for rel, err in report.failed[:10]:
        console.print(f"  [red]failed[/red] {rel}: {err}")


@app.command("ingest-batch")
def ingest_batch(
    ctx: typer.Context,
    path: Annotated[Path | None, typer.Argument(help="Folder to index recursively")] = None,
    manifest: Annotated[Path | None, typer.Option(help="Text file with one path per line")] = None,
    workers: Annotated[int | None, typer.Option(help="Parallel documents")] = None,
    shard: Annotated[
        str | None, typer.Option(help="i/n: only this shard of the paths (multi-machine)")
    ] = None,
) -> None:
    """Resumable, parallel ingestion for large corpora (re-run to resume / retry failures)."""
    engine = _engine(ctx)
    if manifest:
        paths = [
            Path(p.strip()) for p in manifest.read_text(encoding="utf-8").splitlines() if p.strip()
        ]
    elif path:
        paths = engine.discover_files(path)
    else:
        raise typer.BadParameter("Give a folder or --manifest")
    shard_spec = None
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        shard_spec = (i, n)
    from rich.progress import Progress

    with Progress(console=console) as progress:
        task = progress.add_task("Ingesting", total=None)

        def on_progress(info: dict[str, object]) -> None:
            progress.update(task, total=info["total"], completed=info["done"])

        summary = engine.ingest_batch(
            paths, workers=workers, shard=shard_spec, on_progress=on_progress
        )
    console.print(summary)


graph_app = typer.Typer(help="Knowledge graph commands.", no_args_is_help=True)
app.add_typer(graph_app, name="graph")


@graph_app.command("build")
def graph_build(ctx: typer.Context) -> None:
    """Index phrases and facts for HippoRAG 2, then detect communities and write their reports
    (enables global questions)."""
    engine = _engine(ctx)
    with console.status("Building communities and reports..."):
        stats = engine.build_graph_index()
    console.print_json(json.dumps(stats, default=str))


@graph_app.command("search")
def graph_search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument()],
    mode: Annotated[str, typer.Option(help="ppr (HippoRAG 2), local or global")] = "ppr",
) -> None:
    """Run graph retrieval alone and show the ranked evidence (useful for debugging)."""
    results = _engine(ctx).graph_search(query, mode=mode)
    if not results:
        console.print("No graph evidence found (no relevant facts or entities).")
        return
    table = Table("#", "source", "location", "score", "text")
    for i, r in enumerate(results, start=1):
        table.add_row(
            str(i),
            r.chunk.source,
            r.chunk.locator,
            f"{r.score:.4f}",
            r.chunk.text[:160].replace("\n", " "),
        )
    console.print(table)


@graph_app.command("stats")
def graph_stats(ctx: typer.Context) -> None:
    """Node, edge and community counts."""
    console.print_json(json.dumps(_engine(ctx).graph.stats(), default=str))


@app.command()
def services(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Show one service's dependencies")] = None,
) -> None:
    """List discovered microservices, or the dependency facts for one service."""
    engine = _engine(ctx)
    if name:
        facts = engine.graph_retriever.services(name)
        console.print(facts[0].chunk.text if facts else "No such service found.")
        return
    nodes = engine.graph.all_nodes(["service"])
    edges = engine.graph.all_edges(["CALLS"])
    table = Table("service", "calls", "called by", "root")
    for n in sorted(nodes, key=lambda n: n.name):
        out = sorted({e.target.split(":", 1)[-1] for e in edges if e.source == n.id})
        inc = sorted({e.source.split(":", 1)[-1] for e in edges if e.target == n.id})
        table.add_row(n.name, ", ".join(out[:8]), ", ".join(inc[:8]), str(n.data.get("root", "")))
    console.print(table)


def _print_answer(result: Any, show_trace: bool) -> None:
    console.print(Markdown(result.answer))
    if result.citations:
        console.print("\n[bold]Sources[/bold]")
        for c in result.citations:
            where = c.locator or (", ".join(map(str, c.pages)) or "n/a")
            console.print(f"  [{c.marker}] {c.source}, {where}: [dim]{c.snippet[:200]}[/dim]")
    if result.verification and result.verification.get("passed") is False:
        console.print(
            "[yellow]Warning: some statements could not be verified against the sources.[/yellow]"
        )
    meta = f"route={result.route} · {result.latency_ms:.0f} ms · {result.usage.input_tokens}+{result.usage.output_tokens} tokens"
    console.print(f"\n[dim]{meta}{' · cached' if result.cached else ''}[/dim]")
    if show_trace:
        table = Table("step", "ms", "tokens in/out", "detail")
        for e in result.trace:
            table.add_row(
                e.step,
                f"{e.latency_ms:.0f}",
                f"{e.input_tokens}/{e.output_tokens}",
                json.dumps(e.detail, default=str)[:120],
            )
        console.print(table)


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument()],
    doc: Annotated[
        list[str] | None, typer.Option("--doc", "-d", help="Restrict to doc_id(s)")
    ] = None,
    trace: Annotated[bool, typer.Option(help="Show the per-step trace")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print raw JSON")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
) -> None:
    """Answer a question with citations."""
    engine = _engine(ctx)
    with console.status("Thinking..."):
        result = engine.ask(question, doc_ids=doc or None, use_cache=not no_cache)
    if as_json:
        console.print_json(json.dumps(result.to_public(include_trace=trace), default=str))
    else:
        _print_answer(result, trace)


@app.command()
def batch(
    ctx: typer.Context,
    questions: Annotated[
        Path, typer.Argument(exists=True, help="Text file, one question per line")
    ],
    document: Annotated[
        Path | None, typer.Option("--document", "-f", exists=True, help="Ingest this first")
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("output.json"),
) -> None:
    """Answer every question in a file and write JSON (the original notebook workflow)."""
    engine = _engine(ctx)
    doc_ids = None
    if document:
        doc_ids = [engine.ingest(document).doc_id]
    lines = [q.strip() for q in questions.read_text(encoding="utf-8").splitlines() if q.strip()]
    results = []
    with console.status(f"Answering {len(lines)} questions..."):
        for q in lines:
            r = engine.ask(q, doc_ids=doc_ids)
            results.append(
                {
                    "query": q,
                    "answer": r.answer,
                    "abstained": r.abstained,
                    "citations": [c.model_dump() for c in r.citations],
                }
            )
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"Wrote {len(results)} answers to {output}")


@app.command()
def docs(ctx: typer.Context) -> None:
    """List indexed documents."""
    table = Table("doc_id", "source", "title", "pages", "chunks", "ingested")
    for d in _engine(ctx).list_documents():
        table.add_row(
            d["doc_id"],
            d["source"],
            (d["title"] or "")[:40],
            str(d["num_pages"] or "-"),
            str(d["num_chunks"]),
            d["ingested_at"],
        )
    console.print(table)


@app.command()
def delete(ctx: typer.Context, doc_id: str) -> None:
    """Remove a document from the index."""
    _engine(ctx).delete_document(doc_id)
    console.print(f"Deleted {doc_id}")


@app.command(name="eval")
def evaluate(
    ctx: typer.Context,
    dataset: Annotated[Path, typer.Option(exists=True)] = Path("eval/datasets/golden.jsonl"),
    experiment: Annotated[
        list[Path] | None,
        typer.Option(
            "--experiment", "-e", help="Overlay(s) to compare; default: the active config"
        ),
    ] = None,
    out_dir: Annotated[Path, typer.Option("--out")] = Path("eval/reports"),
    limit: Annotated[
        int | None, typer.Option(help="Only the first N questions (smoke test)")
    ] = None,
    judge: Annotated[bool, typer.Option(help="LLM-judge answer correctness")] = True,
) -> None:
    """Run the golden set against one or more configurations and write a report."""
    from ragdoc.evaluation.runner import run_experiments

    report_path = run_experiments(
        dataset=dataset,
        base_config=ctx.obj["config"],
        base_overlays=ctx.obj["overlays"],
        experiments=experiment or [],
        out_dir=out_dir,
        limit=limit,
        judge=judge,
        console=console,
    )
    console.print(f"Report: {report_path}")


@app.command()
def serve(
    ctx: typer.Context,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the REST API (pip install 'ragdoc[api]')."""
    import uvicorn

    from ragdoc.api.app import create_app

    uvicorn.run(create_app(_engine(ctx)), host=host, port=port)


@app.command()
def ui(
    ctx: typer.Context,
    share: Annotated[bool, typer.Option(help="Create a public Gradio link (Colab)")] = False,
    port: int = 7860,
) -> None:
    """Launch the Gradio UI (pip install 'ragdoc[ui]')."""
    from ragdoc.ui.gradio_app import build_ui

    build_ui(_engine(ctx)).launch(share=share, server_port=port)


@app.command()
def mcp(ctx: typer.Context) -> None:
    """Run the MCP server over stdio (pip install 'ragdoc[mcp]')."""
    from ragdoc.mcp_server import run

    run(_engine(ctx))


if __name__ == "__main__":
    app()
