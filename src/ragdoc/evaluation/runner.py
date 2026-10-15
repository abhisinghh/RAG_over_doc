"""Run a golden dataset against one or more configurations and write a comparison report.

Dataset: JSONL, one object per line:
  {"id": "q1", "question": "...", "reference_answer": "...", "gold_pages": [3, 4],
   "doc": "docs/report.pdf", "type": "multi_hop", "answerable": true}
``doc`` is resolved relative to the dataset file and ingested before the run.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from ragdoc.config import load_settings
from ragdoc.engine import RagEngine
from ragdoc.evaluation import metrics as m
from ragdoc.evaluation.judge import judge_correctness, judge_faithfulness
from ragdoc.models.llm import build_llm

log = logging.getLogger(__name__)


class EvalItem(BaseModel):
    id: str
    question: str
    reference_answer: str = ""
    gold_pages: list[int] = Field(default_factory=list)
    doc: str | None = None
    type: str = "single"
    answerable: bool = True


def load_dataset(path: Path, limit: int | None = None) -> list[EvalItem]:
    items = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        data = json.loads(line)
        data.setdefault("id", f"q{i + 1}")
        items.append(EvalItem.model_validate(data))
    return items[:limit] if limit else items


def _evaluate_item(
    engine: RagEngine, judge_llm: Any, item: EvalItem, doc_ids: list[str] | None, judge: bool
) -> dict[str, Any]:
    k = engine.settings.retrieval.final_k
    retrieved = engine.search(item.question, top_k=k, doc_ids=doc_ids)
    pages = [r.chunk.pages for r in retrieved]

    started = time.perf_counter()
    result = engine.ask(item.question, doc_ids=doc_ids, use_cache=False)
    latency = (time.perf_counter() - started) * 1000

    row: dict[str, Any] = {
        "id": item.id,
        "type": item.type,
        "question": item.question,
        "answer": result.answer,
        "abstained": result.abstained,
        "route": result.route,
        "latency_ms": latency,
        "llm_calls": result.usage.calls,
        "tokens": result.usage.input_tokens + result.usage.output_tokens,
        "cited_pages": sorted({p for c in result.citations for p in c.pages}),
    }
    if item.answerable and item.gold_pages:
        row.update(
            hit_at_k=m.hit_at_k(pages, item.gold_pages, k),
            recall_at_k=m.page_recall_at_k(pages, item.gold_pages, k),
            mrr=m.mrr(pages, item.gold_pages),
            ndcg_at_k=m.ndcg_at_k(pages, item.gold_pages, k),
            citation_page_precision=(
                len(set(row["cited_pages"]) & set(item.gold_pages)) / len(row["cited_pages"])
                if row["cited_pages"]
                else None
            ),
        )
    row["abstention_correct"] = result.abstained != item.answerable
    if judge:
        try:
            row["faithfulness"] = judge_faithfulness(judge_llm, result.answer, result.evidence)
            if item.reference_answer:
                verdict = judge_correctness(
                    judge_llm,
                    item.question,
                    item.reference_answer if item.answerable else "Not in the document.",
                    result.answer,
                )
                row["correctness"] = verdict.score
                row["correctness_reason"] = verdict.reason
        except Exception as exc:
            log.warning("Judge failed on %s: %s", item.id, exc)
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows if r.get(name) is not None]

    answerable = [r for r in rows if "hit_at_k" in r]
    return {
        "n": len(rows),
        "hit@k": m.mean([r["hit_at_k"] for r in answerable]),
        "recall@k": m.mean([r["recall_at_k"] for r in answerable]),
        "mrr": m.mean([r["mrr"] for r in answerable]),
        "ndcg@k": m.mean([r["ndcg_at_k"] for r in answerable]),
        "faithfulness": m.mean(col("faithfulness")),
        "correctness": m.mean(col("correctness")),
        "citation_precision": m.mean(col("citation_page_precision")),
        "abstention_acc": m.mean([float(r["abstention_correct"]) for r in rows]),
        "p50_ms": m.percentile(col("latency_ms"), 50),
        "p95_ms": m.percentile(col("latency_ms"), 95),
        "llm_calls": m.mean(col("llm_calls")),
        "tokens": m.mean(col("tokens")),
    }


def run_experiments(
    dataset: Path,
    base_config: Path | None = None,
    base_overlays: list[Path] | None = None,
    experiments: list[Path] | None = None,
    out_dir: Path = Path("eval/reports"),
    limit: int | None = None,
    judge: bool = True,
    console: Console | None = None,
) -> Path:
    console = console or Console()
    items = load_dataset(dataset, limit)
    if not items:
        raise ValueError(f"No questions in {dataset}")
    base_overlays = list(base_overlays or [])
    runs = (
        [("active", base_overlays)]
        if not experiments
        else [(exp.stem, [*base_overlays, exp]) for exp in experiments]
    )

    # The first engine owns the index; later engines share its vector store and embedder,
    # so experiments must only vary retrieval / agent / llm settings.
    first: RagEngine | None = None
    summaries: dict[str, dict[str, float]] = {}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / stamp
    out.mkdir(parents=True, exist_ok=True)

    for name, overlays in runs:
        settings = load_settings(base_config, overlays)
        if first is None:
            engine = RagEngine(settings)
            first = engine
        else:
            if settings.fingerprint("ingestion", "embeddings") != first.settings.fingerprint(
                "ingestion", "embeddings"
            ):
                console.print(
                    f"[yellow]{name}: ingestion/embedding changes are ignored (shared index)[/yellow]"
                )
            llm = first.llm if settings.llm == first.settings.llm else build_llm(settings.llm)
            engine = RagEngine(
                settings, llm=llm, embedder=first.embedder, vector_store=first.vectors
            )

        doc_map: dict[str, str] = {}
        for item in items:
            if item.doc and item.doc not in doc_map:
                doc_map[item.doc] = engine.ingest((dataset.parent / item.doc).resolve()).doc_id

        rows = []
        with console.status(f"[{name}] evaluating {len(items)} questions..."):
            for item in items:
                doc_ids = [doc_map[item.doc]] if item.doc else None
                rows.append(_evaluate_item(engine, first.llm, item, doc_ids, judge))
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        summaries[name] = _summary(rows)

    report = out / "report.md"
    metrics_order = list(next(iter(summaries.values())).keys())
    lines = [
        f"# Evaluation report · {stamp}",
        "",
        f"Dataset: `{dataset}` · {len(items)} questions · k = retrieval.final_k",
        "",
        "| metric | " + " | ".join(summaries) + " |",
        "| --- | " + " | ".join("---" for _ in summaries) + " |",
    ]
    for metric in metrics_order:
        cells = []
        for s in summaries.values():
            v = s[metric]
            cells.append(
                "-"
                if v != v
                else (f"{v:.0f}" if metric in {"n", "p50_ms", "p95_ms", "tokens"} else f"{v:.3f}")
            )
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    table = Table("metric", *summaries.keys(), title="Evaluation summary")
    for line in lines[6:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        table.add_row(*cells)
    console.print(table)
    return report
