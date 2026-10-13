"""Code parsing, repository ingestion, call graph and microservice graph (offline)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from conftest import HashEmbedder, ScriptedLLM, make_settings
from ragdoc.code.parser import parse_braced, parse_python
from ragdoc.code.repo import code_file_to_document, iter_repo_files
from ragdoc.code.services import discover_services
from ragdoc.config import CodeConfig
from ragdoc.engine import RagEngine

PY = textwrap.dedent(
    '''
    """Order handling."""
    import httpx
    from fastapi import FastAPI

    app = FastAPI()


    def total_price(items):
        """Sum of item prices."""
        return sum(i["price"] for i in items)


    class OrderService:
        retries = 3

        def place(self, order):
            amount = total_price(order["items"])
            return amount


    @app.post("/orders")
    def create_order(order: dict):
        amount = total_price(order["items"])
        httpx.post("http://payments:8080/charge", json={"amount": amount})
        publish("order.created", order)
        return {"ok": True}


    def publish(topic, payload):
        pass
    '''
).lstrip()

TS = textwrap.dedent(
    """
    import { Router } from "express";

    export class PaymentClient {
      async charge(amount: number): Promise<boolean> {
        const ok = await this.post("/charge", { amount });
        return validate(ok);
      }
    }

    export function validate(result: unknown): boolean {
      return Boolean(result);
    }

    export const handler = async (req) => {
      return validate(req);
    };
    """
).lstrip()

JS_SERVER = textwrap.dedent(
    """
    const express = require("express");
    const app = express();

    app.post("/charge", (req, res) => {
      res.json({ charged: true });
    });

    consumer.subscribe("order.created", handleOrder);

    function handleOrder(msg) {
      return msg;
    }
    """
).lstrip()

COMPOSE = textwrap.dedent(
    """
    services:
      orders:
        build: ./orders
        environment:
          PAYMENTS_URL: http://payments:8080
        depends_on: [postgres]
      payments:
        build: ./payments
      postgres:
        image: postgres:16
    """
).lstrip()


def test_python_symbols_lines_and_calls() -> None:
    parsed = parse_python(PY, "orders/app.py")
    by_name = {s.qualname: s for s in parsed.symbols}
    assert {"total_price", "OrderService", "OrderService.place", "create_order", "publish"} <= set(
        by_name
    )
    create = by_name["create_order"]
    assert create.line_start < create.line_end and "total_price" in create.calls
    assert by_name["OrderService"].outline and "def place" in by_name["OrderService"].outline
    assert "httpx" in parsed.imports


def test_typescript_symbols_with_brace_matching() -> None:
    parsed = parse_braced(TS, "client.ts", "typescript")
    names = {s.qualname: s for s in parsed.symbols}
    assert "PaymentClient" in names and "PaymentClient.charge" in names
    assert "validate" in names and "handler" in names
    assert names["validate"].line_end > names["validate"].line_start
    assert "validate" in names["PaymentClient.charge"].calls
    assert "express" in parsed.imports


def test_code_document_has_no_duplicate_class_bodies(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text(PY)
    doc, _ = code_file_to_document(f, tmp_path, CodeConfig())
    symbols = [b.meta["symbol"] for b in doc.blocks]
    assert "OrderService" in symbols and "OrderService.place" in symbols
    outline = next(b for b in doc.blocks if b.meta["symbol"] == "OrderService")
    assert "amount = total_price" not in outline.text  # body lives only in the method chunk
    assert all(b.line_start and b.line_end for b in doc.blocks)


def _make_repo(root: Path) -> Path:
    (root / "orders").mkdir(parents=True)
    (root / "payments").mkdir()
    (root / "node_modules" / "junk").mkdir(parents=True)
    (root / "docker-compose.yml").write_text(COMPOSE)
    (root / "orders" / "Dockerfile").write_text("FROM python:3.12\n")
    (root / "orders" / "app.py").write_text(PY)
    (root / "orders" / "README.md").write_text("# Orders\nCreates and tracks customer orders.\n")
    (root / "payments" / "Dockerfile").write_text("FROM node:20\n")
    (root / "payments" / "server.js").write_text(JS_SERVER)
    (root / "node_modules" / "junk" / "index.js").write_text("function junk() {}\n")
    return root


def test_service_discovery(tmp_path: Path) -> None:
    root = _make_repo(tmp_path / "shop")
    files = list(iter_repo_files(root, CodeConfig()))
    assert not any("node_modules" in str(f) for f in files)
    smap = discover_services(root, files)
    assert set(smap.services) == {"orders", "payments"}
    assert "postgres" in smap.datastores
    edges = {(e.source, e.relation, e.target) for e in smap.edges}
    assert ("orders", "CALLS", "payments") in edges
    assert ("orders", "DEPENDS_ON", "postgres") in edges
    assert ("orders", "PUBLISHES", "order.created") in edges
    assert ("payments", "SUBSCRIBES", "order.created") in edges
    labels = {e.label for e in smap.services["orders"].endpoints}
    assert "POST /orders" in labels
    assert "POST /charge" in {e.label for e in smap.services["payments"].endpoints}


@pytest.fixture
def repo_engine(tmp_path: Path) -> RagEngine:
    eng = RagEngine(make_settings(tmp_path), llm=ScriptedLLM(), embedder=HashEmbedder())
    report = eng.ingest_repo(_make_repo(tmp_path / "shop"))
    assert report.services == 2 and not report.failed
    assert report.symbols >= 5
    return eng


def test_repo_ingestion_builds_call_graph(repo_engine: RagEngine) -> None:
    hits = repo_engine.graph_retriever.code("Who calls total_price?")
    facts = hits[0].chunk.text
    assert "create_order" in facts and "OrderService.place" in facts
    assert any(h.chunk.modality == "code" and h.chunk.line_start for h in hits[1:])


def test_service_question_uses_service_graph(repo_engine: RagEngine) -> None:
    assert "code" in repo_engine.available_routes()
    result = repo_engine.ask("Which services call payments, and what is affected if it is down?")
    assert result.route == "code"
    facts = next(e for e in result.evidence if e.source == "graph_services")
    assert "orders" in facts.chunk.text and "CALLS" in facts.chunk.text
    assert "Transitively affected" in facts.chunk.text


def test_reingesting_unchanged_repo_skips_files(repo_engine: RagEngine, tmp_path: Path) -> None:
    report = repo_engine.ingest_repo(tmp_path / "shop")
    assert report.indexed == 0 and report.skipped >= 2
