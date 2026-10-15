"""API smoke tests (skipped when the [api] extra is not installed)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import FIXTURES
from ragdoc.api.app import create_app
from ragdoc.engine import RagEngine


def test_health_documents_and_ask(engine: RagEngine) -> None:
    client = TestClient(create_app(engine))
    assert client.get("/health").json()["documents"] == 1
    assert len(client.get("/documents").json()) == 1

    body = client.post("/ask", json={"question": "Who is eligible for remote work?"}).json()
    assert body["citations"] and "trace" not in body

    debug = client.post(
        "/ask", json={"question": "Who is eligible for remote work?", "debug": True}
    ).json()
    assert debug["trace"]


def test_upload_and_stream(engine: RagEngine) -> None:
    client = TestClient(create_app(engine))
    with (FIXTURES / "handbook.md").open("rb") as fh:
        resp = client.post("/documents", files={"file": ("handbook.md", fh, "text/markdown")})
    assert resp.status_code == 201 and resp.json()["skipped"] is True

    with client.stream(
        "POST", "/ask/stream", json={"question": "Who is eligible for remote work?"}
    ) as resp:
        text = "".join(resp.iter_text())
    assert "event: step" in text and "event: result" in text


def test_rejects_unsupported_upload(engine: RagEngine) -> None:
    client = TestClient(create_app(engine))
    resp = client.post("/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert resp.status_code == 415
