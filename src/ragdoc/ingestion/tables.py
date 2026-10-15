"""Tables: keep them intact for reading, and queryable for computing.

* Retrieval: tables become their own chunks (header repeated on every row-group chunk) with
  contextual headers, so "what does Table 3 say about X" works like any text question.
* Computation: every table (and every chart's extracted data) is also loaded into SQLite.
  Aggregate or numeric questions ("total revenue across regions", "which quarter grew
  fastest") are answered with a read-only, LLM-written SQL query over the real cells instead
  of asking the model to do arithmetic on text.
* Tables split across page breaks are stitched back together first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragdoc.models.llm import LLM, structured
from ragdoc.prompts import render
from ragdoc.retrieval.bm25 import tokenize
from ragdoc.types import Block, Chunk, ScoredChunk

log = logging.getLogger(__name__)

_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
_NUM_RE = re.compile(r"^[\(\-+]?[$€£¥₹]?\s*-?[\d,]*\.?\d+\s*[%kKmMbB]?\)?$")


# --------------------------------------------------------------------------- markdown tables


def parse_markdown_table(md: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return [], []

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    body = [cells(ln) for ln in lines[1:] if not _SEP_RE.match(ln)]
    width = len(header)
    rows = [(r + [""] * width)[:width] for r in body]
    return header, rows


def to_markdown(header: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(fmt(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def merge_multipage_tables(blocks: list[Block]) -> list[Block]:
    """Join a table that continues on the next page (same header, or same width + no header)."""
    out: list[Block] = []
    for block in blocks:
        prev = out[-1] if out else None
        if (
            prev is not None
            and block.kind == "table"
            and prev.kind == "table"
            and prev.page is not None
            and block.page is not None
            and block.page - prev.page == 1
        ):
            h1, r1 = parse_markdown_table(prev.text)
            h2, r2 = parse_markdown_table(block.text)
            if h1 and h2 and len(h1) == len(h2):
                continuation = r2 if h1 == h2 else [h2, *r2]  # no repeated header → h2 is data
                merged = to_markdown(h1, r1 + continuation)
                out[-1] = prev.model_copy(
                    update={"text": merged, "meta": {**prev.meta, "pages": [prev.page, block.page]}}
                )
                continue
        out.append(block)
    return out


# --------------------------------------------------------------------------- SQL store


def _parse_number(value: str) -> float | None:
    v = value.strip()
    if not v or not _NUM_RE.match(v):
        return None
    negative = (v.startswith("(") and v.endswith(")")) or v.startswith("-")
    multiplier = {"k": 1e3, "m": 1e6, "b": 1e9}.get(v.rstrip(")")[-1:].lower(), 1.0)
    digits = re.sub(r"[^\d.]", "", v)
    if not digits or digits == ".":
        return None
    number = float(digits) * multiplier
    return -number if negative else number


def _column_names(headers: list[str]) -> list[str]:
    names: list[str] = []
    for i, h in enumerate(headers):
        base = re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_") or f"col{i + 1}"
        if base[0].isdigit():
            base = f"c_{base}"
        name, n = base, 2
        while name in names:
            name, n = f"{base}_{n}", n + 1
        names.append(name)
    return names


class TableInfo(BaseModel):
    table_id: str
    doc_id: str
    source: str
    page: int | None
    title: str
    columns: list[str]
    headers: list[str]
    n_rows: int
    origin: str = "document"  # document | chart
    approximate: bool = False


class TableStore:
    """Every extracted table as a real SQLite table, plus a registry for discovery."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS _registry (table_id TEXT PRIMARY KEY, doc_id TEXT, data TEXT)"
        )

    def add_table(
        self,
        *,
        doc_id: str,
        source: str,
        page: int | None,
        title: str,
        markdown: str,
        origin: str = "document",
    ) -> str | None:
        headers, rows = parse_markdown_table(markdown)
        if not headers or not rows:
            return None
        table_id = hashlib.sha256(f"{doc_id}|{page}|{markdown}".encode()).hexdigest()[:12]
        columns = _column_names(headers)
        numeric = [
            all(_parse_number(r[i]) is not None for r in rows if r[i].strip())
            and any(r[i].strip() for r in rows)
            for i in range(len(columns))
        ]
        col_defs = ", ".join(
            f'"{c}" {"REAL" if numeric[i] else "TEXT"}' for i, c in enumerate(columns)
        )
        values = [
            [(_parse_number(v) if numeric[i] else v) for i, v in enumerate(row)] for row in rows
        ]
        info = TableInfo(
            table_id=table_id,
            doc_id=doc_id,
            source=source,
            page=page,
            title=title[:200],
            columns=columns,
            headers=headers,
            n_rows=len(rows),
            origin=origin,
            approximate=origin == "chart",
        )
        with self._lock, self._db:
            self._db.execute(f'DROP TABLE IF EXISTS "t_{table_id}"')
            self._db.execute(f'CREATE TABLE "t_{table_id}" ({col_defs})')
            self._db.executemany(
                f'INSERT INTO "t_{table_id}" VALUES ({",".join("?" * len(columns))})', values
            )
            self._db.execute(
                "INSERT OR REPLACE INTO _registry VALUES (?, ?, ?)",
                (table_id, doc_id, info.model_dump_json()),
            )
        return table_id

    def tables(self, doc_ids: list[str] | None = None) -> list[TableInfo]:
        with self._lock:
            if doc_ids:
                q = f"SELECT data FROM _registry WHERE doc_id IN ({','.join('?' * len(doc_ids))})"
                rows = self._db.execute(q, doc_ids).fetchall()
            else:
                rows = self._db.execute("SELECT data FROM _registry").fetchall()
        return [TableInfo.model_validate_json(r[0]) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM _registry").fetchone()[0])

    def get(self, table_ids: list[str]) -> list[TableInfo]:
        ids = list(dict.fromkeys(table_ids))
        if not ids:
            return []
        with self._lock:
            rows = self._db.execute(
                f"SELECT data FROM _registry WHERE table_id IN ({','.join('?' * len(ids))})", ids
            ).fetchall()
        found = {t.table_id: t for t in (TableInfo.model_validate_json(r[0]) for r in rows)}
        return [found[i] for i in ids if i in found]

    def schema_text(self, info: TableInfo, sample_rows: int = 3) -> str:
        with self._lock:
            sample = self._db.execute(
                f'SELECT * FROM "t_{info.table_id}" LIMIT {sample_rows}'
            ).fetchall()
            types = self._db.execute(f'PRAGMA table_info("t_{info.table_id}")').fetchall()
        cols = ", ".join(
            f"{name} {ctype} -- {header!r}"
            for (_, name, ctype, *_), header in zip(types, info.headers, strict=False)
        )
        note = " (values approximate, read from a chart)" if info.approximate else ""
        return (
            f'Table "t_{info.table_id}": {info.title or "untitled"} — {info.source}, '
            f"page {info.page or 'n/a'}, {info.n_rows} rows{note}\n"
            f"Columns: {cols}\nSample rows: {json.dumps(sample, default=str)}"
        )

    def run_sql(self, sql: str, max_rows: int = 200) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Execute a single read-only SELECT. Writes, PRAGMAs and ATTACH are refused."""
        statement = sql.strip().rstrip(";")
        if ";" in statement or not re.match(r"^(select|with)\b", statement, re.IGNORECASE):
            raise ValueError("Only a single SELECT statement is allowed")

        def authorizer(action: int, *args: Any) -> int:
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        with self._lock:
            self._db.set_authorizer(authorizer)
            try:
                cursor = self._db.execute(statement)
                rows = cursor.fetchmany(max_rows)
                columns = [d[0] for d in cursor.description or []]
            finally:
                self._db.set_authorizer(None)
        return columns, rows

    def delete_document(self, doc_id: str) -> None:
        for info in self.tables([doc_id]):
            with self._lock, self._db:
                self._db.execute(f'DROP TABLE IF EXISTS "t_{info.table_id}"')
                self._db.execute("DELETE FROM _registry WHERE table_id = ?", (info.table_id,))


# --------------------------------------------------------------------------- text-to-SQL


class SqlPlan(BaseModel):
    answerable: bool = Field(description="False if none of the tables can answer the question")
    sql: str = ""
    table_ids: list[str] = Field(default_factory=list)
    explanation: str = ""


def candidate_tables(
    store: TableStore,
    question: str,
    evidence: list[ScoredChunk],
    doc_ids: list[str] | None = None,
    limit: int = 5,
) -> list[TableInfo]:
    """Tables behind retrieved table/chart chunks first, then keyword matches on titles/headers."""
    ids: list[str] = []
    for e in evidence:
        ids += [t for t in e.chunk.metadata.get("table_ids", []) if t not in ids]
    picked = store.get(ids)
    # Keyword scan of the registry only when it is small or scoped; at scale the table
    # chunks found by retrieval are the candidates.
    if len(picked) < limit and (doc_ids or store.count() <= 5000):
        q = set(tokenize(question))
        scored = sorted(
            (
                (len(q & set(tokenize(t.title + " " + " ".join(t.headers)))), t)
                for t in store.tables(doc_ids)
                if t.table_id not in ids
            ),
            key=lambda pair: -pair[0],
        )
        picked += [t for score, t in scored if score > 0][: limit - len(picked)]
    return picked[:limit]


def answer_with_sql(
    llm: LLM, store: TableStore, question: str, tables: list[TableInfo], max_rows: int = 200
) -> ScoredChunk | None:
    """Plan and run a SQL query; returns the result as an evidence chunk (or None)."""
    if not tables:
        return None
    schemas = "\n\n".join(store.schema_text(t) for t in tables)
    user = f"Question: {question}\n\nTables:\n{schemas}"
    error = ""
    for _attempt in range(2):
        plan = structured(
            llm,
            system=render("text_to_sql"),
            user=user + (f"\n\nYour previous query failed: {error}" if error else ""),
            schema=SqlPlan,
            role="strong",
            task="text_to_sql",
        )
        if not plan.answerable or not plan.sql:
            return None
        try:
            columns, rows = store.run_sql(plan.sql, max_rows)
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.info("SQL attempt failed: %s", error)
    else:
        return None

    used = [t for t in tables if t.table_id in plan.table_ids or f"t_{t.table_id}" in plan.sql]
    used = used or tables[:1]
    pages = sorted({t.page for t in used if t.page})
    approx = any(t.approximate for t in used)
    text = (
        f"Computed from table(s): {', '.join(t.title or t.table_id for t in used)}"
        f"{' (chart values are approximate)' if approx else ''}\n"
        f"Query: {plan.sql}\n"
        f"Result ({len(rows)} row{'s' if len(rows) != 1 else ''}):\n"
        f"{to_markdown(columns, [list(r) for r in rows]) if rows else '(no rows)'}"
    )
    chunk = Chunk(
        chunk_id=f"sql:{hashlib.sha256(plan.sql.encode()).hexdigest()[:10]}",
        doc_id=used[0].doc_id,
        source=used[0].source,
        text=text,
        pages=pages,
        modality="table",
        metadata={"sql": plan.sql, "table_ids": [t.table_id for t in used]},
    )
    return ScoredChunk(chunk=chunk, score=1.0, source="sql", relevance=3)
