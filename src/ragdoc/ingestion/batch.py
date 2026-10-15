"""Resumable, parallel batch ingestion for large corpora.

A job table (SQLite) records every file's status, so a crashed or interrupted run of a
million-document corpus resumes where it stopped and retries failures with a cap. Workers
are threads (the work is I/O-bound: model APIs and the vector DB). For multi-machine scale,
run one ``ragdoc ingest-batch`` per shard (``--shard i/n``) against a shared Qdrant server
and graph store; shards are disjoint by path hash, so no coordination service is needed.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragdoc.engine import RagEngine

log = logging.getLogger(__name__)


class JobTable:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (path TEXT PRIMARY KEY, status TEXT, doc_id TEXT, "
            "error TEXT, attempts INTEGER DEFAULT 0, updated REAL)"
        )

    def register(self, paths: Iterable[str]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR IGNORE INTO jobs (path, status, attempts, updated) VALUES (?, 'pending', 0, ?)",
                [(p, time.time()) for p in paths],
            )

    def pending(self, max_attempts: int) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT path FROM jobs WHERE status IN ('pending', 'failed', 'running') AND attempts < ? ORDER BY path",
                (max_attempts,),
            ).fetchall()
        return [r[0] for r in rows]

    def mark(
        self, path: str, status: str, doc_id: str | None = None, error: str | None = None
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE jobs SET status = ?, doc_id = COALESCE(?, doc_id), error = ?, updated = ?, "
                "attempts = attempts + CASE WHEN ? = 'running' THEN 1 ELSE 0 END WHERE path = ?",
                (status, doc_id, error, time.time(), status, path),
            )

    def summary(self) -> dict[str, int]:
        with self._lock:
            return dict(
                self._db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
            )

    def failures(self, limit: int = 20) -> list[tuple[str, str]]:
        with self._lock:
            return self._db.execute(
                "SELECT path, error FROM jobs WHERE status = 'failed' ORDER BY updated DESC LIMIT ?",
                (limit,),
            ).fetchall()


def in_shard(path: str, shard: tuple[int, int] | None) -> bool:
    if shard is None:
        return True
    index, total = shard
    return int(hashlib.md5(path.encode()).hexdigest(), 16) % total == index


class BatchIngestor:
    def __init__(self, engine: RagEngine, jobs_path: Path | None = None) -> None:
        self.engine = engine
        self.jobs = JobTable(jobs_path or engine.settings.collection_dir / "jobs.sqlite")

    def run(
        self,
        paths: Iterable[Path],
        workers: int | None = None,
        max_attempts: int = 3,
        shard: tuple[int, int] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, int]:
        selected = [str(p) for p in paths if in_shard(str(p), shard)]
        self.jobs.register(selected)
        todo = self.jobs.pending(max_attempts)
        workers = workers or self.engine.settings.ingestion.workers
        log.info("Batch ingest: %d pending of %d registered", len(todo), len(selected))

        def one(path: str) -> tuple[str, str | None, str | None]:
            self.jobs.mark(path, "running")
            try:
                report = self.engine.ingest(path)
                return path, report.doc_id, None
            except Exception as exc:
                return path, None, f"{type(exc).__name__}: {exc}"[:2000]

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(one, p) for p in todo]
            for future in as_completed(futures):
                path, doc_id, error = future.result()
                self.jobs.mark(path, "failed" if error else "done", doc_id, error)
                done += 1
                if on_progress:
                    on_progress({"done": done, "total": len(todo), "path": path, "error": error})
        return self.jobs.summary()
