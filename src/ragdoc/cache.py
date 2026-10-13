"""Exact-match answer cache (SQLite).

The key covers the question, document filter, the indexed corpus fingerprint and the
answer-affecting config, so re-ingesting a document or changing models never serves a
stale answer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from ragdoc.types import AnswerResult


class AnswerCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, data TEXT)")

    @staticmethod
    def key(question: str, doc_ids: list[str] | None, corpus_fp: str, config_fp: str) -> str:
        normalised = " ".join(question.lower().split())
        payload = json.dumps([normalised, sorted(doc_ids or []), corpus_fp, config_fp])
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> AnswerResult | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM answers WHERE key = ?", (key,)).fetchone()
        return AnswerResult.model_validate_json(row[0]) if row else None

    def put(self, key: str, result: AnswerResult) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO answers (key, data) VALUES (?, ?)",
                (key, result.model_dump_json()),
            )

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM answers")
