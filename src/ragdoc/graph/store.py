"""Knowledge graph storage: SQLite (local, zero services) or Neo4j (scale, Cypher, GDS).

One graph holds three layers, distinguished by node ``type``:
* document knowledge: person, organization, concept, ... extracted by the LLM;
* code: file, symbol (DEFINES, CALLS, IMPORTS);
* architecture: service, endpoint, topic, datastore (CALLS, EXPOSES, PUBLISHES,
  SUBSCRIBES, DEPENDS_ON).
Every node and edge keeps ``source_ids`` (chunk ids), so graph answers cite real text.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

MAX_DESCRIPTION_CHARS = 1500
MAX_SOURCES = 200


def normalise_name(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"^(the|a|an)\s+", "", n)
    n = re.sub(r"[^\w\s./:#@+-]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def node_id(node_type: str, name: str) -> str:
    """Entities share one namespace (types disagree across chunks); structural nodes don't."""
    structural = {"file", "symbol", "service", "endpoint", "topic", "datastore", "module"}
    prefix = node_type if node_type in structural else "entity"
    return f"{prefix}:{normalise_name(name)}"


class Node(BaseModel):
    id: str
    name: str
    type: str
    description: str = ""
    source_ids: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    degree: int = 0
    community: str | None = None


class GEdge(BaseModel):
    source: str
    target: str
    relation: str
    description: str = ""
    weight: float = 1.0
    source_ids: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}|{self.relation}|{self.target}"


class Community(BaseModel):
    id: str
    level: int
    parent: str | None = None
    node_ids: list[str]
    title: str = ""
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    rating: float = 0.0
    source_ids: list[str] = Field(default_factory=list)


class GraphStore(Protocol):
    def upsert_nodes(self, nodes: Sequence[Node]) -> None: ...
    def upsert_edges(self, edges: Sequence[GEdge]) -> None: ...
    def get_nodes(self, ids: Iterable[str]) -> list[Node]: ...
    def find_nodes(
        self, names: Iterable[str], types: Sequence[str] | None = None
    ) -> list[Node]: ...
    def edges_for(
        self, ids: Iterable[str], relations: Sequence[str] | None = None, limit: int = 200
    ) -> list[GEdge]: ...
    def all_nodes(self, types: Sequence[str] | None = None) -> list[Node]: ...
    def all_edges(self, types: Sequence[str] | None = None) -> list[GEdge]: ...
    def set_communities(self, communities: Sequence[Community]) -> None: ...
    def communities(self, level: int | None = None) -> list[Community]: ...
    def delete_document(self, doc_id: str) -> None: ...
    def stats(self) -> dict[str, Any]: ...


def _merge_text(a: str, b: str) -> str:
    if not b or b in a:
        return a
    if not a:
        return b[:MAX_DESCRIPTION_CHARS]
    return (a + " | " + b)[:MAX_DESCRIPTION_CHARS]


def _merge_list(a: list[str], b: list[str], cap: int = MAX_SOURCES) -> list[str]:
    return list(dict.fromkeys([*a, *b]))[:cap]


class SQLiteGraphStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, type TEXT, name TEXT, norm TEXT, data TEXT);
            CREATE INDEX IF NOT EXISTS idx_nodes_norm ON nodes(norm);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
            CREATE TABLE IF NOT EXISTS edges (key TEXT PRIMARY KEY, source TEXT, target TEXT, relation TEXT, data TEXT);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
            CREATE TABLE IF NOT EXISTS communities (id TEXT PRIMARY KEY, level INTEGER, data TEXT);
            CREATE TABLE IF NOT EXISTS mentions (node_id TEXT, doc_id TEXT);
            CREATE INDEX IF NOT EXISTS idx_mentions_doc ON mentions(doc_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mentions_u ON mentions(node_id, doc_id);
            """
        )

    # ---------------------------------------------------------------- writes

    def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        with self._lock, self._db:
            existing = {n.id: n for n in self.get_nodes(n.id for n in nodes)}
            for n in nodes:
                old = existing.get(n.id)
                if old is not None:
                    n = old.model_copy(
                        update={
                            "description": _merge_text(old.description, n.description),
                            "source_ids": _merge_list(old.source_ids, n.source_ids),
                            "doc_ids": _merge_list(old.doc_ids, n.doc_ids),
                            "data": {**old.data, **n.data},
                            "type": old.type if old.type != "concept" else n.type,
                        }
                    )
                existing[n.id] = n
                self._db.execute(
                    "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?)",
                    (n.id, n.type, n.name, normalise_name(n.name), n.model_dump_json()),
                )
                self._db.executemany(
                    "INSERT OR IGNORE INTO mentions VALUES (?, ?)", [(n.id, d) for d in n.doc_ids]
                )

    def upsert_edges(self, edges: Sequence[GEdge]) -> None:
        with self._lock, self._db:
            keys = [e.key for e in edges]
            existing: dict[str, GEdge] = {}
            for start in range(0, len(keys), 500):
                part = keys[start : start + 500]
                rows = self._db.execute(
                    f"SELECT data FROM edges WHERE key IN ({','.join('?' * len(part))})", part
                ).fetchall()
                for (data,) in rows:
                    e = GEdge.model_validate_json(data)
                    existing[e.key] = e
            for e in edges:
                old = existing.get(e.key)
                if old is not None:
                    e = old.model_copy(
                        update={
                            "description": _merge_text(old.description, e.description),
                            "weight": old.weight + e.weight,
                            "source_ids": _merge_list(old.source_ids, e.source_ids),
                            "data": {**old.data, **e.data},
                        }
                    )
                existing[e.key] = e
                self._db.execute(
                    "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?)",
                    (e.key, e.source, e.target, e.relation, e.model_dump_json()),
                )

    def set_communities(self, communities: Sequence[Community]) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM communities")
            self._db.executemany(
                "INSERT INTO communities VALUES (?, ?, ?)",
                [(c.id, c.level, c.model_dump_json()) for c in communities],
            )

    def delete_document(self, doc_id: str) -> None:
        """Drop a document's contributions; nodes/edges with no remaining sources go too."""
        with self._lock, self._db:
            ids = [
                r[0]
                for r in self._db.execute(
                    "SELECT DISTINCT node_id FROM mentions WHERE doc_id = ?", (doc_id,)
                )
            ]
            prefix = f"{doc_id}:"
            for node in self.get_nodes(ids):
                sources = [s for s in node.source_ids if not s.startswith(prefix)]
                docs = [d for d in node.doc_ids if d != doc_id]
                if not docs:
                    self._db.execute("DELETE FROM nodes WHERE id = ?", (node.id,))
                    self._db.execute(
                        "DELETE FROM edges WHERE source = ? OR target = ?", (node.id, node.id)
                    )
                else:
                    updated = node.model_copy(update={"source_ids": sources, "doc_ids": docs})
                    self._db.execute(
                        "UPDATE nodes SET data = ? WHERE id = ?",
                        (updated.model_dump_json(), node.id),
                    )
            self._db.execute("DELETE FROM mentions WHERE doc_id = ?", (doc_id,))

    # ---------------------------------------------------------------- reads

    def get_nodes(self, ids: Iterable[str]) -> list[Node]:
        ids = list(dict.fromkeys(ids))
        out: list[Node] = []
        with self._lock:
            for start in range(0, len(ids), 500):
                part = ids[start : start + 500]
                rows = self._db.execute(
                    f"SELECT data FROM nodes WHERE id IN ({','.join('?' * len(part))})", part
                ).fetchall()
                out += [Node.model_validate_json(r[0]) for r in rows]
        return out

    def find_nodes(self, names: Iterable[str], types: Sequence[str] | None = None) -> list[Node]:
        norms = list({normalise_name(n) for n in names if n.strip()})
        if not norms:
            return []
        sql = f"SELECT data FROM nodes WHERE norm IN ({','.join('?' * len(norms))})"
        params: list[str] = list(norms)
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            params += list(types)
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [Node.model_validate_json(r[0]) for r in rows]

    def edges_for(
        self, ids: Iterable[str], relations: Sequence[str] | None = None, limit: int = 200
    ) -> list[GEdge]:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        sql = f"SELECT data FROM edges WHERE (source IN ({marks}) OR target IN ({marks}))"
        params: list[str] = ids + ids
        if relations:
            sql += f" AND relation IN ({','.join('?' * len(relations))})"
            params += list(relations)
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        edges = [GEdge.model_validate_json(r[0]) for r in rows]
        return sorted(edges, key=lambda e: -e.weight)[:limit]

    def all_nodes(self, types: Sequence[str] | None = None) -> list[Node]:
        with self._lock:
            if types:
                rows = self._db.execute(
                    f"SELECT data FROM nodes WHERE type IN ({','.join('?' * len(types))})",
                    list(types),
                ).fetchall()
            else:
                rows = self._db.execute("SELECT data FROM nodes").fetchall()
        return [Node.model_validate_json(r[0]) for r in rows]

    def all_edges(self, types: Sequence[str] | None = None) -> list[GEdge]:
        with self._lock:
            if types:
                rows = self._db.execute(
                    f"SELECT data FROM edges WHERE relation IN ({','.join('?' * len(types))})",
                    list(types),
                ).fetchall()
            else:
                rows = self._db.execute("SELECT data FROM edges").fetchall()
        return [GEdge.model_validate_json(r[0]) for r in rows]

    def communities(self, level: int | None = None) -> list[Community]:
        with self._lock:
            if level is None:
                rows = self._db.execute("SELECT data FROM communities").fetchall()
            else:
                rows = self._db.execute(
                    "SELECT data FROM communities WHERE level = ?", (level,)
                ).fetchall()
        return [Community.model_validate_json(r[0]) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            types = dict(
                self._db.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type").fetchall()
            )
            rels = dict(
                self._db.execute(
                    "SELECT relation, COUNT(*) FROM edges GROUP BY relation"
                ).fetchall()
            )
            n_comm = self._db.execute("SELECT COUNT(*) FROM communities").fetchone()[0]
        return {"nodes": types, "edges": rels, "communities": n_comm}


class Neo4jGraphStore:
    """Neo4j backend for large graphs. Same interface; community detection can run in-database
    with the Graph Data Science plugin (see ``graph.communities``)."""

    def __init__(
        self, uri: str, user: str, password: str | None, database: str | None = None
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password or ""))
        self._database = database
        with self._session() as s:
            s.run("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
            s.run("CREATE INDEX node_norm IF NOT EXISTS FOR (n:Node) ON (n.norm)")
            s.run("CREATE INDEX node_doc IF NOT EXISTS FOR (n:Node) ON (n.doc_ids)")

    def _session(self) -> Any:
        return self._driver.session(database=self._database)

    @staticmethod
    def _node(record: Any) -> Node:
        return Node.model_validate_json(record["data"])

    def upsert_nodes(self, nodes: Sequence[Node]) -> None:
        existing = {n.id: n for n in self.get_nodes(n.id for n in nodes)}
        rows = []
        for n in nodes:
            old = existing.get(n.id)
            if old is not None:
                n = old.model_copy(
                    update={
                        "description": _merge_text(old.description, n.description),
                        "source_ids": _merge_list(old.source_ids, n.source_ids),
                        "doc_ids": _merge_list(old.doc_ids, n.doc_ids),
                        "data": {**old.data, **n.data},
                    }
                )
            existing[n.id] = n
            rows.append(
                {
                    "id": n.id,
                    "type": n.type,
                    "norm": normalise_name(n.name),
                    "doc_ids": n.doc_ids,
                    "data": n.model_dump_json(),
                }
            )
        with self._session() as s:
            s.run(
                "UNWIND $rows AS r MERGE (n:Node {id: r.id}) "
                "SET n.type = r.type, n.norm = r.norm, n.doc_ids = r.doc_ids, n.data = r.data",
                rows=rows,
            )

    def upsert_edges(self, edges: Sequence[GEdge]) -> None:
        rows = [
            {
                "key": e.key,
                "s": e.source,
                "t": e.target,
                "rel": e.relation,
                "w": e.weight,
                "data": e.model_dump_json(),
            }
            for e in edges
        ]
        with self._session() as s:
            s.run(
                "UNWIND $rows AS r MATCH (a:Node {id: r.s}), (b:Node {id: r.t}) "
                "MERGE (a)-[e:REL {key: r.key}]->(b) "
                "ON CREATE SET e.relation = r.rel, e.weight = r.w, e.data = r.data "
                "ON MATCH SET e.weight = e.weight + r.w, e.data = r.data",
                rows=rows,
            )

    def get_nodes(self, ids: Iterable[str]) -> list[Node]:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        with self._session() as s:
            return [
                self._node(r)
                for r in s.run("MATCH (n:Node) WHERE n.id IN $ids RETURN n.data AS data", ids=ids)
            ]

    def find_nodes(self, names: Iterable[str], types: Sequence[str] | None = None) -> list[Node]:
        norms = list({normalise_name(n) for n in names if n.strip()})
        query = "MATCH (n:Node) WHERE n.norm IN $norms" + (" AND n.type IN $types" if types else "")
        with self._session() as s:
            return [
                self._node(r)
                for r in s.run(
                    query + " RETURN n.data AS data", norms=norms, types=list(types or [])
                )
            ]

    def edges_for(
        self, ids: Iterable[str], relations: Sequence[str] | None = None, limit: int = 200
    ) -> list[GEdge]:
        query = (
            "MATCH (a:Node)-[e:REL]-(b:Node) WHERE a.id IN $ids"
            + (" AND e.relation IN $rels" if relations else "")
            + " RETURN DISTINCT e.data AS data, e.weight AS w ORDER BY w DESC LIMIT $limit"
        )
        with self._session() as s:
            return [
                GEdge.model_validate_json(r["data"])
                for r in s.run(query, ids=list(ids), rels=list(relations or []), limit=limit)
            ]

    def all_nodes(self, types: Sequence[str] | None = None) -> list[Node]:
        query = (
            "MATCH (n:Node)"
            + (" WHERE n.type IN $types" if types else "")
            + " RETURN n.data AS data"
        )
        with self._session() as s:
            return [self._node(r) for r in s.run(query, types=list(types or []))]

    def all_edges(self, types: Sequence[str] | None = None) -> list[GEdge]:
        query = (
            "MATCH ()-[e:REL]->()"
            + (" WHERE e.relation IN $types" if types else "")
            + " RETURN e.data AS data"
        )
        with self._session() as s:
            return [
                GEdge.model_validate_json(r["data"]) for r in s.run(query, types=list(types or []))
            ]

    def set_communities(self, communities: Sequence[Community]) -> None:
        with self._session() as s:
            s.run("MATCH (c:Community) DETACH DELETE c")
            s.run(
                "UNWIND $rows AS r CREATE (c:Community {id: r.id, level: r.level, data: r.data})",
                rows=[
                    {"id": c.id, "level": c.level, "data": c.model_dump_json()} for c in communities
                ],
            )

    def communities(self, level: int | None = None) -> list[Community]:
        query = (
            "MATCH (c:Community)"
            + (" WHERE c.level = $level" if level is not None else "")
            + " RETURN c.data AS data"
        )
        with self._session() as s:
            return [Community.model_validate_json(r["data"]) for r in s.run(query, level=level)]

    def delete_document(self, doc_id: str) -> None:
        with self._session() as s:
            s.run(
                "MATCH (n:Node) WHERE $doc IN n.doc_ids "
                "SET n.doc_ids = [d IN n.doc_ids WHERE d <> $doc] "
                "WITH n WHERE size(n.doc_ids) = 0 DETACH DELETE n",
                doc=doc_id,
            )

    def stats(self) -> dict[str, Any]:
        with self._session() as s:
            nodes = {
                r["t"]: r["c"] for r in s.run("MATCH (n:Node) RETURN n.type AS t, count(*) AS c")
            }
            edges = {
                r["t"]: r["c"]
                for r in s.run("MATCH ()-[e:REL]->() RETURN e.relation AS t, count(*) AS c")
            }
            comm = s.run("MATCH (c:Community) RETURN count(c) AS c").single()["c"]
        return {"nodes": nodes, "edges": edges, "communities": comm}
