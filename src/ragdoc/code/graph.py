"""Deterministic code and service graphs (no LLM calls).

Code graph:    file —DEFINES→ symbol, symbol —CALLS→ symbol, file —IMPORTS→ module
Service graph: service —CALLS/DEPENDS_ON/PUBLISHES/SUBSCRIBES/EXPOSES→ service/datastore/
               topic/endpoint, with file:line evidence on every edge.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ragdoc.code.parser import CodeFile
from ragdoc.code.services import ServiceMap, _norm
from ragdoc.graph.store import GEdge, Node, node_id
from ragdoc.types import Chunk


def symbol_node_id(path: str, qualname: str) -> str:
    return node_id("symbol", f"{path}#{qualname}")


def code_graph(
    files: Sequence[tuple[CodeFile, str, list[Chunk], str | None]],
) -> tuple[list[Node], list[GEdge]]:
    """``files``: (parsed file, doc_id, its indexed chunks, service)."""
    nodes: list[Node] = []
    edges: list[GEdge] = []
    by_name: dict[str, list[tuple[str, str, str | None]]] = defaultdict(
        list
    )  # short → (id, path, service)

    for parsed, doc_id, chunks, service in files:
        file_id = node_id("file", parsed.path)
        chunk_by_symbol: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            sym = str(c.metadata.get("symbol", "")).split(" (part")[0]
            chunk_by_symbol[sym].append(c.chunk_id)
        nodes.append(
            Node(
                id=file_id,
                name=parsed.path,
                type="file",
                description=f"{parsed.language} file, {parsed.n_lines} lines, {len(parsed.symbols)} symbols",
                source_ids=chunk_by_symbol.get("<module>", [])[:5],
                doc_ids=[doc_id],
                data={"language": parsed.language, "service": service, "path": parsed.path},
            )
        )
        for imp in parsed.imports[:80]:
            mod_id = node_id("module", imp)
            nodes.append(Node(id=mod_id, name=imp, type="module", doc_ids=[doc_id]))
            edges.append(GEdge(source=file_id, target=mod_id, relation="IMPORTS", source_ids=[]))
        for sym in parsed.symbols:
            if sym.kind == "window":
                continue
            sid = symbol_node_id(parsed.path, sym.qualname)
            sources = chunk_by_symbol.get(sym.qualname, [])
            nodes.append(
                Node(
                    id=sid,
                    name=sym.name,
                    type="symbol",
                    description=(sym.docstring or sym.signature)[:300],
                    source_ids=sources,
                    doc_ids=[doc_id],
                    data={
                        "qualname": sym.qualname,
                        "kind": sym.kind,
                        "path": parsed.path,
                        "lines": [sym.line_start, sym.line_end],
                        "service": service,
                    },
                )
            )
            edges.append(
                GEdge(source=file_id, target=sid, relation="DEFINES", source_ids=sources[:1])
            )
            by_name[sym.name].append((sid, parsed.path, service))

    # Resolve calls by name: same file > same service > unique in repo (ambiguous names skipped).
    for parsed, _doc_id, _chunks, service in files:
        for sym in parsed.symbols:
            if not sym.calls:
                continue
            sid = symbol_node_id(parsed.path, sym.qualname)
            for callee in sym.calls:
                candidates = by_name.get(callee, [])
                if not candidates:
                    continue
                same_file = [c for c in candidates if c[1] == parsed.path]
                same_service = [c for c in candidates if c[2] == service]
                pick = (
                    same_file
                    or (same_service if len(same_service) <= 2 else [])
                    or (candidates if len(candidates) == 1 else [])
                )
                for target_id, _path, _svc in pick[:2]:
                    if target_id != sid:
                        edges.append(
                            GEdge(source=sid, target=target_id, relation="CALLS", weight=1.0)
                        )
    return nodes, edges


def service_graph(
    smap: ServiceMap, service_chunks: dict[str, list[str]], doc_id: str
) -> tuple[list[Node], list[GEdge]]:
    """Nodes/edges for services, topics, datastores and endpoints."""
    nodes: dict[str, Node] = {}

    def service_id(name: str) -> str:
        return node_id("service", _norm(name))

    for name, info in smap.services.items():
        sid = service_id(name)
        nodes[sid] = Node(
            id=sid,
            name=_norm(name),
            type="service",
            description=(info.summary or f"Service at {info.root or '.'}")[:500],
            source_ids=service_chunks.get(name, []),
            doc_ids=[doc_id],
            data={
                "display": name,
                "root": info.root,
                "languages": info.languages,
                "image": info.image,
            },
        )
    edges: list[GEdge] = []
    for name, info in smap.services.items():
        for ep in info.endpoints[:300]:
            eid = node_id("endpoint", f"{_norm(name)} {ep.label}")
            nodes.setdefault(
                eid,
                Node(
                    id=eid,
                    name=ep.label,
                    type="endpoint",
                    doc_ids=[doc_id],
                    data={"file": ep.file, "line": ep.line},
                ),
            )
            edges.append(
                GEdge(
                    source=service_id(name),
                    target=eid,
                    relation="EXPOSES",
                    data={"evidence": f"{ep.file}:{ep.line}"},
                )
            )
    for e in smap.edges:
        src = service_id(e.source)
        if e.relation in ("PUBLISHES", "SUBSCRIBES"):
            tgt = node_id("topic", e.target)
            nodes.setdefault(tgt, Node(id=tgt, name=e.target, type="topic", doc_ids=[doc_id]))
        elif e.relation == "DEPENDS_ON":
            tgt = node_id("datastore", e.target)
            nodes.setdefault(tgt, Node(id=tgt, name=e.target, type="datastore", doc_ids=[doc_id]))
        else:
            tgt = service_id(e.target)
            nodes.setdefault(
                tgt,
                Node(
                    id=tgt,
                    name=_norm(e.target),
                    type="service",
                    doc_ids=[doc_id],
                    data={"display": e.target},
                ),
            )
        if src not in nodes:
            nodes[src] = Node(id=src, name=_norm(e.source), type="service", doc_ids=[doc_id])
        edges.append(
            GEdge(
                source=src,
                target=tgt,
                relation=e.relation,
                description=f"via {e.via}" if e.via else "",
                data={"evidence": e.evidence, "via": e.via},
            )
        )
    return list(nodes.values()), edges
