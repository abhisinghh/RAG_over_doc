"""Microservice discovery and service dependency graph.

Signals (no running system needed, only the repositories):
* docker-compose: service names, build contexts, depends_on, environment, images
* Kubernetes manifests: Deployments/StatefulSets, Services, container env
* service directories: Dockerfile or build manifests (package.json, go.mod, pom.xml...)
* HTTP endpoints from OpenAPI specs and route declarations (FastAPI, Flask, Express,
  Spring, Gin/net-http, ASP.NET)
* gRPC services from .proto files, and their generated clients in other services
* outbound calls: URLs and *_URL/_HOST env vars that name another service
* messaging: topics/queues published and consumed (Kafka, RabbitMQ, SNS/SQS, NATS...)
* datastores: database/cache/broker images and connection-string schemes

The result answers questions like "what breaks if payments is down?", "who consumes
order.created?", "which service exposes GET /orders/{id}?" through the graph, and gives
every service a searchable summary chunk.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ragdoc.code.parser import detect_language

log = logging.getLogger(__name__)

MANIFESTS = {
    "package.json",
    "pyproject.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Cargo.toml",
    "Gemfile",
    "composer.json",
    "setup.py",
    "requirements.txt",
}
DATASTORE_RE = re.compile(
    r"\b(postgres(?:ql)?|mysql|mariadb|mongo(?:db)?|redis|kafka|rabbitmq|elasticsearch|opensearch|"
    r"cassandra|dynamodb|memcached|nats|zookeeper|clickhouse|minio|s3|sqs|sns|pubsub)\b",
    re.IGNORECASE,
)
_SCHEME_RE = re.compile(
    r"\b(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|kafka|nats)://([\w.-]+)"
)
_URL_HOST_RE = re.compile(r"https?://([a-zA-Z][\w-]*)(?::\d+)?")
_ENV_REF_RE = re.compile(
    r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*?)_(?:SERVICE_)?(?:URL|HOST|ADDR|ADDRESS|ENDPOINT|URI)\b"
)
_ROUTE_PATTERNS: list[tuple[re.Pattern[str], int | None, int]] = [
    # (pattern, method group or None, path group)
    (re.compile(r"@\w+\.(get|post|put|delete|patch|head|options)\(\s*[\"']([^\"']+)"), 1, 2),
    (re.compile(r"@\w+\.route\(\s*[\"']([^\"']+)[\"'][^)]*?methods\s*=\s*\[([^\]]+)\]"), 2, 1),
    (re.compile(r"@\w+\.route\(\s*[\"']([^\"']+)[\"']"), None, 1),
    (
        re.compile(
            r"\b(?:app|router|server|api)\.(get|post|put|delete|patch)\(\s*[\"'`]([^\"'`]+)"
        ),
        1,
        2,
    ),
    (
        re.compile(
            r"@(Get|Post|Put|Delete|Patch)Mapping\(\s*(?:value\s*=\s*|path\s*=\s*)?\"([^\"]+)"
        ),
        1,
        2,
    ),
    (re.compile(r"@RequestMapping\(\s*(?:value\s*=\s*|path\s*=\s*)?\"([^\"]+)"), None, 1),
    (re.compile(r"\.(GET|POST|PUT|DELETE|PATCH)\(\s*\"([^\"]+)\""), 1, 2),
    (re.compile(r"HandleFunc\(\s*\"([^\"]+)\""), None, 1),
    (re.compile(r"\[Http(Get|Post|Put|Delete|Patch)\(\s*\"([^\"]*)\""), 1, 2),
    (re.compile(r"\.Map(Get|Post|Put|Delete|Patch)\(\s*\"([^\"]+)\""), 1, 2),
]
_PUBLISH_RE = re.compile(
    r"(?:publish|produce|send|emit|sendMessage|send_message|basic_publish)\w*\(\s*(?:topic\s*=\s*)?[\"']([\w.\-:/]+)[\"']",
    re.IGNORECASE,
)
_SUBSCRIBE_RE = re.compile(
    r"(?:subscribe|consume|listen|on_message|basic_consume)\w*\(\s*(?:topics?\s*=\s*|queue\s*=\s*)?\[?[\"']([\w.\-:/]+)[\"']"
    r"|@KafkaListener\([^)]*topics\s*=\s*\{?\s*\"([^\"]+)\"",
    re.IGNORECASE,
)
_PROTO_SERVICE_RE = re.compile(r"^\s*service\s+(\w+)\s*\{", re.MULTILINE)
_PROTO_RPC_RE = re.compile(
    r"rpc\s+(\w+)\s*\(\s*(?:stream\s+)?([\w.]+)\s*\)\s*returns\s*\(\s*(?:stream\s+)?([\w.]+)"
)
_GRPC_CLIENT_RE = re.compile(r"\b(?:New)?(\w+?)(?:Client|Stub|BlockingStub|FutureStub)\b\s*\(")


class Endpoint(BaseModel):
    method: str
    path: str
    file: str
    line: int
    protocol: str = "http"

    @property
    def label(self) -> str:
        return (
            f"{self.method.upper()} {self.path}" if self.protocol == "http" else f"rpc {self.path}"
        )


class Edge(BaseModel):
    source: str
    target: str
    relation: str  # CALLS | DEPENDS_ON | PUBLISHES | SUBSCRIBES
    via: str = ""  # http | grpc | env | compose | k8s | connection-string | topic name
    evidence: str = ""  # file:line


class ServiceInfo(BaseModel):
    name: str
    root: str = ""
    languages: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    endpoints: list[Endpoint] = Field(default_factory=list)
    grpc_services: list[str] = Field(default_factory=list)
    image: str | None = None
    ports: list[str] = Field(default_factory=list)
    readme: str = ""
    summary: str = ""

    def describe(self, edges: list[Edge]) -> str:
        out = [e for e in edges if e.source == self.name]
        inc = [e for e in edges if e.target == self.name and e.relation == "CALLS"]

        def fmt(es: list[Edge], rel: str) -> str:
            items = sorted(
                {f"{e.target} ({e.via})" if e.via else e.target for e in es if e.relation == rel}
            )
            return ", ".join(items) or "none found"

        lines = [
            f"Service: {self.name}",
            f"Location: {self.root or '.'} · languages: {', '.join(self.languages) or 'unknown'}"
            + (f" · image: {self.image}" if self.image else ""),
        ]
        if self.summary:
            lines.append(f"Responsibility: {self.summary}")
        if self.endpoints:
            shown = "; ".join(e.label for e in self.endpoints[:40])
            more = f" (+{len(self.endpoints) - 40} more)" if len(self.endpoints) > 40 else ""
            lines.append(f"Exposes: {shown}{more}")
        if self.grpc_services:
            lines.append(f"gRPC services: {', '.join(self.grpc_services)}")
        lines += [
            f"Calls: {fmt(out, 'CALLS')}",
            f"Called by: {', '.join(sorted({e.source for e in inc})) or 'none found'}",
            f"Publishes: {fmt(out, 'PUBLISHES')}",
            f"Subscribes to: {fmt(out, 'SUBSCRIBES')}",
            f"Depends on: {fmt(out, 'DEPENDS_ON')}",
        ]
        return "\n".join(lines)


class ServiceMap(BaseModel):
    repo: str
    services: dict[str, ServiceInfo] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    datastores: list[str] = Field(default_factory=list)

    def service_for(self, rel_path: str) -> str | None:
        best: tuple[int, str | None] = (-1, None)
        for name, info in self.services.items():
            root = info.root.rstrip("/")
            if root == "" or rel_path == root or rel_path.startswith(root + "/"):
                depth = len(root)
                if depth > best[0]:
                    best = (depth, name)
        return best[1]

    def overview(self, max_services: int = 60) -> list[str]:
        """System overview text, split into pages for very large estates."""
        names = sorted(self.services)
        calls = defaultdict(set)
        for e in self.edges:
            if e.relation == "CALLS":
                calls[e.source].add(e.target)
        pages = []
        for i in range(0, len(names), max_services):
            part = names[i : i + max_services]
            lines = [
                f"System overview for {self.repo}: {len(names)} services, {len(self.topics)} "
                f"topics/queues, datastores: {', '.join(self.datastores) or 'none found'}."
            ]
            lines += [f"- {n}: calls {', '.join(sorted(calls[n])) or 'nothing'}" for n in part]
            pages.append("\n".join(lines))
        return pages


def _norm(name: str) -> str:
    n = name.lower().replace("_", "-")
    for suffix in ("-service", "-svc", "-api", "-server", "-app", "service"):
        if n.endswith(suffix) and len(n) > len(suffix):
            n = n[: -len(suffix)].rstrip("-")
    return n


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _load_yaml_docs(text: str) -> list[Any]:
    try:
        return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    except yaml.YAMLError:
        return []


def _env_items(env: Any) -> list[tuple[str, str]]:
    if isinstance(env, dict):
        return [(str(k), str(v)) for k, v in env.items()]
    if isinstance(env, list):
        out = []
        for item in env:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                out.append((k, v))
            elif isinstance(item, dict) and "name" in item:
                out.append((str(item["name"]), str(item.get("value", ""))))
        return out
    return []


def discover_services(root: Path, files: list[Path], repo_name: str | None = None) -> ServiceMap:
    root = root.resolve()
    files = [f.resolve() for f in files]
    repo = repo_name or root.name
    smap = ServiceMap(repo=repo)
    env_refs: list[tuple[str, str, str]] = []  # (service, env text, evidence)
    infra: set[str] = set()

    def rel(p: Path) -> str:
        return p.resolve().relative_to(root).as_posix()

    # 1. docker-compose
    for f in files:
        if re.match(r"^(docker-)?compose[\w.-]*\.ya?ml$", f.name):
            for doc in _load_yaml_docs(_read(f)):
                for name, spec in (doc.get("services") or {}).items():
                    if not isinstance(spec, dict):
                        continue
                    image = spec.get("image")
                    if image and DATASTORE_RE.search(str(image)) and not spec.get("build"):
                        infra.add(name)
                        continue
                    build = spec.get("build")
                    context = build.get("context") if isinstance(build, dict) else build
                    svc_root = ""
                    if context:
                        svc_root = (
                            rel((f.parent / str(context)).resolve())
                            if (f.parent / str(context)).resolve().is_relative_to(root.resolve())
                            else ""
                        )
                    info = smap.services.setdefault(name, ServiceInfo(name=name, root=svc_root))
                    info.image = image or info.image
                    info.ports = [str(p) for p in spec.get("ports") or []]
                    deps = spec.get("depends_on") or []
                    for dep in deps if isinstance(deps, list) else list(deps):
                        smap.edges.append(
                            Edge(
                                source=name,
                                target=str(dep),
                                relation="CALLS",
                                via="compose",
                                evidence=rel(f),
                            )
                        )
                    for k, v in _env_items(spec.get("environment")):
                        env_refs.append((name, f"{k}={v}", rel(f)))

    # 2. Kubernetes manifests
    for f in files:
        if f.suffix.lower() in (".yaml", ".yml"):
            text = _read(f)
            if "apiVersion" not in text or "kind:" not in text:
                continue
            for doc in _load_yaml_docs(text):
                kind = doc.get("kind")
                name = (doc.get("metadata") or {}).get("name")
                if kind in ("Deployment", "StatefulSet", "DaemonSet") and name:
                    info = smap.services.setdefault(name, ServiceInfo(name=name))
                    containers = (
                        ((doc.get("spec") or {}).get("template") or {}).get("spec") or {}
                    ).get("containers") or []
                    for c in containers:
                        info.image = info.image or c.get("image")
                        for k, v in _env_items(c.get("env")):
                            env_refs.append((name, f"{k}={v}", rel(f)))

    # 3. service directories (Dockerfile first, then build manifests)
    dirs_with_docker = {f.parent for f in files if f.name.startswith("Dockerfile")}
    dirs_with_manifest = {f.parent for f in files if f.name in MANIFESTS or f.suffix == ".csproj"}
    candidates = dirs_with_docker or dirs_with_manifest
    roots = {rel(d) if d != root else "" for d in candidates}
    nested = {r for r in roots if any(o and r != o and r.startswith(o + "/") for o in roots)}
    known_roots = {i.root for i in smap.services.values()}
    for r in sorted(roots - nested):
        if r in known_roots:
            continue
        name = Path(r).name if r else repo
        if name in smap.services and smap.services[name].root:
            continue
        smap.services.setdefault(name, ServiceInfo(name=name)).root = r
    if not smap.services:
        smap.services[repo] = ServiceInfo(name=repo, root="")

    # alias table for resolving references
    aliases: dict[str, str] = {}
    for name in smap.services:
        aliases[name.lower()] = name
        aliases[_norm(name)] = name

    def resolve(token: str) -> str | None:
        t = token.lower().replace("_", "-")
        return aliases.get(t) or aliases.get(_norm(t))

    # 4. scan source files
    proto_owner: dict[str, str] = {}
    pending_clients: list[tuple[str, str, str]] = []
    for f in files:
        language = detect_language(f)
        if language is None:
            if f.name.lower().startswith("readme"):
                owner = smap.service_for(rel(f))
                if owner and not smap.services[owner].readme:
                    smap.services[owner].readme = _read(f)[:3000]
            continue
        r = rel(f)
        owner = smap.service_for(r)
        if owner is None:
            continue
        info = smap.services[owner]
        if language not in info.languages and language not in (
            "yaml",
            "json",
            "toml",
            "xml",
            "dockerfile",
            "make",
        ):
            info.languages.append(language)
        info.files.append(r)
        text = _read(f)
        if not text:
            continue

        # OpenAPI specs
        if language in ("yaml", "json") and ("openapi" in text[:400] or "swagger" in text[:400]):
            try:
                spec = json.loads(text) if language == "json" else yaml.safe_load(text)
            except Exception:
                spec = None
            if isinstance(spec, dict):
                for path_, ops in (spec.get("paths") or {}).items():
                    for method in ops or {}:
                        if method.lower() in (
                            "get",
                            "post",
                            "put",
                            "delete",
                            "patch",
                            "head",
                            "options",
                        ):
                            info.endpoints.append(
                                Endpoint(method=method, path=str(path_), file=r, line=1)
                            )
            continue

        if language == "proto":
            for svc in _PROTO_SERVICE_RE.findall(text):
                info.grpc_services.append(svc)
                proto_owner[svc] = owner
            for rpc, _req, _resp in _PROTO_RPC_RE.findall(text):
                info.endpoints.append(
                    Endpoint(method="rpc", path=rpc, file=r, line=1, protocol="grpc")
                )
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, method_group, path_group in _ROUTE_PATTERNS:
                m = pattern.search(line)
                if m:
                    method = m.group(method_group) if method_group else "ANY"
                    method = method.replace("'", "").replace('"', "").split(",")[0].strip() or "ANY"
                    info.endpoints.append(
                        Endpoint(method=method, path=m.group(path_group), file=r, line=lineno)
                    )
                    break
            for m in _PUBLISH_RE.finditer(line):
                smap.edges.append(
                    Edge(
                        source=owner,
                        target=m.group(1),
                        relation="PUBLISHES",
                        via="topic",
                        evidence=f"{r}:{lineno}",
                    )
                )
            for m in _SUBSCRIBE_RE.finditer(line):
                topic = m.group(1) or m.group(2)
                if topic:
                    smap.edges.append(
                        Edge(
                            source=owner,
                            target=topic,
                            relation="SUBSCRIBES",
                            via="topic",
                            evidence=f"{r}:{lineno}",
                        )
                    )
            for m in _URL_HOST_RE.finditer(line):
                target = resolve(m.group(1))
                if target and target != owner:
                    smap.edges.append(
                        Edge(
                            source=owner,
                            target=target,
                            relation="CALLS",
                            via="http",
                            evidence=f"{r}:{lineno}",
                        )
                    )
            for m in _SCHEME_RE.finditer(line):
                store = (
                    m.group(1)
                    .split("+")[0]
                    .replace("postgresql", "postgres")
                    .replace("mongodb", "mongo")
                )
                smap.edges.append(
                    Edge(
                        source=owner,
                        target=store,
                        relation="DEPENDS_ON",
                        via="connection-string",
                        evidence=f"{r}:{lineno}",
                    )
                )
                infra.add(store)
            for m in _ENV_REF_RE.finditer(line):
                target = resolve(m.group(1))
                if target and target != owner:
                    smap.edges.append(
                        Edge(
                            source=owner,
                            target=target,
                            relation="CALLS",
                            via="env",
                            evidence=f"{r}:{lineno}",
                        )
                    )
            for m in _GRPC_CLIENT_RE.finditer(line):
                pending_clients.append((owner, m.group(1), f"{r}:{lineno}"))

    # 5. env references from compose / k8s
    for owner, env_text, evidence in env_refs:
        for m in _URL_HOST_RE.finditer(env_text):
            target = resolve(m.group(1))
            if target and target != owner:
                smap.edges.append(
                    Edge(
                        source=owner, target=target, relation="CALLS", via="env", evidence=evidence
                    )
                )
            elif m.group(1) in infra or DATASTORE_RE.search(m.group(1)):
                smap.edges.append(
                    Edge(
                        source=owner,
                        target=m.group(1),
                        relation="DEPENDS_ON",
                        via="env",
                        evidence=evidence,
                    )
                )
        for m in _SCHEME_RE.finditer(env_text):
            smap.edges.append(
                Edge(
                    source=owner,
                    target=m.group(1),
                    relation="DEPENDS_ON",
                    via="env",
                    evidence=evidence,
                )
            )

    # 6. gRPC clients → the service owning the .proto definition
    for owner, client, evidence in pending_clients:
        target = proto_owner.get(client)
        if target and target != owner:
            smap.edges.append(
                Edge(source=owner, target=target, relation="CALLS", via="grpc", evidence=evidence)
            )

    # compose depends_on on datastores → DEPENDS_ON, not CALLS
    for e in smap.edges:
        if e.relation == "CALLS" and e.target in infra:
            e.relation = "DEPENDS_ON"

    # dedupe edges (keep first evidence)
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[Edge] = []
    for e in smap.edges:
        key = (e.source, e.target, e.relation, e.via)
        if key not in seen and e.source != e.target:
            seen.add(key)
            unique.append(e)
    smap.edges = unique
    smap.topics = sorted({e.target for e in unique if e.relation in ("PUBLISHES", "SUBSCRIBES")})
    smap.datastores = sorted(infra | {e.target for e in unique if e.relation == "DEPENDS_ON"})
    for info in smap.services.values():
        seen_ep: set[str] = set()
        info.endpoints = [
            ep
            for ep in info.endpoints
            if not (ep.label in seen_ep or seen_ep.add(ep.label))  # type: ignore[func-returns-value]
        ]
    log.info(
        "Discovered %d services, %d edges, %d topics",
        len(smap.services),
        len(unique),
        len(smap.topics),
    )
    return smap
