"""Typed configuration.

Non-secret settings come from YAML (``configs/default.yaml`` plus optional overlay files)
and can be overridden with ``RAGDOC__SECTION__KEY`` environment variables. Secrets (API keys)
come only from the environment or a ``.env`` file and are never written to YAML.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
ENV_PREFIX = "RAGDOC__"


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "openai_compatible"] = "openai"
    base_url: str | None = None
    fast_model: str = "gpt-5-mini"
    strong_model: str = "gpt-5"
    vision_model: str | None = None  # None = strong_model (must accept image input)
    temperature: float | None = None
    max_output_tokens: int = 2048
    request_timeout_s: float = 60
    max_retries: int = 3


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "openai_compatible", "local"] = "openai"
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    batch_size: int = 128
    dimensions: int | None = None


class IngestionConfig(BaseModel):
    parser: Literal["auto", "docling", "pypdf"] = "auto"
    chunk_tokens: int = 600
    chunk_overlap_tokens: int = 80
    min_chunk_tokens: int = 40
    contextualize: bool = True
    context_max_doc_tokens: int = 60_000
    raptor: bool = False
    raptor_group_size: int = 6
    redact_pii: bool = False
    doc_summary: bool = False  # one summary vector per document (needed for doc routing)
    workers: int = 8  # parallel LLM calls per document / documents per batch


class MultimodalConfig(BaseModel):
    describe_images: bool = True  # VLM description + OCR for every figure / keyframe
    chart_to_table: bool = True  # charts → approximate data table (queryable via SQL)
    min_image_px: int = 120  # skip icons, logos and rules smaller than this
    max_images_per_doc: int = 200
    image_embeddings: bool = False  # CLIP vectors for text→image search (ragdoc[local])
    image_embedding_model: str = "clip-ViT-B-32"
    answer_with_images: bool = True  # attach top figures to the generator (vision models)
    max_answer_images: int = 3


class TableConfig(BaseModel):
    merge_multipage: bool = True
    rows_per_chunk: int = 40
    sql: bool = True  # load tables into SQLite for text-to-SQL questions
    max_sql_rows: int = 200


class MediaConfig(BaseModel):
    transcriber: Literal["openai", "local"] = "openai"
    transcription_model: str = "whisper-1"
    local_whisper_model: str = "small"
    language: str | None = None
    window_seconds: int = 90  # transcript chunk length
    keyframe_interval_s: int = 30
    keyframe_scene_threshold: float | None = None  # e.g. 0.3 = scene-change detection
    max_keyframes: int = 120


class CodeConfig(BaseModel):
    max_file_bytes: int = 1_000_000
    symbol_max_lines: int = 150  # longer symbols are split into overlapping windows
    window_overlap_lines: int = 15
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "vendor",
            "dist",
            "build",
            "target",
            "__pycache__",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
            ".next",
            "coverage",
        ]
    )
    services: bool = True  # detect microservices and build the service graph
    summarize_services: bool = True


class GraphConfig(BaseModel):
    enabled: bool = False  # entity/relation extraction costs one LLM call per chunk
    store: Literal["sqlite", "neo4j"] = "sqlite"
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "person",
            "organization",
            "location",
            "product",
            "technology",
            "concept",
            "event",
            "metric",
            "document",
        ]
    )
    local_top_entities: int = 10
    max_hops: int = 1
    max_relations: int = 40
    community_max_size: int = 40
    community_min_size: int = 3
    global_top_communities: int = 12


class RetrievalConfig(BaseModel):
    vector_store: Literal["qdrant", "memory"] = "qdrant"
    qdrant_url: str | None = None  # None = embedded local Qdrant under index_dir
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    dense_k: int = 60
    bm25_k: int = 60
    rrf_k: int = 60
    fused_k: int = 40
    reranker: Literal["none", "llm", "cross_encoder", "cohere"] = "llm"
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    cohere_model: str = "rerank-v3.5"
    final_k: int = 8
    # Scale: "memory" keeps BM25 in-process (fine to ~100k chunks); "qdrant_sparse" stores
    # BM25 sparse vectors in Qdrant and fuses server-side (millions of chunks).
    lexical: Literal["memory", "qdrant_sparse"] = "memory"
    quantization: Literal["none", "int8"] = "none"
    on_disk: bool = False
    doc_routing: bool = False  # search document summaries first, then chunks of top docs
    doc_routing_top_docs: int = 50


class AgentConfig(BaseModel):
    enabled: bool = True
    routes: list[str] = Field(
        default_factory=lambda: [
            "simple",
            "multi_hop",
            "global",
            "relational",
            "structured",
            "code",
        ]
    )
    max_rewrites: int = 2
    max_sub_questions: int = 4
    min_relevant_chunks: int = 1
    relevance_threshold: int = 2
    verify_answers: bool = True
    max_revisions: int = 1
    use_hyde: bool = True


class CacheConfig(BaseModel):
    answers: bool = True
    embeddings: bool = True


class TracingConfig(BaseModel):
    otlp_endpoint: str | None = None
    service_name: str = "ragdoc"


class Settings(BaseModel):
    index_dir: Path = Path(".ragdoc")
    collection: str = "default"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    tables: TableConfig = Field(default_factory=TableConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    code: CodeConfig = Field(default_factory=CodeConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)

    @property
    def collection_dir(self) -> Path:
        return Path(self.index_dir) / self.collection

    def fingerprint(self, *sections: str) -> str:
        """Stable hash of selected config sections (used to key caches)."""
        data = self.model_dump(mode="json", include=set(sections) or None)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class Secrets(BaseSettings):
    """API keys, read from the environment or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None
    # Key for an OpenAI-compatible server (Ollama ignores it; vLLM/LM Studio may need it).
    llm_api_key: SecretStr | None = None
    neo4j_password: SecretStr | None = None

    def get(self, name: str) -> str | None:
        value = getattr(self, name, None)
        return value.get_secret_value() if value else None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """Turn RAGDOC__RETRIEVAL__FINAL_K=5 into {"retrieval": {"final_k": 5}}."""
    result: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in key[len(ENV_PREFIX) :].split("__") if part]
        if not path:
            continue
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = yaml.safe_load(raw)  # parses ints, bools, null
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(
    config_path: str | Path | None = None,
    overlays: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Load settings: defaults → base YAML → overlay YAMLs → env vars → explicit overrides."""
    environ = dict(os.environ) if environ is None else environ
    path = Path(config_path or environ.get("RAGDOC_CONFIG", DEFAULT_CONFIG_PATH))
    data: dict[str, Any] = _read_yaml(path) if path.exists() else {}
    for overlay in overlays or []:
        data = _deep_merge(data, _read_yaml(Path(overlay)))
    data = _deep_merge(data, _env_overrides(environ))
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings.model_validate(data)
