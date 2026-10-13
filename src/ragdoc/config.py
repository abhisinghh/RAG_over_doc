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


class AgentConfig(BaseModel):
    enabled: bool = True
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
