# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- HippoRAG 2 retrieval: Personalized PageRank over phrase and passage nodes, seeded by facts
  that an LLM recognises as relevant (`graph/ppr.py`, `graph/hipporag.py`). Used for relational
  and multi-hop questions; falls back to local graph search when no fact is relevant.
- Synonym edges between entities with near-identical names.
- `ragdoc graph search "<query>" --mode ppr|local|global` for inspecting graph retrieval.
- `graph.ppr` configuration section and `hipporag` / `graph_local` experiment overlays.
- `GraphStore.get_edges` and `GraphStore.revision`.
- This changelog.

## [0.3.0]

### Added
- GraphRAG: entity and relation extraction, communities with reports, local and global search,
  SQLite and Neo4j stores.
- Multimodal ingestion: figures and charts (vision model, chart-to-table), tables with
  text-to-SQL, audio and video with timestamped citations.
- Code repositories and microservices: symbol-level chunking, call graph, service discovery and
  dependency graph with impact analysis.
- Scale features: server-side hybrid search in Qdrant, int8 quantization, document routing,
  resumable sharded batch ingestion.

## [0.2.0]

### Added
- Installable `ragdoc` package replacing the original notebook scripts: contextual hybrid
  retrieval, reranking, agentic loop with grading, rewriting and verification, citations,
  evaluation harness, CLI, REST API, Gradio UI, MCP server, Docker and CI.
