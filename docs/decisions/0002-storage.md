# ADR 0002: SQLite for chunks, pluggable vector store, in-memory BM25

**Status:** accepted · 2026-10

## Decision

* Chunk text and document records live in SQLite (`docs.sqlite`), the source of truth.
* Vectors live behind a `VectorStore` protocol: embedded Qdrant by default, a Qdrant server in
  production, a numpy store for tests.
* BM25 is rebuilt in memory from SQLite when the corpus changes.

## Why

The original repo kept chunks and embeddings in a pandas DataFrame and re-embedded the PDF on
every run. Persisting chunks separately from vectors makes re-indexing, citations, page reads
and backend swaps independent of each other, with zero extra services for local use.

## Revisit when

The corpus passes ~100k chunks (move lexical search to Qdrant sparse vectors or OpenSearch) or
multiple API replicas need shared state (move chunks to Postgres and use pgvector).
