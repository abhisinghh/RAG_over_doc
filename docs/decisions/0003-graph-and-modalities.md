# ADR 0003: One chunk model for all modalities; one graph for knowledge, code and services

**Status:** accepted · 2026-10

## Decision

* Every modality is normalised into the same `Chunk` (with `modality` and a locator), so
  retrieval, grading, generation, citation and evaluation are shared code.
* Numbers in tables and charts are answered with read-only SQL over extracted cells, not by
  the language model reading text.
* Entities, code symbols and services live in one graph store with typed nodes, so a question
  can move from a design document to the service that implements it.
* Code and service graphs are built deterministically (no LLM), so large repositories cost
  embeddings only; LLM extraction is reserved for prose.

## Consequences

* Adding a modality means writing a parser that emits blocks; nothing downstream changes.
* The SQLite graph store is for single-node use; Neo4j is the scale path and the place to run
  community detection in-database.
* Static service discovery misses runtime-only dependencies; traces can be merged into the
  same graph later.
