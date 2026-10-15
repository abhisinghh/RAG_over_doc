# ADR 0001: Explicit LangGraph state machine instead of a free-form tool-calling agent

**Status:** accepted · 2026-10

## Context

Agentic RAG can be built as (a) a ReAct-style agent that freely calls a `search` tool, or
(b) an explicit graph of steps with conditional edges (route, grade, rewrite, verify).

## Decision

Use an explicit LangGraph `StateGraph`. Node functions are plain Python that take the state
and a dependency bundle, so they are testable without the framework.

## Consequences

* Predictable cost and latency: every loop has a budget and the worst case is known.
* Each decision (route, grade, verify) is a typed, logged, evaluable output.
* Less flexible than a free-form agent on open-ended research tasks. Retrieval is also
  exposed as MCP tools (`search_document`, `read_pages`) so an external tool-calling agent can
  take the free-form approach on top of the same index.
