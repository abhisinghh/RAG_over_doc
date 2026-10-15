# ragdoc

**Agentic, multimodal retrieval-augmented generation with GraphRAG.**

[![CI](https://github.com/abhisinghh/RAG_over_doc/actions/workflows/ci.yml/badge.svg)](https://github.com/abhisinghh/RAG_over_doc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhisinghh/RAG_over_doc/blob/main/notebooks/demo.ipynb)

ragdoc answers questions over documents, tables, charts, images, audio, video, source code
and microservice architectures. Every answer cites the exact source behind each claim (a
page, a figure, a timestamp or a line range). When the indexed content does not contain the
answer, ragdoc says so instead of guessing.

For each question, a router picks the best retrieval strategy: contextual hybrid search,
knowledge-graph search, SQL over extracted tables, or code and service-dependency graphs.
Each strategy runs inside a self-correcting agent loop with a fixed step budget. An
evaluation harness measures retrieval quality, faithfulness, latency and cost on your own
question set, so you can compare configurations with numbers.

> **Status:** active development. The system has not yet been benchmarked against public
> datasets. Benchmark results are on the roadmap.

## Contents

- [Key features](#key-features)
- [Supported content](#supported-content)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [Development](#development)
- [Roadmap](#roadmap)
- [References](#references)

## Key features

**Retrieval**
- Layout-aware parsing with Docling (reading order, tables, figures), with pypdf as a fallback.
- Structure-aware, token-sized chunks that never cross section boundaries.
- Contextual retrieval: an LLM-written context header is added to each chunk before
  embedding and keyword indexing.
- Hybrid search combining BM25 keyword search and dense vectors, merged with Reciprocal Rank
  Fusion, then reranked by an LLM, a local cross-encoder or Cohere.

**Agent**
- Question routing: simple, multi-hop, global, relational, structured, code or out of scope.
  The router only offers routes the current index can serve.
- Multi-part questions are split into self-contained sub-questions.
- Retrieved passages are graded for relevance. Weak evidence triggers a rewritten query
  (with HyDE) and a retry, and the system abstains once the retry budget is spent.
- Answers cite every factual sentence, and each claim is checked against its sources before
  the answer is returned.

**Multimodal**
- Figures and charts are described by a vision model, with OCR of any text in the image.
  Charts are also converted to data tables.
- Tables stay intact, tables split across pages are stitched back together, and every table
  is loaded into SQLite. Numeric questions are answered with read-only SQL over the actual
  cells.
- Audio is transcribed with timestamps. Video is indexed through its transcript plus
  keyframes described in context.
- Optional CLIP image embeddings support text-to-image search.

**Knowledge graph (GraphRAG)**
- Entity and relationship extraction, with resolution of duplicate names.
- Hierarchical communities, each with an LLM-written report, used for corpus-wide questions.
- Local search over entity neighbourhoods and global search over community reports.
- SQLite storage for local use; Neo4j for large graphs.

**Code and microservices**
- Code is chunked at function level, with class outlines and exact line ranges. Python is
  parsed with its own AST; twelve other languages are parsed by matching declarations and braces.
- A call graph answers "who calls this function" and "what does it call".
- Services, endpoints, calls between services, message topics and datastores are discovered
  from docker-compose files, Kubernetes manifests, OpenAPI and protobuf specs, route
  declarations and connection strings.
- Impact analysis lists which services are affected, directly or transitively, if a
  component fails.

**Production**
- Typed configuration in YAML with environment overrides. Secrets are read only from `.env`.
- Interfaces: CLI, REST API with streaming, Gradio UI, MCP server and a Python API.
- Every answer carries a per-step trace of latency and tokens, which can be exported to
  OpenTelemetry (Arize Phoenix, Langfuse).
- Answer and embedding caches, prompt-injection screening and optional PII redaction.
- Built to scale: server-side hybrid search in Qdrant, int8 vector quantization, an optional
  document-routing first stage, and resumable batch ingestion that can be sharded across
  machines.

## Supported content

| Content | Formats | Processing | Citation |
| --- | --- | --- | --- |
| Documents | PDF, DOCX, Markdown, TXT | Layout parsing, sections, contextual chunks | Page |
| Tables | Within documents | Row-group chunks, multi-page stitching, SQL store | Page or SQL result |
| Figures and charts | Within documents, PNG, JPG, WEBP, TIFF | Vision-model description, OCR, chart-to-table | Figure and page |
| Audio | MP3, WAV, M4A, FLAC, OGG and others | Whisper transcription, 90-second windows | `12:40–14:10` |
| Video | MP4, MOV, MKV, WEBM and others | Transcript plus described keyframes | `frame at 03:30` |
| Source code | Python, TypeScript, JavaScript, Go, Java, Kotlin, C#, Rust, C/C++, Scala, Swift, PHP and others | Function-level chunks, call graph | File, line range, symbol |
| Microservices | Compose, Kubernetes, OpenAPI, protobuf, route declarations | Service map and dependency graph | Service and `file:line` |

The design rationale for each content type, for GraphRAG and for scaling to millions of
documents is in [docs/design/multimodal-graph-scale.md](docs/design/multimodal-graph-scale.md).

## Installation

**Requirements**
- Python 3.11 or 3.12.
- An API key for OpenAI or Anthropic, or a local OpenAI-compatible server such as Ollama or vLLM.
- `ffmpeg` on your `PATH` to ingest audio or video.

```bash
git clone https://github.com/abhisinghh/RAG_over_doc.git
cd RAG_over_doc
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,ui]"
cp .env.example .env    # then add OPENAI_API_KEY or ANTHROPIC_API_KEY
```

**Optional extras**

| Extra | Adds |
| --- | --- |
| `docling` | Layout-aware PDF parsing with tables and figure rendering (recommended) |
| `local` | Local embeddings, cross-encoder reranking and CLIP image search |
| `media` | Local speech-to-text with faster-whisper |
| `neo4j` | Neo4j graph store |
| `scale` | Server-side BM25 sparse vectors in Qdrant |
| `cohere` | Cohere reranker |
| `api`, `ui`, `mcp`, `tracing` | REST API, Gradio UI, MCP server, OpenTelemetry export |
| `all` | Everything above |

## Quick start

```bash
ragdoc ingest path/to/report.pdf
ragdoc ask "What were the main risks identified in 2024?"
```

The answer comes back with numbered citations, each pointing to a file and location, plus
the route the agent took, the latency and the token usage. Add `--trace` to see every step.

## Usage

### Command line

| Command | Purpose |
| --- | --- |
| `ragdoc ingest <path>` | Index a file or folder of documents, images, audio or video |
| `ragdoc ingest-repo <path>` | Index a code repository, its call graph and its service graph |
| `ragdoc ingest-batch <path> --workers 16 --shard 0/4` | Resumable, parallel ingestion for large corpora |
| `ragdoc ask "<question>" [--trace] [--json]` | Answer a question with citations |
| `ragdoc batch questions.txt -f report.pdf` | Answer a file of questions and write `output.json` |
| `ragdoc services [name]` | List microservices, or show one service's dependencies and impact |
| `ragdoc graph build` | Build knowledge-graph communities to enable global questions |
| `ragdoc docs`, `ragdoc delete <doc_id>` | List or remove indexed documents |
| `ragdoc eval` | Run the evaluation harness |
| `ragdoc serve`, `ragdoc ui`, `ragdoc mcp` | Start the REST API, Gradio UI or MCP server |

### Python

```python
from ragdoc.engine import RagEngine

engine = RagEngine.from_config()            # configs/default.yaml + .env
engine.ingest("report.pdf")
engine.ingest_repo("~/code/platform")

result = engine.ask("Which services depend on payments?")
print(result.answer)
for citation in result.citations:
    print(citation.marker, citation.source, citation.locator)
```

### REST API

Start the server with `ragdoc serve`. Interactive documentation is available at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and index statistics |
| `GET` | `/documents` | List indexed documents (paginated) |
| `POST` | `/documents` | Upload and ingest a file |
| `DELETE` | `/documents/{doc_id}` | Remove a document |
| `POST` | `/search` | Hybrid search with reranking, without generation |
| `POST` | `/ask` | Cited answer; `"debug": true` includes the trace |
| `POST` | `/ask/stream` | Server-Sent Events: one event per agent step, then the result |
| `GET` | `/assets/{path}` | The figure or video frame behind a citation |
| `GET` | `/graph/stats` | Knowledge-graph statistics |
| `POST` | `/graph/build` | Rebuild communities and reports |
| `GET` | `/services/{name}` | Dependency facts for one microservice |

### MCP server

`ragdoc mcp` runs over stdio and exposes these tools to MCP clients such as Claude Desktop
and IDE agents: `search_document`, `read_pages`, `ask_documents`, `list_documents`,
`service_dependencies` and `code_references`.

```json
{ "mcpServers": { "ragdoc": { "command": "ragdoc", "args": ["mcp"] } } }
```

## Configuration

All settings live in [`configs/default.yaml`](configs/default.yaml). You can override any
value with an overlay file (`ragdoc -o configs/experiments/dense.yaml ...`) or an
environment variable of the form `RAGDOC__<SECTION>__<KEY>`:

```bash
export RAGDOC__LLM__PROVIDER=anthropic
export RAGDOC__LLM__FAST_MODEL=claude-haiku-5-5
export RAGDOC__LLM__STRONG_MODEL=claude-sonnet-5-5
```

Models are assigned by role. A fast model handles routing, grading, query rewriting and
contextualization; a strong model writes the answer; a vision model (by default the strong
model) reads figures.

| Goal | Settings |
| --- | --- |
| Run fully locally | `llm.provider: openai_compatible`, `llm.base_url`, `embeddings.provider: local` |
| Reduce ingestion cost | `ingestion.contextualize: false`, `multimodal.describe_images: false` |
| Enable GraphRAG | `graph.enabled: true`, then `ragdoc graph build` |
| Large graphs | `graph.store: neo4j`, `graph.neo4j_uri` |
| Text-to-image search | `multimodal.image_embeddings: true` (requires `local`) |
| Local transcription | `media.transcriber: local` (requires `media`) |
| Higher reranking quality | `retrieval.reranker: cross_encoder` (requires `local`) |
| Millions of documents | `retrieval.lexical: qdrant_sparse`, `retrieval.quantization: int8`, `ingestion.doc_summary: true`, `retrieval.doc_routing: true` |
| Single-pass baseline | `agent.enabled: false` |

## Evaluation

The harness runs a golden question set against one or more configurations and writes a
comparison report.

```bash
cp eval/datasets/golden.example.jsonl eval/datasets/golden.jsonl
mkdir -p eval/datasets/docs && cp report.pdf eval/datasets/docs/
python eval/run_eval.py      # compares dense, hybrid and agentic configurations
```

| Layer | Metrics |
| --- | --- |
| Retrieval | Hit@k, page recall@k, MRR, nDCG@k |
| Generation | Faithfulness and correctness (LLM judge), citation precision |
| Agent | Abstention accuracy on unanswerable questions, LLM calls per answer |
| System | p50 and p95 latency, tokens per answer |

A useful starting set has 50 to 100 questions over two or three representative documents.
It should mix single-fact, table, multi-hop and summary questions, with 10 to 15 percent
unanswerable ones.

## Deployment

```bash
docker compose -f docker/docker-compose.yml up --build
```

This starts four services:
- The API, with ffmpeg included.
- A Qdrant server, with server-side hybrid search enabled.
- Neo4j with the Graph Data Science plugin.
- Arize Phoenix for traces, at `http://localhost:6006`.

The API has no built-in authentication. Place it behind an authenticating gateway before
exposing it to a network.

## Project structure

```
src/ragdoc/
  agent/           routing, retrieval strategies, grading, generation, verification
  api/             FastAPI application
  code/            code parsing, repository ingestion, call graph, service discovery
  evaluation/      metrics, LLM judges, experiment runner
  generation/      source formatting and citation handling
  graph/           graph stores, entity extraction, communities, graph search
  guardrails/      prompt-injection screening, PII redaction
  ingestion/       parsers, figures, tables, media, chunking, pipeline, batch jobs
  models/          LLM, vision, embedding and transcription clients
  observability/   tracing and OpenTelemetry export
  prompts/         versioned prompt templates
  retrieval/       vector stores, BM25, rank fusion, reranking
  ui/              Gradio application
  cli.py           command-line interface
  engine.py        RagEngine, the entry point used by every interface
  mcp_server.py    MCP server
configs/           default configuration and experiment overlays
docs/              architecture, design notes and decision records
eval/              golden dataset template and evaluation script
notebooks/         Colab demo
tests/             offline test suite
```

## Development

```bash
pip install -e ".[dev,api]"
make test     # offline suite; uses a scripted LLM and a hashing embedder, no API keys required
make lint     # ruff
```

Continuous integration runs linting and the offline tests on every push. When an API key and
a golden set with its documents are configured, it also runs a ten-question smoke evaluation on each pull request.

## Roadmap

- Page-image retrieval (ColPali) for slide decks and scanned documents
- Speaker diarisation for audio and video
- PostgreSQL chunk store and pgvector backend for multi-replica deployments
- Merging runtime traces into the service graph
- Published benchmark results comparing dense, graph and agentic retrieval

## References

- Lewis et al. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). 2020.
- Singh et al. [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136). 2025.
- Anthropic. [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval). 2024.
- Edge et al. [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). 2024.
- Guo et al. [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779). 2024.
- Jeong et al. [Adaptive-RAG](https://arxiv.org/abs/2403.14403). 2024.
- Yan et al. [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884). 2024.
- Asai et al. [Self-RAG](https://arxiv.org/abs/2310.11511). 2023.
- Gao et al. [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496). 2022.
- Dhuliawala et al. [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495). 2023.
- Sarthi et al. [RAPTOR](https://arxiv.org/abs/2401.18059). 2024.
- Liu et al. [Lost in the Middle](https://arxiv.org/abs/2307.03172). 2023.
- Faysse et al. [ColPali](https://arxiv.org/abs/2407.01449). 2024.
- Es et al. [RAGAS](https://arxiv.org/abs/2309.15217). 2023.
- Mendelevitch and Bao. [Hands-On RAG for Production](https://www.oreilly.com/library/view/hands-on-rag-for/9798341621701/). O'Reilly, 2026. [Companion code](https://github.com/ofermend/hands-on-rag).
