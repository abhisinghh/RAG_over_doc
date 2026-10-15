"""RAPTOR-style summary tree (Sarthi et al., 2024), simplified.

Instead of clustering embeddings, consecutive leaf chunks are grouped in reading order
(documents are already topically ordered) and summarised, level by level, until a single
root summary remains. Summary nodes are indexed alongside leaf chunks, so broad questions
("what is this document about?", "summarise section 3") retrieve a summary while detailed
questions still hit leaves.
"""

from __future__ import annotations

from ragdoc.ingestion.contextualize import run_parallel
from ragdoc.models.llm import LLM, Message
from ragdoc.prompts import render
from ragdoc.types import Chunk, ParsedDocument


def build_summary_tree(
    llm: LLM, doc: ParsedDocument, leaves: list[Chunk], group_size: int = 6, workers: int = 8
) -> list[Chunk]:
    group_size = max(2, group_size)
    summaries: list[Chunk] = []
    level_nodes = leaves
    level = 1
    while len(level_nodes) > 1:
        groups = [level_nodes[i : i + group_size] for i in range(0, len(level_nodes), group_size)]

        def summarise(args: tuple[int, list[Chunk]], level: int = level) -> Chunk:
            index, group = args
            passages = "\n\n".join(f"[{c.page_label}] {c.text}" for c in group)
            response = llm.generate(
                [
                    Message(
                        role="user", content=render("summarize", title=doc.title, passages=passages)
                    )
                ],
                role="fast",
                task="summarize",
            )
            pages = sorted({p for c in group for p in c.pages})
            return Chunk(
                chunk_id=f"{doc.doc_id}:s{level}-{index:03d}",
                doc_id=doc.doc_id,
                source=doc.source,
                text=response.text.strip(),
                pages=pages,
                heading_path=[f"Summary level {level}"],
                kind="summary",
                level=level,
                position=index,
                metadata={"children": [c.chunk_id for c in group]},
            )

        level_nodes = run_parallel(summarise, list(enumerate(groups)), workers)
        summaries.extend(level_nodes)
        level += 1
    return summaries
