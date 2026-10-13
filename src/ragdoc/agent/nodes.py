"""Agent nodes. Each node takes the state and returns a partial state update.

Techniques: Adaptive-RAG routing over retrieval strategies (hybrid, GraphRAG local/global,
text-to-SQL over tables, code/service graphs), query decomposition, CRAG-style evidence
grading with query rewriting + HyDE, cited (multimodal) generation, and
Chain-of-Verification style claim checking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ragdoc.agent.state import ROUTE_DESCRIPTIONS, AgentState, Grades, RouteDecision, Verification
from ragdoc.config import Settings
from ragdoc.generation.citations import (
    ABSTAIN_TEXT,
    build_citations,
    format_sources,
    is_abstention,
    strip_invalid_markers,
)
from ragdoc.guardrails import neutralise
from ragdoc.models.llm import LLM, ImageInput, Message, guess_mime, structured
from ragdoc.observability.tracing import TraceRecorder
from ragdoc.prompts import render
from ragdoc.retrieval.query_transform import decompose, rewrite_query
from ragdoc.retrieval.retriever import HybridRetriever
from ragdoc.types import ScoredChunk

if TYPE_CHECKING:
    from ragdoc.graph.search import GraphRetriever
    from ragdoc.ingestion.assets import AssetStore
    from ragdoc.ingestion.tables import TableStore

log = logging.getLogger(__name__)

OUT_OF_SCOPE_TEXT = (
    "I can only answer questions about the indexed documents. Ask me something about their content."
)
PINNED_SOURCES = ("graph_", "sql")  # computed evidence that bypasses the relevance grader
_SERVICE_HINT = re.compile(
    r"\b(service|microservice|endpoint|api|depend|call|topic|queue|kafka|grpc|deploy|outage|down)\w*",
    re.IGNORECASE,
)


@dataclass
class AgentDeps:
    settings: Settings
    llm: LLM
    retriever: HybridRetriever
    recorder: TraceRecorder
    documents: list[str] = field(default_factory=list)  # titles, for the router prompt
    graph: GraphRetriever | None = None
    tables: TableStore | None = None
    assets: AssetStore | None = None
    available_routes: list[str] = field(default_factory=lambda: ["simple", "multi_hop"])


def _merge_evidence(*groups: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
    """Dedupe by chunk id, keep the best-scoring copy, preserve first-seen order."""
    best: dict[str, ScoredChunk] = {}
    order: list[str] = []
    for group in groups:
        for item in group:
            cid = item.chunk.chunk_id
            if cid not in best:
                order.append(cid)
                best[cid] = item
            elif (item.relevance or 0, item.score) > (best[cid].relevance or 0, best[cid].score):
                best[cid] = item
    return [best[cid] for cid in order][:limit]


def _pinned(item: ScoredChunk) -> bool:
    return item.source.startswith(PINNED_SOURCES)


# --------------------------------------------------------------------------- nodes


def route(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    allowed = [r for r in deps.available_routes if r in ROUTE_DESCRIPTIONS] + ["out_of_scope"]
    routes_text = "\n".join(f'- "{r}": {ROUTE_DESCRIPTIONS[r]}' for r in allowed)
    with deps.recorder.span("route") as d:
        try:
            decision = structured(
                deps.llm,
                system=render(
                    "route",
                    documents=", ".join(deps.documents) or "uploaded documents",
                    routes=routes_text,
                ),
                user=state["question"],
                schema=RouteDecision,
                role="fast",
                task="route",
            )
            route_name = decision.route if decision.route in allowed else "simple"
            d["reason"] = decision.reason
        except Exception as exc:  # never block a question on a routing failure
            log.warning("Routing failed (%s); defaulting to simple", exc)
            route_name = "simple"
        d["route"] = route_name
    update: dict[str, Any] = {"route": route_name}
    if route_name == "out_of_scope":
        update.update(answer=OUT_OF_SCOPE_TEXT, abstained=True, citations=[])
    return update


def plan(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    with deps.recorder.span("plan") as d:
        try:
            subs = decompose(deps.llm, state["question"], deps.settings.agent.max_sub_questions)
        except Exception as exc:
            log.warning("Planning failed (%s); using the original question", exc)
            subs = [state["question"]]
        d["sub_questions"] = subs
    return {"sub_questions": subs}


def _strategy_evidence(
    state: AgentState, deps: AgentDeps, detail: dict[str, Any]
) -> list[ScoredChunk]:
    """Route-specific evidence gathered on the first round (graph facts, SQL results...)."""
    question = state["question"]
    doc_ids = state.get("doc_ids")
    route_name = state.get("route", "simple")
    out: list[ScoredChunk] = []
    try:
        if route_name == "global" and deps.graph is not None:
            out = deps.graph.global_(question)
        elif route_name == "relational" and deps.graph is not None:
            out = deps.graph.local(question, doc_ids)
        elif route_name == "code" and deps.graph is not None:
            out = deps.graph.code(question, doc_ids)
            if _SERVICE_HINT.search(question):
                out += deps.graph.services(question)
        elif route_name == "structured" and deps.tables is not None:
            from ragdoc.ingestion.tables import answer_with_sql, candidate_tables

            table_hits = deps.retriever.search(
                question, doc_ids=doc_ids, modalities=["table", "chart"], top_k=6
            )
            tables = candidate_tables(deps.tables, question, table_hits, doc_ids)
            sql = answer_with_sql(
                deps.llm, deps.tables, question, tables, deps.settings.tables.max_sql_rows
            )
            out = ([sql] if sql else []) + table_hits
            detail["sql"] = sql.chunk.metadata.get("sql") if sql else None
    except Exception as exc:  # a failing strategy degrades to plain hybrid retrieval
        log.warning("Strategy %s failed: %s", route_name, exc)
        detail["strategy_error"] = str(exc)
    detail["strategy"] = route_name
    detail["strategy_hits"] = len(out)
    return out


def retrieve(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    cfg = deps.settings.retrieval
    first_round = not state.get("queries_tried")
    if first_round and state.get("sub_questions"):
        queries = state["sub_questions"]
    else:
        queries = [state.get("current_query") or state["question"]]
    hyde = state.get("hyde_text") if not first_round else None
    route_name = state.get("route", "simple")
    modalities = ["code", "service", "text"] if route_name == "code" else None

    with deps.recorder.span("retrieve", queries=queries, hyde=bool(hyde)) as d:
        strategy = _strategy_evidence(state, deps, d) if first_round else []
        per_query = max(3, cfg.final_k // max(1, len(queries))) if len(queries) > 1 else cfg.final_k
        groups = [
            deps.retriever.search(
                q,
                dense_text=hyde,
                doc_ids=state.get("doc_ids"),
                top_k=per_query,
                modalities=modalities,
            )
            for q in queries
        ]
        # Keep evidence already graded relevant (or pinned) from earlier rounds.
        threshold = deps.settings.agent.relevance_threshold
        previous = [
            e for e in state.get("evidence", []) if _pinned(e) or (e.relevance or 0) >= threshold
        ]
        limit = cfg.final_k if len(queries) == 1 else min(cfg.final_k * 2, 16)
        evidence = _merge_evidence(strategy, previous, *groups, limit=limit + len(strategy))
        d["retrieved"] = [e.chunk.chunk_id for e in evidence]
    return {"evidence": evidence, "queries_tried": [*state.get("queries_tried", []), *queries]}


def grade(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    pinned = [e.model_copy(update={"relevance": e.relevance or 3}) for e in evidence if _pinned(e)]
    to_grade = [e for e in evidence if not _pinned(e)]
    agent_cfg = deps.settings.agent
    with deps.recorder.span("grade", candidates=len(to_grade), pinned=len(pinned)) as d:
        if not to_grade:
            d["relevant"] = len(pinned)
            return {"evidence": pinned}
        passages = "\n\n".join(
            f'<passage id="{i}">\n{neutralise(e.chunk.index_text[:1200])}\n</passage>'
            for i, e in enumerate(to_grade)
        )
        question = state["question"]
        if state.get("sub_questions"):
            question += "\n(Parts: " + "; ".join(state["sub_questions"]) + ")"
        try:
            grades = structured(
                deps.llm,
                system=render("grade"),
                user=f"Question: {question}\n\n{passages}",
                schema=Grades,
                role="fast",
                task="grade",
            )
            scores = {g.id: g.score for g in grades.grades}
        except Exception as exc:  # fail open: let the generator and verifier decide
            log.warning("Grading failed (%s); treating all passages as relevant", exc)
            scores = {i: agent_cfg.relevance_threshold for i in range(len(to_grade))}

        graded = [
            e.model_copy(update={"relevance": scores.get(i, 0)}) for i, e in enumerate(to_grade)
        ]
        relevant = [e for e in graded if (e.relevance or 0) >= agent_cfg.relevance_threshold]
        relevant.sort(key=lambda e: (-(e.relevance or 0), -e.score))
        d["relevant"] = len(relevant) + len(pinned)
        d["scores"] = {e.chunk.chunk_id: e.relevance for e in graded}
    return {"evidence": pinned + relevant}


def decide_after_grade(state: AgentState, deps: AgentDeps) -> str:
    agent_cfg = deps.settings.agent
    if len(state.get("evidence", [])) >= agent_cfg.min_relevant_chunks:
        return "generate"
    if state.get("rewrites", 0) < agent_cfg.max_rewrites:
        return "rewrite"
    return "abstain"


def rewrite(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    with deps.recorder.span("rewrite") as d:
        try:
            result = rewrite_query(deps.llm, state["question"], state.get("queries_tried", []))
            query, hyde = result.query, result.hypothetical_passage
        except Exception as exc:
            log.warning("Rewrite failed (%s)", exc)
            query, hyde = state["question"], ""
        d.update(query=query, hyde=bool(hyde))
    return {
        "current_query": query,
        "hyde_text": hyde if deps.settings.agent.use_hyde and hyde else None,
        "rewrites": state.get("rewrites", 0) + 1,
        "sub_questions": [],  # later rounds search with the single rewritten query
    }


def generate(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    feedback = state.get("feedback")
    with deps.recorder.span("generate", sources=len(evidence), revision=bool(feedback)) as d:
        if not evidence:
            d["abstained"] = True
            return {"answer": ABSTAIN_TEXT, "citations": [], "abstained": True}
        user = f"<sources>\n{format_sources(evidence)}\n</sources>\n\nQuestion: {state['question']}"
        if state.get("sub_questions"):
            user += "\nMake sure to cover: " + "; ".join(state["sub_questions"])
        messages = [Message(role="user", content=user)]
        if feedback and state.get("answer"):
            messages += [
                Message(role="assistant", content=state["answer"]),
                Message(
                    role="user",
                    content=(
                        "A fact-check found problems with that answer:\n"
                        f"{feedback}\n\nRewrite the answer so every statement is supported by and "
                        "cited to the sources. Remove anything the sources do not support."
                    ),
                ),
            ]
        images = _answer_images(evidence, deps)
        d["images"] = len(images)
        response = deps.llm.generate(
            messages,
            system=render("generate", abstain_text=ABSTAIN_TEXT),
            role="vision" if images else "strong",
            task="generate",
            images=images or None,
        )
        answer = strip_invalid_markers(response.text.strip(), len(evidence))
        abstained = is_abstention(answer)
        citations = [] if abstained else build_citations(answer, evidence)
        d.update(citations=len(citations), abstained=abstained)
    update: dict[str, Any] = {"answer": answer, "citations": citations, "abstained": abstained}
    if feedback:
        update["revisions"] = state.get("revisions", 0) + 1
        update["feedback"] = None
    return update


def _answer_images(evidence: list[ScoredChunk], deps: AgentDeps) -> list[ImageInput]:
    """Top figure/chart/frame images among the evidence, for vision-capable generators."""
    mm = deps.settings.multimodal
    if not mm.answer_with_images or deps.assets is None:
        return []
    images: list[ImageInput] = []
    for e in evidence:
        c = e.chunk
        if c.asset and c.modality in ("image", "chart") and deps.assets.exists(c.asset):
            data = deps.assets.read(c.asset)
            images.append(ImageInput(data=data, mime=guess_mime(data)))
            if len(images) >= mm.max_answer_images:
                break
    return images


def verify(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    answer = state.get("answer", "")
    with deps.recorder.span("verify") as d:
        if state.get("abstained") or is_abstention(answer):
            d["skipped"] = "abstained"
            return {"verification": {"passed": True, "skipped": True}}
        if not state.get("citations"):
            d["passed"] = False
            return {
                "verification": {"passed": False, "reason": "no citations"},
                "feedback": "The answer has no [n] citations. Cite a source for every factual sentence.",
            }
        try:
            result = structured(
                deps.llm,
                system=render("verify"),
                user=(
                    f"<sources>\n{format_sources(state.get('evidence', []))}\n</sources>\n\n"
                    f"<answer>\n{answer}\n</answer>"
                ),
                schema=Verification,
                role="fast",
                task="verify",
            )
        except Exception as exc:
            log.warning("Verification failed to run (%s); accepting answer unverified", exc)
            d["error"] = str(exc)
            return {"verification": {"passed": True, "unverified": True}}
        d.update(claims=len(result.claims), unsupported=len(result.unsupported))
        verification = {
            "passed": result.passed,
            "claims": [c.model_dump() for c in result.claims],
        }
    if result.passed:
        return {"verification": verification, "feedback": None}
    feedback = "\n".join(f"- Unsupported: {c.claim}" for c in result.unsupported)
    return {"verification": verification, "feedback": feedback}


def decide_after_verify(state: AgentState, deps: AgentDeps) -> str:
    verification = state.get("verification") or {}
    if verification.get("passed", True):
        return "done"
    if state.get("revisions", 0) < deps.settings.agent.max_revisions:
        return "revise"
    return "done"  # keep the answer but surface verification.passed = False


def abstain(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    with deps.recorder.span("abstain", rewrites=state.get("rewrites", 0)):
        pass
    return {"answer": ABSTAIN_TEXT, "citations": [], "abstained": True}
