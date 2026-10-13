"""LangGraph wiring for the agentic loop (and the single-pass baseline).

Agentic:   route ─┬─ out_of_scope ───────────────────────────────────────────▶ END
                  ├─ simple ─────────────▶ retrieve ─▶ grade ─┬─ generate ─▶ verify ─┬─ done ─▶ END
                  └─ multi_hop ─▶ plan ─▶ retrieve            ├─ rewrite ─▶ retrieve └─ revise ─▶ generate
                                                              └─ abstain ─▶ END
Baseline:  retrieve ─▶ generate ─▶ END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ragdoc.agent import nodes
from ragdoc.agent.nodes import AgentDeps
from ragdoc.agent.state import AgentState


def _bind(fn: Any, deps: AgentDeps) -> Any:
    """Close a node function over its dependencies (LangGraph nodes take only the state)."""

    def node(state: AgentState) -> dict[str, Any]:
        return fn(state, deps)

    node.__name__ = fn.__name__
    return node


def build_graph(deps: AgentDeps) -> Any:
    graph = StateGraph(AgentState)
    agentic = deps.settings.agent.enabled

    graph.add_node("retrieve", _bind(nodes.retrieve, deps))
    graph.add_node("generate", _bind(nodes.generate, deps))

    if not agentic:
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    graph.add_node("route", _bind(nodes.route, deps))
    graph.add_node("plan", _bind(nodes.plan, deps))
    graph.add_node("grade", _bind(nodes.grade, deps))
    graph.add_node("rewrite", _bind(nodes.rewrite, deps))
    graph.add_node("abstain", _bind(nodes.abstain, deps))

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda s: {"out_of_scope": "end", "multi_hop": "plan"}.get(
            s.get("route", "simple"), "retrieve"
        ),
        {"end": END, "plan": "plan", "retrieve": "retrieve"},
    )
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        lambda s: nodes.decide_after_grade(s, deps),
        {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("abstain", END)

    if deps.settings.agent.verify_answers:
        graph.add_node("verify", _bind(nodes.verify, deps))
        graph.add_edge("generate", "verify")
        graph.add_conditional_edges(
            "verify",
            lambda s: nodes.decide_after_verify(s, deps),
            {"done": END, "revise": "generate"},
        )
    else:
        graph.add_edge("generate", END)

    return graph.compile()


def initial_state(question: str, doc_ids: list[str] | None = None) -> AgentState:
    return AgentState(
        question=question,
        doc_ids=doc_ids,
        route="simple",
        sub_questions=[],
        current_query=question,
        hyde_text=None,
        queries_tried=[],
        evidence=[],
        rewrites=0,
        answer="",
        citations=[],
        verification=None,
        revisions=0,
        feedback=None,
        abstained=False,
    )
