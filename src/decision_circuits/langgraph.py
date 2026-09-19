"""LangGraph and LangChain adapters. No dependency on either: a node is a
function of the graph state, and a router is a function returning an
edge key. Import from here and wire them in with LangGraph's own API.

    from decision_circuits.langgraph import as_node, route_on

    graph.add_node("triage", as_node(circuit, backend, state_key="text"))
    graph.add_conditional_edges("triage", route_on("route"), {
        "billing": "billing_agent", "technical": "tech_agent",
        "abstain": "human", "escalate": "human",
    })

`as_node` reads the item to judge from `state[state_key]`, runs the
circuit, and writes `{"circuit": {"answers": ..., "gates": ...}}` into
the graph state (key configurable). `route_on("gate")` reads that gate's
result: its value when the gate decided, otherwise the outcome
("abstain", "escalate", or "default" as its default value), so the
uncertain paths are ordinary edges.

For a LangChain runnable, `as_runnable(circuit, backend)` returns a
`RunnableLambda` if langchain-core is installed; otherwise the plain
function from `as_node` works anywhere a callable does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def as_node(
    circuit: Any, backend: Any, state_key: str = "input", out_key: str = "circuit", model: str | None = None
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def node(state: dict[str, Any]) -> dict[str, Any]:
        item = state[state_key]
        answers = backend.answer(item, circuit.questions, model=model or circuit.model)
        return {out_key: {"answers": answers, "gates": circuit.evaluate(answers)}}

    node.__name__ = f"circuit_{out_key}"
    return node


def route_on(gate: str, out_key: str = "circuit") -> Callable[[dict[str, Any]], str]:
    """Edge key for a gate: str(value) when decided, else the outcome."""

    def router(state: dict[str, Any]) -> str:
        r = state[out_key]["gates"][gate]
        if r["outcome"] == "decided":
            return str(r["value"])
        if r["outcome"] == "default":
            return str(r["value"])
        return str(r["outcome"])

    router.__name__ = f"route_on_{gate}"
    return router


def as_runnable(circuit: Any, backend: Any, **kw: Any) -> Any:
    fn = as_node(circuit, backend, **kw)
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError:
        return fn
    return RunnableLambda(fn)
