"""Shared plumbing for the agent-SDK integrations: turn a circuit's gate
results into one of three actions, and describe why.

    allow   run the tool / accept the input
    block   refuse, with the gate's probability and trace in the message
    ask     hand the decision to a human (interrupt, permission prompt)

Which gate value maps to which action is a plain dict, so a categorical
gate can route too ({"safe": "allow", "risky": "block", "unclear": "ask"}).
The uncertain outcomes (`abstain`, `escalate`) get their own entries;
the default sends them to a human where the SDK can, and blocks where
it cannot. Nothing here imports an SDK.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

Action = Literal["allow", "block", "ask"]


class GateResultDict(TypedDict):
    """`GateResult.to_dict()`: what `Circuit.evaluate` returns per gate."""

    value: Any
    p: float | None
    confidence: float | None
    uncertain: bool
    outcome: str
    trace: list[str]


GateResults = dict[str, GateResultDict]


class RunOutput(TypedDict):
    answers: dict[str, Any]
    gates: GateResults


DEFAULT_ACTIONS: dict[Any, Action] = {True: "block", False: "allow", "abstain": "ask", "escalate": "ask", "default": "allow"}


def decide(results: Mapping[str, GateResultDict], gate: str, actions: Mapping[Any, Action] | None = None, *, uncertain_fallback: Action = "block") -> Action:
    """Map one gate's result to an action.

    A decided gate is looked up by its value (bools for threshold gates,
    the option or bucket for categorical ones). An undecided gate is
    looked up by its outcome. A value with no entry is treated as
    `uncertain_fallback`, so an unlisted route never silently allows."""
    table = {**DEFAULT_ACTIONS, **(actions or {})}
    r = results[gate]
    key = r["value"] if r["outcome"] in ("decided", "default") else r["outcome"]
    if key in table:
        return table[key]
    if isinstance(key, str) and key.lower() in table:
        return table[key.lower()]
    return uncertain_fallback


def explain(results: Mapping[str, GateResultDict], gate: str) -> str:
    r = results[gate]
    p = f" p={r['p']:.2f}" if r.get("p") is not None else ""
    conf = f" confidence={r['confidence']:.2f}" if r.get("confidence") is not None else ""
    trace = "; ".join(r.get("trace") or [])
    return f"gate `{gate}` -> {r['value']!r} ({r['outcome']}{p}{conf}). {trace}"


def tool_state(
    tool_name: str, tool_args: Any, *, description: str | None = None, messages: Any = None, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The state a tool-call circuit judges: the proposed call, the tool's
    own description, and recent conversation. Everything in it is data
    to the model, never instructions; say so in the circuit's questions."""
    state: dict[str, Any] = {"tool_call": {"name": tool_name, "args": _jsonable(tool_args)}}
    if description:
        state["tool_description"] = description
    if messages is not None:
        state["messages"] = _jsonable(messages)
    if extra:
        state.update(extra)
    return state


def _jsonable(x: Any) -> Any:
    """Best-effort conversion of SDK message objects to plain JSON."""
    if isinstance(x, str | int | float | bool) or x is None:
        return x
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    for attr in ("model_dump", "to_dict", "dict"):
        f = getattr(x, attr, None)
        if callable(f):
            try:
                return _jsonable(f())
            except (TypeError, ValueError):
                continue
    if hasattr(x, "content") and hasattr(x, "type"):
        return {"type": x.type, "content": _jsonable(x.content)}
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)
