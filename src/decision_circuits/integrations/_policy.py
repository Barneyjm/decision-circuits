"""The one piece every agent-SDK integration shares.

A `CircuitPolicy` runs a circuit on a state and turns one gate's result
into an action:

    allow   run the tool / accept the input
    block   refuse, with the gate's probability and trace in the message
    ask     hand the decision to a human (interrupt, permission prompt)

Which result maps to which action is a plain dict, so a categorical
gate can route too ({"safe": "allow", "risky": "block"}). Keys are
whatever `result_key` returns for the gate: its value when it decided
(True/False for a threshold, the option for argmax/majority/verify, the
bucket for order), or its outcome ("abstain", "escalate") when it did
not. The defaults send uncertain results to a human; an SDK with no way
to ask blocks them instead. A result with no entry is never allowed.

An SDK adapter is then a few lines: build the state the SDK gives you,
call `policy.judge(state)`, map the action to the SDK's own result
type. Nothing here imports an SDK.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from decision_circuits.dsl import Circuit, RunOutput
from decision_circuits.gates import GateResultDict, result_key
from decision_circuits.types import Backend

Action = Literal["allow", "block", "ask"]

DEFAULT_ACTIONS: dict[Any, Action] = {True: "block", False: "allow", "abstain": "ask", "escalate": "ask"}


@dataclass(frozen=True)
class Judgment:
    action: Action
    reason: str  # the gate's value, outcome, probability, and trace, for messages and logs
    output: RunOutput  # full answers and gate results


class CircuitPolicy:
    """`judge(state)` -> Judgment. Built once, used per call.

    Args:
        circuit: the circuit to run.
        backend: anything with `answer(state, questions, model=None)`.
        gate: the gate whose result decides.
        actions: overrides for result key -> action. Merged over
            DEFAULT_ACTIONS.
        uncertain_fallback: the action for a result key with no entry
            (default "block"). SDKs that cannot ask a human pass "block"
            for the uncertain outcomes too via `actions`.
    """

    def __init__(self, circuit: Circuit, backend: Backend, *, gate: str, actions: Mapping[Any, Action] | None = None, uncertain_fallback: Action = "block"):
        self.circuit = circuit
        self.backend = backend
        self.gate = gate
        self.actions: dict[Any, Action] = {**DEFAULT_ACTIONS, **(actions or {})}
        self.uncertain_fallback = uncertain_fallback
        self.last: Judgment | None = None

    def judge(self, state: Any) -> Judgment:
        out = self.circuit.run(self.backend, state)
        results = out["gates"]
        if self.gate not in results:
            raise KeyError(f"gate {self.gate!r} is not in the circuit; gates are {sorted(results)}")
        self.last = Judgment(self.actions.get(result_key(results[self.gate]), self.uncertain_fallback), explain(results, self.gate), out)
        return self.last


def explain(results: Mapping[str, GateResultDict], gate: str) -> str:
    r = results[gate]
    p = f" p={r['p']:.2f}" if r.get("p") is not None else ""
    conf = f" confidence={r['confidence']:.2f}" if r.get("confidence") is not None else ""
    trace = "; ".join(r.get("trace") or [])
    return f"gate `{gate}` -> {r['value']!r} ({r['outcome']}{p}{conf}). {trace}"


def tool_state(tool_name: str, tool_args: Any, *, description: str | None = None, messages: Any = None, **extra: Any) -> dict[str, Any]:
    """The state a tool-call circuit judges: the proposed call, the tool's
    own description, and recent conversation. Backends convert SDK
    objects to JSON themselves (`to_jsonable`), so pass them as they are.
    Everything in it is data to the model, never instructions; say so in
    the circuit's questions."""
    state: dict[str, Any] = {"tool_call": {"name": tool_name, "args": tool_args}}
    if description:
        state["tool_description"] = description
    if messages is not None:
        state["messages"] = messages
    state.update({k: v for k, v in extra.items() if v is not None})
    return state
