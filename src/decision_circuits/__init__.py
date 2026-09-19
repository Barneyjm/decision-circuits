"""Decision circuits: deterministic gates over calibrated model answers.

Build a circuit with the DSL, send it to any System One style server
(TypeSafe's Jev, or an open model speaking the same contract), and get
back typed decisions with probabilities, traces, and explicit
uncertainty handling. Gates are evaluated by the server when it
supports them and here on the client otherwise, so the result is the
same either way.

    from decision_circuits import Circuit, Q, argmax

    c = Circuit()
    c.noul("pii", "Does this text contain PII about a private individual?")
    c.choice("dept", "Which team?", {"billing": None, "technical": None, "other": None})
    c.gate("redact", Q("pii") >= 0.7, on_uncertain="escalate")
    c.gate("route", argmax("dept", min_confidence=0.35))

    out = c.run(httpx.Client(), state, url="https://api.typesafe.ai/v1/systemone",
                headers={"Authorization": f"Bearer {key}"})
    out["gates"]["redact"]   # {"value": True, "p": 0.91, "outcome": "decided", "trace": [...]}
"""

from decision_circuits.dsl import (
    And,
    Categorical,
    Circuit,
    Expr,
    G,
    GateDef,
    Not,
    Or,
    Q,
    Threshold,
    argmax,
    majority,
    order,
    render_mermaid,
    to_mermaid,
    verify,
)
from decision_circuits.gates import Gate, GateResult, evaluate_gates

__version__ = "0.1.0"

__all__ = [
    "And",
    "Categorical",
    "Circuit",
    "Expr",
    "G",
    "Gate",
    "GateDef",
    "GateResult",
    "Not",
    "Or",
    "Q",
    "Threshold",
    "__version__",
    "argmax",
    "evaluate_gates",
    "majority",
    "order",
    "render_mermaid",
    "to_mermaid",
    "verify",
]
