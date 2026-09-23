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

    from decision_circuits.backends import SystemOne
    out = c.run(SystemOne(api_key=key, model="jev-latest"), state)   # or any object with .answer()
    out["gates"]["redact"]   # {"value": True, "p": 0.91, "outcome": "decided", "trace": [...]}
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

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
    RunOutput,
    Threshold,
    argmax,
    majority,
    order,
    render_mermaid,
    to_mermaid,
    verify,
)
from decision_circuits.gates import Gate, GateResult, GateResultDict, evaluate_gates, result_key
from decision_circuits.interventions import Interventions, drop, set_to
from decision_circuits.media import Audio, Image, Media
from decision_circuits.types import (
    Answer,
    Answers,
    Backend,
    Question,
    answer_distributions,
    answer_from_probabilities,
    normalized_confidence,
    option_keys,
    to_jsonable,
)

try:
    __version__ = _version("decision-circuits")  # one source: pyproject.toml
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0+unknown"

__all__ = [
    "And",
    "Answer",
    "Answers",
    "Audio",
    "Backend",
    "Categorical",
    "Circuit",
    "Expr",
    "G",
    "Gate",
    "GateDef",
    "GateResult",
    "GateResultDict",
    "Image",
    "Interventions",
    "Media",
    "Not",
    "Or",
    "Q",
    "Question",
    "RunOutput",
    "Threshold",
    "__version__",
    "answer_distributions",
    "answer_from_probabilities",
    "argmax",
    "drop",
    "evaluate_gates",
    "majority",
    "normalized_confidence",
    "option_keys",
    "order",
    "render_mermaid",
    "result_key",
    "set_to",
    "to_jsonable",
    "to_mermaid",
    "verify",
]
