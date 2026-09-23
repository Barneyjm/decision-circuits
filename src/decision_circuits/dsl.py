"""A small expression language for decision circuits.

Symbolic references combine with Python operators and compile to the
`gates` block the server evaluates. Same idiom as Django `Q` objects
or Polars expressions: build an expression, hand it to the request.

    from decision_circuits import Circuit, Q, G, argmax, majority, verify, order

    c = Circuit()
    c.noul("pii", "Does this text contain PII about a private individual?",
           true="Email, phone, home address, ID, card number", false="No PII or business-only")
    c.noul("business", "Are all identifying details about a business, not a person?")
    c.choice("dept", "Which team?", {"billing": "...", "technical": "...", "other": "..."})
    c.score("urgency", "How urgent?", ["Low", "Medium", "High", "Critical"])

    c.gate("redact", ((Q("pii") >= 0.7) & ~Q("business")) >= 0.6, on_uncertain="escalate")
    c.gate("route", argmax("dept", min_confidence=0.35))
    c.gate("tier", order("urgency", [1.0, 2.0, 2.6]))
    c.gate("human", (Q("angry") | Q("urgency")[3]) >= 0.6, on_uncertain="escalate")
    c.gate("bill_hot", (G("route")["billing"] & G("tier")[3]).at(0.5))

    body = c.request(state)          # the wire body: TypeSafe's request plus a `gates` block
    out = c.run(SystemOne(api_key=KEY), state)   # any backend; see decision_circuits.backends

Semantics, in one paragraph. `Q("x")` is the probability behind a
question: a noul's P(yes), `Q("choice")["option"]` or `Q("score")[level]`
for one outcome (the `"choice:option"` string form also works), or
`G("name")` for an earlier gate, indexed the same way for categorical
gates. `~` flips it (a `not`
gate), `&` multiplies (an `and` gate, independence assumed and printed
in the trace), `|` is 1 - prod(1 - p) (an `or` gate). `>= tau` turns a
probability into a boolean at that threshold with an uncertainty band
around it; an expression used as a gate without `>=` thresholds at 0.5.
`argmax`, `majority`, `verify`, `order` are the categorical/ordinal
gates. Every gate carries `.on_uncertain("abstain" | "escalate" | "default", default=...)`
and `.band(width)`.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from decision_circuits import tracing
from decision_circuits.gates import Gate, GateResultDict, evaluate_gates
from decision_circuits.types import Answers, Backend, answer_distributions

if TYPE_CHECKING:
    from decision_circuits.interventions import Interventions


class RunOutput(TypedDict):
    """What `Circuit.run` returns."""

    model: str | None
    answers: Answers
    gates: dict[str, GateResultDict]
    gates_evaluated_by: str  # "client" (here) or "server" (a server that evaluates circuits)


# ---------------------------------------------------------------- expressions


class Expr:
    """Base for probability-valued expressions."""

    def __and__(self, other: Expr) -> And:
        return And([self, _as_expr(other)])

    def __or__(self, other: Expr) -> Or:
        return Or([self, _as_expr(other)])

    def __invert__(self) -> Not:
        return Not(self)

    def __ge__(self, tau: float) -> Threshold:
        """`expr >= tau` builds a Threshold node; it does not compare.
        `bool(Q("pii") >= 0.7)` is always True. Use `.at(tau)` where an
        operator would read as a runtime comparison."""
        return Threshold(self, float(tau))

    def at(self, tau: float) -> Threshold:
        """Named form of `>=`: `Q("pii").at(0.7)`."""
        return Threshold(self, float(tau))


def _as_expr(x: Any) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, str):
        return Q(x)
    raise TypeError(f"cannot use {x!r} in a circuit expression")


@dataclass(frozen=True)
class Q(Expr):
    """A question reference. A noul id is a probability on its own;
    index a choice or score to get one option's probability:
    `Q("dept")["billing"]`, `Q("urgency")[3]` (the `"dept:billing"`
    string form is accepted too)."""

    ref: str

    def __getitem__(self, key: Any) -> Q:
        if ":" in self.ref:
            raise KeyError(f"{self.ref!r} already names an option")
        return Q(f"{self.ref}:{key}")


@dataclass(frozen=True)
class G(Expr):
    """A reference to an earlier gate. A boolean gate is a probability on
    its own; index a categorical gate for one value: `G("route")["billing"]`."""

    ref: str

    def __getitem__(self, key: Any) -> G:
        if ":" in self.ref:
            raise KeyError(f"{self.ref!r} already names a value")
        return G(f"{self.ref}:{key}")


@dataclass(frozen=True)
class Not(Expr):
    inner: Expr


@dataclass(frozen=True)
class And(Expr):
    parts: list[Expr]

    def __and__(self, other: Expr) -> And:  # flatten chains
        return And([*self.parts, _as_expr(other)])


@dataclass(frozen=True)
class Or(Expr):
    parts: list[Expr]

    def __or__(self, other: Expr) -> Or:
        return Or([*self.parts, _as_expr(other)])


@dataclass(frozen=True)
class Threshold(Expr):
    inner: Expr
    tau: float


# categorical / ordinal gate constructors ----------------------------------


@dataclass(frozen=True)
class Categorical:
    op: str
    input: str | None = None
    inputs: list[str] | None = None
    check: Expr | None = None
    tau: float = 0.5
    min_confidence: float = 0.0
    cutpoints: list[float] | None = None
    k: int | None = None
    relation: str | None = None


def _ref(x: str | Q | G) -> str:
    return x.ref if isinstance(x, Q | G) else x


def _pool(refs: tuple[str | Q | G, ...]) -> dict[str, Any]:
    """One name is a multi question, read option by option; several are the references themselves."""
    names = [_ref(r) for r in refs]
    if not names:
        raise ValueError("name a multi question, or two or more references")
    return {"input": names[0]} if len(names) == 1 else {"inputs": names}


def argmax(choice_id: str, min_confidence: float = 0.0) -> Categorical:
    return Categorical("argmax", input=choice_id, min_confidence=min_confidence)


def majority(*choice_ids: str, min_confidence: float = 0.0) -> Categorical:
    return Categorical("majority", inputs=list(choice_ids), min_confidence=min_confidence)


def verify(choice_id: str, check: Expr | str, tau: float = 0.6, min_confidence: float = 0.0) -> Categorical:
    return Categorical("verify", input=choice_id, check=_as_expr(check), tau=tau, min_confidence=min_confidence)


def order(score_id: str, cutpoints: list[float]) -> Categorical:
    return Categorical("order", input=score_id, cutpoints=list(cutpoints))


def at_least(k: int, *refs: str | Q | G, tau: float = 0.5) -> Categorical:
    """True when at least `k` hold: `at_least(2, "issues")` over a multi's options, or
    `at_least(2, "late", "damaged", Q("dept")["billing"])` over references. Independence is
    assumed and printed in the trace; k=1 is OR, k=all is AND."""
    return Categorical("at_least", k=k, tau=tau, **_pool(refs))


def count(*refs: str | Q | G, min_confidence: float = 0.0) -> Categorical:
    """How many hold: the most likely count, with the whole distribution in the result, so
    `G("n")[2]` is P(exactly two)."""
    return Categorical("count", min_confidence=min_confidence, **_pool(refs))


def consistent(a: str | Q | G, b: str | Q | G, relation: str = "same") -> Categorical:
    """Check two answers against each other: "same" (one question asked two ways),
    "complement" (a question and its negation), "implies" (a cannot be likelier than b).
    Uncertain, so `on_uncertain` applies, when the violation exceeds the band."""
    return Categorical("consistent", inputs=[_ref(a), _ref(b)], relation=relation)


# ---------------------------------------------------------------- circuit


@dataclass
class GateDef:
    name: str
    body: Expr | Categorical
    on_uncertain_: str = "abstain"
    default_: Any = None
    band_: float = 0.1

    def on_uncertain(self, policy: str, default: Any = None) -> GateDef:
        self.on_uncertain_ = policy
        self.default_ = default
        return self

    def band(self, width: float) -> GateDef:
        self.band_ = width
        return self


@dataclass
class Circuit:
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)
    gates: list[GateDef] = field(default_factory=list)
    model: str | None = None  # None: the backend picks

    # questions ---------------------------------------------------------
    def noul(self, qid: str, instructions: Any, true: str | None = None, false: str | None = None, **extra: Any) -> Circuit:
        q: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if true or false:
            q["criteria"] = {"true": true, "false": false}
        self.questions[qid] = {**q, **extra}
        return self

    def choice(self, qid: str, instructions: Any, criteria: dict[str, Any], **extra: Any) -> Circuit:
        self.questions[qid] = {"type": "choice", "instructions": instructions, "criteria": criteria, **extra}
        return self

    def multi(self, qid: str, instructions: Any, criteria: dict[str, Any], **extra: Any) -> Circuit:
        """Every option that applies, each with its own probability (circuit v2 models, text states)."""
        self.questions[qid] = {"type": "multi", "instructions": instructions, "criteria": criteria, **extra}
        return self

    def locate(self, qid: str, instructions: Any, none: str | None = None, **extra: Any) -> Circuit:
        """Which part of the state answers it: a field, a list element or a sentence, or "none"
        (described by `none`). Circuit v2 models, text states."""
        q: dict[str, Any] = {"type": "locate", "instructions": instructions, **extra}
        if none:
            q["criteria"] = none
        self.questions[qid] = q
        return self

    def score(self, qid: str, instructions: Any, levels: list[Any], **extra: Any) -> Circuit:
        self.questions[qid] = {"type": "score", "instructions": instructions, "criteria": levels, **extra}
        return self

    # gates -------------------------------------------------------------
    def gate(
        self,
        name: str,
        body: Expr | Categorical | str,
        *,
        on_uncertain: str | None = None,
        default: Any = None,
        band: float | None = None,
    ) -> GateDef:
        """Add a gate. Policy can be given as keywords here or chained on
        the returned GateDef (`.on_uncertain(...)`, `.band(...)`). Names starting
        with "_" are reserved for the helper gates `compile` generates."""
        if name.startswith("_"):
            raise ValueError(f"gate name {name!r}: names starting with '_' are reserved for generated helpers")
        g = GateDef(name, _as_expr(body) if isinstance(body, str) else body)
        if on_uncertain is not None:
            g.on_uncertain(on_uncertain, default)
        if band is not None:
            g.band(band)
        self.gates.append(g)
        return g

    # rendering ---------------------------------------------------------
    def to_mermaid(self, results: dict[str, Any] | None = None, answers: dict[str, Any] | None = None, direction: str = "LR", plain: bool = False) -> str:
        """Schematic of the compiled circuit; see `render_mermaid`."""
        return render_mermaid(self, results, answers, direction, plain)

    # compile -----------------------------------------------------------
    def compile(self) -> dict[str, dict[str, Any]]:
        """Lower the expression trees to the server's flat `gates` map.
        Sub-expressions become auto-named helper gates (`_name_N`) so
        every intermediate probability shows up in the trace. Recompiled
        on every call (it is cheap) so `GateDef.band()` / `.on_uncertain()`
        edits are always honored."""
        out: dict[str, dict[str, Any]] = {}
        counter = itertools.count(1)

        def helper(prefix: str, spec: dict[str, Any]) -> str:
            name = f"_{prefix}_{next(counter)}"
            out[name] = spec
            return name

        def ref_of(e: Expr, prefix: str, band: float) -> str:
            """A string reference the server accepts for this expression,
            emitting helper gates where needed. A threshold inside an expression
            is a decision, so it is referenced by its value ("helper:True": 1 or
            0), with the enclosing gate's band so a probability near its tau
            makes the whole gate uncertain rather than silently rounding."""
            if isinstance(e, Q | G):
                return e.ref
            if isinstance(e, Not):
                return helper(prefix, {"op": "not", "input": ref_of(e.inner, prefix, band), "tau": 0.5, "band": 0.0})
            if isinstance(e, And | Or):
                parts = [ref_of(p, prefix, band) for p in e.parts]
                return helper(prefix, {"op": "and" if isinstance(e, And) else "or", "inputs": parts, "tau": 0.5, "band": 0.0})
            if isinstance(e, Threshold):
                return helper(prefix, {"op": "threshold", "input": ref_of(e.inner, prefix, band), "tau": e.tau, "band": band}) + ":True"
            raise TypeError(f"unsupported expression {e!r}")

        for g in self.gates:
            common = {"on_uncertain": g.on_uncertain_, "band": g.band_}
            if g.on_uncertain_ == "default":
                common["default"] = g.default_
            b = g.body
            if isinstance(b, Categorical):
                spec: dict[str, Any] = {"op": b.op, "tau": b.tau, "min_confidence": b.min_confidence}
                if b.input is not None:
                    spec["input"] = b.input
                if b.inputs is not None:
                    spec["inputs"] = b.inputs
                if b.check is not None:
                    spec["check"] = ref_of(b.check, g.name, g.band_)
                if b.cutpoints is not None:
                    spec["cutpoints"] = b.cutpoints
                if b.k is not None:
                    spec["k"] = b.k
                if b.relation is not None:
                    spec["relation"] = b.relation
                out[g.name] = {**spec, **common}
                continue
            # boolean expression: the top node becomes the named gate
            tau = 0.5
            e = b
            if isinstance(e, Threshold):
                tau, e = e.tau, e.inner
            if isinstance(e, Q | G | Threshold):  # a nested threshold: the outer one thresholds its decision
                out[g.name] = {"op": "threshold", "input": ref_of(e, g.name, g.band_), "tau": tau, **common}
            elif isinstance(e, Not):
                out[g.name] = {"op": "not", "input": ref_of(e.inner, g.name, g.band_), "tau": tau, **common}
            elif isinstance(e, And | Or):
                parts = [ref_of(p, g.name, g.band_) for p in e.parts]
                out[g.name] = {"op": "and" if isinstance(e, And) else "or", "inputs": parts, "tau": tau, **common}
            else:
                raise TypeError(f"unsupported gate body {b!r}")
        return out

    def request(self, state: Any) -> dict[str, Any]:
        """The wire request: TypeSafe's body plus a `gates` block. `model`
        is the circuit's own; `SystemOne` fills in its default when None."""
        return {"state": state, "model": self.model, "questions": dict(self.questions), "gates": self.compile()}

    def evaluate(self, answers: Answers) -> dict[str, GateResultDict]:
        """Evaluate the compiled gates locally against answers already in hand."""
        res = evaluate_gates({k: Gate.from_dict(v) for k, v in self.compile().items()}, answers)
        return {k: v.to_dict() for k, v in res.items() if not k.startswith("_")}

    def run(self, backend: Backend, state: Any, *, model: str | None = None) -> RunOutput:
        """Answer the questions with `backend` and evaluate the gates.

        A backend is anything with `answer(state, questions, model=...)`
        (see `decision_circuits.backends`; `SystemOne` wraps any System
        One HTTP server, TypeSafe's Jev included). A backend may also
        implement `answer_with_gates(state, questions, gates, model=...)`
        to let a server evaluate the circuit itself; `SystemOne` does,
        and falls back to answering when the server rejects gates."""
        model = model or self.model or getattr(backend, "model", None)
        public = [g.name for g in self.gates]
        with tracing.span(
            "decision_circuits.run",
            **{
                "gen_ai.operation.name": "decision_circuits.run",
                "gen_ai.request.model": model,
                "decision_circuits.backend": type(backend).__name__,
                "decision_circuits.questions": list(self.questions),
                "decision_circuits.question_types": [q["type"] for q in self.questions.values()],
                "decision_circuits.gates": public,
            },
        ) as span:
            out = self._run(backend, state, model)
            _record(span, backend, out)
            return out

    def _run(self, backend: Backend, state: Any, model: str | None) -> RunOutput:
        with_gates = getattr(backend, "answer_with_gates", None)
        with tracing.span("decision_circuits.backend", **{"decision_circuits.backend": type(backend).__name__, "gen_ai.request.model": model}):
            if callable(with_gates):
                answers, gates = with_gates(state, self.questions, self.compile(), model=model)
                if gates is not None:
                    return {
                        "model": model,
                        "answers": answers,
                        "gates": {k: v for k, v in gates.items() if not k.startswith("_")},
                        "gates_evaluated_by": "server",
                    }
            else:
                answers = backend.answer(state, self.questions, model=model)
        missing = [q for q in self.questions if q not in answers]
        if missing:
            raise ValueError(f"{type(backend).__name__} returned no answer for {', '.join(map(repr, missing))}")
        return {"model": model, "answers": answers, "gates": self.evaluate(answers), "gates_evaluated_by": "client"}

    def intervene(self, backend: Backend, state: Any, interventions: Mapping[str, Any], **kw: Any) -> Interventions:
        """Run the circuit on `state` and on each edited state; report which gates flipped
        and how every probability moved. See `decision_circuits.interventions`."""
        from decision_circuits.interventions import intervene

        return intervene(self, backend, state, interventions, **kw)

    def ablate(self, backend: Backend, state: Any, **kw: Any) -> Interventions:
        """Remove each sentence (text state) or field (dict state) in turn; see `intervene`."""
        from decision_circuits.interventions import ablate

        return ablate(self, backend, state, **kw)


def _record(span: Any, backend: Any, out: RunOutput) -> None:
    """A run's result on its span: the response ids, an event per answer and per gate, and
    the gates that went to a person or held back. Never the state."""
    last = getattr(backend, "last_response", None) or {}
    span.set_attribute("decision_circuits.gates_evaluated_by", out["gates_evaluated_by"])
    for key, value in (("gen_ai.response.model", last.get("model")), ("gen_ai.response.id", last.get("request_id"))):
        if value:
            span.set_attribute(key, value)
    if (last.get("usage") or {}).get("input_tokens") is not None:
        span.set_attribute("gen_ai.usage.input_tokens", int(last["usage"]["input_tokens"]))
    for qid, a in out["answers"].items():
        for suffix, dist in answer_distributions(a).items():
            pick = max(dist, key=dist.__getitem__)
            tracing.event(span, "decision_circuits.answer", question=qid + suffix, type=a.get("type"), pick=pick, p=round(dist[pick], 4))
    held: dict[str, list[str]] = {"escalate": [], "abstain": []}
    for gid, g in out["gates"].items():
        tracing.event(
            span,
            "decision_circuits.gate",
            gate=gid,
            value=g.get("value"),
            outcome=g.get("outcome"),
            p=g.get("p"),
            uncertain=g.get("uncertain"),
            trace="; ".join(g.get("trace") or []),
        )
        if g.get("outcome") in held:
            held[g["outcome"]].append(gid)
    span.set_attribute("decision_circuits.escalated", held["escalate"])
    span.set_attribute("decision_circuits.abstained", held["abstain"])


# ---------------------------------------------------------------- rendering


def _short(s: str, n: int = 38) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def render_mermaid(
    circuit: Circuit, results: dict[str, Any] | None = None, answers: dict[str, Any] | None = None, direction: str = "LR", plain: bool = False
) -> str:
    """Render the compiled circuit as a schematic. `plain=True` omits the
    init directive and inline HTML styling for stricter renderers (some
    hosted Mermaid builds reject them); layout is the same.

    Schematic: an input column of
    questions, a logic column of gates, and a decisions column for the
    gates nothing else consumes. Threshold and NOT helpers are folded
    into edge labels rather than drawn as nodes. Only decisions are
    coloured: green yes / grey no / amber abstained or escalated."""
    compiled = circuit.compile()

    # which gates feed other gates (their base name before ':')
    consumers: dict[str, set[str]] = {g: set() for g in compiled}
    for gid, spec in compiled.items():
        refs = ([spec["input"]] if spec.get("input") else []) + list(spec.get("inputs", [])) + ([spec["check"]] if spec.get("check") else [])
        for ref in refs:
            base = ref.partition(":")[0]
            if base in consumers:
                consumers[base].add(gid)

    # helpers that fold into an edge: threshold/not with one consumer
    folded: dict[str, tuple[str, str]] = {}  # helper -> (source ref, label)
    for gid, spec in compiled.items():
        if gid.startswith("_") and spec["op"] in ("threshold", "not") and len(consumers[gid]) == 1:
            src_ref = spec["input"]
            lab = "NOT" if spec["op"] == "not" else f"≥ {spec['tau']:g}"
            # chain: a folded helper feeding a folded helper
            while src_ref in folded:
                inner_src, inner_lab = folded[src_ref]
                lab = f"{inner_lab} · {lab}"
                src_ref = inner_src
            folded[gid] = (src_ref, lab)

    def qnode(qid: str) -> str:
        return f"q_{qid}"

    def gnode(gid: str) -> str:
        return "g_" + gid.replace("-", "_")

    init = (
        "%%{init: {'theme': 'base', 'flowchart': {'nodeSpacing': 40, 'rankSpacing': 120, 'curve': 'basis', 'useMaxWidth': false, 'htmlLabels': true}, "
        "'themeVariables': {'fontFamily': 'IBM Plex Sans, system-ui, sans-serif', 'fontSize': '13px', 'lineColor': '#5F6B78'}}}%%"
    )
    L = [
        *([] if plain else [init]),
        f"flowchart {direction}",
        "  classDef q fill:#FFFFFF,stroke:#1264A3,stroke-width:1.5px,color:#1B1F24;",
        "  classDef logic fill:#F7F8F6,stroke:#5F6B78,stroke-width:1.5px,color:#1B1F24;",
        "  classDef yes fill:#DDF3E4,stroke:#2E7D4F,stroke-width:2.5px,color:#0F3D22;",
        "  classDef no fill:#EEF0F2,stroke:#98A2AD,stroke-width:2px,color:#3B4550;",
        "  classDef hold fill:#FFF1CC,stroke:#9A6B00,stroke-width:2.5px,color:#4A3300;",
        "  classDef col fill:none,stroke:#D9DEE3,stroke-dasharray:3 3,color:#5F6B78;",
    ]

    # --- inputs
    L.append('  subgraph IN["Inputs"]')
    L.append("    direction TB")
    for qid, q in circuit.questions.items():
        kind = q["type"]
        if answers and qid in answers:
            a = answers[qid]
            if kind == "noul":
                val = f"yes {a['noul']:.0%}"
            elif kind == "choice":
                val = f"{a['choice']} {a['probabilities'][a['choice']]:.0%}"
            elif kind == "multi":
                val = ", ".join(a["selected"]) or "none apply"
            elif kind == "rank":
                val = " > ".join(a["order"][:3])
            elif kind == "match":
                val = f"{sum(m['match'] != 'none' for m in a['matches'].values())} of {len(a['matches'])} matched"
            elif kind == "locate":
                val = f"none {a['none']:.0%}" if a["none"] >= 0.5 or not a["located"] else f"{a['located'][0]['path']} {a['located'][0]['probability']:.0%}"
            else:
                val = f"{a['score']:.1f} / {len(q['criteria']) - 1}"
            label = f"<b>{qid}</b><br/>{val}"
        else:
            label = f"<b>{qid}</b><br/>{kind}" if plain else f"<b>{qid}</b><br/><span style='color:#5F6B78'>{kind}</span>"
        L.append(f'    {qnode(qid)}["{label}"]:::q')
    L.append("  end")
    L.append("  class IN col")

    # --- logic and decisions
    OP_LABEL = {
        "and": "AND",
        "or": "OR",
        "not": "NOT",
        "threshold": "≥",
        "argmax": "PICK",
        "majority": "VOTE",
        "verify": "VERIFY",
        "order": "BUCKET",
        "at_least": "AT LEAST",
        "count": "COUNT",
        "consistent": "CONSISTENT",
    }
    SHAPES = {"and": ("{{", "}}"), "or": (">", "]"), "not": ("((", "))"), "threshold": ("{", "}")}
    logic, decisions = [], []
    for gid, spec in compiled.items():
        if gid in folded:
            continue
        terminal = not gid.startswith("_") and not consumers[gid]
        (decisions if terminal else logic).append(gid)

    def edge_lines(gid: str, spec: dict[str, Any]) -> list[str]:
        out = []
        refs = ([spec["input"]] if spec.get("input") else []) + list(spec.get("inputs", [])) + ([spec["check"]] if spec.get("check") else [])
        for ref in refs:
            lab = "check" if spec.get("check") == ref else ""
            base, _, opt = ref.partition(":")
            if base.startswith("_") and opt == "True":  # a threshold helper referenced by its decision
                opt = ""
            if base in folded:
                base, flab = folded[base]
                base, _, opt = base.partition(":")
                lab = flab + (" · " + lab if lab else "")
            if opt:
                q = circuit.questions.get(base)
                shown = f"level {opt}" if (q and q["type"] == "score") else opt
                lab = (f"{shown} " + lab).strip()
            node = gnode(base) if base in compiled else qnode(base)
            out.append(f"  {node} --{'>' if not lab else f'>|{lab}|'} {gnode(gid)}")
        return out

    def node_line(gid: str, spec: dict[str, Any], terminal: bool) -> str:
        op = spec["op"]
        head = OP_LABEL[op]
        if op == "order":
            head += " " + " | ".join(f"{c:g}" for c in spec.get("cutpoints", []))
        if op == "at_least":
            head += f" {spec['k']}"
        if op == "consistent":
            head += f"<br/>{spec['relation']}"
        if op in ("argmax", "majority", "verify") and spec.get("min_confidence"):
            head += f"<br/>conf ≥ {spec['min_confidence']:g}" if plain else f"<br/><span style='color:#5F6B78'>conf ≥ {spec['min_confidence']:g}</span>"
        if op in ("and", "or") and not terminal and spec.get("tau", 0.5) != 0.5:
            head += f" ≥ {spec['tau']:g}"
        if terminal:
            label = f"<b>{gid.replace('_', ' ').title()}</b><br/>{head}"
            if op in ("and", "or", "threshold", "not"):
                label += f"{'' if op == 'threshold' else ' ≥'} {spec.get('tau', 0.5):g}"  # a threshold's head is already "≥"
            cls = "logic"
            if results and gid in results:
                r = results[gid]
                val, pr, outc = r.get("value"), r.get("p"), r.get("outcome", "decided")
                if outc != "decided":
                    verdict, cls = f"⚠ {outc.upper()}", "hold"
                elif val is True:
                    verdict, cls = "✓ YES", "yes"
                elif val is False:
                    verdict, cls = "✗ NO", "no"
                else:
                    verdict, cls = f"✓ {val}", "yes"
                label += f"<br/><b>{verdict}</b>" + (f" · {pr:.0%}" if pr is not None else "")
            return f'  {gnode(gid)}["{label}"]:::{cls}'
        label = f"<b>{gid.replace('_', ' ')}</b><br/>{head}" if not gid.startswith("_") else head
        if results and gid in results:
            r = results[gid]
            if r.get("outcome", "decided") != "decided":
                label += f"<br/>⚠ {r['outcome']}"
            elif op in ("argmax", "majority", "verify", "order"):
                label += f"<br/>→ <b>{r.get('value')}</b>" + (f" · {r['p']:.0%}" if r.get("p") is not None else "")
            elif r.get("p") is not None:
                label += f"<br/>{r['p']:.0%}"
        lo, hi = SHAPES.get(op, ("[", "]"))
        return f'  {gnode(gid)}{lo}"{label}"{hi}:::logic'

    if logic:
        L.append('  subgraph LOGIC["Logic"]')
        L.append("    direction TB")
        for gid in logic:
            L.append("  " + node_line(gid, compiled[gid], False))
        L.append("  end")
        L.append("  class LOGIC col")
    L.append('  subgraph OUT["Decisions"]')
    L.append("    direction TB")
    for gid in decisions:
        L.append("  " + node_line(gid, compiled[gid], True))
    L.append("  end")
    L.append("  class OUT col")
    for gid in logic + decisions:
        L += edge_lines(gid, compiled[gid])
    return "\n".join(L)


to_mermaid = render_mermaid  # module-level alias
