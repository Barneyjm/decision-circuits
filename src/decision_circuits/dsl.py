"""A small expression language for decision circuits.

Symbolic references combine with Python operators and compile to the
`gates` block (evaluated here; a System One server that knows gates accepts the same block). Same idiom as Django `Q` objects
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, TypedDict

from decision_circuits import tracing
from decision_circuits.gates import Gate, GateResultDict, evaluate_gates
from decision_circuits.types import Answers, Backend, answer_distributions

if TYPE_CHECKING:
    from decision_circuits.chips import Chip
    from decision_circuits.interventions import Interventions


class RunOutput(TypedDict):
    """What `Circuit.run` returns."""

    model: str | None
    answers: Answers
    gates: dict[str, GateResultDict]  # evaluated here, by this package


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
    rules: list[tuple[Any, Expr]] | None = None
    otherwise: Any = None


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


def route(rules: Sequence[tuple[Any, Expr | str]], otherwise: Any = None) -> Categorical:
    """One action from a priority list: the first `(action, condition)` whose condition holds,
    else `otherwise`. A gate in a condition counts as its decision (`G("hot")` is "hot decided
    yes"), so `G("a") & ~G("b")` is logic over decisions. A condition too close to call
    stops the route (escalate, by default with `Circuit.route`) instead of trying the next."""
    if not rules:
        raise ValueError("route needs at least one (action, condition) rule")
    return Categorical("route", rules=[(a, _as_expr(c)) for a, c in rules], otherwise=otherwise)


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
    mounts: dict[str, dict[str, Any]] = field(default_factory=dict)  # namespace -> {chip, pins, outputs}; see `mount`

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

    def route(
        self, name: str, rules: Sequence[tuple[Any, Expr | str]], otherwise: Any = None, *, on_uncertain: str = "escalate", default: Any = None
    ) -> GateDef:
        """The circuit's last step: one action, from the first rule that holds.

            c.route("action", [
                ("page on-call", G("urgent_yes") & G("outage")),
                ("technical queue", G("technical")),
            ], otherwise="general queue")

        Escalates when a rule it needs is too close to call; see `route`."""
        return self.gate(name, route(rules, otherwise), on_uncertain=on_uncertain, default=default)

    # chips -------------------------------------------------------------
    def mount(self, chip: Chip, ns: str, pins: Mapping[str, str | Q | G] | None = None, params: Mapping[str, str] | None = None) -> dict[str, G]:
        """Wire a `Chip` into this circuit under namespace `ns`: its questions and gates are
        copied in as `ns.<name>`, each input pin reads the host reference it is wired to, and
        the chip's output pins come back as `G` references:

            esc = c.mount(escalation, "esc", {"angry": "angry", "dept": "dept"})
            c.gate("page_oncall", esc["out"] >= 0.5)

        `params` fill the chip's `{param}` placeholders in its question text, so each copy
        can ask about its own part of the state. Mounting the same chip twice under two
        namespaces gives two independent copies."""
        from decision_circuits.chips import mount

        return mount(self, chip, ns, pins or {}, params)

    @property
    def question_types(self) -> list[str]:
        """The question types this circuit asks, sorted: what a backend has to answer."""
        return sorted({q["type"] for q in self.questions.values()})

    # serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able data; `Circuit.from_dict` reads it back. Gates keep their
        expression trees (not the compiled helpers), so a loaded circuit can still be mounted,
        edited and rendered."""
        d: dict[str, Any] = {"format": FORMAT, "model": self.model, "questions": dict(self.questions), "gates": [_gate_to_dict(g) for g in self.gates]}
        if self.mounts:
            d["mounts"] = self.mounts
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Circuit:
        """Read `to_dict` output. A dict with a `chip` block comes back as a `Chip`."""
        if d.get("format", FORMAT) != FORMAT:
            raise ValueError(f"unknown circuit format {d.get('format')!r}; this version reads {FORMAT!r}")
        if "chip" in d and cls is Circuit:
            from decision_circuits.chips import Chip

            return Chip.from_dict(d)
        c = cls()
        _load_into(c, d)
        return c

    # rendering ---------------------------------------------------------
    def to_mermaid(
        self,
        results: dict[str, Any] | None = None,
        answers: dict[str, Any] | None = None,
        direction: str = "LR",
        plain: bool = False,
        text: bool = True,
        state: str | None = "State",
    ) -> str:
        """Schematic of the compiled circuit; see `render_mermaid`."""
        return render_mermaid(self, results, answers, direction, plain, text, state)

    def describe(self, results: Mapping[str, Any] | None = None, answers: Mapping[str, Any] | None = None) -> str:
        """The circuit in plain English, as Markdown: what the model is asked, what each gate
        decides and when it holds back, how each chip is wired. Pass a run's `gates` and
        `answers` to say what happened. See `decision_circuits.describe`."""
        from decision_circuits.describe import describe

        return describe(self, results, answers)

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

        boolean = {g.name for g in self.gates if isinstance(g.body, Expr) or (isinstance(g.body, Categorical) and g.body.op in ("at_least", "consistent"))}

        def decisions(e: Expr) -> Expr:
            """A route condition reads a boolean gate as its decision ("gate:True", 1 or 0, or its
            probability while uncertain, uncertainty included), not as the probability behind it."""
            if isinstance(e, G) and e.ref in boolean:
                return G(e.ref + ":True")
            if isinstance(e, Not):
                return Not(decisions(e.inner))
            if isinstance(e, And | Or):
                return type(e)([decisions(p) for p in e.parts])
            if isinstance(e, Threshold):
                return Threshold(decisions(e.inner), e.tau)
            return e

        def as_condition(e: Expr) -> Expr:
            e = decisions(e)
            return e if isinstance(e, Threshold) else Threshold(e, 0.5)

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
                if b.rules is not None:
                    spec["rules"] = [[action, ref_of(as_condition(c), g.name, g.band_)] for action, c in b.rules]
                    spec["otherwise"] = b.otherwise
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
        """Answer the questions with `backend` and evaluate the gates here.

        A backend is anything with `answer(state, questions, model=...)` (see
        `decision_circuits.backends`; `SystemOne` wraps any System One HTTP server, TypeSafe's
        Jev included). The gates never leave this process: they are the version of this
        package you installed, the same arithmetic whichever backend answered."""
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
            with tracing.span("decision_circuits.backend", **{"decision_circuits.backend": type(backend).__name__, "gen_ai.request.model": model}):
                answers = backend.answer(state, self.questions, model=model)
            missing = [q for q in self.questions if q not in answers]
            if missing:
                raise ValueError(f"{type(backend).__name__} returned no answer for {', '.join(map(repr, missing))}")
            out: RunOutput = {"model": model, "answers": answers, "gates": self.evaluate(answers)}
            _record(span, backend, out)
            return out

    def intervene(self, backend: Backend, state: Any, interventions: Mapping[str, Any], **kw: Any) -> Interventions:
        """Run the circuit on `state` and on each edited state; report which gates flipped
        and how every probability moved. See `decision_circuits.interventions`."""
        from decision_circuits.interventions import intervene

        return intervene(self, backend, state, interventions, **kw)

    def ablate(self, backend: Backend, state: Any, **kw: Any) -> Interventions:
        """Remove each sentence (text state) or field (dict state) in turn; see `intervene`."""
        from decision_circuits.interventions import ablate

        return ablate(self, backend, state, **kw)


# ---------------------------------------------------------------- serialization

FORMAT = "decision-circuits/1"
_CAT_DEFAULTS = {f.name: f.default for f in fields(Categorical)}


def expr_to_dict(e: Expr | Categorical) -> dict[str, Any]:
    """An expression as plain data: {"q": ref}, {"g": ref}, {"not": e}, {"and": [e, ...]},
    {"or": [...]}, {"at": tau, "of": e}, or a categorical gate {"op": ..., its fields}."""
    if isinstance(e, Q):
        return {"q": e.ref}
    if isinstance(e, G):
        return {"g": e.ref}
    if isinstance(e, Not):
        return {"not": expr_to_dict(e.inner)}
    if isinstance(e, And | Or):
        return {"and" if isinstance(e, And) else "or": [expr_to_dict(p) for p in e.parts]}
    if isinstance(e, Threshold):
        return {"at": e.tau, "of": expr_to_dict(e.inner)}
    if isinstance(e, Categorical):
        d: dict[str, Any] = {"op": e.op}
        for name, default in _CAT_DEFAULTS.items():
            v = getattr(e, name)
            if name in ("op", "rules") or v == default:
                continue
            d[name] = expr_to_dict(v) if name == "check" else v
        if e.rules is not None:
            d["rules"] = [{"action": a, "when": expr_to_dict(c)} for a, c in e.rules]
            d["otherwise"] = e.otherwise
        return d
    raise TypeError(f"cannot serialize {e!r}")


def expr_from_dict(d: Mapping[str, Any]) -> Expr | Categorical:
    if "q" in d:
        return Q(d["q"])
    if "g" in d:
        return G(d["g"])
    if "not" in d:
        return Not(_expr(d["not"]))
    if "and" in d or "or" in d:
        parts = [_expr(p) for p in d.get("and", d.get("or"))]
        return And(parts) if "and" in d else Or(parts)
    if "at" in d:
        return Threshold(_expr(d["of"]), float(d["at"]))
    if "op" in d:
        unknown = set(d) - set(_CAT_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)} in a {d['op']!r} gate")
        kw = dict(d)
        if "check" in kw:
            kw["check"] = _expr(kw["check"])
        if "rules" in kw:
            kw["rules"] = [(r["action"], _expr(r["when"])) for r in kw["rules"]]
        return Categorical(**kw)
    raise ValueError(f"not an expression: {dict(d)!r}")


def _expr(d: Mapping[str, Any]) -> Expr:
    e = expr_from_dict(d)
    if not isinstance(e, Expr):
        raise TypeError(f"a {d.get('op')!r} gate cannot sit inside an expression")
    return e


def _gate_to_dict(g: GateDef) -> dict[str, Any]:
    d: dict[str, Any] = {"name": g.name, "body": expr_to_dict(g.body), "on_uncertain": g.on_uncertain_, "band": g.band_}
    if g.on_uncertain_ == "default":
        d["default"] = g.default_
    return d


def _load_into(c: Circuit, d: Mapping[str, Any]) -> None:
    c.model = d.get("model")
    c.questions = {k: dict(v) for k, v in (d.get("questions") or {}).items()}
    c.gates = [
        GateDef(g["name"], expr_from_dict(g["body"]), g.get("on_uncertain", "abstain"), g.get("default"), float(g.get("band", 0.1)))
        for g in d.get("gates") or []
    ]
    c.mounts = {k: dict(v) for k, v in (d.get("mounts") or {}).items()}


def _record(span: Any, backend: Any, out: RunOutput) -> None:
    """A run's result on its span: the response ids, an event per answer and per gate, and
    the gates that went to a person or held back. Never the state."""
    last = getattr(backend, "last_response", None) or {}
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


def mount_owner(circuit: Circuit, name: str) -> str | None:
    """The namespace of the innermost mounted chip a question or gate came from, or None
    for the circuit's own."""
    bare = name.lstrip("_")
    found = [ns for ns in circuit.mounts if bare.startswith(ns + ".")]
    return max(found, key=len) if found else None


def question_text(q: Mapping[str, Any]) -> str:
    """A question's instructions as one line of text (structured instructions are flattened)."""
    ins = q.get("instructions")
    if isinstance(ins, str):
        return " ".join(ins.split())
    if isinstance(ins, Mapping):
        return " ".join(f"{k}: {v}" for k, v in ins.items() if isinstance(v, str))
    if isinstance(ins, list):
        return " ".join(str(x) for x in ins if isinstance(x, str))
    return ""


def answer_summary(q: Mapping[str, Any], a: Mapping[str, Any]) -> str:
    """One answer in a few words: "yes 89%", "billing 80%", "level 2.4 of 3"."""
    kind = q["type"]
    if kind == "noul":
        return f"yes {a['noul']:.0%}"
    if kind == "choice":
        return f"{a['choice']} {a['probabilities'][a['choice']]:.0%}"
    if kind == "multi":
        return ", ".join(a["selected"]) or "none apply"
    if kind == "rank":
        return " > ".join(a["order"][:3])
    if kind == "match":
        return f"{sum(m['match'] != 'none' for m in a['matches'].values())} of {len(a['matches'])} matched"
    if kind == "locate":
        return f"none {a['none']:.0%}" if a["none"] >= 0.5 or not a["located"] else f"{a['located'][0]['path']} {a['located'][0]['probability']:.0%}"
    levels = q["criteria"]
    top = max(a["probabilities"], key=a["probabilities"].__getitem__)
    name = levels[int(top)] if top.isdigit() and int(top) < len(levels) and isinstance(levels[int(top)], str) else top
    return f"{name} ({a['score']:.1f} of {len(levels) - 1})"


def _wrap(s: str, width: int, lines: int) -> list[str]:
    out: list[str] = []
    for word in s.split():
        if out and len(out[-1]) + 1 + len(word) <= width:
            out[-1] += " " + word
        else:
            out.append(word)
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][: width - 1].rstrip() + "…"
    return out


def _mermaid_safe(s: str) -> str:
    """User text inside a quoted Mermaid label: entity codes for what would end or break it."""
    for ch, code in (("#", "#35;"), ("&", "#amp;"), ('"', "#quot;"), ("<", "#lt;"), (">", "#gt;")):
        s = s.replace(ch, code)
    return s


def render_mermaid(
    circuit: Circuit,
    results: dict[str, Any] | None = None,
    answers: dict[str, Any] | None = None,
    direction: str = "LR",
    plain: bool = False,
    text: bool = True,
    state: str | None = "State",
) -> str:
    """Render the compiled circuit as a schematic. `plain=True` omits the
    init directive and inline HTML styling for stricter renderers (some
    hosted Mermaid builds reject them); layout is the same. `text=False`
    leaves out each question's wording, for a compact diagram. `state` labels
    the node the questions are asked about ("Conversation", "Ticket"); None
    leaves it out. A `route` gate is drawn as the circuit's one action, every
    possible action listed and the one taken marked.

    Schematic: an input column of
    questions, a logic column of gates, and a decisions column for the
    gates nothing else consumes. Threshold and NOT helpers are folded
    into edge labels rather than drawn as nodes. Only decisions are
    coloured: green yes / grey no / amber abstained or escalated. Each
    mounted chip is drawn as its own box, with wires into it labelled by
    the pin they land on."""
    compiled = circuit.compile()

    def spec_refs(spec: dict[str, Any]) -> list[str]:
        return (
            ([spec["input"]] if spec.get("input") else [])
            + list(spec.get("inputs", []))
            + ([spec["check"]] if spec.get("check") else [])
            + [ref for _, ref in spec.get("rules", [])]
        )

    # which gates feed other gates (their base name before ':')
    consumers: dict[str, set[str]] = {g: set() for g in compiled}
    for gid, spec in compiled.items():
        for ref in spec_refs(spec):
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
        return "q_" + qid.replace(".", "__")

    def gnode(gid: str) -> str:
        return "g_" + gid.replace("-", "_").replace(".", "__")

    # mounted chips are drawn as boxes: each node belongs to the top-level chip it came from
    def owner(name: str) -> str | None:
        return mount_owner(circuit, name)

    def shown(name: str) -> str:
        """A node's name inside its chip's box: without the namespace."""
        own = owner(name)
        return name[len(own) + 1 :] if own and not name.startswith("_") else name

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
        "  classDef chip fill:#F2F4F7,stroke:#1B1F24,stroke-width:2px,color:#1B1F24;",
        "  classDef state fill:#1B1F24,stroke:#1B1F24,color:#FFFFFF;",
    ]
    if state:
        L.append(f'  STATE(["<b>{_mermaid_safe(state)}</b>"]):::state')

    # --- inputs
    L.append('  subgraph IN["Inputs"]')
    L.append("    direction TB")

    def qnode_line(qid: str, q: dict[str, Any]) -> str:
        head = f"<b>{shown(qid)}</b>"
        if text:
            asked = "<br/>".join(_mermaid_safe(line) for line in _wrap(question_text(q), 26, 4))
            head += f"<br/>{asked}" if plain else f"<br/><span style='color:#3B4550;font-size:12px'>{asked}</span>"
        if answers and qid in answers:
            label = f"{head}<br/><b>→ {_mermaid_safe(answer_summary(q, answers[qid]))}</b>"
        else:
            label = f"{head}<br/>{q['type']}" if plain else f"{head}<br/><span style='color:#5F6B78'>{q['type']}</span>"
        return f'    {qnode(qid)}["{label}"]:::q'

    for qid, q in circuit.questions.items():
        if not owner(qid):
            L.append(qnode_line(qid, q))
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
        "route": "ROUTE",
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
        rule_of = {ref: f"{i}. {action}" for i, (action, ref) in enumerate(spec.get("rules", []), 1)}
        for ref in spec_refs(spec):
            lab = "check" if spec.get("check") == ref else ""
            base, _, opt = ref.partition(":")
            if opt == "True" and base in compiled:  # a gate read as its decision: the wire says nothing more
                opt = ""
            if base in folded:
                base, flab = folded[base]
                base, _, opt = base.partition(":")
                lab = flab + (" · " + lab if lab else "")
            if opt:
                q = circuit.questions.get(base)
                shown = f"level {opt}" if (q and q["type"] == "score") else opt
                lab = (f"{shown} " + lab).strip()
            if ref in rule_of:  # a route's rule: the wire is named for the action it leads to
                lab = rule_of[ref]
            node = gnode(base) if base in compiled else qnode(base)
            dest = owner(gid)
            if dest and owner(ref) != dest:  # a wire into a chip: name the pin it lands on
                pin = next((p for p, t in circuit.mounts[dest]["pins"].items() if t == ref or t == ref.partition(":")[0]), None)
                if pin and pin != ref.partition(":")[0]:
                    lab = f"{pin}: {lab}" if lab else pin
            out.append(f"  {node} --{'>' if not lab else f'>|{lab}|'} {gnode(gid)}")
        return out

    def route_line(gid: str, spec: dict[str, Any]) -> str:
        r = (results or {}).get(gid)
        taken = r.get("value") if r and r.get("outcome") in ("decided", "default") else None
        rows = [(f"{i}.", action) for i, (action, _) in enumerate(spec["rules"], 1)] + [("else", spec.get("otherwise"))]
        lines = []
        for mark, action in rows:
            name = _mermaid_safe(str(action))
            lines.append(f"<b>✓ {name}</b>" if r and action == taken else f"{mark} {name}")
        cls = "logic"
        if r:
            cls = "yes" if taken is not None else "hold"
            if taken is None:
                lines.append(f"<b>⚠ {str(r.get('outcome')).upper()}</b>: too close to call")
        return f'  {gnode(gid)}[["<b>{shown(gid).replace("_", " ").title()}</b><br/>{"<br/>".join(lines)}"]]:::{cls}'

    def node_line(gid: str, spec: dict[str, Any], terminal: bool) -> str:
        op = spec["op"]
        if op == "route":
            return route_line(gid, spec)
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
        if op == "threshold" and not terminal and not gid.startswith("_"):
            head += f" {spec['tau']:g}"
        if terminal:
            label = f"<b>{shown(gid).replace('_', ' ').title()}</b><br/>{head}"
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
        label = f"<b>{shown(gid).replace('_', ' ')}</b><br/>{head}" if not gid.startswith("_") else head
        cls = "logic"
        if results and gid in results:
            r = results[gid]
            if r.get("outcome", "decided") != "decided":
                label += f"<br/>⚠ {r['outcome']}"
                cls = "hold"
            elif op in ("argmax", "majority", "verify", "order"):
                label += f"<br/>→ <b>{r.get('value')}</b>" + (f" · {r['p']:.0%}" if r.get("p") is not None else "")
            elif r.get("p") is not None:
                label += f"<br/>{'✓' if r.get('value') is True else '✗'} {r['p']:.0%}"
            if r.get("value") is True:  # the path the run took lights up; what didn't hold greys out
                cls = "yes"
            elif r.get("value") is False:
                cls = "no"
        lo, hi = SHAPES.get(op, ("[", "]"))
        return f'  {gnode(gid)}{lo}"{label}"{hi}:::{cls}'

    def chip_box(ns: str) -> list[str]:
        """A mounted chip as a box, with any chips it mounted drawn inside it."""
        sid = "M_" + ns.replace(".", "__")
        out = [f'  subgraph {sid}["{ns.rpartition(".")[2]} · {circuit.mounts[ns]["chip"]}"]', "    direction TB"]
        out += [qnode_line(qid, q) for qid, q in circuit.questions.items() if owner(qid) == ns]
        out += ["  " + node_line(gid, compiled[gid], gid in decisions) for gid in logic + decisions if owner(gid) == ns]
        for inner in circuit.mounts:
            if inner.startswith(ns + ".") and "." not in inner[len(ns) + 1 :]:
                out += chip_box(inner)
        return [*out, "  end", f"  class {sid} chip"]

    for ns in circuit.mounts:
        if "." not in ns:
            L += chip_box(ns)
    if any(not owner(g) for g in logic):
        L.append('  subgraph LOGIC["Logic"]')
        L.append("    direction TB")
        for gid in logic:
            if owner(gid):
                continue
            L.append("  " + node_line(gid, compiled[gid], False))
        L.append("  end")
        L.append("  class LOGIC col")
    if any(not owner(g) for g in decisions) or not circuit.mounts:
        only_routes = all(compiled[g]["op"] == "route" for g in decisions if not owner(g))
        L.append(f'  subgraph OUT["{"Action" if only_routes and decisions else "Decisions"}"]')
        L.append("    direction TB")
        for gid in decisions:
            if not owner(gid):
                L.append("  " + node_line(gid, compiled[gid], True))
        L.append("  end")
        L.append("  class OUT col")
    edges = [f"  STATE --> {qnode(qid)}" for qid in circuit.questions] if state else []
    for gid in logic + decisions:
        edges += edge_lines(gid, compiled[gid])
    L += edges
    # the wire into each route's chosen action, drawn heavy
    for gid, spec in compiled.items():
        r = (results or {}).get(gid)
        if spec["op"] != "route" or not r or r.get("outcome") != "decided":
            continue
        taken = next((f"|{i}. {a}| {gnode(gid)}" for i, (a, _) in enumerate(spec["rules"], 1) if a == r.get("value")), None)
        L += [f"  linkStyle {i} stroke:#2E7D4F,stroke-width:3.5px" for i, e in enumerate(edges) if taken and e.endswith(taken)]
    return "\n".join(L)


to_mermaid = render_mermaid  # module-level alias
