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
import textwrap
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, TypedDict

from decision_circuits import tracing
from decision_circuits.gates import Gate, GateResultDict, evaluate_gates, pick
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


@dataclass(frozen=True)
class P:
    """A chip setting, filled in when the chip is mounted: `Q("risk") >= P("strictness")`,
    `at_least(P("hold_at"), ...)`, a route action, a choice's options. Declared with a default in
    `Chip(params={...})`; `{name}` in question text is the same setting as words."""

    name: str

    def __format__(self, spec: str) -> str:  # a datasheet prints the setting's name
        return "{" + self.name + "}"


def _num(x: Any) -> Any:
    return x if isinstance(x, P) else float(x)


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
        return Threshold(self, _num(tau))

    def at(self, tau: float) -> Threshold:
        """Named form of `>=`: `Q("pii").at(0.7)`."""
        return Threshold(self, _num(tau))


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
    return Categorical("order", input=score_id, cutpoints=cutpoints if isinstance(cutpoints, P) else list(cutpoints))


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
    def mount(self, chip: Chip, ns: str, pins: Mapping[str, str | Q | G] | None = None, params: Mapping[str, Any] | None = None) -> dict[str, G]:
        """Wire a `Chip` into this circuit under namespace `ns`: its questions and gates are
        copied in as `ns.<name>`, each input pin reads the host reference it is wired to, and
        the chip's output pins come back as `G` references:

            esc = c.mount(escalation, "esc", {"angry": "angry", "dept": "dept"})
            c.gate("page_oncall", esc["out"] >= 0.5)

        `params` are the chip's settings for this copy: `{name}` in its question text (so each
        copy can ask about its own part of the state) and `P("name")` in its logic. Mounting the
        same chip twice under two namespaces gives two independent copies."""
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
        d: dict[str, Any] = {
            "format": FORMAT,
            "model": self.model,
            "questions": encode_params(dict(self.questions)),
            "gates": [_gate_to_dict(g) for g in self.gates],
        }
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

        boolean = {g.name for g in self.gates if gate_output(g.body) == "noul"}

        def as_condition(e: Expr) -> Expr:
            """A route condition reads a boolean gate as its decision ("gate:True", 1 or 0, or its
            probability while uncertain, uncertainty included), not as the probability behind it."""
            e = map_expr(e, leaf=lambda n: G(n.ref + ":True") if isinstance(n, G) and n.ref in boolean else n)
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
        with tracing.span("decision_circuits.evaluate", **{"decision_circuits.gates": [g.name for g in self.gates]}) as span:
            gates = self._evaluate(answers)
            _record_gates(span, self, gates)
            return gates

    def _evaluate(self, answers: Answers) -> dict[str, GateResultDict]:
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
            out: RunOutput = {"model": model, "answers": answers, "gates": self._evaluate(answers)}
            _record(span, backend, out)
            _record_gates(span, self, out["gates"])
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


# ---------------------------------------------------------------- walking expressions


def map_expr(e: Any, leaf: Callable[[Q | G], Q | G] | None = None, value: Callable[[Any], Any] | None = None) -> Any:
    """The same expression rebuilt: `leaf` on every reference (a categorical gate's `input` and
    `inputs` arrive as `Q`), `value` on every other field that can hold a setting (a threshold's
    tau, a categorical's numbers and actions). The one walk renaming, settings and compiling share."""
    leaf = leaf or (lambda n: n)
    value = value or (lambda v: v)
    if isinstance(e, Q | G):
        return leaf(e)
    if isinstance(e, Not):
        return Not(map_expr(e.inner, leaf, value))
    if isinstance(e, And | Or):
        return type(e)([map_expr(p, leaf, value) for p in e.parts])
    if isinstance(e, Threshold):
        return Threshold(map_expr(e.inner, leaf, value), value(e.tau))
    if isinstance(e, Categorical):
        kw: dict[str, Any] = {}
        for f in fields(Categorical):
            v = getattr(e, f.name)
            if f.name in ("op", "relation") or v is None:
                kw[f.name] = v
            elif f.name == "input":
                kw[f.name] = leaf(Q(v)).ref
            elif f.name == "inputs":
                kw[f.name] = [leaf(Q(r)).ref for r in v]
            elif f.name == "check":
                kw[f.name] = map_expr(v, leaf, value)
            elif f.name == "rules":
                kw[f.name] = [(value(a), map_expr(c, leaf, value)) for a, c in v]
            else:
                kw[f.name] = value(v)
        return Categorical(**kw)
    raise TypeError(f"unsupported expression {e!r}")


def rename_refs(e: Any, f: Callable[[str], str]) -> Any:
    """The same expression with every reference passed through `f`."""
    return map_expr(e, leaf=lambda n: type(n)(f(n.ref)))


def expr_refs(e: Any) -> list[str]:
    """Every reference an expression reads, in order."""
    out: list[str] = []
    map_expr(e, leaf=lambda n: out.append(n.ref) or n)
    return out


def gate_output(body: Expr | Categorical) -> str:
    """What a gate gives: "noul" (a yes/no with a probability), "choice" (one of several options),
    or "order" / "count" (a bucket or a number)."""
    if isinstance(body, Expr) or body.op in ("at_least", "consistent"):
        return "noul"
    if body.op in ("argmax", "majority", "verify", "route"):
        return "choice"
    return body.op


# ---------------------------------------------------------------- serialization

FORMAT = "decision-circuits/1"
_CAT_DEFAULTS = {f.name: f.default for f in fields(Categorical)}


def encode_params(x: Any) -> Any:
    """Plain data with every `P` as {"parameter": name}."""
    if isinstance(x, P):
        return {"parameter": x.name}
    if isinstance(x, Mapping):
        return {k: encode_params(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [encode_params(v) for v in x]
    return x


def decode_params(x: Any) -> Any:
    if isinstance(x, Mapping):
        if set(x) == {"parameter"}:
            return P(x["parameter"])
        return {k: decode_params(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decode_params(v) for v in x]
    return x


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
        return {"at": encode_params(e.tau), "of": expr_to_dict(e.inner)}
    if isinstance(e, Categorical):
        d: dict[str, Any] = {"op": e.op}
        for name, default in _CAT_DEFAULTS.items():
            v = getattr(e, name)
            if name in ("op", "rules") or v == default:
                continue
            d[name] = expr_to_dict(v) if name == "check" else encode_params(v)
        if e.rules is not None:
            d["rules"] = [{"action": encode_params(a), "when": expr_to_dict(c)} for a, c in e.rules]
            d["otherwise"] = encode_params(e.otherwise)
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
        return Threshold(_expr(d["of"]), _num(decode_params(d["at"])))
    if "op" in d:
        unknown = set(d) - set(_CAT_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)} in a {d['op']!r} gate")
        kw = {k: (v if k in ("check", "rules") else decode_params(v)) for k, v in d.items()}
        if "check" in kw:
            kw["check"] = _expr(kw["check"])
        if "rules" in kw:
            kw["rules"] = [(decode_params(r["action"]), _expr(r["when"])) for r in kw["rules"]]
        return Categorical(**kw)
    raise ValueError(f"not an expression: {dict(d)!r}")


def _expr(d: Mapping[str, Any]) -> Expr:
    e = expr_from_dict(d)
    if not isinstance(e, Expr):
        raise TypeError(f"a {d.get('op')!r} gate cannot sit inside an expression")
    return e


def _gate_to_dict(g: GateDef) -> dict[str, Any]:
    d: dict[str, Any] = {"name": g.name, "body": expr_to_dict(g.body), "on_uncertain": g.on_uncertain_, "band": encode_params(g.band_)}
    if g.on_uncertain_ == "default":
        d["default"] = encode_params(g.default_)
    return d


def _load_into(c: Circuit, d: Mapping[str, Any]) -> None:
    c.model = d.get("model")
    c.questions = {k: decode_params(dict(v)) for k, v in (d.get("questions") or {}).items()}
    c.gates = [
        GateDef(
            g["name"], expr_from_dict(g["body"]), g.get("on_uncertain", "abstain"), decode_params(g.get("default")), _num(decode_params(g.get("band", 0.1)))
        )
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


def _record_gates(span: Any, circuit: Circuit, gates: Mapping[str, GateResultDict]) -> None:
    """Each gate's result: an event on the span (value, outcome, p, trace, a route's rule
    statuses, the chip it came from) and one count on `decision_circuits.gate.results`."""
    held: dict[str, list[str]] = {"escalate": [], "abstain": []}
    for gid, g in gates.items():
        ns = mount_owner(circuit, gid)
        chip = circuit.mounts[ns] if ns else {}
        where = {"chip": chip.get("chip"), "chip_version": chip.get("version")}
        tracing.event(
            span,
            "decision_circuits.gate",
            gate=gid,
            value=g.get("value"),
            outcome=g.get("outcome"),
            p=g.get("p"),
            uncertain=g.get("uncertain"),
            trace="; ".join(g.get("trace") or []),
            rules=g.get("rules"),
            **where,
        )
        decided = g.get("outcome") in ("decided", "default")
        tracing.count(
            "decision_circuits.gate.results",
            "Gate results by gate, outcome and value",
            **{
                "decision_circuits.gate": gid,
                "decision_circuits.outcome": g.get("outcome"),
                "decision_circuits.value": str(g.get("value")) if decided else None,
                "decision_circuits.chip": where["chip"],
                "decision_circuits.chip.version": where["chip_version"],
            },
        )
        if g.get("outcome") in held:
            held[g["outcome"]].append(gid)
    span.set_attribute("decision_circuits.escalated", held["escalate"])
    span.set_attribute("decision_circuits.abstained", held["abstain"])


# ---------------------------------------------------------------- rendering


def _short(s: str, n: int = 38) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def local_name(circuit: Circuit, name: str) -> str:
    """A question or gate's name inside its chip: without the namespace (helpers keep theirs)."""
    own = mount_owner(circuit, name)
    return name[len(own) + 1 :] if own and not name.startswith("_") else name


def child_mounts(circuit: Circuit, ns: str | None) -> list[str]:
    """The chips mounted directly in `ns` (None: in the circuit itself)."""
    return [m for m in circuit.mounts if (m.rpartition(".")[0] or None) == ns]


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


def answer_outcomes(q: Mapping[str, Any], a: Mapping[str, Any]) -> tuple[str, list[str]]:
    """What an answer landed on, and the outcomes it didn't: ("no 89%", ["yes 11%"]),
    ("refund 80%", ["replacement 15%", "info 5%"]). A yes/no question shows both sides, a
    pick-one every option, so the outcomes that weren't taken are on the page too."""
    kind = q["type"]
    if kind == "noul":
        p = float(a["noul"])
        yes, no = f"yes {p:.0%}", f"no {1 - p:.0%}"
        return (yes, [no]) if p >= 0.5 else (no, [yes])
    if kind == "choice":
        probs = {k: float(v) for k, v in a["probabilities"].items()}
        top = pick(dict(a))  # the pick the gates act on
        rest = sorted((k for k in probs if k != top), key=lambda k: -probs[k])
        return f"{top} {probs[top]:.0%}", [f"{k} {probs[k]:.0%}" for k in rest]
    return _summary(q, a), []


def answer_summary(q: Mapping[str, Any], a: Mapping[str, Any]) -> str:
    """One answer in a few words: "no 89%", "billing 80%", "level 2.4 of 3"."""
    return answer_outcomes(q, a)[0]


def _summary(q: Mapping[str, Any], a: Mapping[str, Any]) -> str:
    kind = q["type"]
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
    return textwrap.wrap(s, width, max_lines=lines, placeholder="…", break_long_words=False)


def verdict(r: Mapping[str, Any]) -> tuple[str, str, str]:
    """A gate result in words: ("yes" / "no" / its value / its outcome, " (91%)" or "", and the
    diagram class "yes" / "no" / "hold"). The one wording the diagram and `describe` share."""
    p = r.get("p")
    pct = f" ({p:.0%})" if isinstance(p, int | float) else ""
    if r.get("outcome", "decided") not in ("decided", "default"):
        return str(r.get("outcome")), pct, "hold"
    v = r.get("value")
    return ("yes", pct, "yes") if v is True else ("no", pct, "no") if v is False else (str(v), pct, "yes")


SKIP = "skip"  # a wire on a part of a route that never ran
COLUMN_STYLE = "fill:none,stroke:#D9DEE3,stroke-dasharray:3 3,color:#5F6B78"


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
    """Render the circuit as a schematic, left to right in columns: the state (`state` names it,
    None leaves it out), Asked (each question, with its wording unless `text=False`), Checks (the
    gates), Decisions (gates nothing else reads), and for a route nothing reads, Decide (its
    yes/no ladder) and Outcome (every action). Each mounted chip is its own box, wires into it
    labelled by pin. Threshold and NOT helpers fold into wire labels.

    Given a run's `results` and `answers`: green is what held and the way through, grey what ran
    and said no, amber too close to call, dashed what never ran; the path to the action is one
    heavy line. `plain=True` leaves out the init directive and inline styling, for stricter
    renderers."""
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
    owners: dict[str, str | None] = {}

    def owner(name: str) -> str | None:  # asked for every node, often more than once: worked out once each
        if name not in owners:
            owners[name] = mount_owner(circuit, name)
        return owners[name]

    def shown(name: str) -> str:
        return local_name(circuit, name)

    init = (
        "%%{init: {'theme': 'base', 'flowchart': {'nodeSpacing': 40, 'rankSpacing': 120, 'curve': 'basis', 'useMaxWidth': false, 'htmlLabels': true}, "
        "'themeVariables': {'fontFamily': 'IBM Plex Sans, system-ui, sans-serif', 'fontSize': '13px', 'lineColor': '#5F6B78'}}}%%"
    )
    L = [
        *([] if plain else [init]),
        f"flowchart {direction}",
        "  classDef q fill:#FFFFFF,stroke:#1264A3,stroke-width:1.5px,color:#1B1F24;",
        "  classDef qyes fill:#EEF7F1,stroke:#1264A3,stroke-width:1.5px,color:#1B1F24;",
        "  classDef qno fill:#EEF0F2,stroke:#98A2AD,stroke-width:1.5px,color:#3B4550;",
        "  classDef logic fill:#F7F8F6,stroke:#5F6B78,stroke-width:1.5px,color:#1B1F24;",
        "  classDef yes fill:#DDF3E4,stroke:#2E7D4F,stroke-width:2.5px,color:#0F3D22;",
        "  classDef no fill:#EEF0F2,stroke:#98A2AD,stroke-width:2px,color:#3B4550;",
        "  classDef hold fill:#FFF1CC,stroke:#9A6B00,stroke-width:2.5px,color:#4A3300;",
        "  classDef state fill:#1B1F24,stroke:#1B1F24,color:#FFFFFF;",
        "  classDef act fill:#FFFFFF,stroke:#1B1F24,stroke-width:1.5px,color:#1B1F24;",
        "  classDef faded fill:#F4F5F7,stroke:#C9CED4,stroke-width:1px,color:#9AA3AD;",
        "  classDef passed fill:#FFFFFF,stroke:#2E7D4F,stroke-width:2.5px,color:#1B1F24;",
    ]
    if state:
        L.append(f'  STATE(["<b>{_mermaid_safe(state)}</b>"]):::state')

    def column(sid: str, title: str, lines: list[str]) -> list[str]:
        """One of the diagram's columns (Asked, Checks, Decisions, Decide, Outcome)."""
        return [f'  subgraph {sid}["{title}"]', "    direction TB", *lines, "  end", f"  style {sid} {COLUMN_STYLE}"]

    def qnode_line(qid: str, q: dict[str, Any]) -> str:
        head = f"<b>{shown(qid)}</b>"
        if text:
            asked = "<br/>".join(_mermaid_safe(line) for line in _wrap(question_text(q), 26, 4))
            head += f"<br/>{asked}" if plain else f"<br/><span style='color:#3B4550;font-size:12px'>{asked}</span>"
        if answers and qid in answers:
            main, others = answer_outcomes(q, answers[qid])
            label = f"{head}<br/><b>→ {_mermaid_safe(main)}</b>"
            if others:  # the outcomes it didn't land on
                rest = "<br/>".join(_mermaid_safe(x) for x in _wrap(" · ".join(others), 30, 2))
                label += f"<br/>{rest}" if plain else f"<br/><span style='color:#5F6B78'>{rest}</span>"
        else:
            label = f"{head}<br/>{q['type']}" if plain else f"{head}<br/><span style='color:#5F6B78'>{q['type']}</span>"
        cls = "q"
        if answers and qid in answers and q["type"] == "noul":  # a yes lights up, a no greys out
            cls = "qyes" if answers[qid]["noul"] >= 0.5 else "qno"
        return f'    {qnode(qid)}["{label}"]:::{cls}'

    L += column("IN", "Asked", [qnode_line(qid, q) for qid, q in circuit.questions.items() if not owner(qid)])

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
    ladder_of = {gid for gid in decisions if compiled[gid]["op"] == "route"}  # a route nothing reads is drawn as its ladder

    # A ladder rule's AND / OR / NOT is drawn inside its diamond, not as loose junctions: each
    # rule is wired straight from the gates and answers it reads.
    absorbed: set[str] = set()

    def rule_leaves(ref: str, negated: bool = False) -> list[tuple[str, bool]]:
        base = ref.partition(":")[0]
        if not (base.startswith("_") and base in compiled):
            return [(ref, negated)]
        absorbed.add(base)
        spec = compiled[base]
        if spec["op"] == "not":
            return rule_leaves(spec["input"], not negated)
        return [leaf for r in spec_refs(spec) for leaf in rule_leaves(r, negated)]

    rule_inputs = {gid: [rule_leaves(ref) for _, ref in compiled[gid]["rules"]] for gid in ladder_of}

    # A rule that reads one gate's decision branches from that gate: the gate already answered,
    # so a diamond asking it again would only repeat it. Diamonds are kept for rules that combine.
    branch_at: dict[tuple[str, int], str] = {}
    for gid in ladder_of:
        for i, leaves in enumerate(rule_inputs[gid], 1):
            if len(leaves) == 1:
                ((leaf, negated),) = leaves
                base, _, opt = leaf.partition(":")
                if not negated and opt in ("", "True") and base in compiled and not base.startswith("_") and compiled[base]["op"] != "route":
                    branch_at[(gid, i)] = base

    # A host gate that exists only to feed one rule is drawn in the ladder at that rule's place,
    # so the path doesn't dive back into the Logic column and out again.
    uses = Counter([*branch_at.values(), *(leaf.partition(":")[0] for gid in ladder_of for leaves in rule_inputs[gid] for leaf, _ in leaves)])
    in_ladder = {
        base
        for base in set(branch_at.values())
        if not owner(base) and uses[base] == 2 and all(c in absorbed for c in consumers[base])  # its rule's two mentions, nothing else
    }
    logic = [gid for gid in logic if gid not in in_ladder and gid not in absorbed]
    # Branch straight from a gate only where that keeps the path in one place: a gate drawn in
    # the ladder, or a chip's output. A gate other parts of the circuit also read stays in the
    # Checks column, and its rule gets a diamond in the ladder instead of a detour back to it.
    branch_at = {k: base for k, base in branch_at.items() if base in in_ladder or owner(base)}

    def rule_node(gid: str, i: int) -> str:
        return gnode(branch_at[(gid, i)]) if (gid, i) in branch_at else f"{gnode(gid)}__r{i}"

    gate_defs = {g.name: g for g in circuit.gates}

    def cond_text(e: Any) -> str:
        """A rule's condition in a few words: "wants refund AND has reason"."""
        if isinstance(e, Q | G):
            base, _, opt = e.ref.partition(":")
            name = shown(base).replace("_", " ")
            return name if opt in ("", "True") else f"NOT {name}" if opt == "False" else f"{name}: {opt}"
        if isinstance(e, Not):
            inner = cond_text(e.inner)
            return f"NOT ({inner})" if isinstance(e.inner, And | Or) else f"NOT {inner}"
        if isinstance(e, And | Or):
            word = " AND " if isinstance(e, And) else " OR "
            return word.join(f"({cond_text(p)})" if isinstance(p, And | Or) else cond_text(p) for p in e.parts)
        if isinstance(e, Threshold):
            return f"{cond_text(e.inner)} ≥ {e.tau:g}"
        return "?"

    def carries(base: str, opt: str) -> bool | None:
        """Whether a wire carried a yes on this run: a noul answered yes, the picked option, a gate
        that decided yes or the value read. None when it can't be said (no run, a helper, unsure)."""
        if answers and base in answers:
            a = answers[base]
            kind = a.get("type")
            if kind == "noul" and not opt:
                return float(a["noul"]) >= 0.5
            if kind == "choice" and opt:
                return pick(dict(a)) == opt  # the same pick the gates act on
            if kind == "score" and opt:
                probs = a.get("probabilities") or {}
                return bool(probs) and max(probs, key=probs.__getitem__) == opt
            if kind == "multi" and opt:
                return opt in a.get("selected", [])
            return None
        r = (results or {}).get(base)
        if not r or r.get("outcome") != "decided":
            return None
        v = r.get("value")
        if not opt:
            return v if isinstance(v, bool) else None
        return v is True if opt == "True" else v is False if opt == "False" else str(v) == opt

    def node_of(base: str) -> str:
        return gnode(base) if base in compiled else qnode(base)

    def edge_lines(gid: str, spec: dict[str, Any]) -> list[tuple[str, bool | str | None]]:
        """(Mermaid line, whether the wire carried a yes on the run) per input of a gate."""
        out = []
        if gid in ladder_of:  # each rule's diamond, wired straight from what its condition reads
            status = rule_status(gid, len(spec["rules"]))
            for i, leaves in enumerate(rule_inputs[gid], 1):
                if (gid, i) in branch_at:  # the gate is the decision point itself: no wire into it
                    continue
                for leaf, negated in leaves:
                    base, _, opt = leaf.partition(":")
                    shown_opt = "" if opt in ("", "True") else opt
                    lab = " ".join(x for x in ("NOT" if negated else "", shown_opt) if x)
                    # a checked rule's wires decided its answer: solid, unlabelled (the box says yes or no)
                    live = SKIP if status[i - 1] == "skipped" else None
                    out.append((f"  {node_of(base)} --{'>' if not lab else f'>|{lab}|'} {gnode(gid)}__r{i}", live))
            return out
        # a route something else reads is one box; its wires are named for the action they lead to
        rule_of = {ref: f"{i}. {action}" for i, (action, ref) in enumerate(spec.get("rules", []), 1)}
        rule_no = {ref: i for i, (_, ref) in enumerate(spec.get("rules", []), 1)}
        route_r = (results or {}).get(gid) if spec["op"] == "route" else None
        for ref in spec_refs(spec):
            lab = "check" if spec.get("check") == ref else ""
            base, _, opt = ref.partition(":")
            live = carries(base, opt)
            if ref in rule_of and route_r and route_r.get("outcome") == "decided":
                live = (route_r.get("rules") or [])[rule_no[ref] - 1] == "held"  # only the rule taken carried
            if opt == "True" and base in compiled:  # a gate read as its decision: the wire says nothing more
                opt = ""
            if base in folded:
                base, flab = folded[base]
                base, _, opt = base.partition(":")
                lab = flab + (" · " + lab if lab else "")
                if ref not in rule_of:
                    src = carries(base, opt)  # a folded NOT carries the opposite; a folded threshold can't be read here
                    live = (not src if src is not None else None) if flab == "NOT" else None
            if opt:
                q = circuit.questions.get(base)
                opt_label = f"level {opt}" if (q and q["type"] == "score") else opt
                lab = (f"{opt_label} " + lab).strip()
            if ref in rule_of:
                lab = rule_of[ref]
            dest = owner(gid)
            if dest and owner(ref) != dest:  # a wire into a chip: name the pin it lands on
                pin = next((p for p, t in circuit.mounts[dest]["pins"].items() if t == ref or t == ref.partition(":")[0]), None)
                if pin and pin != ref.partition(":")[0]:
                    lab = f"{pin}: {lab}" if lab else pin
            out.append((f"  {node_of(base)} --{'>' if not lab else f'>|{lab}|'} {gnode(gid)}", live))
        return out

    def rule_status(gid: str, n: int) -> list[str | None]:
        """Per rule of a route on this run: "held", "no", "unsure", or "skipped" (never reached)."""
        r = (results or {}).get(gid)
        return list(r["rules"]) if r and r.get("rules") else [None] * n

    drawn: dict[str, tuple[list[str], list[str], list[tuple[str, bool | str | None]]]] = {}

    def ladder(gid: str, spec: dict[str, Any]) -> tuple[list[str], list[str], list[tuple[str, bool | str | None]]]:
        """A route as the decision ladder it is: a diamond per rule, yes to its action, no to the
        next rule, the last no to `otherwise`; an escalation leaves from the rule it stopped at.
        (steps, ends, edges): the Decide column's nodes, the Outcome column's, and the wires."""
        if gid not in drawn:
            drawn[gid] = _ladder(gid, spec)
        return drawn[gid]

    def _ladder(gid: str, spec: dict[str, Any]) -> tuple[list[str], list[str], list[tuple[str, bool | str | None]]]:
        g = gnode(gid)
        rules = spec["rules"]
        st = rule_status(gid, len(rules))
        r = (results or {}).get(gid)
        taken = r.get("value") if r and r.get("outcome") in ("decided", "default") else None
        ran = r is not None
        steps: list[str] = []
        ends: list[str] = []
        edges: list[tuple[str, bool | str | None]] = []
        n = len(rules)

        def arrow(i: int) -> str:
            """A link long enough that every outcome lands in the same column (rule i sits i-1
            ranks right of rule 1; its outcome needs n-i+1 more)."""
            return "-" * (n - i + 2) + ">"

        # every rule the route checked is on the path: a no passes through (outlined), a yes stops
        # there (filled); only rules it never reached fade
        cls_rule = {"held": "yes", "no": "passed", "unsure": "hold", "skipped": "faded", None: "logic"}
        for i, (action, _) in enumerate(rules, 1):
            name = _mermaid_safe(str(action))
            here = rule_node(gid, i)
            if (gid, i) not in branch_at:  # a combining rule gets a diamond asking its condition
                asks = cond_text(gate_defs[gid].body.rules[i - 1][1]) if gid in gate_defs else str(action)
                question = "<br/>".join(_mermaid_safe(x) for x in _wrap(f"{asks}?", 24, 4))
                steps.append(f'  {here}{{"<b>{i}.</b> {question}"}}:::{cls_rule[st[i - 1]]}')
            took = ran and st[i - 1] == "held"
            ends.append(f'  {g}__a{i}(["{"✓ " if took else ""}{name}"]):::{"yes" if took else "faded" if ran else "act"}')
            yes_label = "|yes|" if took or not ran else ""  # a branch not taken is faded and needs no label
            edges.append((f"  {here} {arrow(i)}{yes_label} {g}__a{i}", (True if took else SKIP) if ran else None))
            nxt = rule_node(gid, i + 1) if i < len(rules) else f"{g}__else"
            edges.append((f"  {here} -->|no| {nxt}", (True if st[i - 1] == "no" else SKIP) if ran else None))
            if st[i - 1] == "unsure":
                edges.append((f"  {here} {arrow(i)}|too close to call| {g}__esc", True))
        fell = ran and all(x == "no" for x in st)
        other = _mermaid_safe(str(spec.get("otherwise")))
        ends.append(f'  {g}__else(["{"✓ " if fell else ""}{other}"]):::{"yes" if fell else "faded" if ran else "act"}')
        if ran and taken is None:
            ends.append(f'  {g}__esc(["⚠ {"a person" if r.get("outcome") == "escalate" else str(r.get("outcome"))}"]):::hold')
        return steps, ends, edges

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
            head += f" ≥ {spec['tau']:.0%}"
        if op == "threshold" and not terminal and not gid.startswith("_"):
            head += f" {spec['tau']:.0%}"
        if terminal:
            label = f"<b>{shown(gid).replace('_', ' ').title()}</b><br/>{head}"
            if op in ("and", "or", "threshold", "not"):
                label += f"{'' if op == 'threshold' else ' ≥'} {spec.get('tau', 0.5):.0%}"  # a threshold's head is already "≥"
            cls = "logic"
            if results and gid in results:
                word, pct, cls = verdict(results[gid])
                mark = f"⚠ {word.upper()}" if cls == "hold" else ("✗ " if cls == "no" else "✓ ") + word
                label += f"<br/><b>{mark}</b>{pct}"
            return f'  {gnode(gid)}["{label}"]:::{cls}'
        nums = sorted(i for (_, i), gate in branch_at.items() if gate == gid)  # a gate a route branches from shows its rule number
        num = f"{'/'.join(map(str, nums))}. " if nums else ""
        label = f"<b>{num}{shown(gid).replace('_', ' ')}</b><br/>{head}" if not gid.startswith("_") else head
        cls = "logic"
        if results and gid in results:
            r = results[gid]
            if r.get("outcome", "decided") != "decided":
                label += f"<br/>⚠ {r['outcome']}"
                cls = "hold"
            elif op in ("argmax", "majority", "verify", "order"):
                label += f"<br/>→ <b>{r.get('value')}</b>" + (f" · {r['p']:.0%}" if r.get("p") is not None else "")
            elif r.get("p") is not None:
                label += f"<br/><b>{'✓ yes' if r.get('value') is True else '✗ no'}</b> ({r['p']:.0%})"
            if r.get("value") is True:  # the path the run took lights up; what didn't hold greys out
                cls = "yes"
            elif r.get("value") is False:
                cls = "no"
        on_path = [rule_status(rg, len(compiled[rg]["rules"]))[i - 1] for (rg, i), gate in branch_at.items() if gate == gid]
        if "no" in on_path:  # a route passed through this gate on its no: it is on the path
            cls = "passed"
        elif gid in in_ladder and on_path and all(x == "skipped" for x in on_path):
            cls = "faded"  # a ladder step the route never reached, like the diamonds around it
        lo, hi = SHAPES.get(op, ("[", "]"))
        return f'  {gnode(gid)}{lo}"{label}"{hi}:::{cls}'

    def verdict_word(r: Mapping[str, Any]) -> str:
        word, pct, cls = verdict(r)
        return ("⚠ " if cls == "hold" else "") + word + pct

    def chip_box(ns: str) -> list[str]:
        """A mounted chip as a box, with any chips it mounted drawn inside it."""
        sid = "M_" + ns.replace(".", "__")
        title = f"{ns.rpartition('.')[2]} · {circuit.mounts[ns]['chip']}"
        verdicts = [f"{o}: {verdict_word((results or {})[f'{ns}.{o}'])}" for o in circuit.mounts[ns].get("outputs", []) if f"{ns}.{o}" in (results or {})]
        if verdicts:  # what the chip decided, on its lid
            title += " → " + ", ".join(verdicts)
        out = [f'  subgraph {sid}["{_mermaid_safe(title)}"]', "    direction TB"]
        out += [qnode_line(qid, q) for qid, q in circuit.questions.items() if owner(qid) == ns]
        for gid in logic + decisions:
            if owner(gid) != ns:
                continue
            if gid in ladder_of:
                steps, ends, _ = ladder(gid, compiled[gid])
                out += ["  " + n for n in [*steps, *ends]]
            else:
                out.append("  " + node_line(gid, compiled[gid], gid in decisions))
        for inner in child_mounts(circuit, ns):
            out += chip_box(inner)
        return [*out, "  end", f"  style {sid} fill:#F2F4F7,stroke:#1B1F24,stroke-width:2px,color:#1B1F24"]  # style, not class: survives being an edge target

    for ns in circuit.mounts:
        if "." not in ns:
            L += chip_box(ns)
    host_logic = [gid for gid in logic if not owner(gid)]
    if host_logic:
        L += column("LOGIC", "Checks", ["  " + node_line(gid, compiled[gid], False) for gid in host_logic])
    host_decisions = [gid for gid in decisions if not owner(gid) and gid not in ladder_of]  # a host route is drawn below
    if host_decisions or not (circuit.mounts or ladder_of):
        L += column("OUT", "Decisions", ["  " + node_line(gid, compiled[gid], True) for gid in host_decisions])
    host_ladders = [gid for gid in decisions if gid in ladder_of and not owner(gid)]
    if host_ladders:
        steps, ends = [], []
        for gid in host_ladders:
            s_, e_, _ = ladder(gid, compiled[gid])
            steps += s_ + [node_line(base, compiled[base], False) for (rg, _), base in sorted(branch_at.items()) if rg == gid and base in in_ladder]
            ends += e_
        L += column("DECIDE", "Decide", ["  " + x for x in steps])
        L += column("OUTCOME", "Outcome", ["  " + x for x in ends])
    wired: list[tuple[str, bool | str | None]] = []
    if state:  # one arrow into each box of questions, not one per question
        boxes = ["IN"] if any(not owner(q) for q in circuit.questions) else []
        boxes += [
            "M_" + ns.replace(".", "__") for ns in circuit.mounts if "." not in ns and any(owner(q) and owner(q).split(".")[0] == ns for q in circuit.questions)
        ]
        wired += [(f"  STATE --> {b}", None) for b in boxes]
        # Invisible links keep every question one step from the state, so the Asked column lines
        # up; without them Mermaid ranks questions by what they feed and wires cut through boxes.
        wired += [(f"  STATE ~~~ {qnode(q)}", None) for q in circuit.questions]
    for gid in logic + decisions + sorted(in_ladder):  # a gate drawn in the ladder keeps its own inputs
        wired += edge_lines(gid, compiled[gid])
    path_start = len(wired)
    for gid in ladder_of:
        wired += ladder(gid, compiled[gid])[2]
    edges = [line for line, _ in wired]
    L += edges
    # Everything upstream runs on every request, so a wire that carried a no ran: solid, light.
    # Only what never ran (a rule the route didn't reach, a branch not taken) is dashed.
    said_no = [str(i) for i, (_, live) in enumerate(wired) if live is False]
    if said_no:
        L.append(f"  linkStyle {','.join(said_no)} stroke:#AEB6BF,stroke-width:1px")
    never = [str(i) for i, (_, live) in enumerate(wired) if live == SKIP]
    if never:
        L.append(f"  linkStyle {','.join(never)} stroke:#D5DAE0,stroke-width:1px,stroke-dasharray:4 3")
    path = [i for i in range(path_start, len(wired)) if wired[i][1] is True]
    unsure = [str(i) for i in path if "|too close to call|" in wired[i][0]]
    decided = [str(i) for i in path if "|too close to call|" not in wired[i][0]]
    if decided:  # the way through each route's ladder to its action, drawn heavy
        L.append(f"  linkStyle {','.join(decided)} stroke:#2E7D4F,stroke-width:3.5px")
    if unsure:  # the hand-off to a person, heavy amber: it went somewhere, but nothing was decided
        L.append(f"  linkStyle {','.join(unsure)} stroke:#9A6B00,stroke-width:3.5px")
    return "\n".join(L)


to_mermaid = render_mermaid  # module-level alias
