"""Decision circuits: deterministic gates over calibrated answers.

A circuit is the `questions` map of a normal request plus a `gates`
map. Each gate reads one or more answers (or earlier gates) and emits a
typed value with a probability and a trace. The model never sees the
gates; code evaluates them, so a circuit is versionable and testable
offline, and each item's path is auditable.

Gate ops (inputs may reference question ids or earlier gate ids):

  threshold  input: noul id, "choice_id:option", "score_id:level",
             or "gate_id:value" for a categorical gate   value: bool
             p = P(input); passes if p >= tau
  not        input: noul or gate                       value: bool, p = 1 - p_in
  and / or   inputs: k nouls/gates, or input: a multi id (all its options)   value: bool
             p = product / 1 - product(1-p) (independence assumption; reported as such)
  at_least   inputs or input (as and/or), k               value: bool
             p = P(at least k of the inputs hold), from the count distribution under
             independence; k=1 is `or`, k=len is `and`, "at most k" is `not` over k+1
  count      inputs or input (as and/or)                 value: most likely count
             p = P(that count); confidence = the same; expected count in the trace;
             `probabilities` holds the whole distribution, so "count_id:k" reads P(exactly k)
  majority   inputs: k choice ids over the same option set   value: option
             votes by argmax; p = mean probability of the winner; margin reported
  argmax     input: choice id                           value: option or abstain
             abstains when confidence < min_confidence
  verify     input: choice id, check: noul id           value: option or escalate
             the check asks "is <that answer> supported?"; escalates when
             P(check) < tau or the choice is below min_confidence
  order      input: score id, cutpoints: [c1, c2, ...]  value: bucket index (0..len)
             bucket = number of cutpoints <= expected score
  consistent inputs: [a, b], relation                   value: bool
             checks two answers against each other: "same" (one question asked two
             ways: P(a) = P(b)), "complement" (a question and its negation: P(a) +
             P(b) = 1), "implies" (a implies b: P(a) <= P(b)). p = 1 - the violation;
             uncertain, so on_uncertain applies, when the violation exceeds band.
             Catches a model that is confused about the case, which calibration of
             each answer on its own cannot show.

  route      rules: [[action, condition], ...], otherwise    value: an action
             the first rule whose condition holds gives the action, `otherwise` when none
             does. A condition too close to call stops the route (on_uncertain applies)
             rather than falling through to a lower rule on a guess.

References may also name a multi option ("multi_id:option") or a locate question's
"none" ("locate_id:none").

Every gate has `on_uncertain`: "abstain" | "escalate" | "default" (with
`default` value). A gate is uncertain when the probability it acts on
is within `band` of tau, exclusive (band defaults to 0.1), or when an
argmax/verify confidence check fails. Uncertainty is surfaced, never
silently resolved. An uncertain input to and / or / at_least counts only
when it could change the result: AND with a decided no is no, OR with a
decided yes is yes, whatever the uncertain input turns out to be.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from decision_circuits.types import answer_distributions

OPS = ("threshold", "not", "and", "or", "at_least", "count", "majority", "argmax", "verify", "order", "consistent", "route")
POOLED = ("and", "or", "at_least", "count")  # ops that read a set: `inputs`, or every option of a multi `input`
RELATIONS = ("same", "complement", "implies")
POLICIES = ("abstain", "escalate", "default")
OUTCOMES = ("decided", "abstain", "escalate", "default")


@dataclass
class Gate:
    """One gate in the wire format. `from_dict` validates a plain dict."""

    op: str
    input: str | None = None
    inputs: list[str] | None = None
    check: str | None = None
    tau: float = 0.5
    band: float = 0.1
    min_confidence: float = 0.0
    cutpoints: list[float] | None = None
    on_uncertain: str = "abstain"
    default: Any = None
    relation: str | None = None
    k: int | None = None
    rules: list[list[Any]] | None = None  # route: [[action, condition ref], ...] in priority order
    otherwise: Any = None  # route: the action when no rule holds

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"unknown gate op {self.op!r}; expected one of {OPS}")
        if self.on_uncertain not in POLICIES:
            raise ValueError(f"on_uncertain must be one of {POLICIES}, got {self.on_uncertain!r}")
        if self.op in ("threshold", "not", "argmax", "verify", "order") and not self.input:
            raise ValueError(f"gate op {self.op!r} needs `input`")
        if self.op == "majority" and not self.inputs:
            raise ValueError("gate op 'majority' needs `inputs`")
        if self.op in POOLED and not (self.inputs or self.input):
            raise ValueError(f"gate op {self.op!r} needs `inputs` (or a multi `input`)")
        if self.op == "at_least":
            if self.k is None or int(self.k) != self.k or self.k < 0:
                raise ValueError("gate op 'at_least' needs `k`, a whole number >= 0")
            self.k = int(self.k)
        if self.op == "route":
            if not self.rules or any(not isinstance(r, list | tuple) or len(r) != 2 for r in self.rules):
                raise ValueError("gate op 'route' needs `rules`: [[action, condition], ...]")
            self.rules = [[a, c] for a, c in self.rules]
        if self.op == "verify" and not self.check:
            raise ValueError("gate op 'verify' needs `check`")
        if self.op == "order" and self.cutpoints is None:
            raise ValueError("gate op 'order' needs `cutpoints`")
        if self.op == "consistent":
            if not self.inputs or len(self.inputs) != 2:
                raise ValueError("gate op 'consistent' needs exactly two `inputs`")
            if self.relation not in RELATIONS:
                raise ValueError(f"gate op 'consistent' needs `relation`, one of {RELATIONS}")
        self.tau = float(self.tau)
        self.band = float(self.band)
        self.min_confidence = float(self.min_confidence)
        if self.cutpoints is not None:
            self.cutpoints = [float(c) for c in self.cutpoints]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Gate:
        if not isinstance(d, dict):
            raise TypeError(f"gate spec must be a dict, got {type(d).__name__}")
        unknown = set(d) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown gate fields {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # pydantic-style aliases, kept so callers written against 0.1 keep working
    model_validate = from_dict
    model_dump = to_dict


class GateResultDict(TypedDict):
    """`GateResult.to_dict()`: one gate's result as plain data."""

    value: Any
    p: float | None
    confidence: float | None
    uncertain: bool
    outcome: str
    trace: list[str]
    probabilities: dict[str, float] | None
    rules: list[str] | None


@dataclass
class GateResult:
    value: Any
    p: float | None = None
    confidence: float | None = None
    uncertain: bool = False
    outcome: str = "decided"
    trace: list[str] = field(default_factory=list)
    probabilities: dict[str, float] | None = None  # per value, when the gate has a distribution (count)
    rules: list[str] | None = None  # route: each rule's "held", "no", "unsure" or "skipped" (never reached)

    def to_dict(self) -> GateResultDict:
        return asdict(self)  # type: ignore[return-value]

    model_dump = to_dict


def result_key(result: GateResultDict | GateResult) -> Any:
    """What a gate result routes on: its value when it decided (or fell
    back to its default), otherwise its outcome ("abstain" or "escalate").
    Used by every integration so an actions table and a graph edge map
    agree on the key for a given result."""
    r = result.to_dict() if isinstance(result, GateResult) else result
    return r["value"] if r["outcome"] in ("decided", "default") else r["outcome"]


EPS = 1e-9  # float slack at a band's edge: 0.9 against 0.8 +/- 0.1 is on the edge, not inside


def _near(p: float, tau: float, band: float) -> bool:
    """Within `band` of `tau`, exclusive, up to float error (0.9 - 0.8 is 0.0999... in floats)."""
    return abs(p - tau) < band - EPS


def _noul_p(answers: dict[str, Any], results: dict[str, GateResult], ref: str) -> tuple[float, str, bool]:
    """(probability, trace, upstream_uncertain) behind a reference.

    A reference is a noul id; 'choice_id:option'; 'score_id:level'; an
    earlier boolean gate (contributes P(its condition), whatever its
    value); or 'gate_id:value' for a categorical gate. If the upstream
    gate was uncertain, the flag propagates so the consumer's
    on_uncertain policy applies."""
    if ref in results:
        r = results[ref]
        if r.p is None:
            raise ValueError(f"gate {ref!r} carries no probability; use '{ref}:<value>'")
        note = " (uncertain)" if r.uncertain else ""
        return float(r.p), f"gate {ref} p={r.p:.2f}{note}", r.uncertain
    if ":" in ref:
        qid, opt = ref.split(":", 1)
        if qid in results:
            r = results[qid]
            note = " (uncertain)" if r.uncertain else ""
            if r.probabilities is None:
                raise ValueError(f"gate {qid!r} has no distribution over values; reference it as a probability, {qid!r}")
            pm = float(r.probabilities.get(opt, 0.0))
            return pm, f"gate {qid}={r.value} -> P({opt})={pm:.2f}{note}", r.uncertain
        views = answer_distributions(answers[qid])
        if f"[{opt}]" in views and "yes" in views[f"[{opt}]"]:  # a multi option: P(it applies)
            p = views[f"[{opt}]"]["yes"]
        elif opt in views.get("", {}):
            p = views[""][opt]
        else:
            raise ValueError(f"{qid!r} has no option {opt!r} (a match answers per item; read it in code)")
        return p, f"{qid}[{opt}] p={p:.2f}", False
    a = answers[ref]
    if a["type"] != "noul":
        raise ValueError(f"{ref!r} is not a noul; use 'choice_id:option' for a choice option")
    return float(a["noul"]), f"{ref} p={a['noul']:.2f}", False


def _pooled(g: Gate, answers: dict[str, Any], results: dict[str, GateResult], trace: list[str]) -> tuple[list[float], list[bool]]:
    """Probabilities behind a pooled gate, its `inputs` or every option of its multi `input`, and
    which of them came from an upstream gate that was uncertain."""
    refs = g.inputs
    if not refs:
        a = answers.get(g.input or "")
        if not a or a["type"] != "multi":
            raise ValueError(f"gate op {g.op!r} reads `inputs`, or a multi as `input`; {g.input!r} is neither")
        refs = [f"{g.input}:{opt}" for opt in a["probabilities"]]
    ps, unc = [], []
    for ref in refs:
        p, t, u = _noul_p(answers, results, ref)
        ps.append(p)
        unc.append(u)
        trace.append(t)
    return ps, unc


def _settled_anyway(ps: list[float], unc: list[bool], k: int, tau: float, band: float) -> bool:
    """Whether "at least k of these hold" decides the same way whatever the uncertain inputs turn
    out to be. P(at least k) only rises as any input does, so it is enough to try them all at 0
    and all at 1: AND with a decided no, or OR with a decided yes, is settled by that input alone."""
    lo = sum(_count_dist([0.0 if u else p for p, u in zip(ps, unc, strict=True)])[k:])
    hi = sum(_count_dist([1.0 if u else p for p, u in zip(ps, unc, strict=True)])[k:])
    return (lo >= tau and not _near(lo, tau, band)) or (hi < tau and not _near(hi, tau, band))


def _count_dist(ps: list[float]) -> list[float]:
    """P(exactly j hold) for j = 0..len(ps), independence assumed (Poisson binomial)."""
    dist = [1.0]
    for p in ps:
        dist = [(dist[j] if j < len(dist) else 0.0) * (1 - p) + (dist[j - 1] * p if j else 0.0) for j in range(len(dist) + 1)]
    return dist


def pick(a: dict[str, Any]) -> str:
    """A choice answer's pick: an argmax of its probabilities, the backend's `choice` when that
    attains the maximum. A `choice` the probabilities do not support is not trusted, since a
    reference could then read another option likelier than the pick (proofs/Pick.lean,
    `trusting_choice_fails`)."""
    probs = a["probabilities"]
    top = max(probs.values())
    return a["choice"] if probs.get(a["choice"], -1.0) >= top else max(probs, key=probs.__getitem__)


def _settle(
    g: Gate,
    value: Any,
    p: float | None,
    uncertain: bool,
    trace: list[str],
    confidence: float | None = None,
    probabilities: dict[str, float] | None = None,
    rules: list[str] | None = None,
) -> GateResult:
    if probabilities is None and isinstance(value, bool) and p is not None:
        # A decided boolean is a fact downstream ("gate:True" is 1 or 0); an uncertain one passes
        # on its probability, and its uncertainty with it.
        probabilities = {"True": p, "False": 1.0 - p} if uncertain else {"True": float(value), "False": float(not value)}
    kw = {"p": p, "confidence": confidence, "probabilities": probabilities, "trace": trace, "rules": rules}
    if not uncertain:
        return GateResult(value=value, **kw)
    if g.on_uncertain == "default":
        trace.append(f"uncertain -> default {g.default!r}")
        return GateResult(value=g.default, uncertain=True, outcome="default", **kw)
    trace.append(f"uncertain -> {g.on_uncertain}")
    return GateResult(value=None, uncertain=True, outcome=g.on_uncertain, **kw)


def evaluate_gates(gates: dict[str, Gate], answers: dict[str, Any]) -> dict[str, GateResult]:
    """Evaluate gates in dependency order (declaration order must respect
    dependencies; a forward reference raises)."""
    results: dict[str, GateResult] = {}
    for gid, g in gates.items():
        trace: list[str] = []
        if g.op == "threshold":
            p, t, unc = _noul_p(answers, results, g.input)
            trace.append(t)
            results[gid] = _settle(g, p >= g.tau, p, unc or _near(p, g.tau, g.band), trace)
        elif g.op == "not":
            p, t, unc = _noul_p(answers, results, g.input)
            trace += [t, f"not -> p={1 - p:.2f}"]
            results[gid] = _settle(g, (1 - p) >= g.tau, 1 - p, unc or _near(1 - p, g.tau, g.band), trace)
        elif g.op in POOLED:
            # one count distribution for all four: `and` is "all of them", `or` "at least one"
            ps, unc_in = _pooled(g, answers, results, trace)
            dist = _count_dist(ps)
            if g.op == "count":
                n = max(range(len(dist)), key=dist.__getitem__)
                trace.append(f"count of {len(ps)} under independence -> {n} p={dist[n]:.2f} (expected {sum(ps):.2f})")
                probs = {str(j): q for j, q in enumerate(dist)}
                results[gid] = _settle(g, n, dist[n], any(unc_in) or dist[n] < g.min_confidence, trace, confidence=dist[n], probabilities=probs)
                continue
            k = {"and": len(ps), "or": 1}.get(g.op, g.k)
            p = sum(dist[k:])
            if g.op == "at_least":
                trace.append(f"at least {k} of {len(ps)} under independence -> p={p:.2f} (expected {sum(ps):.2f})")
            else:
                trace.append(f"{g.op} under independence -> p={p:.2f}")
            unc = any(unc_in)
            if unc and _settled_anyway(ps, unc_in, k, g.tau, g.band):
                unc = False
                trace.append("the uncertain inputs cannot change it")
            results[gid] = _settle(g, p >= g.tau, p, unc or _near(p, g.tau, g.band), trace)
        elif g.op == "majority":
            votes: dict[str, list[float]] = {}
            for ref in g.inputs or []:
                a = answers[ref]
                if a["type"] != "choice":
                    raise ValueError(f"majority input {ref!r} must be a choice")
                chosen = pick(a)
                votes.setdefault(chosen, []).append(float(a["probabilities"][chosen]))
                trace.append(f"{ref} -> {chosen} ({a['probabilities'][chosen]:.2f})")
            n = len(g.inputs or [])
            winner, ps = max(votes.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))
            margin = len(ps) / n
            p = sum(ps) / len(ps)
            trace.append(f"majority {winner} {len(ps)}/{n}, mean p={p:.2f}")
            shares = {k: len(v) / n for k, v in votes.items()}  # vote shares: the winner's is the largest (proofs/Pick.lean)
            results[gid] = _settle(g, winner, p, margin <= 0.5 or p < g.min_confidence, trace, confidence=margin, probabilities=shares)
        elif g.op == "argmax":
            a = answers[g.input]
            if a["type"] != "choice":
                raise ValueError(f"argmax input {g.input!r} must be a choice")
            conf = float(a["confidence"])
            chosen = pick(a)
            p = float(a["probabilities"][chosen])
            trace.append(f"{g.input} -> {chosen} p={p:.2f} conf={conf:.2f} (min {g.min_confidence})")
            dist = {k: float(v) for k, v in a["probabilities"].items()}
            results[gid] = _settle(g, chosen, p, conf < g.min_confidence, trace, confidence=conf, probabilities=dist)
        elif g.op == "verify":
            a = answers[g.input]
            if a["type"] != "choice":
                raise ValueError(f"verify input {g.input!r} must be a choice")
            conf = float(a["confidence"])
            p_check, t, u = _noul_p(answers, results, g.check)
            chosen = pick(a)
            trace += [f"{g.input} -> {chosen} conf={conf:.2f}", f"check {t} (tau {g.tau})"]
            unc = u or conf < g.min_confidence or p_check < g.tau
            dist = {k: float(v) for k, v in a["probabilities"].items()}
            results[gid] = _settle(g, chosen, p_check, unc, trace, confidence=conf, probabilities=dist)
        elif g.op == "order":
            a = answers[g.input]
            if a["type"] != "score":
                raise ValueError(f"order input {g.input!r} must be a score")
            s = float(a["score"])
            cuts = g.cutpoints or []
            bucket = sum(1 for c in cuts if s >= c)
            near = any(_near(s, c, g.band) for c in cuts)
            trace.append(f"{g.input} score={s:.2f} cutpoints={cuts} -> bucket {bucket}" + (" (near a cutpoint)" if near else ""))
            # The bucket is a function of the expected score, so it reads 1 and the others 0; summing
            # level mass per bucket can read another bucket likelier (proofs/Pick.lean, levelMassReading_fails).
            onehot = {str(b): float(b == bucket) for b in range(len(cuts) + 1)}
            results[gid] = _settle(g, bucket, None, near, trace, confidence=float(a["confidence"]), probabilities=onehot)
        elif g.op == "consistent":
            ref_a, ref_b = g.inputs  # exactly two, and a known relation: checked in __post_init__
            (pa, ta, ua), (pb, tb, ub) = _noul_p(answers, results, ref_a), _noul_p(answers, results, ref_b)
            gap = {"same": abs(pa - pb), "complement": abs(pa + pb - 1.0), "implies": max(0.0, pa - pb)}[g.relation]
            trace += [ta, tb, f"{g.relation}: violation {gap:.2f} (band {g.band})"]
            results[gid] = _settle(g, gap <= g.band + EPS, 1.0 - gap, ua or ub or gap > g.band + EPS, trace)
        elif g.op == "route":
            # First rule that holds wins. A rule too close to call stops the route: falling through to
            # a lower rule would act on a guess about the higher one.
            action, unc = g.otherwise, False
            status = ["skipped"] * len(g.rules or [])  # what happened to each rule, for whoever draws or explains it
            for i, (name, ref) in enumerate(g.rules or [], 1):
                p, t, u = _noul_p(answers, results, ref)
                if u:
                    unc, status[i - 1] = True, "unsure"
                    trace.append(f"rule {i} {name!r}: too close to call ({t})")
                    break
                if p >= 0.5:
                    action, status[i - 1] = name, "held"
                    trace.append(f"rule {i} {name!r} holds -> {name}")
                    break
                status[i - 1] = "no"
                trace.append(f"rule {i} {name!r} does not hold")
            else:
                trace.append(f"no rule holds -> {g.otherwise!r}")
            # a decided action reads as certain ("route:hold" is 1 or 0); an undecided route reads 0
            # with its uncertainty, so a gate or route built on it escalates too
            results[gid] = _settle(g, action, None, unc, trace, probabilities={} if unc else {str(action): 1.0}, rules=status)
    return results
