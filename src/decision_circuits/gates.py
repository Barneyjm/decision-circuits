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
  and / or   inputs: k nouls/gates                     value: bool
             p = product / 1 - product(1-p) (independence assumption; reported as such)
  majority   inputs: k choice ids over the same option set   value: option
             votes by argmax; p = mean probability of the winner; margin reported
  argmax     input: choice id                           value: option or abstain
             abstains when confidence < min_confidence
  verify     input: choice id, check: noul id           value: option or escalate
             the check asks "is <that answer> supported?"; escalates when
             P(check) < tau or the choice is below min_confidence
  order      input: score id, cutpoints: [c1, c2, ...]  value: bucket index (0..len)
             bucket = number of cutpoints <= expected score

Every gate has `on_uncertain`: "abstain" | "escalate" | "default" (with
`default` value). A gate is uncertain when the probability it acts on
falls inside [tau - band, tau + band] (band defaults to 0.1), or when an
argmax/verify confidence check fails. Uncertainty is surfaced, never
silently resolved.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Gate(BaseModel):
    op: Literal["threshold", "not", "and", "or", "majority", "argmax", "verify", "order"]
    input: str | None = None
    inputs: list[str] | None = None
    check: str | None = None
    tau: float = 0.5
    band: float = 0.1
    min_confidence: float = 0.0
    cutpoints: list[float] | None = None
    on_uncertain: Literal["abstain", "escalate", "default"] = "abstain"
    default: Any = None


class GateResult(BaseModel):
    value: Any
    p: float | None = None
    confidence: float | None = None
    uncertain: bool = False
    outcome: Literal["decided", "abstain", "escalate", "default"] = "decided"
    trace: list[str] = Field(default_factory=list)


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
            p = float(r.p if r.p is not None else 1.0)
            match = str(r.value) == opt
            pm = p if match else max(0.0, 1.0 - p)
            note = " (uncertain)" if r.uncertain else ""
            return pm, f"gate {qid}={r.value} -> P({opt})={pm:.2f}{note}", r.uncertain
        a = answers[qid]
        if a["type"] in ("choice", "score"):
            p = float(a["probabilities"][opt])
            return p, f"{qid}[{opt}] p={p:.2f}", False
        raise ValueError(f"{qid!r} is not a choice or score")
    a = answers[ref]
    if a["type"] != "noul":
        raise ValueError(f"{ref!r} is not a noul; use 'choice_id:option' for a choice option")
    return float(a["noul"]), f"{ref} p={a['noul']:.2f}", False


def _settle(g: Gate, value: Any, p: float | None, uncertain: bool, trace: list[str], confidence: float | None = None) -> GateResult:
    if not uncertain:
        return GateResult(value=value, p=p, confidence=confidence, trace=trace)
    if g.on_uncertain == "default":
        trace.append(f"uncertain -> default {g.default!r}")
        return GateResult(value=g.default, p=p, confidence=confidence, uncertain=True, outcome="default", trace=trace)
    trace.append(f"uncertain -> {g.on_uncertain}")
    return GateResult(value=None, p=p, confidence=confidence, uncertain=True, outcome=g.on_uncertain, trace=trace)


def evaluate_gates(gates: dict[str, Gate], answers: dict[str, Any]) -> dict[str, GateResult]:
    """Evaluate gates in dependency order (declaration order must respect
    dependencies; a forward reference raises)."""
    results: dict[str, GateResult] = {}
    for gid, g in gates.items():
        trace: list[str] = []
        if g.op == "threshold":
            p, t, unc = _noul_p(answers, results, g.input)
            trace.append(t)
            results[gid] = _settle(g, p >= g.tau, p, unc or abs(p - g.tau) < g.band, trace)
        elif g.op == "not":
            p, t, unc = _noul_p(answers, results, g.input)
            trace += [t, f"not -> p={1 - p:.2f}"]
            results[gid] = _settle(g, (1 - p) >= g.tau, 1 - p, unc or abs((1 - p) - g.tau) < g.band, trace)
        elif g.op in ("and", "or"):
            ps = []
            unc = False
            for ref in g.inputs or []:
                p, t, u = _noul_p(answers, results, ref)
                ps.append(p)
                unc = unc or u
                trace.append(t)
            if g.op == "and":
                p = 1.0
                for x in ps:
                    p *= x
            else:
                q = 1.0
                for x in ps:
                    q *= 1 - x
                p = 1 - q
            trace.append(f"{g.op} under independence -> p={p:.2f}")
            results[gid] = _settle(g, p >= g.tau, p, unc or abs(p - g.tau) < g.band, trace)
        elif g.op == "majority":
            votes: dict[str, list[float]] = {}
            for ref in g.inputs or []:
                a = answers[ref]
                if a["type"] != "choice":
                    raise ValueError(f"majority input {ref!r} must be a choice")
                votes.setdefault(a["choice"], []).append(float(a["probabilities"][a["choice"]]))
                trace.append(f"{ref} -> {a['choice']} ({a['probabilities'][a['choice']]:.2f})")
            n = len(g.inputs or [])
            winner, ps = max(votes.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))
            margin = len(ps) / n
            p = sum(ps) / len(ps)
            trace.append(f"majority {winner} {len(ps)}/{n}, mean p={p:.2f}")
            results[gid] = _settle(g, winner, p, margin <= 0.5 or p < g.min_confidence, trace, confidence=margin)
        elif g.op == "argmax":
            a = answers[g.input]
            if a["type"] != "choice":
                raise ValueError(f"argmax input {g.input!r} must be a choice")
            conf = float(a["confidence"])
            p = float(a["probabilities"][a["choice"]])
            trace.append(f"{g.input} -> {a['choice']} p={p:.2f} conf={conf:.2f} (min {g.min_confidence})")
            results[gid] = _settle(g, a["choice"], p, conf < g.min_confidence, trace, confidence=conf)
        elif g.op == "verify":
            a = answers[g.input]
            if a["type"] != "choice":
                raise ValueError(f"verify input {g.input!r} must be a choice")
            conf = float(a["confidence"])
            p_check, t, u = _noul_p(answers, results, g.check)
            trace += [f"{g.input} -> {a['choice']} conf={conf:.2f}", f"check {t} (tau {g.tau})"]
            unc = u or conf < g.min_confidence or p_check < g.tau
            results[gid] = _settle(g, a["choice"], p_check, unc, trace, confidence=conf)
        elif g.op == "order":
            a = answers[g.input]
            if a["type"] != "score":
                raise ValueError(f"order input {g.input!r} must be a score")
            s = float(a["score"])
            cuts = g.cutpoints or []
            bucket = sum(1 for c in cuts if s >= c)
            near = any(abs(s - c) < g.band for c in cuts)
            trace.append(f"{g.input} score={s:.2f} cutpoints={cuts} -> bucket {bucket}" + (" (near a cutpoint)" if near else ""))
            results[gid] = _settle(g, bucket, None, near, trace, confidence=float(a["confidence"]))
    return results
