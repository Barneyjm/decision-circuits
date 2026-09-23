"""Interventions: change the input, re-run the circuit, see what the decision does.

`explain` on a System One server removes one sentence at a time and reports how one
answer moves. This is the same idea one level up and on any backend: each intervention
is an edited state, the whole circuit runs on it, and the result says which gates
changed their decision and by how much every probability moved. It is Pearl's do():
the effect is measured by changing the input, so it is a fact about what the circuit
does, not a story about why.

    from decision_circuits import Circuit, drop, set_to

    effects = c.intervene(backend, state, {
        "no receipt": drop("receipt"),               # remove a field of a dict state
        "vip": set_to("tier", "vip"),                # do(tier = "vip")
        "calm": lambda s: {**s, "message": "Hi, could you check my order?"},
    })
    effects["effects"]["vip"]["flipped"]            # gates whose decision changed

    c.ablate(backend, state)                         # one intervention per sentence (text) or field (dict)

Each intervention is one more call to the backend; the baseline is one more. Calls run
`workers` at a time.
"""

from __future__ import annotations

import copy
import itertools
import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypedDict

from decision_circuits.gates import result_key

if TYPE_CHECKING:
    from decision_circuits.dsl import Circuit, RunOutput
    from decision_circuits.types import Backend

Edit = Callable[[Any], Any]
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def drop(*path: str | int) -> Edit:
    """Remove the value at `path` (keys and list indices) from a dict or list state."""
    if not path:
        raise ValueError("drop needs a path")

    def edit(state: Any) -> Any:
        out = copy.deepcopy(state)
        parent = out
        for k in path[:-1]:
            parent = parent[k]
        del parent[path[-1]]
        return out

    return edit


def set_to(*path_and_value: Any) -> Edit:
    """do(path = value): `set_to("tier", "vip")`, `set_to("order", "items", 0, "qty", 3)`."""
    *path, value = path_and_value
    if not path:
        raise ValueError("set_to needs a path and a value")

    def edit(state: Any) -> Any:
        out = copy.deepcopy(state)
        parent = out
        for k in path[:-1]:
            parent = parent[k]
        parent[path[-1]] = value
        return out

    return edit


class GateEffect(TypedDict):
    before: Any  # the gate's routing key (value, or "abstain"/"escalate"), as `result_key`
    after: Any
    p_before: float | None
    p_after: float | None
    dp: float | None  # p_after - p_before, when both carry a probability


class AnswerEffect(TypedDict):
    """Keyed by question id, or "id[option]" for a multi and "id[item]" for a match."""

    option: str  # the baseline's pick; the probabilities below are of this option
    p_before: float
    p_after: float
    dp: float


class Effect(TypedDict):
    state: Any
    run: RunOutput
    flipped: list[str]  # gates whose decision changed
    gates: dict[str, GateEffect]
    answers: dict[str, AnswerEffect]


class Interventions(TypedDict):
    baseline: RunOutput
    effects: dict[str, Effect]


def distributions(answer: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """An answer as named distributions: one for most types, one per option for multi
    (applies or not), one per item for match. Keys are "" or "[option]" / "[item]"."""
    t = answer.get("type")
    if t == "noul":
        p = float(answer["noul"])
        return {"": {"yes": p, "no": 1.0 - p}}
    if t == "multi":
        return {f"[{k}]": {"yes": float(v), "no": 1.0 - float(v)} for k, v in answer["probabilities"].items()}
    if t == "locate":
        return {"": {"found": 1.0 - float(answer["none"]), "none": float(answer["none"])}}
    if t == "match":
        return {f"[{k}]": {o: float(v) for o, v in m["probabilities"].items()} for k, m in answer["matches"].items()}
    return {"": {k: float(v) for k, v in answer["probabilities"].items()}}


def compare(baseline: RunOutput, run: RunOutput) -> tuple[list[str], dict[str, GateEffect], dict[str, AnswerEffect]]:
    """What changed between two runs of the same circuit."""
    gates: dict[str, GateEffect] = {}
    flipped = []
    for gid, before in baseline["gates"].items():
        after = run["gates"][gid]
        kb, ka = result_key(before), result_key(after)
        pb, pa = before.get("p"), after.get("p")
        gates[gid] = {"before": kb, "after": ka, "p_before": pb, "p_after": pa, "dp": None if pb is None or pa is None else pa - pb}
        if kb != ka:
            flipped.append(gid)
    answers: dict[str, AnswerEffect] = {}
    for qid, a in baseline["answers"].items():
        after = distributions(run["answers"][qid])
        for suffix, dist in distributions(a).items():
            option = max(dist, key=dist.__getitem__)
            pb, pa = dist[option], after.get(suffix, {}).get(option, 0.0)
            answers[qid + suffix] = {"option": option, "p_before": pb, "p_after": pa, "dp": pa - pb}
    return flipped, gates, answers


def intervene(
    circuit: Circuit,
    backend: Backend,
    state: Any,
    interventions: Mapping[str, Any],
    *,
    model: str | None = None,
    workers: int = 4,
    baseline: RunOutput | None = None,
) -> Interventions:
    """Run the circuit on `state` and on each intervention; report what changed.

    An intervention is a function of the state (`drop`, `set_to`, or your own) or a
    replacement state. Pass `baseline` to reuse a run you already have."""
    names = list(interventions)
    states = [v(state) if callable(v) else v for v in interventions.values()]
    jobs = ([] if baseline else [state]) + states
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        runs = list(pool.map(lambda s: circuit.run(backend, s, model=model), jobs))
    base = baseline or runs.pop(0)
    effects: dict[str, Effect] = {}
    for name, s, run in zip(names, states, runs, strict=True):
        flipped, gates, answers = compare(base, run)
        effects[name] = {"state": s, "run": run, "flipped": flipped, "gates": gates, "answers": answers}
    return {"baseline": base, "effects": effects}


def segments(state: Any, unit: str = "auto", limit: int = 12) -> dict[str, Edit]:
    """One removal per segment: each sentence (or line) of a text state, or each top-level
    field of a dict state. Named by the segment's text or the field's key."""
    if unit not in ("auto", "field", "sentence", "line"):
        raise ValueError(f"unit must be 'auto', 'field', 'sentence' or 'line', got {unit!r}")
    if unit == "auto":
        unit = "field" if isinstance(state, dict) else "sentence"
    if unit == "field":
        if not isinstance(state, dict):
            raise TypeError("unit='field' needs a dict state")
        return {f"-{k}": drop(k) for k in list(state)[:limit]}
    if not isinstance(state, str):
        raise TypeError(f"unit={unit!r} needs a text state; use unit='field' for a dict")
    # Each segment keeps the whitespace after it, so removing one leaves the rest of the text,
    # newlines and all, exactly as written.
    if unit == "line":
        spans = state.splitlines(keepends=True)
    else:
        cuts = [0, *(m.end() for m in _SENTENCE.finditer(state)), len(state)]
        spans = [state[a:b] for a, b in itertools.pairwise(cuts)]
    real = [i for i, t in enumerate(spans) if t.strip()]

    def without(i: int) -> Edit:
        return lambda _s: "".join(spans[:i] + spans[i + 1 :])

    return {f"-[{n}] {spans[i].strip()}": without(i) for n, i in enumerate(real[:limit])}  # the first `limit`; the rest stay in


def ablate(circuit: Circuit, backend: Backend, state: Any, *, unit: str = "auto", limit: int = 12, model: str | None = None, workers: int = 4) -> Interventions:
    """Remove each segment in turn (see `segments`) and report what the decision does."""
    return intervene(circuit, backend, state, segments(state, unit, limit), model=model, workers=workers)
