"""Chips: reusable sub-circuits with named pins.

A chip is a circuit packaged like an integrated circuit: input pins it reads, output pins
(gates) it drives, and the questions and gates in between. Wire it into a host circuit with
`Circuit.mount`; test it on its own with `Chip.evaluate` and the test vectors it carries; ship
it as JSON (`to_dict` / `Circuit.from_dict`) through PyPI or git like any other code.

    from decision_circuits import Chip, Circuit, G, Q, at_least

    esc = Chip("escalation", inputs=["angry", "pii", "dept"], outputs=["out"])
    esc.noul("threat", "Does the writer threaten legal action or a chargeback?")
    esc.gate("hot", at_least(2, "angry", "pii", "threat", Q("dept")["billing"]))
    esc.gate("out", G("hot") >= 0.6, on_uncertain="escalate")
    esc.add_test({"angry": 0.9, "pii": 0.1, "threat": 0.9, "dept": {"billing": 0.9, "other": 0.1}}, {"out": True})

    c = Circuit()
    c.noul("angry", "Is the writer angry?")
    ...
    pins = c.mount(esc, "esc", {"angry": "angry", "pii": "pii", "dept": "dept"})
    c.gate("page_oncall", pins["out"] >= 0.5)

Inside the chip an input pin is referenced like a question (`"angry"`, `Q("dept")["billing"]`);
mounting rewrites it to whatever the pin is wired to: a host question, a host gate (`G(...)`),
or one option of either (`Q("dept")["billing"]`). Params fill `{name}` in the chip's question
text per mount (`params=["who"]`, then `{"who": "the agent's reply"}`), so two copies can ask
about two parts of the same state. The chip's own questions and gates become
`ns.<name>` in the host, so a chip mounted twice is two independent copies, and the backend is
asked the chip's questions alongside the host's in one request.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from decision_circuits.dsl import FORMAT, And, Categorical, Circuit, Expr, G, GateDef, Not, Or, Q, RunOutput, Threshold, _load_into
from decision_circuits.gates import GateResultDict, result_key
from decision_circuits.types import Backend, normalized_confidence


class Chip(Circuit):
    """A circuit with declared input and output pins. Build it like a `Circuit` (questions it
    asks itself, gates over those and over its input pins), then `mount` it into a host."""

    def __init__(
        self,
        name: str,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        *,
        params: Sequence[str] = (),
        version: str | None = None,
        description: str | None = None,
    ):
        super().__init__()
        self.name = name
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.params = list(params)
        self.version = version
        self.description = description
        self.tests: list[dict[str, Any]] = []
        for pin in [*self.inputs, *self.outputs, *self.params]:
            _check_name(pin, "pin")

    def __repr__(self) -> str:
        return f"Chip({self.name!r}, inputs={self.inputs}, outputs={self.outputs})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Chip) and self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]  # mutable, like Circuit

    # checking ----------------------------------------------------------
    def check(self) -> None:
        """Raise unless the chip is well formed: pins don't collide with its own names, every
        output pin is a gate, and every reference inside reads an input pin, one of its own
        questions, or one of its own gates."""
        own = set(self.questions) | {g.name for g in self.gates}
        clash = own & set(self.inputs)
        if clash:
            raise ValueError(f"chip {self.name!r}: {sorted(clash)} are both input pins and its own questions or gates")
        gates = {g.name for g in self.gates}
        missing = [o for o in self.outputs if o not in gates]
        if missing:
            raise ValueError(f"chip {self.name!r}: output pins {missing} are not gates of the chip")
        known = own | set(self.inputs)
        for g in self.gates:
            for ref in _refs(g.body):
                if ref.partition(":")[0] not in known:
                    raise ValueError(f"chip {self.name!r}, gate {g.name!r}: {ref!r} is not an input pin, question or gate of the chip")

    # offline use -------------------------------------------------------
    def evaluate(self, answers: Mapping[str, Any]) -> dict[str, GateResultDict]:  # type: ignore[override]
        """Evaluate the chip on pin values and answers to its own questions, no model needed.
        Shorthands are accepted: a number is a noul's P(yes), a dict of option -> probability
        is a choice; anything with a `type` is a wire-format answer as is."""
        self.check()
        full = {k: _coerce(v) for k, v in answers.items()}
        missing = [n for n in [*self.inputs, *self.questions] if n not in full]
        if missing:
            raise ValueError(f"chip {self.name!r}: no value for {', '.join(map(repr, missing))}")
        return super().evaluate(full)

    def run(self, backend: Backend, state: Any, *, model: str | None = None) -> RunOutput:
        if self.inputs:
            raise TypeError(f"chip {self.name!r} has input pins: mount it in a circuit to run it, or test it with evaluate()")
        return super().run(backend, state, model=model)

    # test vectors ------------------------------------------------------
    def add_test(self, answers: Mapping[str, Any], expect: Mapping[str, Any], name: str | None = None) -> Chip:
        """Store a test vector that ships with the chip. `expect` maps gate -> its value when
        decided (or defaulted), else its outcome: True, "billing", 2, "escalate", "abstain"."""
        case: dict[str, Any] = {"answers": dict(answers), "expect": dict(expect)}
        if name:
            case["name"] = name
        self.tests.append(case)
        return self

    def test(self, cases: Sequence[Mapping[str, Any]] | str | Path | None = None) -> list[dict[str, Any]]:
        """Run test vectors (the chip's own by default, a list, or a JSONL file of
        {"answers", "expect", "name"?} lines) and return the failures; empty means all pass."""
        if isinstance(cases, str | Path):
            cases = [json.loads(line) for line in Path(cases).read_text().splitlines() if line.strip()]
        failures = []
        for i, case in enumerate(self.tests if cases is None else cases):
            label = case.get("name", f"#{i}")
            results = self.evaluate(case["answers"])
            for gate, want in case["expect"].items():
                if gate not in results:
                    failures.append({"case": label, "gate": gate, "expected": want, "got": "no such gate", "trace": []})
                    continue
                got = result_key(results[gate])
                if got != want:
                    failures.append({"case": label, "gate": gate, "expected": want, "got": got, "trace": results[gate]["trace"]})
        return failures

    # serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        meta: dict[str, Any] = {"name": self.name, "inputs": self.inputs, "outputs": self.outputs, "requires": self.question_types}
        if self.params:
            meta["params"] = self.params
        if self.version:
            meta["version"] = self.version
        if self.description:
            meta["description"] = self.description
        if self.tests:
            meta["tests"] = self.tests
        return {"format": d.pop("format"), "chip": meta, **d}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Chip:
        if d.get("format", FORMAT) != FORMAT:
            raise ValueError(f"unknown circuit format {d.get('format')!r}; this version reads {FORMAT!r}")
        meta = d.get("chip")
        if not meta:
            raise ValueError("not a chip: no `chip` block")
        chip = cls(
            meta["name"],
            meta.get("inputs", ()),
            meta.get("outputs", ()),
            params=meta.get("params", ()),
            version=meta.get("version"),
            description=meta.get("description"),
        )
        _load_into(chip, d)
        chip.tests = [dict(t) for t in meta.get("tests", [])]
        chip.check()
        return chip


# ---------------------------------------------------------------- mounting


def mount(host: Circuit, chip: Chip, ns: str, pins: Mapping[str, str | Q | G], params: Mapping[str, str] | None = None) -> dict[str, G]:
    """`Circuit.mount`: copy `chip` into `host` under `ns` and return its output pins."""
    if not isinstance(chip, Chip):
        raise TypeError(f"mount takes a Chip, got {type(chip).__name__}")
    chip.check()
    _check_name(ns, "namespace")
    taken = set(host.questions) | {g.name for g in host.gates} | set(host.mounts)
    if any(n == ns or n.startswith(ns + ".") for n in taken):
        raise ValueError(f"namespace {ns!r} is already used in this circuit")
    unknown = set(pins) - set(chip.inputs)
    if unknown:
        raise ValueError(f"chip {chip.name!r} has no input pins {sorted(unknown)}; its inputs are {chip.inputs}")
    params = dict(params or {})
    if set(params) != set(chip.params):
        raise ValueError(f"chip {chip.name!r} takes params {chip.params}; got {sorted(params)}")
    unwired = [p for p in chip.inputs if p not in pins]
    if unwired:
        raise ValueError(f"chip {chip.name!r}: input pins {unwired} are not wired")
    wired = {p: (t.ref if isinstance(t, Q | G) else str(t)) for p, t in pins.items()}
    own = set(chip.questions) | {g.name for g in chip.gates}

    def remap(ref: str) -> str:
        base, sep, opt = ref.partition(":")
        if base in wired:
            target = wired[base]
            if sep and ":" in target:
                raise ValueError(f"chip {chip.name!r} reads {ref!r}, but pin {base!r} is wired to {target!r}, which already names an option")
            return target + sep + opt
        if base in own:
            return f"{ns}.{ref}"
        raise ValueError(f"chip {chip.name!r}: {ref!r} is not an input pin, question or gate of the chip")

    # build everything first, so a wiring mistake leaves the host as it was
    gates = [GateDef(f"{ns}.{g.name}", _rename(g.body, remap), g.on_uncertain_, copy.deepcopy(g.default_), g.band_) for g in chip.gates]
    mounts = {ns: {"chip": chip.name, "pins": wired, "outputs": list(chip.outputs)}}
    for inner, m in chip.mounts.items():  # a chip built from chips: keep the inner boards, rewired
        mounts[f"{ns}.{inner}"] = {**m, "pins": {p: remap(t) for p, t in m["pins"].items()}}
    host.questions.update({f"{ns}.{qid}": _fill(copy.deepcopy(q), params) for qid, q in chip.questions.items()})
    host.gates.extend(gates)
    host.mounts.update(mounts)
    return {o: G(f"{ns}.{o}") for o in chip.outputs}


def _fill(x: Any, params: Mapping[str, str]) -> Any:
    """`{param}` in every string of a question replaced by its value; other braces are left alone."""
    if isinstance(x, str):
        for k, v in params.items():
            x = x.replace("{" + k + "}", str(v))
        return x
    if isinstance(x, dict):
        return {k: _fill(v, params) for k, v in x.items()}
    if isinstance(x, list):
        return [_fill(v, params) for v in x]
    return x


def _check_name(name: str, what: str) -> None:
    if not name or ":" in name or "." in name or name.startswith("_"):
        raise ValueError(f"{what} {name!r}: use a plain name, no ':' or '.', not starting with '_'")


def _rename(e: Any, f: Callable[[str], str]) -> Any:
    """The same expression with every reference passed through `f`."""
    if isinstance(e, Q):
        return Q(f(e.ref))
    if isinstance(e, G):
        return G(f(e.ref))
    if isinstance(e, Not):
        return Not(_rename(e.inner, f))
    if isinstance(e, And):
        return And([_rename(p, f) for p in e.parts])
    if isinstance(e, Or):
        return Or([_rename(p, f) for p in e.parts])
    if isinstance(e, Threshold):
        return Threshold(_rename(e.inner, f), e.tau)
    if isinstance(e, Categorical):
        return Categorical(
            e.op,
            input=f(e.input) if e.input is not None else None,
            inputs=[f(r) for r in e.inputs] if e.inputs is not None else None,
            check=_rename(e.check, f) if e.check is not None else None,
            tau=e.tau,
            min_confidence=e.min_confidence,
            cutpoints=list(e.cutpoints) if e.cutpoints is not None else None,
            k=e.k,
            relation=e.relation,
            rules=[(a, _rename(c, f)) for a, c in e.rules] if e.rules is not None else None,
            otherwise=e.otherwise,
        )
    raise TypeError(f"unsupported expression {e!r}")


def _refs(e: Expr | Categorical) -> list[str]:
    out: list[str] = []
    _rename(e, lambda r: out.append(r) or r)
    return out


def _coerce(v: Any) -> Any:
    """Pin-value shorthands to wire-format answers."""
    if isinstance(v, bool):
        raise TypeError("a pin value is a probability, not a bool: use 1.0 or 0.0")
    if isinstance(v, int | float):
        return {"type": "noul", "noul": float(v)}
    if isinstance(v, Mapping) and "type" not in v:
        probs = {str(k): float(p) for k, p in v.items()}
        return {"type": "choice", "choice": max(probs, key=probs.__getitem__), "probabilities": probs, "confidence": normalized_confidence(probs)}
    return v


__all__ = ["Chip", "mount"]
