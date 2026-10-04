"""Chips: reusable sub-circuits with typed pins and settings.

A chip is a circuit packaged like an integrated circuit: input pins it reads, output pins
(gates) it drives, settings that configure it, and the questions and gates in between. Wire it
into any host circuit with `Circuit.mount`; test it on its own with `Chip.evaluate` and the test
vectors it carries; ship it as JSON (`to_dict` / `Circuit.from_dict`) like any other code.

    from decision_circuits import Chip, Circuit, G, P, Pin, Q, at_least

    risk = Chip(
        "risk",
        inputs=[Pin("new_account", ask="Was the account behind {request} opened in the last 30 days?"),
                Pin("high_value", ask="Is the amount in {request} unusually high?")],
        outputs=["risky"],
        params={"request": "the request", "needed": 2},
    )
    risk.noul("story", "Is the account of what happened in {request} inconsistent?")
    risk.gate("risky", at_least(P("needed"), "new_account", "high_value", "story"), on_uncertain="escalate")

    c = Circuit()
    c.noul("big", "Order over $500 (computed).")
    out = c.mount(risk, "risk", {"high_value": "big"}, params={"request": "`ticket`"})

**Pins.** An input pin is referenced inside the chip like a question (`"new_account"`,
`Q("dept")["billing"]`). Mounting wires it to a host question, a host gate (`G(...)`), or one
option of either. A `Pin` declares what it takes (`type="noul"`, `"choice"` with `options`,
`"score"`, or `"any"`) and the mount refuses a wire of the wrong kind up front. A pin with `ask`
is optional: wire it when the host knows the answer, leave it and the chip asks the model itself.

**Settings.** `params` are the chip's settings, with defaults (or none, for one the host must
give). `{name}` in question text is a setting as words: what to call the thing the chip reads
("`ticket`", "the agent's reply"). `P("name")` is a setting in the logic: a threshold, a count,
a choice's options, a route's action. A mounted chip is its settings filled in.

The chip's own questions and gates become `ns.<name>` in the host, so a chip mounted twice is two
independent copies, and the backend answers the chip's questions with the host's, in one request.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_circuits.dsl import (
    FORMAT,
    And,
    Categorical,
    Circuit,
    Expr,
    G,
    GateDef,
    Not,
    Or,
    P,
    Q,
    RunOutput,
    Threshold,
    _load_into,
    decode_params,
    encode_params,
)
from decision_circuits.gates import GateResultDict, result_key
from decision_circuits.types import Backend, normalized_confidence

PIN_TYPES = ("noul", "choice", "score", "any")
REQUIRED = object()  # a setting with no default: the host must give it


@dataclass(frozen=True)
class Pin:
    """An input pin. `type` is what it takes: "noul" (a probability: a yes/no answer, one option
    of a choice, or a gate's decision), "choice" (a pick-one; `options` it must offer), "score",
    or "any". `ask` makes it optional: the question the chip asks itself when the pin is not
    wired. A string is the question's text; a dict is a full question spec."""

    name: str
    type: str = "noul"
    options: Sequence[str] | None = None
    ask: str | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _check_name(self.name, "pin")
        if self.type not in PIN_TYPES:
            raise ValueError(f"pin {self.name!r}: type must be one of {PIN_TYPES}, got {self.type!r}")

    def question(self) -> dict[str, Any]:
        """The question an unwired optional pin asks."""
        if isinstance(self.ask, Mapping):
            return dict(self.ask)
        q: dict[str, Any] = {"type": "noul" if self.type == "any" else self.type, "instructions": self.ask}
        if self.type == "choice":
            q["criteria"] = {o: None for o in self.options or ()}
        if self.type == "score":
            q["criteria"] = list(self.options or ())
        return q

    def to_dict(self) -> dict[str, Any] | str:
        if self.type == "any" and self.ask is None:
            return self.name
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.options is not None:
            d["options"] = list(self.options)
        if self.ask is not None:
            d["ask"] = encode_params(self.ask)
        return d

    @classmethod
    def of(cls, x: str | Pin | Mapping[str, Any]) -> Pin:
        if isinstance(x, Pin):
            return x
        if isinstance(x, str):
            return cls(x, type="any")
        return cls(x["name"], x.get("type", "noul"), x.get("options"), decode_params(x.get("ask")))


class Chip(Circuit):
    """A circuit with declared pins and settings. Build it like a `Circuit` (questions it asks
    itself, gates over those and over its input pins), then `mount` it into a host."""

    def __init__(
        self,
        name: str,
        inputs: Sequence[str | Pin] = (),
        outputs: Sequence[str] = (),
        *,
        params: Sequence[str] | Mapping[str, Any] = (),
        version: str | None = None,
        description: str | None = None,
    ):
        super().__init__()
        self.name = name
        self.pins = {p.name: p for p in map(Pin.of, inputs)}
        self.outputs = list(outputs)
        # settings: name -> default, REQUIRED for one the host must give
        self.settings: dict[str, Any] = dict(params) if isinstance(params, Mapping) else dict.fromkeys(params, REQUIRED)
        self.version = version
        self.description = description
        self.tests: list[dict[str, Any]] = []
        for name_ in [*self.outputs, *self.settings]:
            _check_name(name_, "pin" if name_ in self.outputs else "setting")

    @property
    def inputs(self) -> list[str]:
        return list(self.pins)

    @property
    def params(self) -> list[str]:
        return list(self.settings)

    def __repr__(self) -> str:
        return f"Chip({self.name!r}, inputs={self.inputs}, outputs={self.outputs})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Chip) and self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]  # mutable, like Circuit

    # checking ----------------------------------------------------------
    def check(self) -> None:
        """Raise unless the chip is well formed: pins don't collide with its own names, every
        output pin is a gate, every reference inside reads an input pin, one of its own questions
        or one of its own gates, and every `P` names a declared setting."""
        own = set(self.questions) | {g.name for g in self.gates}
        clash = own & set(self.pins)
        if clash:
            raise ValueError(f"chip {self.name!r}: {sorted(clash)} are both input pins and its own questions or gates")
        gates = {g.name for g in self.gates}
        missing = [o for o in self.outputs if o not in gates]
        if missing:
            raise ValueError(f"chip {self.name!r}: output pins {missing} are not gates of the chip")
        known = own | set(self.pins)
        for g in self.gates:
            for ref in _refs(g.body):
                if ref.partition(":")[0] not in known:
                    raise ValueError(f"chip {self.name!r}, gate {g.name!r}: {ref!r} is not an input pin, question or gate of the chip")
        used = _settings_used([self.questions, [(g.body, g.band_, g.default_) for g in self.gates], [p.ask for p in self.pins.values()]])
        undeclared = sorted(used - set(self.settings))
        if undeclared:
            raise ValueError(f"chip {self.name!r} uses settings {undeclared} it does not declare in `params`")

    # settings ----------------------------------------------------------
    def configured(self, values: Mapping[str, Any] | None = None, *, strict: bool = True) -> Chip:
        """A copy with its settings filled in: `values` over the defaults. `strict` refuses an
        unknown setting or a required one left out; otherwise unfilled ones stay as they are."""
        values = dict(values or {})
        unknown = sorted(set(values) - set(self.settings))
        if unknown:
            raise ValueError(f"chip {self.name!r} has no settings {unknown}; its settings are {self.params}")
        full = {k: v for k, v in self.settings.items() if v is not REQUIRED} | values
        if strict:
            missing = [k for k in self.settings if k not in full]
            if missing:
                raise ValueError(f"chip {self.name!r} needs settings {missing}")
        out = Chip(self.name, [], self.outputs, params=self.settings, version=self.version, description=self.description)
        out.pins = {n: Pin(p.name, p.type, _resolve(p.options, full), _resolve(p.ask, full)) for n, p in self.pins.items()}
        out.model = self.model
        out.questions = {k: _resolve(copy.deepcopy(q), full) for k, q in self.questions.items()}
        out.gates = [
            GateDef(g.name, _resolve(g.body, full), g.on_uncertain_, _resolve(copy.deepcopy(g.default_), full), _resolve(g.band_, full)) for g in self.gates
        ]
        out.mounts = copy.deepcopy(self.mounts)
        out.tests = copy.deepcopy(self.tests)
        return out

    def _unfilled(self) -> bool:
        return bool(_settings_used([self.questions, [(g.body, g.band_, g.default_) for g in self.gates]]))

    def compile(self) -> dict[str, dict[str, Any]]:
        """Compiled with the settings' defaults (a mounted chip has its own)."""
        if self._unfilled():
            return self.configured().compile()
        return super().compile()

    def describe(self, results: Mapping[str, Any] | None = None, answers: Mapping[str, Any] | None = None) -> str:
        from decision_circuits.describe import describe

        return describe(self.configured(strict=False), results, answers)

    def to_mermaid(self, *args: Any, **kw: Any) -> str:
        return Circuit.to_mermaid(self.configured(), *args, **kw) if self._unfilled() else super().to_mermaid(*args, **kw)

    # offline use -------------------------------------------------------
    def evaluate(self, answers: Mapping[str, Any]) -> dict[str, GateResultDict]:  # type: ignore[override]
        """Evaluate the chip on pin values and answers to its own questions, no model needed,
        with its default settings. Shorthands: a number is a noul's P(yes), a dict of option ->
        probability is a choice; anything with a `type` is a wire-format answer as is."""
        self.check()
        full = {k: _coerce(v) for k, v in answers.items()}
        missing = [n for n in [*self.pins, *self.questions] if n not in full]
        if missing:
            raise ValueError(f"chip {self.name!r}: no value for {', '.join(map(repr, missing))}")
        return Circuit.evaluate(self.configured() if self._unfilled() else self, full)

    def run(self, backend: Backend, state: Any, *, model: str | None = None) -> RunOutput:
        """Run a chip on its own: every optional pin asks its own question. A chip with a pin it
        cannot ask must be mounted."""
        must_wire = [n for n, p in self.pins.items() if p.ask is None]
        if must_wire:
            raise TypeError(f"chip {self.name!r} has input pins {must_wire} to wire: mount it in a circuit to run it, or test it with evaluate()")
        cfg = self.configured()
        alone = Circuit(questions={**{n: p.question() for n, p in cfg.pins.items()}, **cfg.questions}, gates=cfg.gates, model=cfg.model, mounts=cfg.mounts)
        return alone.run(backend, state, model=model)

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
        meta: dict[str, Any] = {
            "name": self.name,
            "inputs": [p.to_dict() for p in self.pins.values()],
            "outputs": self.outputs,
            "requires": self.question_types,
        }
        if self.settings:
            meta["params"] = [{"name": k} if v is REQUIRED else {"name": k, "default": encode_params(v)} for k, v in self.settings.items()]
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
        raw = meta.get("params", ())
        if raw and isinstance(raw[0], Mapping):  # [{"name", "default"?}]
            params: Any = {p["name"]: decode_params(p["default"]) if "default" in p else REQUIRED for p in raw}
        else:  # an older chip: a list of required names
            params = list(raw)
        chip = cls(
            meta["name"],
            [Pin.of(p) for p in meta.get("inputs", ())],
            meta.get("outputs", ()),
            params=params,
            version=meta.get("version"),
            description=meta.get("description"),
        )
        _load_into(chip, d)
        chip.tests = [dict(t) for t in meta.get("tests", [])]
        chip.check()
        return chip


# ---------------------------------------------------------------- mounting


def mount(host: Circuit, chip: Chip, ns: str, pins: Mapping[str, str | Q | G], params: Mapping[str, Any] | None = None) -> dict[str, G]:
    """`Circuit.mount`: copy `chip` into `host` under `ns` and return its output pins."""
    if not isinstance(chip, Chip):
        raise TypeError(f"mount takes a Chip, got {type(chip).__name__}")
    chip.check()
    _check_name(ns, "namespace")
    taken = set(host.questions) | {g.name for g in host.gates} | set(host.mounts)
    if any(n == ns or n.startswith(ns + ".") for n in taken):
        raise ValueError(f"namespace {ns!r} is already used in this circuit")
    unknown = set(pins) - set(chip.pins)
    if unknown:
        raise ValueError(f"chip {chip.name!r} has no input pins {sorted(unknown)}; its inputs are {chip.inputs}")
    chip = chip.configured(params)  # settings filled in; refuses unknown or missing ones
    unwired = [p for p, pin in chip.pins.items() if p not in pins and pin.ask is None]
    if unwired:
        raise ValueError(f"chip {chip.name!r}: input pins {unwired} are not wired (and don't ask for themselves)")
    wired = {p: (t.ref if isinstance(t, Q | G) else str(t)) for p, t in pins.items()}
    for p, target in wired.items():
        _check_wire(host, chip, chip.pins[p], target)
    asked = {p: pin for p, pin in chip.pins.items() if p not in wired}  # optional pins the chip answers itself
    wired |= {p: f"{ns}.{p}" for p in asked}
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
    mounts = {ns: {"chip": chip.name, "pins": {p: t for p, t in wired.items() if p not in asked}, "outputs": list(chip.outputs)}}
    if asked:
        mounts[ns]["asked"] = sorted(asked)
    for inner, m in chip.mounts.items():  # a chip built from chips: keep the inner boards, rewired
        mounts[f"{ns}.{inner}"] = {**m, "pins": {p: remap(t) for p, t in m["pins"].items()}}
    questions = {f"{ns}.{p}": pin.question() for p, pin in asked.items()}
    questions |= {f"{ns}.{qid}": copy.deepcopy(q) for qid, q in chip.questions.items()}
    host.questions.update(questions)
    host.gates.extend(gates)
    host.mounts.update(mounts)
    return {o: G(f"{ns}.{o}") for o in chip.outputs}


def kind_of(host: Circuit, ref: str) -> tuple[str, list[str] | None]:
    """What a reference in `host` carries, as a pin type, and the options it offers when it is a
    pick-one: a question's own type; a probability for one option, a boolean gate or "gate:value";
    a choice for a gate that picks an option (argmax, majority, verify)."""
    base, _, opt = ref.partition(":")
    if isinstance(host, Chip) and base in host.pins:  # a chip built from chips: wired to its own input pin
        pin = host.pins[base]
        return ("noul", None) if opt else (pin.type, list(pin.options) if pin.options else None)
    if base in host.questions:
        q = host.questions[base]
        if opt or q["type"] == "noul":
            return "noul", None
        crit = q.get("criteria")
        return q["type"], list(crit) if isinstance(crit, Mapping) else None
    gates = {g.name: g.body for g in host.gates}
    if base in gates:
        body = gates[base]
        if opt or isinstance(body, Expr) or body.op in ("at_least", "consistent"):
            return "noul", None
        if body.op in ("argmax", "verify"):
            return "choice", kind_of(host, body.input or "")[1]
        if body.op == "majority":
            return "choice", kind_of(host, (body.inputs or [""])[0])[1]
        return body.op, None  # order, count, route: a value, not a probability or a pick
    raise ValueError(f"{ref!r} is not a question or gate of the circuit")


def _check_wire(host: Circuit, chip: Chip, pin: Pin, target: str) -> None:
    try:
        kind, options = kind_of(host, target)
    except ValueError as e:
        raise ValueError(f"chip {chip.name!r}, pin {pin.name!r}: {e}") from None
    if pin.type == "any" or kind == "any":
        return
    if kind != pin.type:
        raise ValueError(f"chip {chip.name!r}, pin {pin.name!r} takes a {pin.type}; {target!r} is a {kind}")
    if pin.options and options is not None:
        short = [o for o in pin.options if o not in options]
        if short:
            raise ValueError(f"chip {chip.name!r}, pin {pin.name!r} needs options {short} that {target!r} does not offer")


# ---------------------------------------------------------------- settings


def _resolve(x: Any, values: Mapping[str, Any]) -> Any:
    """`x` with every `P` replaced by its value and `{name}` in text by the value as words.
    References (Q, G) are left alone: they are names, not text."""
    if isinstance(x, P):
        return values.get(x.name, x)
    if isinstance(x, str):
        for k, v in values.items():
            if isinstance(v, str | int | float):
                x = x.replace("{" + k + "}", str(v))
        return x
    if isinstance(x, Q | G):
        return x
    if isinstance(x, Not):
        return Not(_resolve(x.inner, values))
    if isinstance(x, And | Or):
        return type(x)([_resolve(p, values) for p in x.parts])
    if isinstance(x, Threshold):
        return Threshold(_resolve(x.inner, values), _resolve(x.tau, values))
    if isinstance(x, Categorical):
        return Categorical(
            x.op,
            input=x.input,
            inputs=x.inputs,
            check=_resolve(x.check, values) if x.check is not None else None,
            tau=_resolve(x.tau, values),
            min_confidence=_resolve(x.min_confidence, values),
            cutpoints=_resolve(x.cutpoints, values),
            k=_resolve(x.k, values),
            relation=x.relation,
            rules=[(_resolve(a, values), _resolve(c, values)) for a, c in x.rules] if x.rules is not None else None,
            otherwise=_resolve(x.otherwise, values),
        )
    if isinstance(x, Mapping):
        return {k: _resolve(v, values) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_resolve(v, values) for v in x]
    return x


def _settings_used(x: Any) -> set[str]:
    found: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, P):
            found.add(v.name)
        elif isinstance(v, Threshold):
            walk(v.inner)
            walk(v.tau)
        elif isinstance(v, Not):
            walk(v.inner)
        elif isinstance(v, And | Or):
            for p in v.parts:
                walk(p)
        elif isinstance(v, Categorical):
            for f in (v.check, v.tau, v.min_confidence, v.cutpoints, v.k, v.otherwise, v.rules):
                walk(f)
        elif isinstance(v, Mapping):
            for w in v.values():
                walk(w)
        elif isinstance(v, list | tuple):
            for w in v:
                walk(w)

    walk(x)
    return found


# ---------------------------------------------------------------- helpers


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
            cutpoints=list(e.cutpoints) if isinstance(e.cutpoints, list) else e.cutpoints,
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


__all__ = ["REQUIRED", "Chip", "Pin", "kind_of", "mount"]
