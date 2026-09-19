"""The wire format as plain Python types, and the one protocol an
integration has to satisfy.

Nothing here imports anything outside the standard library. A backend
is any object with an `answer` method that turns state plus questions
into the `answers` map below; the circuit does the rest.

Questions (as built by `Circuit.noul/choice/score`, or by hand):

    {"type": "noul",   "instructions": str, "criteria": {"true": str|None, "false": str|None}}
    {"type": "choice", "instructions": str, "criteria": {option: description|None, ...}}
    {"type": "score",  "instructions": str, "criteria": [level_description, ...]}

Answers (what a System One server returns, and what a backend must produce):

    {"type": "noul",   "noul": P(yes)}
    {"type": "choice", "choice": option, "probabilities": {option: p, ...}, "confidence": c}
    {"type": "score",  "score": expected, "probabilities": {"0": p, "1": p, ...}, "confidence": c}

`confidence` is 1 - H(p) / log N: 1.0 when all mass is on one option,
0.0 when uniform. `normalized_confidence` computes it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable


class NoulQuestion(TypedDict, total=False):
    type: Literal["noul"]
    instructions: Any
    criteria: dict[str, str | None]


class ChoiceQuestion(TypedDict, total=False):
    type: Literal["choice"]
    instructions: Any
    criteria: dict[str, Any]


class ScoreQuestion(TypedDict, total=False):
    type: Literal["score"]
    instructions: Any
    criteria: list[Any]


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer
Answers = dict[str, Answer]


@runtime_checkable
class Backend(Protocol):
    """Anything that can answer a circuit's questions about a state.

    Implementations live in `decision_circuits.backends` (System One
    HTTP, OpenAI-style logprobs, Anthropic) or in your own code. Return
    one answer per question id, in the wire format above. Raise on
    failure; the circuit does not guess."""

    def answer(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Answers: ...


def option_keys(question: Mapping[str, Any]) -> list[str]:
    """The option ids of a question, in the order they are presented."""
    kind = question["type"]
    if kind == "noul":
        return ["yes", "no"]
    if kind == "choice":
        return list(question["criteria"].keys())
    return [str(i) for i in range(len(question["criteria"]))]


def normalized_confidence(probabilities: Mapping[str, float]) -> float:
    """1 - H(p) / log(N); 1.0 for a point mass, 0.0 for uniform."""
    ps = [max(0.0, float(p)) for p in probabilities.values()]
    n = len(ps)
    if n <= 1:
        return 1.0
    z = sum(ps) or 1.0
    h = -sum((p / z) * math.log(p / z) for p in ps if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(n)))


def answer_from_probabilities(question: Mapping[str, Any], probabilities: Mapping[str, float] | Sequence[float]) -> Answer:
    """Build a wire-format answer from a distribution over a question's
    options. Accepts a mapping keyed by option id or a sequence in option
    order; normalizes; derives choice, score, and confidence."""
    keys = option_keys(question)
    if isinstance(probabilities, Mapping):
        raw = [max(0.0, float(probabilities.get(k, 0.0))) for k in keys]
    else:
        raw = [max(0.0, float(p)) for p in probabilities]
        if len(raw) != len(keys):
            raise ValueError(f"expected {len(keys)} probabilities, got {len(raw)}")
    z = sum(raw)
    ps = [p / z for p in raw] if z > 0 else [1.0 / len(keys)] * len(keys)
    dist = dict(zip(keys, ps, strict=True))
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": dist["yes"]}
    conf = normalized_confidence(dist)
    if kind == "choice":
        best = max(dist, key=dist.get)  # type: ignore[arg-type]
        return {"type": "choice", "choice": best, "probabilities": dist, "confidence": conf}
    expected = sum(i * p for i, p in enumerate(ps))
    return {"type": "score", "score": expected, "probabilities": dist, "confidence": conf}


def to_jsonable(x: Any) -> Any:
    """Turn a state made of SDK objects into plain JSON-able data.

    Chat messages from agent frameworks (objects with `type` and
    `content`) become `{"type", "content"[, "tool_calls"]}`, which is
    what a judgment needs and drops the metadata a full dump would
    carry. Pydantic-style objects use `model_dump(mode="json")`.
    Anything else that json can't encode becomes its `str`. Backends
    call this on the state before sending it, so a circuit can be run
    on raw framework objects."""
    if x is None or isinstance(x, str | int | float | bool):
        return x
    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "content") and hasattr(x, "type"):
        out = {"type": x.type, "content": to_jsonable(x.content)}
        calls = getattr(x, "tool_calls", None)
        if calls:
            out["tool_calls"] = to_jsonable(calls)
        return out
    dump = getattr(x, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except (TypeError, ValueError):
            pass
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)
