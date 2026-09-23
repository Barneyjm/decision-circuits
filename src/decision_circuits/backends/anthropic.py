"""Claude as a backend. The Messages API exposes no token
probabilities, so there are two honest ways to get a distribution, and
the trace of every gate records which one produced its number:

  mode="stated"   One call. Claude answers every question at once through
                  a tool whose schema is the answers map, giving a
                  probability per option. Fast and cheap; the numbers are
                  the model's stated confidence, which is not calibrated
                  the way a System One model's output is. Set thresholds
                  with that in mind, or fit a calibration on a labeled
                  sample first.
  mode="sampled"  k calls at temperature 1, each forced to pick one option
                  per question through the same tool. The distribution is
                  the vote frequency (with add-one smoothing). Slower and
                  k times the cost; an empirical distribution rather than
                  a self-report.

Requires the `anthropic` package (`pip install decision-circuits[anthropic]`)
or any client exposing `messages.create`.

    from decision_circuits.backends import Anthropic
    backend = Anthropic(model="claude-sonnet-5")                    # ANTHROPIC_API_KEY from the environment
    backend = Anthropic(model="claude-haiku-4-5-20251001", mode="sampled", k=7)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from decision_circuits.types import V1_TYPES, Answers, answer_from_probabilities, option_keys, require_types, to_jsonable

TOOL_NAME = "submit_answers"


def _schema(questions: Mapping[str, Any], *, pick_one: bool) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for qid, q in questions.items():
        keys = option_keys(q)
        if pick_one:
            props[qid] = {"type": "string", "enum": keys, "description": str(q["instructions"])}
        else:
            props[qid] = {
                "type": "object",
                "description": f"{q['instructions']} Give a probability for every option; they should sum to 1.",
                "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in keys},
                "required": keys,
                "additionalProperties": False,
            }
    return {"type": "object", "properties": props, "required": list(questions), "additionalProperties": False}


def render_prompt(state: Any, questions: Mapping[str, Any]) -> str:
    parts = [
        "Consider the following state and answer every question about it.",
        "",
        "State:",
        state if isinstance(state, str) else json.dumps(to_jsonable(state), ensure_ascii=False, indent=1),
        "",
    ]
    for qid, q in questions.items():
        parts.append(f"Question `{qid}`: {q['instructions']}")
        crit = q.get("criteria")
        keys = option_keys(q)
        if q["type"] == "noul":
            c = crit or {}
            parts.append(f"  yes: {c.get('true') or 'Yes'}")
            parts.append(f"  no: {c.get('false') or 'No'}")
        elif q["type"] == "choice":
            for k in keys:
                d = crit.get(k) if isinstance(crit, Mapping) else None
                parts.append(f"  {k}: {d}" if d else f"  {k}")
        else:
            for k, level in zip(keys, crit or [], strict=False):
                parts.append(f"  {k}: {level}")
        parts.append("")
    parts.append(f"Report your answers by calling the `{TOOL_NAME}` tool.")
    return "\n".join(parts)


class Anthropic:
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        client: Any = None,
        api_key: str | None = None,
        mode: str = "stated",
        k: int = 5,
        max_tokens: int = 2048,
        workers: int = 8,
        system: str = "You are a careful, well-calibrated classifier. When asked for probabilities, spread mass honestly across plausible options; do not round to 0 or 1 unless certain.",
    ):
        if mode not in ("stated", "sampled"):
            raise ValueError("mode must be 'stated' or 'sampled'")
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError("pip install 'decision-circuits[anthropic]' or pass client=") from e
            client = anthropic.Anthropic(api_key=api_key)
        self.client = client
        self.model = model
        self.mode = mode
        self.k = k
        self.max_tokens = max_tokens
        self.workers = workers
        self.system = system

    def _call(self, state: Any, questions: Mapping[str, Any], model: str | None, *, pick_one: bool, temperature: float) -> dict[str, Any]:
        r = self.client.messages.create(
            model=model or self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=self.system,
            tools=[{"name": TOOL_NAME, "description": "Submit answers to every question.", "input_schema": _schema(questions, pick_one=pick_one)}],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": render_prompt(state, questions)}],
        )
        for block in r.content:
            if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
                return dict(block.input)
        raise RuntimeError("Claude did not call the answers tool")

    def answer(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Answers:
        require_types(questions, V1_TYPES, "Anthropic")
        if self.mode == "stated":
            raw = self._call(state, questions, model, pick_one=False, temperature=0)
            return {qid: answer_from_probabilities(q, raw[qid]) for qid, q in questions.items()}
        with ThreadPoolExecutor(max_workers=min(self.workers, self.k)) as pool:
            picks = list(pool.map(lambda _: self._call(state, questions, model, pick_one=True, temperature=1.0), range(self.k)))
        out: Answers = {}
        for qid, q in questions.items():
            keys = option_keys(q)
            counts = {k: 1.0 for k in keys}  # add-one smoothing: k samples never claim certainty
            for p in picks:
                if p.get(qid) in counts:
                    counts[p[qid]] += 1.0
            out[qid] = answer_from_probabilities(q, counts)
        return out
