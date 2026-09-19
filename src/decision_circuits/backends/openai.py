"""Any OpenAI-compatible chat API with logprobs as a backend: OpenAI,
Fireworks, Together, vLLM, and so on.

Each question is scored by prefill: the model is shown the state, the
question, and lettered options, then asked to answer with one letter.
The next-token log-probabilities over the letters become the
distribution. That is a measured probability, not a stated one, and it
is one short request per question (parallel across questions).

Limits: options are lettered A..T (20), the `top_logprobs` cap on most
providers. Requires the `openai` package (`pip install
decision-circuits[openai]`) or any client exposing
`chat.completions.create` with `logprobs=True`.

    from decision_circuits.backends import OpenAILogprobs
    backend = OpenAILogprobs(model="gpt-4.1-mini")            # OPENAI_API_KEY from the environment
    backend = OpenAILogprobs(model="accounts/fireworks/models/qwen3-8b",
                             base_url="https://api.fireworks.ai/inference/v1", api_key=FW_KEY)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from decision_circuits.types import Answers, answer_from_probabilities, option_keys

LETTERS = "ABCDEFGHIJKLMNOPQRST"


def render_question(state: Any, question: Mapping[str, Any]) -> tuple[str, list[str]]:
    """(prompt text, option ids in lettered order)."""
    keys = option_keys(question)
    if len(keys) > len(LETTERS):
        raise ValueError(f"logprob scoring supports at most {len(LETTERS)} options, got {len(keys)}")
    kind = question["type"]
    crit = question.get("criteria")
    lines = []
    if kind == "noul":
        c = crit or {}
        lines = [f"A. Yes{': ' + c['true'] if c.get('true') else ''}", f"B. No{': ' + c['false'] if c.get('false') else ''}"]
    elif kind == "choice":
        for letter, k in zip(LETTERS, keys, strict=False):
            desc = crit.get(k) if isinstance(crit, Mapping) else None
            lines.append(f"{letter}. {k}{': ' + str(desc) if desc else ''}")
    else:
        for letter, level in zip(LETTERS, crit or [], strict=False):
            lines.append(f"{letter}. {level}")
    state_text = state if isinstance(state, str) else _json(state)
    prompt = (
        f"State:\n{state_text}\n\nQuestion: {question['instructions']}\n\nOptions:\n"
        + "\n".join(lines)
        + "\n\nAnswer with the single letter of the best option."
    )
    return prompt, keys


def _json(x: Any) -> str:
    import json

    return json.dumps(x, ensure_ascii=False, indent=1)


class OpenAILogprobs:
    def __init__(
        self,
        model: str,
        client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system: str = "You are a careful classifier. Reply with exactly one letter.",
        top_logprobs: int = 20,
        workers: int = 8,
        temperature: float = 1.0,
    ):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("pip install 'decision-circuits[openai]' or pass client=") from e
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.system = system
        self.top_logprobs = top_logprobs
        self.workers = workers
        self.temperature = temperature

    def score_one(self, state: Any, question: Mapping[str, Any], model: str | None = None) -> dict[str, float]:
        prompt, keys = render_question(state, question)
        r = self.client.chat.completions.create(
            model=model or self.model,
            messages=[{"role": "system", "content": self.system}, {"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=min(self.top_logprobs, 20),
        )
        top = r.choices[0].logprobs.content[0].top_logprobs
        lp: dict[str, float] = {}
        for t in top:
            tok = t.token.strip().upper()
            if len(tok) == 1 and tok in LETTERS[: len(keys)]:
                lp[tok] = max(lp.get(tok, -math.inf), t.logprob)
        if not lp:
            raise RuntimeError(f"no option letter among top logprobs: {[t.token for t in top][:10]}")
        m = max(lp.values())
        weights = {k: math.exp((lp.get(letter, -math.inf) - m) / self.temperature) for letter, k in zip(LETTERS, keys, strict=False)}
        return weights

    def answer(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Answers:
        ids = list(questions)
        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, len(ids)))) as pool:
            dists = list(pool.map(lambda qid: self.score_one(state, questions[qid], model), ids))
        return {qid: answer_from_probabilities(questions[qid], d) for qid, d in zip(ids, dists, strict=True)}
