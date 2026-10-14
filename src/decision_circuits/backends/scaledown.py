"""ScaleDown's Decisions API as a backend: Jev's request and response at
`POST https://api.scaledown.xyz/v1/scaledown`, answered by their classify model.

    from decision_circuits.backends import ScaleDown
    sd = ScaleDown(api_key=KEY)
    out = circuit.run(sd, state)

It differs from Jev in what it reads, and this backend bridges the gaps:

  * Only `state.text` is read, so a structured state is sent as indented JSON text, which
    keeps the field names a question points at (`order.prior_refunds_12mo`).
  * A choice option needs a description; an option without one is described by its name.
  * It answers noul, choice and score; other types are refused before the request.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from decision_circuits.backends.systemone import SystemOne
from decision_circuits.types import V1_TYPES, to_jsonable


def state_text(state: Any) -> str:
    """The text ScaleDown reads: a string as is, anything else as indented JSON."""
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping) and set(state) == {"text"} and isinstance(state["text"], str):
        return state["text"]
    return json.dumps(to_jsonable(state), indent=2, ensure_ascii=False, default=str)


class ScaleDown(SystemOne):
    def __init__(self, api_key: str | None = None, model: str = "classify-1", url: str = "https://api.scaledown.xyz/v1/scaledown", **kwargs: Any):
        """`model` is "classify" (latest) or "classify-1" (pinned). Other keyword
        arguments are SystemOne's: `client`, `timeout`, `retry_for`, `retry_wait`, `headers`."""
        kwargs.setdefault("question_types", V1_TYPES)
        super().__init__(url, api_key=api_key, model=model, **kwargs)

    def _auth(self, api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key}

    def _provider(self, host: str | None) -> str:
        return "scaledown"

    def _body(self, state: Any, questions: Mapping[str, Any], model: str) -> dict[str, Any]:
        sent = {}
        for qid, q in questions.items():
            if q.get("type") == "choice" and isinstance(q.get("criteria"), Mapping):
                q = {**q, "criteria": {k: v if v else str(k).replace("_", " ") for k, v in q["criteria"].items()}}
            sent[qid] = q
        return {"state": {"text": state_text(state)}, "model": model, "questions": sent}
