"""A System One HTTP server as a backend: TypeSafe's Jev, or any server
speaking `POST /v1/systemone` (s1proto).

Uses the standard library's urllib by default, so it needs no extra
dependency. Pass `client=` to use httpx or requests instead (anything
with `.post(url, json=..., headers=...)` returning `.json()`).

    from decision_circuits.backends import SystemOne
    jev = SystemOne("https://api.typesafe.ai/v1/systemone", api_key=KEY, model="jev-latest")
    out = circuit.run(jev, state)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from decision_circuits.types import Answers, to_jsonable


class SystemOneError(RuntimeError):
    def __init__(self, status: int, body: Any):
        super().__init__(f"System One server returned {status}: {str(body)[:300]}")
        self.status = status
        self.body = body


class SystemOne:
    def __init__(
        self,
        url: str = "https://api.typesafe.ai/v1/systemone",
        api_key: str | None = None,
        model: str = "jev-latest",
        client: Any = None,
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ):
        self.url = url
        self.model = model
        self.client = client
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.last_response: dict[str, Any] | None = None  # full body of the last call (usage, model)
        self._server_gates: bool | None = None  # learned on first use: does this server evaluate gates?

    def _post(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if self.client is not None:
            r = self.client.post(self.url, json=body, headers=self.headers)
            return getattr(r, "status_code", 200), r.json()
        data = json.dumps(body, default=to_jsonable).encode()
        req = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read().decode())
            except ValueError:
                payload = {"detail": str(e)}
            return e.code, payload

    def answer(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Answers:
        status, body = self._post({"state": to_jsonable(state), "model": model or self.model, "questions": dict(questions)})
        if status >= 400 or "answers" not in body:
            raise SystemOneError(status, body)
        self.last_response = body
        return body["answers"]

    def answer_with_gates(
        self, state: Any, questions: Mapping[str, Any], gates: Mapping[str, Any], *, model: str | None = None
    ) -> tuple[Answers, dict[str, Any] | None]:
        """Send the circuit's gates too. A server that evaluates circuits
        (s1proto) returns `gates`; one that rejects the extra field or
        ignores it (TypeSafe today) gets the plain request instead, and
        the answer is (answers, None) so the circuit evaluates locally.
        Which kind the server is gets remembered after the first call."""
        if self._server_gates is not False:
            status, body = self._post({"state": to_jsonable(state), "model": model or self.model, "questions": dict(questions), "gates": dict(gates)})
            if status < 400 and "answers" in body and "gates" in body:
                self._server_gates = True
                self.last_response = body
                return body["answers"], body["gates"]
            if status < 400 and "answers" in body:  # ignored the field: a plain server, no second request needed
                self._server_gates = False
                self.last_response = body
                return body["answers"], None
            if status not in (400, 422):  # auth, rate limit, outage: report it, learn nothing
                raise SystemOneError(status, body)
            self._server_gates = False  # rejected the field: a plain server
        return self.answer(state, questions, model=model), None
