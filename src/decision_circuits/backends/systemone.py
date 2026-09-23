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
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

USER_AGENT = "decision-circuits/0.4"
RETRY_STATUSES = (502, 503, 504, 524)  # a model starting up, or a platform timeout in front of it

from decision_circuits import tracing
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
        timeout: float = 120.0,
        headers: Mapping[str, str] | None = None,
        retry_for: float = 240.0,
        retry_wait: float = 5.0,
    ):
        """`timeout` is per request; `retry_for` is how long to keep retrying
        a 502/503/504/524 (a hosted model spinning up from zero takes about
        a minute) before raising. 0 disables retries."""
        self.url = url
        self.model = model
        self.client = client
        self.timeout = timeout
        self.retry_for = retry_for
        self.retry_wait = retry_wait
        self.headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT, **(headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.last_response: dict[str, Any] | None = None  # full body of the last call (usage, model)

    def _post(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        deadline = time.monotonic() + self.retry_for
        wait = self.retry_wait
        while True:
            status, payload = self._post_once(body)
            if status not in RETRY_STATUSES or time.monotonic() + wait > deadline:
                return status, payload
            time.sleep(wait)
            wait = min(wait * 1.5, 30.0)

    def _post_once(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if self.client is not None:
            r = self.client.post(self.url, json=body, headers=tracing.inject(dict(self.headers)))
            status = getattr(r, "status_code", 200)
            try:
                return status, r.json()
            except ValueError:  # a gateway's HTML error page: keep the status, so a 502 is retried
                return status, {"detail": str(getattr(r, "text", ""))[:500]}
        data = json.dumps(body, default=to_jsonable).encode()
        req = urllib.request.Request(self.url, data=data, headers=tracing.inject(dict(self.headers)), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read().decode())
            except ValueError:
                payload = {"detail": str(e)}
            return e.code, payload
        except urllib.error.URLError as e:  # timed out or unreachable: retryable like a 504
            return 504, {"detail": str(e)}

    def answer(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Answers:
        status, body = self._post({"state": to_jsonable(state), "model": model or self.model, "questions": dict(questions)})
        if status >= 400 or "answers" not in body:
            raise SystemOneError(status, body)
        self.last_response = body
        return body["answers"]
