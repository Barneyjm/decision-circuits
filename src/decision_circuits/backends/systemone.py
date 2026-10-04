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
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    USER_AGENT = f"decision-circuits/{_version('decision-circuits')}"
except PackageNotFoundError:  # not installed as a distribution (a bare checkout on sys.path)
    USER_AGENT = "decision-circuits"
RETRY_STATUSES = (429, 502, 503, 504, 524, 529)  # rate limited, a model starting up, a platform timeout, or overloaded


def _retry_after(headers: Any) -> float | None:
    """Seconds from a Retry-After header, when it gives a number (the date form is ignored)."""
    try:
        v = headers.get("Retry-After") if headers is not None else None
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


from decision_circuits import tracing
from decision_circuits.types import V1_TYPES, Answers, require_types, to_jsonable


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
        question_types: Sequence[str] | None = None,
    ):
        """`timeout` is per request; `retry_for` is how long to keep retrying
        a 429/502/503/504/524/529 (a hosted model spinning up from zero takes about
        a minute) before raising. 0 disables retries. `question_types` are the
        types the server answers; a question of another type is refused before
        the request. Defaults to noul, choice and score for TypeSafe's API (all it
        documents) and to no check for other servers."""
        self.url = url
        self.model = model
        self.client = client
        self.timeout = timeout
        self.retry_for = retry_for
        self.retry_wait = retry_wait
        if question_types is None and urllib.parse.urlsplit(url).hostname == "api.typesafe.ai":
            question_types = V1_TYPES
        self.question_types = tuple(question_types) if question_types is not None else None
        self.headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT, **(headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.last_response: dict[str, Any] | None = None  # full body of the last call (usage, model)

    def _post(self, body: dict[str, Any], call: tracing.ModelCall | None = None) -> tuple[int, dict[str, Any]]:
        deadline = time.monotonic() + self.retry_for
        wait = self.retry_wait
        while True:
            status, payload, after = self._post_once(body)
            pause = max(wait, after or 0.0)  # a rate limit says how long; back off at least that much
            if status not in RETRY_STATUSES or time.monotonic() + pause > deadline:
                return status, payload
            if call is not None:
                tracing.event(call.span, "decision_circuits.retry", **{"http.response.status_code": status, "decision_circuits.retry.wait_s": round(pause, 3)})
            time.sleep(pause)
            wait = min(wait * 1.5, 30.0)

    def _post_once(self, body: dict[str, Any]) -> tuple[int, dict[str, Any], float | None]:
        """(status, body, Retry-After seconds or None)."""
        if self.client is not None:
            r = self.client.post(self.url, json=body, headers=tracing.inject(dict(self.headers)))
            status, after = getattr(r, "status_code", 200), _retry_after(getattr(r, "headers", None))
            try:
                return status, r.json(), after
            except ValueError:  # a gateway's HTML error page: keep the status, so a 502 is retried
                return status, {"detail": str(getattr(r, "text", ""))[:500]}, after
        data = json.dumps(body, default=to_jsonable).encode()
        req = urllib.request.Request(self.url, data=data, headers=tracing.inject(dict(self.headers)), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode()), None
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read().decode())
            except ValueError:
                payload = {"detail": str(e)}
            return e.code, payload, _retry_after(e.headers)
        except urllib.error.URLError as e:  # timed out or unreachable: retryable like a 504
            return 504, {"detail": str(e)}, None

    def answer(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Answers:
        if self.question_types is not None:
            require_types(questions, self.question_types, f"SystemOne at {urllib.parse.urlsplit(self.url).hostname}", hint="a circuit v2 server")
        model = model or self.model
        host = urllib.parse.urlsplit(self.url).hostname
        provider = "typesafe" if host and host.endswith("typesafe.ai") else "system_one"
        with tracing.model_call(provider, "answer", model, host) as call:
            status, body = self._post({"state": to_jsonable(state), "model": model, "questions": dict(questions)}, call)
            if status >= 400 or "answers" not in body:
                call.failed(str(status))
                raise SystemOneError(status, body)
            usage = body.get("usage") or {}
            call.response(
                model=body.get("model"), response_id=body.get("request_id"), input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens")
            )
        self.last_response = body
        return body["answers"]
