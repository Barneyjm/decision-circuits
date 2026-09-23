"""OpenTelemetry spans for circuit runs, when `opentelemetry-api` is installed; nothing otherwise.

    pip install "decision-circuits[otel]"      # plus an SDK and exporter of your choice

Each `Circuit.run` is a span, `decision_circuits.run`. Its child `decision_circuits.backend`
covers the model call. The run span carries an event per answer (the pick and its
probability) and per gate (value, outcome, probability, trace), and lists the gates that
escalated or abstained, so "which decisions went to a person today" is a query on your
traces. `intervene` and `ablate` are a `decision_circuits.intervene` span over one run span
per edited state. `SystemOne` sends the W3C `traceparent` header, so a server that traces
joins the same trace.

The state is never recorded: it is where personal data lives. Question ids, option names,
probabilities and gate traces are. Model attributes follow the OpenTelemetry GenAI
conventions (`gen_ai.request.model`, `gen_ai.response.id`, `gen_ai.usage.input_tokens`).
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import propagate, trace
except ImportError:  # the core stays dependency-free
    trace = propagate = None  # type: ignore[assignment]

SCOPE = "decision-circuits"


class _NoSpan:
    def set_attribute(self, key: str, value: Any) -> None: ...
    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None: ...


def enabled() -> bool:
    return trace is not None


def _attr(v: Any) -> Any:
    """OpenTelemetry takes str, bool, int, float, or lists of one of them; None is left out."""
    if v is None or isinstance(v, str | bool | int | float):
        return v
    if isinstance(v, list | tuple) and all(isinstance(x, str | bool | int | float) for x in v):
        return list(v)
    return str(v)


def _clean(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _attr(v) for k, v in attrs.items() if v is not None}


@contextmanager
def span(name: str, /, **attrs: Any) -> Iterator[Any]:
    """A span under the current one, or a stand-in that records nothing."""
    if trace is None:
        yield _NoSpan()
        return
    from decision_circuits import __version__

    with trace.get_tracer(SCOPE, __version__).start_as_current_span(name, attributes=_clean(attrs)) as s:
        yield s


def event(s: Any, name: str, /, **attrs: Any) -> None:
    s.add_event(name, _clean(attrs))


def inject(headers: dict[str, str]) -> dict[str, str]:
    """The same headers plus `traceparent` (and `tracestate`) for the current span."""
    if propagate is not None:
        propagate.inject(headers)
    return headers


def in_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """`fn` bound to the caller's context, so work sent to a thread pool stays in its trace."""
    ctx = contextvars.copy_context()
    return lambda *a, **kw: ctx.copy().run(fn, *a, **kw)
