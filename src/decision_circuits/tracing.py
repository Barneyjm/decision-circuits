"""OpenTelemetry spans for circuit runs, when `opentelemetry-api` is installed; nothing otherwise.

    pip install "decision-circuits[otel]"      # plus an SDK and exporter of your choice

Each `Circuit.run` is a span, `decision_circuits.run`. Its child `decision_circuits.backend`
covers the model call. The run span carries an event per answer (the pick and its
probability) and per gate (value, outcome, probability, trace), and lists the gates that
escalated or abstained, so "which decisions went to a person today" is a query on your
traces. `intervene` and `ablate` are a `decision_circuits.intervene` span over one run span
per edited state. `SystemOne` sends the W3C `traceparent` header, so a server that traces
joins the same trace.

Every model request a backend makes is its own client span, `{operation} {model}`, and feeds
the OpenTelemetry GenAI metrics `gen_ai.client.operation.duration` and
`gen_ai.client.token.usage`, with their standard attributes. A SystemOne retry is an event on
that span. `Circuit.evaluate` on its own is a `decision_circuits.evaluate` span. Each gate
result also counts on `decision_circuits.gate.results` (gate, outcome, value, and the chip it
came from), and a middleware judgment is a `decision_circuits.policy` span and counts on
`decision_circuits.policy.actions`. These are facts, not verdicts: what is a good escalation
rate, or an alert, is yours to decide from them.

The state is never recorded: it is where personal data lives. Question ids, option names,
probabilities and gate traces are. Model attributes follow the OpenTelemetry GenAI
conventions (`gen_ai.request.model`, `gen_ai.response.id`, `gen_ai.usage.input_tokens`).
Nothing is configured here: bring your own SDK, exporters and views.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import metrics, propagate, trace
except ImportError:  # the core stays dependency-free
    trace = propagate = metrics = None  # type: ignore[assignment]

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
    if isinstance(v, list | tuple):  # one element type or OpenTelemetry drops the attribute
        kinds = {type(x) for x in v}
        return list(v) if len(kinds) <= 1 and kinds <= {str, bool, int, float} else [str(x) for x in v]
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


_INSTRUMENTS: dict[str, Any] = {}


def _instrument(kind: str, name: str, unit: str, description: str) -> Any:
    """A histogram or counter, made once per name; None without OpenTelemetry."""
    if metrics is None:
        return None
    if name not in _INSTRUMENTS:
        from decision_circuits import __version__

        meter = metrics.get_meter(SCOPE, __version__)
        make = meter.create_histogram if kind == "histogram" else meter.create_counter
        _INSTRUMENTS[name] = make(name, unit=unit, description=description)
    return _INSTRUMENTS[name]


def count(name: str, /, description: str = "", **attrs: Any) -> None:
    """Add one to a counter, under these attributes."""
    inst = _instrument("counter", name, "{" + name.rsplit(".", 1)[-1] + "}", description)
    if inst is not None:
        inst.add(1, _clean(attrs))


class ModelCall:
    """What a backend learns about one model request, for its span and the GenAI metrics."""

    def __init__(self, span: Any, attrs: dict[str, Any]):
        self.span = span
        self.attrs = attrs
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def response(self, *, model: str | None = None, response_id: str | None = None, input_tokens: Any = None, output_tokens: Any = None) -> None:
        if model:
            self.attrs["gen_ai.response.model"] = model
            self.span.set_attribute("gen_ai.response.model", model)
        if response_id:
            self.span.set_attribute("gen_ai.response.id", response_id)
        if input_tokens is not None:
            self.input_tokens = int(input_tokens)
            self.span.set_attribute("gen_ai.usage.input_tokens", self.input_tokens)
        if output_tokens is not None:
            self.output_tokens = int(output_tokens)
            self.span.set_attribute("gen_ai.usage.output_tokens", self.output_tokens)

    def failed(self, error_type: str) -> None:
        """A request that ended in an error the backend raises, named as `error.type` wants."""
        self.attrs["error.type"] = error_type
        self.span.set_attribute("error.type", error_type)


@contextmanager
def model_call(provider: str, operation: str, model: str | None, server: str | None = None) -> Iterator[ModelCall]:
    """One model request: a client span named `{operation} {model}` and the GenAI
    `gen_ai.client.operation.duration` and `gen_ai.client.token.usage` measurements."""
    attrs = _clean({"gen_ai.operation.name": operation, "gen_ai.provider.name": provider, "gen_ai.request.model": model, "server.address": server})
    start = time.perf_counter()
    if trace is None:
        call = ModelCall(_NoSpan(), attrs)
        yield call
        return
    from decision_circuits import __version__

    tracer = trace.get_tracer(SCOPE, __version__)
    with tracer.start_as_current_span(f"{operation} {model}".strip(), kind=trace.SpanKind.CLIENT, attributes=attrs) as s:
        call = ModelCall(s, attrs)
        try:
            yield call
        except Exception as e:
            call.attrs.setdefault("error.type", type(e).__qualname__)
            raise
        finally:
            duration = _instrument("histogram", "gen_ai.client.operation.duration", "s", "GenAI operation duration")
            if duration is not None:
                duration.record(time.perf_counter() - start, call.attrs)
            tokens = _instrument("histogram", "gen_ai.client.token.usage", "{token}", "Number of input and output tokens used")
            if tokens is not None:
                for kind, n in (("input", call.input_tokens), ("output", call.output_tokens)):
                    if n is not None:
                        tokens.record(n, {**call.attrs, "gen_ai.token.type": kind})


def inject(headers: dict[str, str]) -> dict[str, str]:
    """The same headers plus `traceparent` (and `tracestate`) for the current span."""
    if propagate is not None:
        propagate.inject(headers)
    return headers


def in_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """`fn` bound to the caller's context, so work sent to a thread pool stays in its trace."""
    ctx = contextvars.copy_context()
    return lambda *a, **kw: ctx.copy().run(fn, *a, **kw)
