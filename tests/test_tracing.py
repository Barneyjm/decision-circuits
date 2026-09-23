import json

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from decision_circuits import Circuit, Q, argmax, drop, tracing
from decision_circuits.backends import SystemOne

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)

SECRET = "card 4532 0151 1283 0366"


class Backend:
    model = "m1"

    def __init__(self):
        self.last_response = {"model": "circuit-1.7b@v2.0", "request_id": "r-42", "usage": {"input_tokens": 57}}

    def answer(self, state, questions, *, model=None):
        return {
            "refund": {"type": "noul", "noul": 0.9 if "receipt" in str(state) else 0.55},
            "desk": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "tech": 0.2}, "confidence": 0.3},
        }


def circuit():
    c = Circuit()
    c.noul("refund", "Refund owed?")
    c.choice("desk", "Which desk?", {"billing": None, "tech": None})
    c.gate("pay", Q("refund") >= 0.6, band=0.1, on_uncertain="escalate")
    c.gate("route", argmax("desk"))
    return c


@pytest.fixture(autouse=True)
def fresh():
    EXPORTER.clear()


def spans(name):
    return [s for s in EXPORTER.get_finished_spans() if s.name == name]


def test_a_run_is_a_span_with_its_model_call_answers_and_gates_but_not_the_state():
    circuit().run(Backend(), {"message": SECRET})
    (run,) = spans("decision_circuits.run")
    (call,) = spans("decision_circuits.backend")
    assert call.parent.span_id == run.context.span_id
    a = run.attributes
    assert a["gen_ai.request.model"] == "m1" and a["gen_ai.response.id"] == "r-42" and a["gen_ai.usage.input_tokens"] == 57
    assert list(a["decision_circuits.questions"]) == ["refund", "desk"] and list(a["decision_circuits.escalated"]) == ["pay"]
    gates = {e.attributes["gate"]: e.attributes for e in run.events if e.name == "decision_circuits.gate"}
    assert gates["pay"]["outcome"] == "escalate" and gates["route"]["value"] == "billing"
    picks = {e.attributes["question"]: e.attributes["pick"] for e in run.events if e.name == "decision_circuits.answer"}
    assert picks == {"refund": "yes", "desk": "billing"}
    everything = json.dumps([dict(s.attributes) for s in EXPORTER.get_finished_spans()] + [dict(e.attributes) for e in run.events], default=str)
    assert "4532" not in everything and "message" not in everything


def test_interventions_nest_their_runs_even_across_threads():
    circuit().intervene(Backend(), {"message": "hi", "receipt": "#1"}, {"no receipt": drop("receipt")}, workers=2)
    (top,) = spans("decision_circuits.intervene")
    runs = spans("decision_circuits.run")
    assert len(runs) == 2 and all(r.parent.span_id == top.context.span_id for r in runs)
    (ev,) = [e for e in top.events if e.name == "decision_circuits.intervention"]
    assert ev.attributes["name"] == "no receipt" and list(ev.attributes["flipped"]) == ["pay"]


def test_the_http_backend_sends_traceparent():
    seen = {}

    class Client:
        def post(self, url, json, headers):
            seen.update(headers)

            class R:
                status_code = 200

                def json(self):
                    return {"answers": {"refund": {"type": "noul", "noul": 0.9}}}

            return R()

    c = Circuit()
    c.noul("refund", "Refund owed?")
    with trace.get_tracer("t").start_as_current_span("caller") as caller:
        c.run(SystemOne("http://x/v1/systemone", client=Client()), "text")
    assert seen["traceparent"].split("-")[1] == format(caller.get_span_context().trace_id, "032x")


def test_without_opentelemetry_nothing_is_recorded_and_nothing_breaks(monkeypatch):
    monkeypatch.setattr(tracing, "trace", None)
    monkeypatch.setattr(tracing, "propagate", None)
    out = circuit().run(Backend(), "text")
    assert out["gates"]["route"]["value"] == "billing" and not EXPORTER.get_finished_spans()
    assert tracing.inject({"a": "b"}) == {"a": "b"}


def test_mixed_type_lists_are_kept_as_strings():
    assert tracing._attr([1, "vip"]) == ["1", "vip"] and tracing._attr(["a", "b"]) == ["a", "b"] and tracing._attr([1, 2]) == [1, 2]
