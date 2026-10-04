import json

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from decision_circuits import Circuit, Q, argmax, drop, tracing
from decision_circuits.backends import SystemOne

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)
METRICS = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[METRICS]))

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


# ---- every fundamental call leaves a fact behind ------------------------------------------


def points(name):
    """Every data point of a metric so far, as (attributes, value): a sum's value, a histogram's count and sum."""
    out = []
    data = METRICS.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    for p in m.data.data_points:
                        out.append((dict(p.attributes), getattr(p, "value", None) or (p.count, p.sum)))
    return out


def no_state_anywhere():
    for sp in EXPORTER.get_finished_spans():
        blob = json.dumps({"a": dict(sp.attributes), "e": [dict(e.attributes) for e in sp.events]}, default=str)
        assert "4532" not in blob, sp.name


def test_a_system_one_call_is_a_genai_client_span_with_duration_tokens_and_retries(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr("decision_circuits.backends.systemone.time.sleep", lambda s: None)
    replies = [
        SimpleNamespace(status_code=529, headers={}, json=lambda: {"detail": "overloaded"}),
        SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {
                "model": "jev-1.13.0",
                "request_id": "q-1",
                "usage": {"input_tokens": 300, "output_tokens": 40},
                "answers": {
                    "refund": {"type": "noul", "noul": 0.9},
                    "desk": {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.1, "tech": 0.9}, "confidence": 0.8},
                },
            },
        ),
    ]

    class Client:
        def post(self, url, json=None, headers=None):
            return replies.pop(0)

    circuit().run(SystemOne(client=Client(), retry_wait=0.01), {"message": SECRET})
    (call,) = spans("answer jev-latest")
    a = call.attributes
    assert call.kind == trace.SpanKind.CLIENT
    assert a["gen_ai.provider.name"] == "typesafe" and a["gen_ai.operation.name"] == "answer" and a["server.address"] == "api.typesafe.ai"
    assert a["gen_ai.response.model"] == "jev-1.13.0" and a["gen_ai.usage.input_tokens"] == 300 and a["gen_ai.usage.output_tokens"] == 40
    (retry,) = [e for e in call.events if e.name == "decision_circuits.retry"]
    assert retry.attributes["http.response.status_code"] == 529
    assert call.parent.span_id == spans("decision_circuits.backend")[-1].context.span_id
    durations = [p for p in points("gen_ai.client.operation.duration") if p[0].get("gen_ai.request.model") == "jev-latest"]
    assert durations and durations[-1][0]["gen_ai.response.model"] == "jev-1.13.0"
    tokens = {p[0]["gen_ai.token.type"]: p[1] for p in points("gen_ai.client.token.usage") if p[0].get("gen_ai.request.model") == "jev-latest"}
    assert tokens["input"][1] >= 300 and tokens["output"][1] >= 40
    no_state_anywhere()


def test_a_failed_model_call_names_its_error_type():
    from types import SimpleNamespace

    from decision_circuits.backends import SystemOneError

    class Client:
        def post(self, url, json=None, headers=None):
            return SimpleNamespace(status_code=401, headers={}, json=lambda: {"detail": "bad key"})

    with pytest.raises(SystemOneError):
        circuit().run(SystemOne("https://example.test/v1/systemone", client=Client(), retry_for=0), "s")
    (call,) = spans("answer jev-latest")
    assert call.attributes["error.type"] == "401" and call.attributes["gen_ai.provider.name"] == "system_one"
    assert any(p[0].get("error.type") == "401" for p in points("gen_ai.client.operation.duration"))


def test_openai_and_anthropic_calls_are_client_spans_under_the_run_even_from_a_thread_pool():
    from types import SimpleNamespace

    from decision_circuits.backends import Anthropic, OpenAILogprobs
    from decision_circuits.backends.anthropic import TOOL_NAME

    def openai_create(**kw):
        top = [SimpleNamespace(token="A", logprob=-0.1), SimpleNamespace(token="B", logprob=-2.0)]
        return SimpleNamespace(
            model="gpt-x-2026",
            id="cmpl-1",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=1),
            choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)]))],
        )

    def anthropic_create(**kw):
        payload = {"refund": {"yes": 0.9, "no": 0.1}, "desk": {"billing": 0.2, "tech": 0.8}}
        return SimpleNamespace(
            model="claude-x",
            id="msg-1",
            usage=SimpleNamespace(input_tokens=80, output_tokens=20),
            content=[SimpleNamespace(type="tool_use", name=TOOL_NAME, input=payload)],
        )

    openai = OpenAILogprobs(model="gpt-x", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=openai_create))))
    claude = Anthropic(model="claude-x", client=SimpleNamespace(messages=SimpleNamespace(create=anthropic_create)))
    for backend, name, provider in ((openai, "chat gpt-x", "openai"), (claude, "chat claude-x", "anthropic")):
        EXPORTER.clear()
        circuit().run(backend, {"message": SECRET})
        (outer,) = spans("decision_circuits.backend")
        calls = spans(name)
        assert calls and all(c.parent.span_id == outer.context.span_id for c in calls), name  # the pool kept the context
        assert {c.attributes["gen_ai.provider.name"] for c in calls} == {provider}
        assert all("gen_ai.usage.input_tokens" in c.attributes for c in calls)
        no_state_anywhere()


def test_evaluate_on_its_own_is_a_span_and_every_gate_result_counts_with_its_chip():
    from decision_circuits import G
    from decision_circuits.parts import refund_risk

    c = Circuit()
    c.mount(refund_risk(), "risk")
    c.route("queue", [("fraud", G("risk.action")["hold"])], otherwise="auto")
    answers = {
        f"risk.{q}": {"type": "noul", "noul": 0.95} for q in ("new_account", "high_value", "repeat_refunds", "story", "pressure", "other_payee", "vague")
    }
    c.evaluate(answers)
    (ev,) = spans("decision_circuits.evaluate")
    gates = {e.attributes["gate"]: e.attributes for e in ev.events if e.name == "decision_circuits.gate"}
    assert gates["risk.action"]["chip"] == "refund_risk" and gates["risk.action"]["chip_version"] == "1.0"
    assert list(gates["risk.action"]["rules"]) == ["held", "skipped"] and list(gates["queue"]["rules"]) == ["held"]
    assert "chip" not in gates["queue"]  # the host's own gate
    counted = [p for p in points("decision_circuits.gate.results") if p[0]["decision_circuits.gate"] == "risk.action"]
    assert any(p[0]["decision_circuits.value"] == "hold" and p[0]["decision_circuits.chip"] == "refund_risk" for p in counted)


def test_a_middleware_judgment_is_a_span_with_its_action_and_counts():
    from decision_circuits.integrations import CircuitPolicy

    p = CircuitPolicy(circuit(), Backend(), gate="pay")
    assert p.judge({"message": SECRET + " receipt"}).action == "block"  # pay=True -> block by default
    (pol,) = spans("decision_circuits.policy")
    assert pol.attributes["decision_circuits.action"] == "block" and pol.attributes["decision_circuits.result"] == "True"
    assert spans("decision_circuits.run")[-1].parent.span_id == pol.context.span_id
    assert any(
        p[0] == {"decision_circuits.gate": "pay", "decision_circuits.result": "True", "decision_circuits.action": "block"}
        for p in points("decision_circuits.policy.actions")
    )
    no_state_anywhere()
