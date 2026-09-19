"""Backends are tested against fakes that mimic each SDK's response
objects; no network, no provider packages."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from decision_circuits import Circuit, Q, answer_from_probabilities, argmax, normalized_confidence
from decision_circuits.backends import SystemOne, SystemOneError
from decision_circuits.backends.anthropic import TOOL_NAME, Anthropic, render_prompt
from decision_circuits.backends.openai import OpenAILogprobs, render_question
from decision_circuits.langgraph import as_node, route_on
from decision_circuits.types import Backend

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is this urgent?", "criteria": {"true": "Needs action today", "false": "Can wait"}},
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Money", "technical": "Bugs", "other": None}},
    "sev": {"type": "score", "instructions": "How severe?", "criteria": ["Low", "Medium", "High"]},
}


def circuit() -> Circuit:
    c = Circuit()
    c.questions.update(QUESTIONS)
    c.gate("rush", Q("urgent") >= 0.5, on_uncertain="escalate")
    c.gate("route", argmax("dept", min_confidence=0.2))
    return c


# --- types -----------------------------------------------------------------


def test_answer_from_probabilities_all_kinds():
    a = answer_from_probabilities(QUESTIONS["urgent"], {"yes": 3, "no": 1})
    assert a == {"type": "noul", "noul": 0.75}
    b = answer_from_probabilities(QUESTIONS["dept"], [0.2, 0.7, 0.1])
    assert b["choice"] == "technical" and abs(sum(b["probabilities"].values()) - 1) < 1e-9 and 0 < b["confidence"] < 1
    c = answer_from_probabilities(QUESTIONS["sev"], {"0": 0, "1": 0, "2": 1})
    assert c["score"] == 2.0 and c["confidence"] == 1.0
    assert normalized_confidence({"a": 1, "b": 1}) == 0.0


def test_any_object_with_answer_is_a_backend():
    class Mine:
        def answer(self, state, questions, *, model=None):
            return {qid: answer_from_probabilities(q, [1.0] * len(q["criteria"]) if q["type"] != "noul" else [0.9, 0.1]) for qid, q in questions.items()}

    assert isinstance(Mine(), Backend)
    out = circuit().run(Mine(), "hello")
    assert out["gates_evaluated_by"] == "client" and out["gates"]["rush"]["value"] is True
    assert out["gates"]["route"]["outcome"] == "abstain"  # uniform -> confidence 0 < 0.2


# --- SystemOne -------------------------------------------------------------


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


def test_systemone_backend_with_injected_client():
    calls = []

    class C:
        def post(self, url, json=None, headers=None):
            calls.append((url, json, headers))
            return _Resp(200, {"model": "jev-1.13.0", "answers": {"urgent": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 5}})

    be = SystemOne("https://x/v1/systemone", api_key="k", model="jev-latest", client=C())
    ans = be.answer("s", {"urgent": QUESTIONS["urgent"]})
    assert ans["urgent"]["noul"] == 0.8 and be.last_response["usage"]["input_tokens"] == 5
    _url, body, headers = calls[0]
    assert headers["Authorization"] == "Bearer k" and body["model"] == "jev-latest" and "gates" not in body


def test_systemone_backend_raises_on_error():
    class C:
        def post(self, url, json=None, headers=None):
            return _Resp(400, {"detail": "Invalid request."})

    with pytest.raises(SystemOneError) as e:
        SystemOne(client=C()).answer("s", QUESTIONS)
    assert e.value.status == 400


def test_systemone_backend_stdlib_path(monkeypatch):
    import decision_circuits.backends.systemone as mod

    class FakeHTTP:
        status = 200

        def __init__(self, req):
            self.req = req

        def read(self):
            sent = json.loads(self.req.data.decode())
            return json.dumps({"answers": {qid: {"type": "noul", "noul": 0.6} for qid in sent["questions"]}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=None: FakeHTTP(req))
    ans = SystemOne("https://x/v1/systemone", api_key="k").answer("s", {"urgent": QUESTIONS["urgent"]})
    assert ans == {"urgent": {"type": "noul", "noul": 0.6}}


# --- OpenAI logprobs -------------------------------------------------------


def test_openai_render_letters_options():
    prompt, keys = render_question("the state", QUESTIONS["dept"])
    assert keys == ["billing", "technical", "other"] and "A. billing: Money" in prompt and "C. other" in prompt


class _OpenAIClient:
    def __init__(self, table):
        self.table = table  # first letter of prompt's question -> {letter: logprob}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        prompt = kw["messages"][1]["content"]
        qtext = prompt.split("Question: ")[1].split("\n")[0]
        top = [SimpleNamespace(token=t, logprob=lp) for t, lp in self.table[qtext].items()]
        return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)]))])


def test_openai_backend_turns_logprobs_into_answers():
    import math

    table = {
        "Is this urgent?": {"A": math.log(0.9), "B": math.log(0.1), "The": -5.0},
        "Which team?": {"B": math.log(0.6), "A": math.log(0.3), "C": math.log(0.1)},
        "How severe?": {"C": 0.0, "A": -3.0},
    }
    be = OpenAILogprobs(model="m", client=_OpenAIClient(table))
    ans = be.answer("s", QUESTIONS)
    assert abs(ans["urgent"]["noul"] - 0.9) < 1e-6
    assert ans["dept"]["choice"] == "technical" and abs(ans["dept"]["probabilities"]["billing"] - 0.3) < 1e-6
    assert ans["sev"]["probabilities"]["1"] == 0.0 and ans["sev"]["score"] > 1.9


def test_openai_backend_rejects_more_than_20_options():
    q = {"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(21)}}
    with pytest.raises(ValueError):
        render_question("s", q)


# --- Anthropic -------------------------------------------------------------


class _AnthropicClient:
    def __init__(self, stated=None, picks=None):
        self.stated, self.picks, self.n = stated, picks or [], 0
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        assert kw["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
        schema = kw["tools"][0]["input_schema"]
        if schema["properties"]["urgent"].get("enum"):
            payload = self.picks[self.n % len(self.picks)]
            self.n += 1
        else:
            payload = self.stated
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=TOOL_NAME, input=payload)])


def test_anthropic_stated_mode():
    stated = {"urgent": {"yes": 0.7, "no": 0.3}, "dept": {"billing": 0.5, "technical": 0.4, "other": 0.1}, "sev": {"0": 0.2, "1": 0.5, "2": 0.3}}
    be = Anthropic(model="m", client=_AnthropicClient(stated=stated))
    ans = be.answer("s", QUESTIONS)
    assert ans["urgent"]["noul"] == 0.7 and ans["dept"]["choice"] == "billing" and abs(ans["sev"]["score"] - 1.1) < 1e-9


def test_anthropic_sampled_mode_votes_with_smoothing():
    picks = [{"urgent": "yes", "dept": "billing", "sev": "2"}] * 4 + [{"urgent": "no", "dept": "technical", "sev": "1"}]
    be = Anthropic(model="m", client=_AnthropicClient(picks=picks), mode="sampled", k=5)
    ans = be.answer("s", QUESTIONS)
    # 4 yes + 1 no, plus one pseudo-count each -> 5/7
    assert abs(ans["urgent"]["noul"] - 5 / 7) < 1e-9
    assert ans["dept"]["choice"] == "billing" and ans["dept"]["probabilities"]["other"] > 0


def test_anthropic_prompt_lists_every_option():
    p = render_prompt({"a": 1}, QUESTIONS)
    assert "Question `dept`" in p and "billing: Money" in p and "2: High" in p


# --- LangGraph -------------------------------------------------------------


def test_langgraph_node_and_router():
    class BE:
        def answer(self, state, questions, *, model=None):
            return {
                "urgent": {"type": "noul", "noul": 0.52},  # inside the band -> escalate
                "dept": answer_from_probabilities(QUESTIONS["dept"], [0.8, 0.1, 0.1]),
                "sev": answer_from_probabilities(QUESTIONS["sev"], [1, 0, 0]),
            }

    node = as_node(circuit(), BE(), state_key="text")
    state = {"text": "refund now"}
    state.update(node(state))
    assert route_on("route")(state) == "billing"
    assert route_on("rush")(state) == "escalate"
