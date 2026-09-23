import pytest

from decision_circuits import Circuit, G, Q, argmax, majority, order, verify
from decision_circuits.backends import SystemOne

ANSWERS = {
    "pii": {"type": "noul", "noul": 0.92},
    "business": {"type": "noul", "noul": 0.10},
    "angry": {"type": "noul", "noul": 0.93},
    "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.7, "technical": 0.2, "other": 0.1}, "confidence": 0.55},
    "dept2": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.6, "technical": 0.3, "other": 0.1}, "confidence": 0.4},
    "supported": {"type": "noul", "noul": 0.85},
    "urgency": {"type": "score", "score": 2.88, "legend": {}, "probabilities": {"0": 0.0, "1": 0.02, "2": 0.08, "3": 0.90}, "confidence": 0.73},
}


def test_operators_compile_and_evaluate():
    c = Circuit()
    c.gate("redact", ((Q("pii") >= 0.7) & ~Q("business")) >= 0.6).on_uncertain("escalate")
    c.gate("human", (Q("angry") | Q("urgency:3")) >= 0.6)
    c.gate("route", argmax("dept", min_confidence=0.35))
    c.gate("vote", majority("dept", "dept2"))
    c.gate("checked", verify("dept", Q("supported"), tau=0.8)).on_uncertain("escalate")
    c.gate("tier", order("urgency", [1.0, 2.0, 2.6]))
    c.gate("bill_and_hot", (G("route:billing") & G("tier:3")) >= 0.5)

    compiled = c.compile()
    assert compiled["redact"]["op"] == "and" and compiled["redact"]["tau"] == 0.6 and compiled["redact"]["on_uncertain"] == "escalate"
    # helpers exist for the threshold and the not
    helper_ops = sorted(v["op"] for k, v in compiled.items() if k.startswith("_"))
    assert helper_ops == ["not", "threshold"]

    r = c.evaluate(ANSWERS)
    assert set(r) == {"redact", "human", "route", "vote", "checked", "tier", "bill_and_hot"}  # helpers hidden
    # `Q("pii") >= 0.7` inside the expression is a decision: pii 0.92 passes, so it counts as 1
    assert r["redact"]["value"] is True and r["redact"]["p"] == pytest.approx(1.0 * 0.90, abs=1e-6)
    assert r["human"]["value"] is True and r["human"]["p"] > 0.99
    assert r["route"]["value"] == "billing"
    assert r["vote"]["value"] == "billing"
    assert r["checked"]["value"] == "billing" and r["checked"]["outcome"] == "decided"
    assert r["tier"]["value"] == 3
    assert r["bill_and_hot"]["value"] is True and r["bill_and_hot"]["p"] == pytest.approx(0.7)  # P(billing) x 1: tier decided bucket 3


def test_chained_and_flattens():
    c = Circuit()
    c.gate("all3", (Q("pii") & Q("angry") & Q("supported")) >= 0.5)
    spec = c.compile()["all3"]
    assert spec["op"] == "and" and spec["inputs"] == ["pii", "angry", "supported"]


def test_bare_reference_thresholds_at_half():
    c = Circuit()
    c.gate("is_pii", Q("pii"))
    assert c.compile()["is_pii"] == {"op": "threshold", "input": "pii", "tau": 0.5, "on_uncertain": "abstain", "band": 0.1}


def test_default_policy_carries_value():
    c = Circuit()
    c.gate("tier", order("urgency", [2.85])).on_uncertain("default", default=2).band(0.1)
    r = c.evaluate(ANSWERS)["tier"]
    assert r["outcome"] == "default" and r["value"] == 2


class _FakeClient:
    """Looks like httpx/requests: .post(url, json=..., headers=...) -> .json()."""

    def __init__(self, answers, with_gates: bool):
        self.answers = answers
        self.with_gates = with_gates
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        status = 200
        body = {"answers": self.answers}
        if "gates" in json:
            if self.with_gates:  # a server that evaluates gates itself (s1proto)
                body["gates"] = {"_h_1": {"value": True}, "rush": {"value": True, "p": 0.9, "outcome": "decided"}}
            else:  # TypeSafe rejects unknown fields
                status, body = 400, {"detail": {"error_type": "api_usage_error", "message": "Invalid request."}}

        class R:
            status_code = status

            def json(self):
                return body

        return R()


def _circuit():
    c = Circuit()
    c.noul("urgent", "Is this urgent?", true="Needs action today", false="Can wait")
    c.choice("dept", "Which team?", {"billing": None, "technical": None, "other": None})
    c.gate("rush", Q("urgent") >= 0.5)
    c.gate("route", argmax("dept"))
    return c


def test_run_sends_questions_only_and_evaluates_gates_here():
    client = _FakeClient({"urgent": {"type": "noul", "noul": 0.9}, "dept": ANSWERS["dept"]}, with_gates=True)
    out = _circuit().run(SystemOne("/v1/systemone", api_key="k", client=client), "Card charged twice, please refund today.")
    url, body, headers = client.calls[0]
    assert url == "/v1/systemone" and headers["Authorization"] == "Bearer k" and len(client.calls) == 1
    assert set(body["questions"]) == {"urgent", "dept"} and "gates" not in body
    assert out["gates"]["rush"]["value"] is True and out["gates"]["route"]["value"] == "billing"


def test_indexing_and_keyword_policy_match_string_forms():
    a = Circuit()
    a.gate("hot", (Q("urgency:3") | G("route:billing")) >= 0.5).on_uncertain("escalate").band(0.05)
    b = Circuit()
    b.gate("hot", (Q("urgency")[3] | G("route")["billing"]).at(0.5), on_uncertain="escalate", band=0.05)
    assert a.compile() == b.compile()
    with pytest.raises(KeyError):
        Q("urgency:3")["x"]


def test_to_mermaid_is_a_method():
    c = Circuit()
    c.noul("pii", "PII?")
    c.gate("redact", Q("pii") >= 0.7)
    assert "redact" in c.to_mermaid(plain=True)


def test_questions_carry_fields_this_version_does_not_know_about():
    """A server may accept per-question fields newer than this SDK. Passing one
    through beats waiting for a release, and an unknown field is ignored by any
    server that doesn't use it."""
    c = Circuit()
    c.noul("billing", "Is this about a bill?", weight=2)
    c.choice("dept", "Which team?", {"a": None, "b": None}, weight=3)
    c.score("urgency", "How urgent?", ["low", "high"], weight=4)
    assert c.questions["billing"]["weight"] == 2
    assert c.questions["dept"]["weight"] == 3
    assert c.questions["urgency"]["weight"] == 4
    assert c.questions["billing"]["type"] == "noul"
    assert c.questions["dept"]["criteria"] == {"a": None, "b": None}


def test_v2_questions_and_pooled_gates_compile_and_evaluate():
    from decision_circuits import at_least, consistent, count

    c = Circuit()
    c.multi("issues", "Which problems does the customer report?", {"late": None, "damaged": None, "wrong_item": None})
    c.locate("ask", "Which sentence says what the customer wants?", none="the message does not say")
    c.noul("refund", "Does the customer want a refund?")
    c.noul("keep", "Does the customer want to keep the order?")
    c.gate("several", at_least(2, "issues", tau=0.5))
    c.gate("n", count("issues"))
    c.gate("two_exactly", G("n")[2] >= 0.3)
    c.gate("coherent", consistent("refund", Q("keep"), relation="complement"), on_uncertain="escalate")
    c.gate("unclear_ask", Q("ask")["none"] >= 0.5)
    assert c.questions["ask"] == {"type": "locate", "instructions": "Which sentence says what the customer wants?", "criteria": "the message does not say"}
    spec = c.compile()
    assert spec["several"]["op"] == "at_least" and spec["several"]["k"] == 2 and spec["several"]["input"] == "issues"
    assert spec["coherent"]["inputs"] == ["refund", "keep"] and spec["coherent"]["relation"] == "complement"
    answers = {
        "issues": {"type": "multi", "selected": ["late", "damaged"], "probabilities": {"late": 0.9, "damaged": 0.8, "wrong_item": 0.1}},
        "ask": {"type": "locate", "located": [], "none": 0.2, "confidence": 0.5},
        "refund": {"type": "noul", "noul": 0.9},
        "keep": {"type": "noul", "noul": 0.6},
    }
    r = c.evaluate(answers)
    assert r["several"]["value"] is True and r["n"]["value"] == 2 and r["two_exactly"]["value"] is True
    assert r["coherent"]["outcome"] == "escalate"  # 0.9 + 0.6 is 0.5 over a complement
    assert r["unclear_ask"]["value"] is False
    assert "several" in c.to_mermaid()


def test_pooled_gate_needs_something_to_pool():
    from decision_circuits import at_least

    with pytest.raises(ValueError):
        at_least(1)


def test_an_unpicked_option_of_a_categorical_gate_is_never_likelier_than_the_pick():
    answers = {"dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.34, "technical": 0.33, "other": 0.33}, "confidence": 0.0}}
    c = Circuit()
    c.gate("route", argmax("dept"))
    c.gate("is_billing", G("route")["billing"] >= 0.3)
    c.gate("is_tech", G("route")["technical"] >= 0.5)
    r = c.evaluate(answers)
    assert r["is_billing"]["p"] == pytest.approx(0.34) and r["is_tech"]["p"] == pytest.approx(0.33) and r["is_tech"]["value"] is False


def test_a_threshold_inside_an_expression_is_a_decision_and_uncertain_near_its_tau():
    answers = {"pii": {"type": "noul", "noul": 0.8}, "biz": {"type": "noul", "noul": 0.0}}
    below = Circuit()
    below.gate("a", (Q("pii") >= 0.9) & ~Q("biz"), band=0.05)
    assert below.evaluate(answers)["a"]["value"] is False  # 0.8 misses 0.9: the AND is 0, not 0.8
    near = Circuit()
    near.gate("a", (Q("pii") >= 0.85) & ~Q("biz"), band=0.1, on_uncertain="escalate")
    assert near.evaluate(answers)["a"]["outcome"] == "escalate"  # 0.8 is within the band of 0.85
    nested = Circuit()
    nested.gate("x", (Q("pii") >= 0.3).at(0.7))  # used to raise in compile
    assert nested.evaluate(answers)["x"]["value"] is True


def test_underscore_gate_names_are_reserved():
    with pytest.raises(ValueError):
        Circuit().gate("_internal", Q("x"))
