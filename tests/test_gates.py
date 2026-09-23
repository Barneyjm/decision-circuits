import pytest

from decision_circuits.gates import Gate, evaluate_gates

ANSWERS = {
    "pii": {"type": "noul", "noul": 0.92},
    "business": {"type": "noul", "noul": 0.10},
    "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}, "confidence": 0.55},
    "dept2": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.6, "technical": 0.3, "sales": 0.1}, "confidence": 0.4},
    "dept3": {"type": "choice", "choice": "technical", "probabilities": {"billing": 0.4, "technical": 0.5, "sales": 0.1}, "confidence": 0.3},
    "supported": {"type": "noul", "noul": 0.85},
    "urgency": {
        "type": "score",
        "score": 2.4,
        "legend": {"0": "low", "1": "med", "2": "high", "3": "critical"},
        "probabilities": {"0": 0.0, "1": 0.1, "2": 0.4, "3": 0.5},
        "confidence": 0.6,
    },
}


def g(**kw):
    return Gate.from_dict(kw)


def test_threshold_and_not_and_and():
    r = evaluate_gates(
        {
            "has_pii": g(op="threshold", input="pii", tau=0.8),
            "private": g(op="not", input="business", tau=0.8, band=0.05),
            "redact": g(op="and", inputs=["has_pii", "private"], tau=0.7, on_uncertain="escalate"),
        },
        ANSWERS,
    )
    assert r["has_pii"].value is True and r["has_pii"].p == pytest.approx(0.92)
    assert r["private"].value is True and r["private"].p == pytest.approx(0.90)
    assert r["redact"].value is True and r["redact"].p == pytest.approx(0.92 * 0.90)
    assert "independence" in " ".join(r["redact"].trace)


def test_uncertain_band_escalates():
    r = evaluate_gates({"x": g(op="threshold", input="pii", tau=0.9, band=0.05, on_uncertain="escalate")}, ANSWERS)
    assert r["x"].outcome == "escalate" and r["x"].value is None and r["x"].uncertain


def test_uncertain_default():
    r = evaluate_gates({"x": g(op="threshold", input="pii", tau=0.9, band=0.05, on_uncertain="default", default=False)}, ANSWERS)
    assert r["x"].outcome == "default" and r["x"].value is False


def test_argmax_confidence_gate():
    ok = evaluate_gates({"route": g(op="argmax", input="dept", min_confidence=0.5)}, ANSWERS)["route"]
    assert ok.value == "billing" and ok.outcome == "decided"
    ab = evaluate_gates({"route": g(op="argmax", input="dept", min_confidence=0.6)}, ANSWERS)["route"]
    assert ab.outcome == "abstain" and ab.value is None


def test_majority_over_paraphrases():
    r = evaluate_gates({"vote": g(op="majority", inputs=["dept", "dept2", "dept3"])}, ANSWERS)["vote"]
    assert r.value == "billing" and r.confidence == pytest.approx(2 / 3) and r.outcome == "decided"
    tie = evaluate_gates({"vote": g(op="majority", inputs=["dept", "dept3"])}, ANSWERS)["vote"]
    assert tie.outcome == "abstain"


def test_verify_with_negative_checker():
    ok = evaluate_gates({"v": g(op="verify", input="dept", check="supported", tau=0.8, on_uncertain="escalate")}, ANSWERS)["v"]
    assert ok.value == "billing" and ok.outcome == "decided"
    weak = dict(ANSWERS, supported={"type": "noul", "noul": 0.4})
    esc = evaluate_gates({"v": g(op="verify", input="dept", check="supported", tau=0.8, on_uncertain="escalate")}, weak)["v"]
    assert esc.outcome == "escalate"


def test_order_buckets_and_cutpoint_band():
    r = evaluate_gates({"tier": g(op="order", input="urgency", cutpoints=[1.0, 2.0, 2.8])}, ANSWERS)["tier"]
    assert r.value == 2 and r.outcome == "decided"
    near = evaluate_gates({"tier": g(op="order", input="urgency", cutpoints=[2.45], band=0.1, on_uncertain="escalate")}, ANSWERS)["tier"]
    assert near.outcome == "escalate"


def test_forward_reference_is_an_error():
    with pytest.raises(KeyError):
        evaluate_gates({"a": g(op="and", inputs=["b"]), "b": g(op="threshold", input="pii")}, ANSWERS)


CONSISTENCY = {
    "refund": {"type": "noul", "noul": 0.80},
    "refund_reworded": {"type": "noul", "noul": 0.74},
    "no_refund": {"type": "noul", "noul": 0.55},
    "tags": {"type": "multi", "selected": ["billing"], "probabilities": {"billing": 0.9, "outage": 0.3}},
    "both": {"type": "noul", "noul": 0.6},
    "where": {"type": "locate", "located": [{"path": "a", "text": "x", "probability": 0.7}], "none": 0.3, "confidence": 0.4},
}


def test_consistent_same_within_band():
    r = evaluate_gates({"c": g(op="consistent", inputs=["refund", "refund_reworded"], relation="same")}, CONSISTENCY)["c"]
    assert r.value is True and r.outcome == "decided" and abs(r.p - 0.94) < 1e-9


def test_consistent_complement_violation_escalates():
    gates = {"c": g(op="consistent", inputs=["refund", "no_refund"], relation="complement", on_uncertain="escalate")}
    r = evaluate_gates(gates, CONSISTENCY)["c"]
    assert r.outcome == "escalate" and r.uncertain and abs(r.p - 0.65) < 1e-9  # 0.80 + 0.55 - 1 = 0.35 over


def test_consistent_implies_reads_multi_options():
    # "billing and outage" (0.6) cannot be likelier than "outage" (0.3)
    r = evaluate_gates({"c": g(op="consistent", inputs=["both", "tags:outage"], relation="implies")}, CONSISTENCY)["c"]
    assert r.outcome == "abstain" and "violation 0.30" in r.trace[-2]
    ok = evaluate_gates({"c": g(op="consistent", inputs=["both", "tags:billing"], relation="implies")}, CONSISTENCY)["c"]
    assert ok.value is True


def test_threshold_on_locate_none():
    r = evaluate_gates({"missing": g(op="threshold", input="where:none", tau=0.5)}, CONSISTENCY)["missing"]
    assert r.value is False and abs(r.p - 0.3) < 1e-9


def test_consistent_needs_two_inputs_and_a_relation():
    with pytest.raises(ValueError):
        g(op="consistent", inputs=["refund"], relation="same")
    with pytest.raises(ValueError):
        g(op="consistent", inputs=["refund", "no_refund"])
