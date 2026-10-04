"""The parts library: each part passes its own vectors and drops into unrelated circuits."""

from __future__ import annotations

import json

import pytest

from decision_circuits import Circuit, G
from decision_circuits.parts import refund_risk, tool_guard, verified_classifier

PARTS = [refund_risk, tool_guard, verified_classifier]


@pytest.mark.parametrize("part", PARTS, ids=lambda f: f.__name__)
def test_each_part_passes_its_own_vectors_and_survives_json(part):
    chip = part()
    assert chip.tests and chip.test() == []
    again = Circuit.from_dict(json.loads(json.dumps(chip.to_dict())))
    assert again == chip and again.test() == []
    assert chip.describe().startswith(f"# Chip `{chip.name}` v1.0")


def test_refund_risk_takes_the_facts_a_host_has_and_asks_for_the_rest():
    desk = Circuit()  # a refund desk that knows the order from its database
    desk.noul("big", "Order over $500 (computed).")
    desk.noul("repeat", "Three or more refunds in the last year (computed).")
    desk.mount(refund_risk(), "risk", {"high_value": "big", "repeat_refunds": "repeat"}, params={"request": "`ticket`"})
    assert {q for q in desk.questions if q.startswith("risk.")} == {"risk.new_account", "risk.story", "risk.pressure", "risk.other_payee", "risk.vague"}

    payments = Circuit()  # a payments circuit that knows nothing: the chip asks all seven
    payments.mount(refund_risk(), "risk", params={"request": "the chargeback claim", "hold_at": 3})
    assert len([q for q in payments.questions if q.startswith("risk.")]) == 7
    assert "the chargeback claim" in payments.questions["risk.high_value"]["instructions"]
    assert payments.compile()["risk.hold"]["k"] == 3


def test_tool_guard_reads_where_the_host_keeps_the_call_and_drives_middleware():
    from decision_circuits.integrations import CircuitPolicy

    c = Circuit()
    c.mount(tool_guard(), "guard", params={"call": "`proposed`", "when_requested": "allow"})
    assert "`proposed`" in c.questions["guard.destructive"]["instructions"]

    class Asked:  # an irreversible email the user asked for
        def answer(self, state, questions, *, model=None):
            vals = {"destructive": 0.02, "external": 0.95, "undoable": 0.1, "requested": 0.95, "injected": 0.02}
            return {q: {"type": "noul", "noul": vals[q.split(".")[1]]} for q in questions}

    assert CircuitPolicy(c, Asked(), gate="guard.decision").judge({"proposed": "send_email"}).action == "allow"  # when_requested


def test_verified_classifier_needs_its_question_and_feeds_a_route():
    c = Circuit()
    with pytest.raises(ValueError, match=r"needs settings \['question', 'options'\]"):
        c.mount(verified_classifier(), "kind")
    c.mount(verified_classifier(), "kind", params={"question": "Which team?", "options": {"billing": None, "technical": None}})
    c.route("team", [("person", ~G("kind.grounded")), ("billing", G("kind.label")["billing"])], otherwise="technical")
    assert c.questions["kind.first"]["criteria"] == {"billing": None, "technical": None}
