"""route: the circuit's last step, one action from a priority list of rules."""

from __future__ import annotations

import json

import pytest

from decision_circuits import Chip, Circuit, G, Q, order, route
from decision_circuits.gates import Gate


def noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


def triage() -> Circuit:
    c = Circuit()
    c.noul("urgent", "Must it happen today?")
    c.noul("outage", "Is a service down?")
    c.noul("technical", "Is it a technical problem?")
    c.gate("now", Q("urgent") >= 0.8, on_uncertain="escalate")
    c.gate("down", Q("outage") >= 0.8, on_uncertain="escalate")
    c.gate("tech", Q("technical") >= 0.8, on_uncertain="escalate")
    c.route("action", [("page on-call", G("now") & G("down")), ("technical queue", G("tech"))], otherwise="general queue")
    return c


@pytest.mark.parametrize(
    ("urgent", "outage", "technical", "want"),
    [
        (0.98, 0.95, 0.95, "page on-call"),  # rule 1 wins over rule 2
        (0.20, 0.95, 0.95, "technical queue"),  # rule 1 is decided no: try rule 2
        (0.20, 0.10, 0.10, "general queue"),  # nothing holds
        (0.75, 0.95, 0.95, "escalate"),  # rule 1 is too close to call: stop, don't fall through
        (0.20, 0.95, 0.75, "escalate"),  # rule 2 is too close to call
    ],
)
def test_the_first_rule_that_holds_wins_and_a_close_call_stops_the_route(urgent, outage, technical, want):
    r = triage().evaluate({"urgent": noul(urgent), "outage": noul(outage), "technical": noul(technical)})["action"]
    got = r["value"] if r["outcome"] == "decided" else r["outcome"]
    assert got == want, r["trace"]


def test_a_gate_in_a_rule_counts_as_its_decision_not_its_probability():
    c = Circuit()
    c.noul("a", "?")
    c.gate("strict", Q("a") >= 0.9, band=0.0)  # p=0.7 decides no
    c.route("action", [("go", G("strict"))], otherwise="stop")
    assert c.evaluate({"a": noul(0.7)})["action"]["value"] == "stop"  # not "go" because 0.7 >= 0.5


def test_a_raw_question_in_a_rule_is_read_at_one_half_with_the_band():
    c = Circuit()
    c.noul("a", "?")
    c.route("action", [("go", Q("a"))], otherwise="stop")
    assert c.evaluate({"a": noul(0.9)})["action"]["value"] == "go"
    assert c.evaluate({"a": noul(0.55)})["action"]["outcome"] == "escalate"


def test_route_survives_json_and_mounting_inside_a_chip():
    desk = Chip("desk", inputs=["urgent"], outputs=["action"])
    desk.gate("now", Q("urgent") >= 0.8)
    desk.route("action", [("page", G("now"))], otherwise="queue")
    host = Circuit()
    host.noul("u", "?")
    pins = host.mount(desk, "d", {"urgent": "u"})
    assert host.evaluate({"u": noul(0.95)})[pins["action"].ref]["value"] == "page"
    back = Circuit.from_dict(json.loads(json.dumps(host.to_dict())))
    assert back.compile() == host.compile()
    assert Circuit.from_dict(desk.to_dict()) == desk
    with pytest.raises(ValueError, match="at least one"):
        route([])
    with pytest.raises(ValueError, match="needs `rules`"):
        Gate(op="route", rules=[["only an action"]])


def test_a_probability_on_the_band_edge_is_outside_the_band():
    c = Circuit()
    c.noul("a", "?")
    c.gate("g", Q("a") >= 0.8, on_uncertain="escalate")  # band 0.1: uncertain strictly between 0.7 and 0.9
    assert c.evaluate({"a": noul(0.9)})["g"]["outcome"] == "decided"  # 0.9 - 0.8 is 0.0999... in floats
    assert c.evaluate({"a": noul(0.7)})["g"]["outcome"] == "decided"
    assert c.evaluate({"a": noul(0.8999)})["g"]["outcome"] == "escalate"
    s = Circuit()
    s.score("size", "?", ["s", "m", "l"])
    s.gate("b", order("size", [1.1]))  # band 0.1: 1.0 is on the edge
    assert s.evaluate({"size": {"type": "score", "score": 1.0, "probabilities": {"0": 0, "1": 1, "2": 0}, "confidence": 1.0}})["b"]["outcome"] == "decided"


def test_the_diagram_ends_in_one_action_and_starts_from_the_state():
    c = triage()
    m = c.to_mermaid(plain=True, state="Ticket")
    assert 'STATE(["<b>Ticket</b>"]):::state' in m and "STATE --> q_urgent" in m
    assert 'subgraph OUT["Action"]' in m
    assert "1. page on-call<br/>2. technical queue<br/>else general queue" in m
    assert "-->|1. page on-call| g_action" in m and "-->|2. technical queue| g_action" in m
    answers = {"urgent": noul(0.2), "outage": noul(0.95), "technical": noul(0.95)}
    ran = c.to_mermaid(c.evaluate(answers), answers, plain=True)
    assert "<b>✓ technical queue</b>" in ran and ":::yes" in ran
    assert "STATE" not in c.to_mermaid(plain=True, state=None)


def test_describe_leads_with_the_outcome_and_names_what_was_unsure():
    c = triage()
    assert "1. **page on-call** if the `now` decision is yes and the `down` decision is yes" in c.describe()
    assert "  - otherwise **general queue**" in c.describe()
    ok = {"urgent": noul(0.2), "outage": noul(0.95), "technical": noul(0.95)}
    assert "**Outcome (action):** **technical queue**, by rule 2." in c.describe(c.evaluate(ok), ok)
    none = {"urgent": noul(0.2), "outage": noul(0.1), "technical": noul(0.1)}
    assert "**general queue**, since no rule held." in c.describe(c.evaluate(none), none)
    unsure = {"urgent": noul(0.75), "outage": noul(0.95), "technical": noul(0.95)}
    assert "escalated to a person**: rule 1 (**page on-call**) hinges on `now` (75%) too close to call." in c.describe(c.evaluate(unsure), unsure)


def test_after_a_run_what_said_no_greys_out_and_its_wires_fade():
    c = triage()
    answers = {"urgent": noul(0.2), "outage": noul(0.95), "technical": noul(0.95)}
    m = c.to_mermaid(c.evaluate(answers), answers, plain=True)
    assert "<b>urgent</b>" in m and ":::qno" in m.split("q_urgent[")[1].split("\n")[0]
    assert ":::qyes" in m.split("q_outage[")[1].split("\n")[0]
    lines = m.split("\n")
    wires = [ln.strip() for ln in lines if "-->" in ln]
    faded = {int(i) for ln in lines if ln.strip().startswith("linkStyle") and "dasharray" in ln for i in ln.split()[1].split(",")}
    assert "q_urgent --> g_now" in {wires[i] for i in faded}  # the no from urgent
    assert any(wires[i].endswith("|1. page on-call| g_action") for i in faded)  # the rule not taken
    assert not any(wires[i].endswith("|2. technical queue| g_action") for i in faded)  # the rule taken
    assert "linkStyle" not in c.to_mermaid(plain=True)  # no run, nothing to fade
