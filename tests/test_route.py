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


def test_the_diagram_draws_a_route_as_its_yes_no_ladder_and_starts_from_the_state():
    c = triage()
    m = c.to_mermaid(plain=True, state="Ticket")
    assert 'STATE(["<b>Ticket</b>"]):::state' in m and "STATE --> q_urgent" in m
    assert 'subgraph OUT["Action"]' in m
    assert 'g_action__r1{"<b>1. page on-call?</b><br/>now AND down"}' in m and 'g_action__a1(["page on-call"])' in m
    assert "g_now --> g_action__r1" in m and "g_down --> g_action__r1" in m  # wired straight in
    assert "AND" not in m.replace("now AND down", "")  # no loose junction box for the rule's logic
    for wire in ("g_action__r1 -->|yes| g_action__a1", "g_action__r1 -->|no| g_action__r2", "g_action__r2 -->|no| g_action__else"):
        assert wire in m
    assert 'g_action__else(["general queue"])' in m
    answers = {"urgent": noul(0.2), "outage": noul(0.95), "technical": noul(0.95)}
    ran = c.to_mermaid(c.evaluate(answers), answers, plain=True)
    assert 'g_action__a2(["✓ technical queue"]):::yes' in ran
    assert (
        'g_action__r1{"<b>1. page on-call?</b><br/>now AND down"}:::passed' in ran
    )  # checked, said no: on the path and 'g_action__a1(["page on-call"]):::faded' in ran
    wires = [ln.strip() for ln in ran.split("\n") if "-->" in ln]
    heavy = next(ln for ln in ran.split("\n") if "stroke-width:3.5px" in ln).split()[1].split(",")
    assert {wires[int(i)] for i in heavy} == {"g_action__r1 -->|no| g_action__r2", "g_action__r2 -->|yes| g_action__a2"}  # the way through
    assert "STATE" not in c.to_mermaid(plain=True, state=None)

    unsure = {"urgent": noul(0.75), "outage": noul(0.95), "technical": noul(0.95)}
    held = c.to_mermaid(c.evaluate(unsure), unsure, plain=True)
    assert 'g_action__esc(["⚠ a person"]):::hold' in held and "g_action__r1 -->|too close to call| g_action__esc" in held


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


def test_what_ran_and_said_no_stays_solid_and_only_what_never_ran_is_dashed():
    c = triage()
    answers = {"urgent": noul(0.2), "outage": noul(0.95), "technical": noul(0.95)}
    m = c.to_mermaid(c.evaluate(answers), answers, plain=True)
    assert ":::qno" in m.split("q_urgent[")[1].split("\n")[0]
    assert ":::qyes" in m.split("q_outage[")[1].split("\n")[0]
    lines = m.split("\n")
    wires = [ln.strip() for ln in lines if "-->" in ln]

    def styled(marker: str) -> set[str]:
        return {wires[int(i)] for ln in lines if ln.strip().startswith("linkStyle") and marker in ln for i in ln.split()[1].split(",")}

    said_no = styled("stroke:#AEB6BF")  # ran, carried a no: solid
    never = styled("dasharray")  # never ran: dashed
    assert "q_urgent --> g_now" in said_no and "g_now --> g_action__r1" in said_no
    assert never == {"g_action__r1 -->|yes| g_action__a1", "g_action__r2 -->|no| g_action__else"}  # branches not taken
    assert "g_action__r2 -->|yes| g_action__a2" not in said_no | never  # the branch taken
    assert "linkStyle" not in c.to_mermaid(plain=True)  # no run, nothing styled


def test_a_chip_box_shows_what_the_chip_decided():
    chip = Chip("strict", inputs=["x"], outputs=["yes"])
    chip.gate("yes", Q("x") >= 0.8)
    c = Circuit()
    c.noul("a", "?")
    c.mount(chip, "s", {"x": "a"})
    answers = {"a": noul(0.05)}
    assert 'subgraph M_s["s · strict → yes: no (5%)"]' in c.to_mermaid(c.evaluate(answers), answers, plain=True)
    assert 'subgraph M_s["s · strict"]' in c.to_mermaid(plain=True)


def test_a_rule_that_was_never_reached_has_its_wires_faded_and_not_reads_inverted():
    c = triage()
    c.gates.pop()  # replace the route with one that has a NOT and a third rule
    c.route("action", [("page", G("now")), ("quiet", ~G("tech")), ("tech", G("tech"))], otherwise="none")
    answers = {"urgent": noul(0.95), "outage": noul(0.1), "technical": noul(0.95)}
    m = c.to_mermaid(c.evaluate(answers), answers, plain=True)
    assert "g_tech -->|NOT| g_action__r2" in m and "<b>2. quiet?</b><br/>NOT tech" in m
    lines = m.split("\n")
    wires = [ln.strip() for ln in lines if "-->" in ln]
    faded = {int(i) for ln in lines if ln.strip().startswith("linkStyle") and "dasharray" in ln for i in ln.split()[1].split(",")}
    assert {"g_tech -->|NOT| g_action__r2", "g_tech --> g_action__r3"} <= {wires[i] for i in faded}  # rule 1 held: 2 and 3 never ran
    assert "g_now --> g_action__r1" not in {wires[i] for i in faded}
