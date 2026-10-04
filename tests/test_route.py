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
    assert 'STATE(["<b>Ticket</b>"]):::state' in m and "STATE --> IN" in m and "STATE --> q_" not in m  # one arrow into the column
    assert 'subgraph DECIDE["Decide"]' in m and 'subgraph OUTCOME["Outcome"]' in m
    outcome = m.split('subgraph OUTCOME["Outcome"]')[1].split("  end")[0]
    assert all(f'g_action__{k}(["' in outcome for k in ("a1", "a2", "else"))  # every action in the right-hand column
    assert 'g_action__r1{"<b>1.</b> now AND down?"}' in m and 'g_action__a1(["page on-call"])' in m  # asks the condition
    assert "g_now --> g_action__r1" in m and "g_down --> g_action__r1" in m  # wired straight in (no run: no values)
    assert "AND" not in m.replace("now AND down", "")  # no loose junction box for the rule's logic
    # rule 2 reads one gate, so it branches from that gate: no diamond asking "tech?" again
    # yes links are long enough that every action lands in the same column
    for wire in ("g_action__r1 --->|yes| g_action__a1", "g_action__r1 -->|no| g_tech", "g_tech -->|yes| g_action__a2", "g_tech -->|no| g_action__else"):
        assert wire in m
    assert "g_action__r2" not in m
    assert 'g_action__else(["general queue"])' in m
    answers = {"urgent": noul(0.2), "outage": noul(0.95), "technical": noul(0.95)}
    ran = c.to_mermaid(c.evaluate(answers), answers, plain=True)
    assert 'g_action__a2(["✓ technical queue"]):::yes' in ran
    assert 'g_action__r1{"<b>1.</b> now AND down?"}:::passed' in ran  # checked, said no: on the path
    assert 'g_action__a1(["page on-call"]):::faded' in ran
    assert "g_action__r1 ---> g_action__a1" in ran  # a branch not taken carries no label
    wires = [ln.strip() for ln in ran.split("\n") if "-->" in ln or "~~~" in ln]
    heavy = next(ln for ln in ran.split("\n") if "stroke-width:3.5px" in ln).split()[1].split(",")
    assert {wires[int(i)] for i in heavy} == {"g_action__r1 -->|no| g_tech", "g_tech -->|yes| g_action__a2"}  # the way through
    assert "STATE" not in c.to_mermaid(plain=True, state=None)

    unsure = {"urgent": noul(0.75), "outage": noul(0.95), "technical": noul(0.95)}
    held = c.to_mermaid(c.evaluate(unsure), unsure, plain=True)
    assert 'g_action__esc(["⚠ a person"]):::hold' in held and "g_action__r1 --->|too close to call| g_action__esc" in held
    held_wires = [ln.strip() for ln in held.split("\n") if "-->" in ln or "~~~" in ln]
    amber = next(ln for ln in held.split("\n") if ln.strip().startswith("linkStyle") and "stroke:#9A6B00" in ln).split()[1].split(",")
    assert [held_wires[int(i)] for i in amber] == ["g_action__r1 --->|too close to call| g_action__esc"]


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
    wires = [ln.strip() for ln in lines if "-->" in ln or "~~~" in ln]

    def styled(marker: str) -> set[str]:
        return {wires[int(i)] for ln in lines if ln.strip().startswith("linkStyle") and marker in ln for i in ln.split()[1].split(",")}

    said_no = styled("stroke:#AEB6BF")  # ran, carried a no: solid
    never = styled("dasharray")  # never ran: dashed
    assert "q_urgent --> g_now" in said_no
    # into a rule the route checked: solid and dark; the box it comes from already says no
    assert "g_now --> g_action__r1" in wires and "g_now --> g_action__r1" not in said_no | never
    assert "g_tech -->|yes| g_action__a2" in wires
    assert never == {"g_action__r1 ---> g_action__a1", "g_tech -->|no| g_action__else"}  # branches not taken
    assert "g_tech -->|yes| g_action__a2" not in said_no | never  # the branch taken
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
    assert "g_tech -->|NOT| g_action__r2" in m and "<b>2.</b> NOT tech?" in m
    lines = m.split("\n")
    wires = [ln.strip() for ln in lines if "-->" in ln or "~~~" in ln]
    faded = {int(i) for ln in lines if ln.strip().startswith("linkStyle") and "dasharray" in ln for i in ln.split()[1].split(",")}
    assert "g_tech -->|NOT| g_action__r2" in {wires[i] for i in faded}  # rule 1 held: rule 2 never ran
    assert "g_action__r1{" not in m  # `now` feeds only rule 1: the rule branches from the gate
    assert 'g_action__r3{"<b>3.</b> tech?"}' in m  # `tech` is shared with rule 2: a diamond in the ladder
    assert "g_now ---->|yes| g_action__a1" in wires and "g_now ---->|yes| g_action__a1" not in {wires[i] for i in faded}


def test_a_question_shows_what_it_landed_on_and_the_outcomes_it_did_not():
    c = Circuit()
    c.noul("story", "Does the story add up?")
    c.choice("ask", "What do they want?", {"refund": None, "replacement": None, "info": None})
    c.gate("g", Q("story") >= 0.5)
    answers = {
        "story": noul(0.11),
        "ask": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.8, "replacement": 0.15, "info": 0.05}, "confidence": 0.4},
    }
    m = c.to_mermaid(c.evaluate(answers), answers, plain=True)
    assert "<b>→ no 89%</b><br/>yes 11%" in m  # a no reads as no, with its other side
    assert "<b>→ refund 80%</b><br/>replacement 15% · info 5%" in m
    text = c.describe(c.evaluate(answers), answers)
    assert "→ **no 89%** (yes 11%)" in text and "→ **refund 80%** (replacement 15%, info 5%)" in text


def test_a_gate_that_only_feeds_one_rule_is_drawn_in_the_ladder():
    c = Circuit()
    c.noul("a", "?")
    c.noul("b", "?")
    c.gate("shared", Q("a") >= 0.5)
    c.gate("only_here", Q("b") >= 0.5)
    c.gate("elsewhere", G("shared") >= 0.5)  # something else reads `shared`
    c.route("action", [("x", G("shared")), ("y", G("only_here"))], otherwise="z")
    m = c.to_mermaid(plain=True)
    action = m.split('subgraph DECIDE["Decide"]')[1].split("  end")[0]
    logic = m.split('subgraph LOGIC["Checks"]')[1].split("  end")[0]
    assert "g_only_here" in action and "g_only_here" not in logic  # moved to its rule's place
    assert "q_b --> g_only_here" in m  # and keeps the wire from what it reads
    assert "g_shared" in logic and "g_shared[" not in action and "g_shared{" not in action  # used elsewhere: stays
    assert "g_action__r1 -->|no| g_only_here" in m and "g_only_here -->|no| g_action__else" in m  # shared: a diamond
    answers = {"a": noul(0.9), "b": noul(0.9)}  # rule 1 holds: the route never reaches `only_here`
    ran = c.to_mermaid(c.evaluate(answers), answers, plain=True)
    line = next(ln for ln in ran.split("\n") if ln.strip().startswith("g_only_here"))
    assert line.endswith(":::faded")  # a ladder step never reached fades, whatever the gate said


def test_a_shared_gate_keeps_its_place_and_its_rule_gets_a_diamond():
    c = Circuit()
    c.noul("a", "?")
    c.gate("risky", Q("a") >= 0.5)
    c.gate("also_reads_it", ~G("risky") >= 0.5)
    c.route("action", [("review", G("risky"))], otherwise="ok")
    m = c.to_mermaid(plain=True)
    assert 'g_action__r1{"<b>1.</b> risky?"}' in m and "g_risky --> g_action__r1" in m  # a diamond in the ladder
    assert "g_risky -->|no|" not in m  # no ladder branches from the shared gate itself
