"""The chip interface: typed and optional pins, settings in text and in logic."""

from __future__ import annotations

import json

import pytest

from decision_circuits import Chip, Circuit, G, P, Pin, Q, argmax, at_least
from decision_circuits.chips import REQUIRED


def noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


def risk_chip() -> Chip:
    risk = Chip(
        "risk",
        inputs=[
            Pin("new_account", ask="Was the account behind {request} opened in the last 30 days?"),
            Pin("high_value", ask="Is the amount in {request} unusually high?"),
            Pin("reason", type="choice", options=["damaged", "not_received"]),
        ],
        outputs=["action"],
        params={"request": "the request", "hold_at": 2, "strictness": 0.6, "held": "hold"},
    )
    risk.noul("story", "Is the account of what happened in {request} inconsistent?")
    risk.gate("signals", at_least(P("hold_at"), "new_account", "high_value", "story", Q("reason")["not_received"]))
    risk.gate("hot", G("signals") >= P("strictness"), on_uncertain="escalate")
    risk.route("action", [(P("held"), G("hot"))], otherwise="clear")
    return risk


def host() -> Circuit:
    c = Circuit()
    c.noul("big", "Order over $500 (computed).")
    c.choice("why", "Why do they want a refund?", {"damaged": None, "not_received": None, "other": None})
    return c


def test_an_optional_pin_is_asked_by_the_chip_when_the_host_does_not_wire_it():
    c = host()
    c.mount(risk_chip(), "risk", {"high_value": "big", "reason": "why"}, params={"request": "`ticket`"})
    assert c.questions["risk.new_account"] == {"type": "noul", "instructions": "Was the account behind `ticket` opened in the last 30 days?"}
    assert "risk.high_value" not in c.questions  # wired to the host's own fact
    assert c.questions["risk.story"]["instructions"] == "Is the account of what happened in `ticket` inconsistent?"
    assert c.mounts["risk"]["asked"] == ["new_account"]
    answers = {
        "big": noul(1.0),
        "why": {"type": "choice", "choice": "damaged", "probabilities": {"damaged": 0.9, "not_received": 0.05, "other": 0.05}, "confidence": 0.6},
    }
    answers |= {"risk.new_account": noul(0.95), "risk.story": noul(0.2)}
    assert c.evaluate(answers)["risk.action"]["value"] == "hold"  # 2 of 4: new account and high value


def test_a_wire_of_the_wrong_kind_is_refused_at_mount():
    c = host()
    c.gate("bucket", argmax("why"))
    c.choice("thin", "?", {"damaged": None, "other": None})
    with pytest.raises(ValueError, match=r"pin 'high_value' takes a noul; 'why' is a choice"):
        c.mount(risk_chip(), "a", {"high_value": "why", "reason": "why"})
    with pytest.raises(ValueError, match="needs options \\['not_received'\\]"):
        c.mount(risk_chip(), "b", {"reason": "thin"})
    with pytest.raises(ValueError, match="'nope' is not a question or gate"):
        c.mount(risk_chip(), "c", {"reason": "nope"})
    with pytest.raises(ValueError, match="not wired"):
        c.mount(risk_chip(), "d", {})  # `reason` has no question of its own to fall back on
    ok = c.mount(risk_chip(), "e", {"reason": "bucket", "high_value": Q("why")["damaged"]})  # an argmax gate is a choice
    assert ok == {"action": G("e.action")}
    assert set(c.mounts) == {"e"}  # the refused mounts left nothing behind


def test_settings_reach_thresholds_counts_and_route_actions():
    c = host()
    c.mount(risk_chip(), "lenient", {"high_value": "big", "reason": "why"}, params={"hold_at": 3, "held": "review"})
    c.mount(risk_chip(), "strict", {"high_value": "big", "reason": "why"}, params={"hold_at": 1, "strictness": 0.5})
    compiled = c.compile()
    assert compiled["lenient.signals"]["k"] == 3 and compiled["strict.signals"]["k"] == 1
    assert compiled["strict.hot"]["tau"] == 0.5 and compiled["lenient.hot"]["tau"] == 0.6
    assert compiled["lenient.action"]["rules"][0][0] == "review" and compiled["strict.action"]["rules"][0][0] == "hold"
    with pytest.raises(ValueError, match=r"has no settings \['loud'\]"):
        c.mount(risk_chip(), "x", {"reason": "why"}, params={"loud": True})


def test_a_required_setting_must_be_given_and_an_undeclared_one_is_caught():
    who = Chip("who", outputs=["out"], params=["who"])
    who.noul("q", "Does {who} agree?")
    who.gate("out", Q("q") >= 0.5)
    c = Circuit()
    with pytest.raises(ValueError, match=r"needs settings \['who'\]"):
        c.mount(who, "w")
    sloppy = Chip("sloppy", outputs=["out"])
    sloppy.noul("q", "?")
    sloppy.gate("out", Q("q") >= P("cutoff"))
    with pytest.raises(ValueError, match=r"uses settings \['cutoff'\] it does not declare"):
        sloppy.check()


def test_a_chip_evaluates_runs_and_reads_with_its_default_settings():
    risk = risk_chip()
    out = risk.evaluate({"new_account": 0.9, "high_value": 0.9, "reason": {"damaged": 0.9, "not_received": 0.1}, "story": 0.1})
    assert out["action"]["value"] == "hold"
    assert "Is the account of what happened in the request inconsistent?" in risk.describe()  # defaults filled
    assert "at least 2" in risk.to_mermaid(plain=True).replace("AT LEAST", "at least")

    alone = Chip("alone", inputs=[Pin("big", ask="Is it big?")], outputs=["out"], params={"cut": 0.7})
    alone.gate("out", Q("big") >= P("cut"))

    class Fixed:
        def answer(self, state, questions, *, model=None):
            assert set(questions) == {"big"}  # the optional pin asks, under its own name
            return {"big": noul(0.9)}

    assert alone.run(Fixed(), "s")["gates"]["out"]["value"] is True
    with pytest.raises(TypeError, match=r"input pins \['reason'\] to wire"):
        risk.run(Fixed(), "s")


def test_pins_settings_and_placeholders_round_trip_through_json():
    risk = risk_chip()
    d = json.loads(json.dumps(risk.to_dict()))
    assert d["chip"]["inputs"][2] == {"name": "reason", "type": "choice", "options": ["damaged", "not_received"]}
    assert {"name": "hold_at", "default": 2} in d["chip"]["params"]
    loaded = Circuit.from_dict(d)
    assert loaded == risk and loaded.pins == risk.pins and loaded.settings == risk.settings

    a, b = host(), host()
    a.mount(risk, "r", {"reason": "why"}, params={"hold_at": 3})
    b.mount(loaded, "r", {"reason": "why"}, params={"hold_at": 3})
    assert a.compile() == b.compile() and a.questions == b.questions

    plain = Chip("plain", inputs=["x"], outputs=["y"], params=["who"])  # untyped pins and required settings round-trip too
    plain.gate("y", Q("x") >= 0.5)
    again = Circuit.from_dict(json.loads(json.dumps(plain.to_dict())))
    assert again.pins["x"].type == "any" and again.settings == {"who": REQUIRED}


def test_a_choice_setting_fills_a_question_s_options():
    triage = Chip("triage", outputs=["team"], params={"teams": {"billing": "Money", "other": None}, "floor": 0.4})
    triage.choice("which", "Which team?", P("teams"))
    triage.gate("team", argmax("which", min_confidence=P("floor")), on_uncertain="escalate")
    c = Circuit()
    c.mount(triage, "t", params={"teams": {"technical": "Bugs", "account": "Login", "other": None}})
    assert c.questions["t.which"]["criteria"] == {"technical": "Bugs", "account": "Login", "other": None}
    assert Circuit.from_dict(json.loads(json.dumps(triage.to_dict()))).questions["which"]["criteria"] == P("teams")


def test_a_dict_or_list_setting_reads_as_words_in_question_text():
    c = Chip("words", outputs=["out"], params={"options": {"billing": "Money", "other": None}, "tags": ["a", "b"]})
    c.noul("q", "Pick from: {options}. Tags: {tags}.")
    c.gate("out", Q("q") >= 0.5)
    host = Circuit()
    host.mount(c, "w")
    assert host.questions["w.q"]["instructions"] == "Pick from: billing (Money); other. Tags: a, b."


def test_review_fixes_hold():
    # user data shaped like the old marker survives JSON
    c = Chip("data", outputs=["out"], params={"options": {"param": "a real option"}})
    c.choice("q", "Pick: {options}", P("options"))
    c.gate("out", argmax("q"))
    back = Circuit.from_dict(json.loads(json.dumps(c.to_dict())))
    assert back.settings["options"] == {"param": "a real option"}

    # a setting's value is never itself rewritten by another setting
    text = Chip("text", outputs=["out"], params={"question": "Is {text} about billing?", "text": "the text"})
    text.noul("q", "{question}")
    text.gate("out", Q("q") >= 0.5)
    host = Circuit()
    host.mount(text, "t")
    assert host.questions["t.q"]["instructions"] == "Is {text} about billing?"

    # a broken case is a failure, and the run carries on
    from decision_circuits.parts import tool_guard, verified_classifier

    vc = verified_classifier()
    fails = vc.test([{"name": "no settings", "answers": {}, "expect": {"label": "a"}}, *vc.tests])
    assert [f["case"] for f in fails] == ["no settings"] and fails[0]["got"].startswith("error: chip 'verified_classifier' needs settings")

    # a setting with allowed values is checked at mount, and survives JSON
    with pytest.raises(ValueError, match=r"setting 'when_requested' must be one of \['ask', 'allow', 'block'\], got 'deny'"):
        Circuit().mount(tool_guard(), "g", params={"when_requested": "deny"})
    assert Circuit.from_dict(json.loads(json.dumps(tool_guard().to_dict()))).allowed == {"when_requested": ["ask", "allow", "block"]}
