import pytest

from decision_circuits import Circuit, Q, argmax, drop, set_to
from decision_circuits.interventions import segments


class Keyword:
    """Answers from the state's text: a refund is owed when a receipt is mentioned,
    VIPs go to the priority desk."""

    def __init__(self):
        self.calls = 0

    def answer(self, state, questions, *, model=None):
        self.calls += 1
        text = str(state).lower()
        vip = 0.9 if "vip" in text else 0.1
        return {
            "refund": {"type": "noul", "noul": 0.9 if "receipt" in text else 0.2},
            "desk": {
                "type": "choice",
                "choice": "priority" if vip > 0.5 else "general",
                "probabilities": {"priority": vip, "general": 1 - vip},
                "confidence": 0.5,
            },
            "tags": {"type": "multi", "selected": [], "probabilities": {"late": 0.8 if "late" in text else 0.1}},
        }


def circuit():
    c = Circuit()
    c.noul("refund", "Is a refund owed?")
    c.choice("desk", "Which desk?", {"priority": None, "general": None})
    c.gate("pay", Q("refund") >= 0.5)
    c.gate("route", argmax("desk"))
    return c


STATE = {"message": "My order came late.", "receipt": "#4411", "tier": "standard"}


def test_intervene_reports_flipped_gates_and_moved_answers():
    b = Keyword()
    r = circuit().intervene(b, STATE, {"no receipt": drop("receipt"), "vip": set_to("tier", "vip"), "same": dict(STATE)}, workers=1)
    assert b.calls == 4  # baseline + three
    assert r["baseline"]["gates"]["pay"]["value"] is True
    nr = r["effects"]["no receipt"]
    assert nr["flipped"] == ["pay"] and nr["gates"]["pay"]["after"] is False
    assert nr["answers"]["refund"] == pytest.approx({"option": "yes", "p_before": 0.9, "p_after": 0.2, "dp": -0.7})
    assert "receipt" in STATE and "receipt" not in nr["state"]  # the edit copies
    vip = r["effects"]["vip"]
    assert vip["flipped"] == ["route"] and (vip["gates"]["route"]["before"], vip["gates"]["route"]["after"]) == ("general", "priority")
    assert r["effects"]["same"]["flipped"] == [] and r["effects"]["same"]["answers"]["tags[late]"]["dp"] == pytest.approx(0)


def test_reusing_a_baseline_saves_the_call():
    b, c = Keyword(), circuit()
    base = c.run(b, STATE)
    r = c.intervene(b, STATE, {"no receipt": drop("receipt")}, baseline=base)
    assert b.calls == 2 and r["baseline"] is base


def test_ablate_fields_and_sentences():
    b = Keyword()
    r = circuit().ablate(b, STATE)
    assert list(r["effects"]) == ["-message", "-receipt", "-tier"]
    assert r["effects"]["-receipt"]["flipped"] == ["pay"] and r["effects"]["-message"]["answers"]["tags[late]"]["dp"] < 0
    text = "The parcel was late. I kept the receipt. Please help."
    r = circuit().ablate(Keyword(), text)
    assert [k for k, e in r["effects"].items() if e["flipped"]] == ["-[1] I kept the receipt."]


def test_segment_limit_keeps_the_rest_of_the_text():
    cut = segments("One. Two. Three. Four.", limit=2)
    assert list(cut) == ["-[0] One.", "-[1] Two."]
    assert cut["-[0] One."](None) == "Two. Three. Four."


def test_set_to_nested_path_and_type_errors():
    s = {"order": {"items": [{"qty": 1}]}}
    assert set_to("order", "items", 0, "qty", 3)(s)["order"]["items"][0]["qty"] == 3 and s["order"]["items"][0]["qty"] == 1
    with pytest.raises(TypeError):
        segments("text", unit="field")
    with pytest.raises(ValueError):
        set_to("x")


def test_sentence_ablation_keeps_the_rest_of_the_layout():
    cut = segments("Refund?\nThanks.  Bye.")
    assert list(cut) == ["-[0] Refund?", "-[1] Thanks.", "-[2] Bye."]
    assert cut["-[0] Refund?"](None) == "Thanks.  Bye."
    assert cut["-[1] Thanks."](None) == "Refund?\nBye."
    assert segments("a\n\nb", unit="line")["-[1] b"](None) == "a\n\n"


def test_unknown_unit_and_empty_drop_are_refused():
    with pytest.raises(ValueError):
        segments("x. y.", unit="lines")
    with pytest.raises(ValueError):
        drop()


class Picky(Keyword):
    """Refuses a state without a message, as a server refuses a locate question with no text."""

    def answer(self, state, questions, *, model=None):
        if isinstance(state, dict) and "message" not in state:
            raise ValueError("422: locate needs a state with text in it")
        return super().answer(state, questions, model=model)


def test_one_edited_state_failing_leaves_the_others_reported():
    r = circuit().ablate(Picky(), STATE, workers=1)
    bad = r["effects"]["-message"]
    assert bad["run"] is None and "locate needs a state" in bad["error"] and bad["flipped"] == []
    good = r["effects"]["-receipt"]
    assert good["error"] is None and good["flipped"] == ["pay"]


def test_a_failing_baseline_still_raises():
    with pytest.raises(ValueError):
        circuit().intervene(Picky(), {"receipt": "#1"}, {"x": drop("receipt")})


def test_an_edit_that_raises_fails_only_its_own_intervention():
    def broken(_state):
        raise KeyError("no such field")

    r = circuit().intervene(Keyword(), STATE, {"broken": broken, "no receipt": drop("receipt")}, workers=1)
    assert "KeyError" in r["effects"]["broken"]["error"] and r["effects"]["broken"]["state"] is None
    assert r["effects"]["no receipt"]["flipped"] == ["pay"]
