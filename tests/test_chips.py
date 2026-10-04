"""Chips: packaged sub-circuits with pins, mounted under a namespace."""

from __future__ import annotations

import json

import pytest

from decision_circuits import Chip, Circuit, G, Q, argmax, at_least, consistent, count, majority, order, verify
from decision_circuits.backends import SystemOne


def escalation() -> Chip:
    esc = Chip("escalation", inputs=["angry", "pii", "dept"], outputs=["out"], version="1.0")
    esc.noul("threat", "Does the writer threaten legal action or a chargeback?")
    esc.gate("hot", at_least(2, "angry", "pii", "threat", Q("dept")["billing"]))
    esc.gate("out", G("hot") >= 0.6, on_uncertain="escalate")
    esc.add_test({"angry": 0.9, "pii": 0.1, "threat": 0.9, "dept": {"billing": 0.9, "other": 0.1}}, {"out": True}, name="furious billing")
    esc.add_test({"angry": 0.1, "pii": 0.1, "threat": 0.1, "dept": {"billing": 0.1, "other": 0.9}}, {"out": False})
    return esc


def host() -> Circuit:
    c = Circuit()
    c.noul("angry", "Is the writer angry?")
    c.noul("pii", "Is there personal information?")
    c.choice("dept", "Which team?", {"billing": None, "other": None})
    return c


def noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


ANSWERS = {
    "angry": noul(0.9),
    "pii": noul(0.2),
    "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "other": 0.2}, "confidence": 0.28},
    "esc.threat": noul(0.7),
    "esc2.threat": noul(0.1),
}


def test_mounting_decides_like_the_same_gates_written_out_by_hand():
    c = host()
    pins = c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii", "dept": "dept"})
    c.gate("page", pins["out"] >= 0.5)

    by_hand = host()
    by_hand.noul("esc.threat", "Does the writer threaten legal action or a chargeback?")
    by_hand.gate("esc.hot", at_least(2, "angry", "pii", "esc.threat", Q("dept")["billing"]))
    by_hand.gate("esc.out", G("esc.hot") >= 0.6, on_uncertain="escalate")
    by_hand.gate("page", G("esc.out") >= 0.5)

    assert c.questions == by_hand.questions
    assert c.compile() == by_hand.compile()
    assert c.evaluate(ANSWERS) == by_hand.evaluate(ANSWERS)
    assert pins == {"out": G("esc.out")}


def test_a_chip_mounted_twice_is_two_independent_copies():
    c = host()
    a = c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii", "dept": "dept"})
    b = c.mount(escalation(), "esc2", {"angry": "angry", "pii": "pii", "dept": "dept"})
    r = c.evaluate(ANSWERS)
    assert a != b
    assert r["esc.hot"]["p"] == pytest.approx(0.9204)  # its own threat question said 0.7
    assert r["esc2.hot"]["p"] == pytest.approx(0.7932)  # its own threat question said 0.1
    assert set(c.questions) == {"angry", "pii", "dept", "esc.threat", "esc2.threat"}


def test_params_fill_the_question_text_per_mount():
    side = Chip("side", outputs=["threat"], params=["who"])
    side.noul("q", "Does {who} threaten a chargeback? Answer {strictly}.", true="{who} says chargeback")
    side.gate("threat", Q("q") >= 0.5)
    c = Circuit()
    c.mount(side, "cust", params={"who": "the customer"})
    c.mount(side, "agent", params={"who": "the agent"})
    assert c.questions["cust.q"]["instructions"] == "Does the customer threaten a chargeback? Answer {strictly}."
    assert c.questions["agent.q"]["criteria"]["true"] == "the agent says chargeback"
    assert side.questions["q"]["instructions"].startswith("Does {who}")  # the chip itself is untouched
    with pytest.raises(ValueError, match=r"takes params \['who'\]"):
        c.mount(side, "x")
    with pytest.raises(ValueError, match="takes params"):
        c.mount(side, "y", params={"who": "a", "extra": "b"})
    assert Circuit.from_dict(json.loads(json.dumps(side.to_dict()))).params == ["who"]


def test_a_pin_can_be_wired_to_an_option_or_to_a_host_gate():
    chip = Chip("strict", inputs=["x"], outputs=["yes"])
    chip.gate("yes", Q("x") >= 0.6)
    c = host()
    c.gate("route", argmax("dept"))
    on_option = c.mount(chip, "a", {"x": Q("dept")["billing"]})
    on_gate = c.mount(chip, "b", {"x": G("route")["billing"]})
    on_noul = c.mount(chip, "c", {"x": "pii"})
    r = c.evaluate(ANSWERS)
    assert c.compile()["a.yes"]["input"] == "dept:billing"
    assert r[on_option["yes"].ref]["value"] is True  # 0.8
    assert r[on_gate["yes"].ref]["value"] is True  # the decided route reads 1.0 billing
    assert r[on_noul["yes"].ref]["value"] is False  # 0.2


def test_wiring_mistakes_are_refused_by_name():
    c = host()
    with pytest.raises(ValueError, match=r"not wired"):
        c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii"})
    with pytest.raises(ValueError, match=r"no input pins \['mood'\]"):
        c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii", "dept": "dept", "mood": "angry"})
    with pytest.raises(ValueError, match="already names an option"):
        c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii", "dept": "dept:billing"})
    c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii", "dept": "dept"})
    with pytest.raises(ValueError, match="already used"):
        c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii", "dept": "dept"})
    with pytest.raises(ValueError, match="plain name"):
        c.mount(escalation(), "a.b", {"angry": "angry", "pii": "pii", "dept": "dept"})
    with pytest.raises(TypeError, match="takes a Chip"):
        c.mount(host(), "x", {})  # type: ignore[arg-type]


def test_check_finds_a_stray_reference_and_a_missing_output():
    stray = Chip("stray", inputs=["a"], outputs=["out"])
    stray.gate("out", (Q("a") & Q("b")) >= 0.5)
    with pytest.raises(ValueError, match="'b' is not an input pin"):
        stray.check()
    missing = Chip("missing", inputs=["a"], outputs=["out"])
    missing.gate("other", Q("a") >= 0.5)
    with pytest.raises(ValueError, match=r"output pins \['out'\]"):
        missing.check()


def test_a_chip_evaluates_on_its_own_with_shorthand_values():
    r = escalation().evaluate({"angry": 0.9, "pii": 0.1, "threat": 0.9, "dept": {"billing": 0.9, "other": 0.1}})
    assert r["out"]["value"] is True
    assert r["hot"]["p"] == pytest.approx(r["out"]["p"])
    with pytest.raises(ValueError, match="no value for 'threat'"):
        escalation().evaluate({"angry": 0.9, "pii": 0.1, "dept": {"billing": 1.0}})
    shorthand = escalation()
    shorthand.gate("team", argmax("dept", min_confidence=0.5))  # reads the confidence the shorthand fills in
    assert shorthand.evaluate({"angry": 0.1, "pii": 0.1, "threat": 0.1, "dept": {"billing": 0.5, "other": 0.5}})["team"]["outcome"] == "abstain"
    with pytest.raises(TypeError, match="not a bool"):
        escalation().evaluate({"angry": True, "pii": 0.1, "threat": 0.1, "dept": {"billing": 1.0}})


def test_test_vectors_report_failures_and_read_jsonl(tmp_path):
    esc = escalation()
    assert esc.test() == []
    path = tmp_path / "cases.jsonl"
    near = {"angry": 0.62, "pii": 0.62, "threat": 0.62, "dept": {"billing": 0.62, "other": 0.38}}
    path.write_text(json.dumps({"name": "on the fence", "answers": near, "expect": {"out": False}}) + "\n\n")
    failures = esc.test(path)
    assert [(f["case"], f["gate"], f["expected"], f["got"]) for f in failures] == [("on the fence", "out", False, True)]
    assert esc.test([{"answers": near, "expect": {"nope": 1}}])[0]["got"] == "no such gate"


def test_circuits_and_chips_round_trip_through_json():
    c = host()
    c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii", "dept": "dept"})
    c.gate("page", G("esc.out") >= 0.5).on_uncertain("default", False)
    back = Circuit.from_dict(json.loads(json.dumps(c.to_dict())))
    assert type(back) is Circuit
    assert back.compile() == c.compile()
    assert back.evaluate(ANSWERS) == c.evaluate(ANSWERS)
    assert back.mounts == c.mounts

    esc = escalation()
    d = json.loads(json.dumps(esc.to_dict()))
    assert d["chip"]["requires"] == ["noul"] and d["chip"]["version"] == "1.0"
    loaded = Circuit.from_dict(d)
    assert isinstance(loaded, Chip) and loaded == esc
    assert loaded.test() == []

    with pytest.raises(ValueError, match="unknown circuit format"):
        Circuit.from_dict({**d, "format": "decision-circuits/99"})


def test_a_chip_built_from_chips_mounts_with_its_inner_boards_rewired():
    inner = Chip("pass", inputs=["x"], outputs=["y"])
    inner.gate("y", Q("x") >= 0.5)
    outer = Chip("wrap", inputs=["a"], outputs=["z"])
    got = outer.mount(inner, "in", {"x": "a"})
    outer.gate("z", ~got["y"] >= 0.5)
    assert outer.test([{"answers": {"a": 0.9}, "expect": {"z": False}}]) == []

    c = host()
    pins = c.mount(outer, "w", {"a": "angry"})
    m = c.to_mermaid(plain=True)
    assert m.index('subgraph M_w["w · wrap"]') < m.index('subgraph M_w__in["in · pass"]') < m.index("<b>y</b>")  # the inner box sits in the outer
    assert 'subgraph OUT["Decisions"]' not in m  # every decision is inside a chip: no empty column
    assert [g.name for g in c.gates] == ["w.in.y", "w.z"]
    assert c.mounts["w.in"]["pins"] == {"x": "angry"}
    assert c.evaluate(ANSWERS)[pins["z"].ref]["value"] is False


def test_a_chip_with_inputs_does_not_run_on_its_own():
    with pytest.raises(TypeError, match="mount it"):
        escalation().run(object(), "state")  # type: ignore[arg-type]

    class Fixed:
        def answer(self, state, questions, *, model=None):
            return {"q": noul(0.9)}

    alone = Chip("alone", outputs=["out"])
    alone.noul("q", "?")
    alone.gate("out", Q("q") >= 0.5)
    assert alone.run(Fixed(), "s")["gates"]["out"]["value"] is True


def test_mermaid_draws_each_chip_as_a_box_with_its_pins():
    c = host()
    c.mount(escalation(), "esc", {"angry": "pii", "pii": "angry", "dept": "dept"})
    m = c.to_mermaid(plain=True)
    assert 'subgraph M_esc["esc · escalation"]' in m
    assert "q_esc__threat" in m and "<b>threat</b>" in m
    assert "q_angry -->|pii| g_esc__hot" in m  # crossed wires are labelled with the pin
    assert "q_dept -->|dept: billing| g_esc__hot" not in m and "q_dept -->|billing| g_esc__hot" in m
    assert "<b>Out</b><br/>≥ 60%" in m  # a decision nothing consumes, drawn inside its chip
    c.gate("page", G("esc.out") >= 0.5)
    assert "<b>out</b><br/>≥ 60%" in c.to_mermaid(plain=True)  # consumed: a logic node, its tau shown


def test_question_types_and_typesafe_refuses_what_it_does_not_answer():
    c = host()
    c.multi("issues", "Which apply?", {"late": None, "damaged": None})
    assert c.question_types == ["choice", "multi", "noul"]

    class Client:
        def post(self, url, json=None, headers=None):
            raise AssertionError("refused before any request")

    with pytest.raises(ValueError, match="'issues' is multi"):
        SystemOne(client=Client()).answer("s", c.questions)
    assert SystemOne("http://localhost:8901/v1/systemone").question_types is None
    assert SystemOne(question_types=["noul", "multi"]).question_types == ("noul", "multi")


def test_describe_says_what_is_asked_and_decided_in_words():
    c = host()
    c.mount(escalation(), "esc", {"angry": "angry", "pii": "pii", "dept": "dept"})
    c.gate("page", G("esc.out") >= 0.5, on_uncertain="escalate")
    text = c.describe()
    assert "- **angry** (yes/no): Is the writer angry?" in text
    assert "*[billing / other]*" in text
    assert "## Chip `esc`: escalation" in text
    assert "- `dept` ← “Which team?”" in text
    assert "- **threat** (yes/no): Does the writer threaten legal action or a chargeback?" in text  # under the chip, short name
    assert "2 or more of these hold" in text and "  - the answer to “Which team?” is billing" in text
    assert "yes when it is at least 60% likely that the `hot` decision is yes." in text  # same chip: short name
    assert "yes when it is at least 50% likely that the `esc.out` decision is yes." in text  # from the host: full name
    assert "*Between 50% and 70% it is too close to call, so it escalates to a person.*" in text

    r = c.evaluate(ANSWERS)
    ran = c.describe(r, ANSWERS)
    assert "Is the writer angry? → **yes 90%**" in ran
    assert "Result: **yes** (92%)." in ran

    near = escalation()
    out = near.evaluate({"angry": 0.62, "pii": 0.62, "threat": 0.62, "dept": {"billing": 0.62, "other": 0.38}})
    said = near.describe(out)
    assert sum(said.count(s) for s in ("Result:", "Escalated", "Abstained")) == 2  # one outcome line per gate
    datasheet = escalation().describe()
    assert datasheet.startswith("# Chip `escalation` v1.0")
    assert "**Input pins:** `angry`, `pii`, `dept`" in datasheet
    assert "the `angry` input pin is yes" in datasheet


def test_describe_reads_every_gate_kind():
    c = Circuit()
    c.choice("a", "Which colour?", {"red": None, "blue": None})
    c.choice("b", "Which colour, asked again?", {"red": None, "blue": None})
    c.noul("ok", "Is that supported by the text?")
    c.score("size", "How big?", ["small", "medium", "large"])
    c.multi("tags", "Which apply?", {"x": None, "y": None})
    c.gate("pick", argmax("a", min_confidence=0.4))
    c.gate("vote", majority("a", "b"))
    c.gate("checked", verify("a", "ok", tau=0.7))
    c.gate("bucket", order("size", [0.5, 1.5]))
    c.gate("n", count("tags"))
    c.gate("same", consistent("a", "b"))
    c.gate("big", ~Q("size")[0] >= 0.5, on_uncertain="default", default=False)
    text = c.describe()
    assert "picks the most likely answer to “Which colour?”. If its confidence is under 0.4, it abstains." in text
    assert "picks the answer most of these agree on" in text
    assert "then checks it with the answer to “Is that supported by the text?” is yes; if the check is under 70%" in text
    assert "into 3 buckets by its expected level, cut at 0.5, 1.5" in text
    assert "counts how many of the options of “Which apply?” hold" in text
    assert "checks that these two agree" in text
    assert "not (the answer to “How big?” is small)" in text and "it falls back to False" in text
    assert "*[small < medium < large]*" in text


def test_mermaid_shows_what_each_question_asks():
    c = host()
    c.noul("quote", 'Does the "customer" say <urgent> #now?')
    m = c.to_mermaid(plain=True)
    assert "<b>angry</b><br/>Is the writer angry?<br/>noul" in m
    assert "#quot;customer#quot;" in m and "#lt;urgent#gt;" in m and "#35;now" in m
    assert "Is the writer angry?" not in c.to_mermaid(plain=True, text=False)
