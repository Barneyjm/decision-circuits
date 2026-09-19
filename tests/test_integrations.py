"""Integration tests run against the real SDKs (installed as dev deps),
with a scripted backend so no network is involved. Each block skips when
its SDK is missing."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from decision_circuits import Circuit, Q, answer_from_probabilities, result_key
from decision_circuits.integrations import CircuitPolicy, tool_state


def guard_circuit() -> Circuit:
    c = Circuit()
    c.noul("risky", "Would executing `tool_call` be risky?", true="Destructive or unauthorized", false="Safe")
    c.noul("authorized", "Did the user explicitly ask for this?")
    c.gate("block", (Q("risky") & ~Q("authorized")) >= 0.5, band=0.15, on_uncertain="escalate")
    return c


class ScriptedBackend:
    """Answers from a table keyed by tool name (or 'input' for run input)."""

    def __init__(self, table: dict[str, tuple[float, float]]):
        self.table = table
        self.states: list[Any] = []

    def answer(self, state, questions, *, model=None):
        self.states.append(state)
        key = state["tool_call"]["name"] if isinstance(state, dict) and "tool_call" in state else "input"
        risky, auth = self.table[key]
        return {"risky": {"type": "noul", "noul": risky}, "authorized": {"type": "noul", "noul": auth}}


TABLE = {
    "delete_file": (0.95, 0.05),  # block: 0.95 * 0.95 = 0.90
    "read_file": (0.05, 0.50),  # allow: 0.025
    "send_email": (0.60, 0.20),  # 0.48 -> inside the band around 0.5 -> escalate -> ask
    "input": (0.95, 0.05),
}


# --- policy ------------------------------------------------------------------


def test_policy_judges_and_explains():
    be = ScriptedBackend(TABLE)
    policy = CircuitPolicy(guard_circuit(), be, gate="block")
    for tool, expected in [("delete_file", "block"), ("read_file", "allow"), ("send_email", "ask")]:
        assert policy.judge(tool_state(tool, {})).action == expected, tool
    j = policy.judge(tool_state("send_email", {}))
    assert "gate `block`" in j.reason and "escalate" in j.reason and j.output["gates"]["block"]["outcome"] == "escalate"
    assert policy.last is j
    strict = CircuitPolicy(guard_circuit(), be, gate="block", actions={"escalate": "block"})
    assert strict.judge(tool_state("send_email", {})).action == "block"
    with pytest.raises(KeyError):
        CircuitPolicy(guard_circuit(), be, gate="nope").judge(tool_state("read_file", {}))


def test_policy_categorical_and_fallback():
    c = Circuit()
    c.choice("route", "Where?", {"billing": None, "risky": None})
    from decision_circuits import argmax

    c.gate("route", argmax("route"))

    class BE:
        def answer(self, state, questions, *, model=None):
            return {"route": answer_from_probabilities(questions["route"], [0.9, 0.1])}

    assert result_key({"value": "billing", "outcome": "decided", "p": 0.9, "confidence": 0.5, "uncertain": False, "trace": []}) == "billing"
    assert CircuitPolicy(c, BE(), gate="route", actions={"billing": "allow"}).judge("x").action == "allow"
    assert CircuitPolicy(c, BE(), gate="route").judge("x").action == "block"  # unlisted value never silently allows
    assert CircuitPolicy(c, BE(), gate="route", uncertain_fallback="ask").judge("x").action == "ask"


# --- LangChain ---------------------------------------------------------------


def _tools(executed: list[str]):
    from langchain_core.tools import StructuredTool

    def read_file(path: str) -> str:
        executed.append(f"read_file:{path}")
        return "contents"

    def delete_file(path: str) -> str:
        executed.append(f"delete_file:{path}")
        return "deleted"

    def send_email(to: str) -> str:
        executed.append(f"send_email:{to}")
        return "sent"

    return [
        StructuredTool.from_function(read_file, name="read_file", description="Read a file."),
        StructuredTool.from_function(delete_file, name="delete_file", description="Delete a file permanently."),
        StructuredTool.from_function(send_email, name="send_email", description="Send an email."),
    ]


def _fake_model(*msgs):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class ToolFake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return ToolFake(messages=iter(msgs))


def _make_agent(guard, tool_calls, checkpointer=None):
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage

    executed: list[str] = []
    model = _fake_model(
        AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{i}"} for i, (n, a) in enumerate(tool_calls)]),
        AIMessage(content="done"),
    )
    agent = create_agent(model, tools=_tools(executed), middleware=[guard], checkpointer=checkpointer)
    return agent, executed


def test_langchain_tool_guard_blocks_and_allows():
    pytest.importorskip("langchain.agents")
    from langchain_core.messages import ToolMessage

    from decision_circuits.integrations.langchain import CircuitToolGuard

    be = ScriptedBackend(TABLE)
    guard = CircuitToolGuard(guard_circuit(), be, gate="block", tools=["delete_file", "send_email"])
    agent, executed = _make_agent(guard, [("read_file", {"path": "a"}), ("delete_file", {"path": "b"})])
    out = agent.invoke({"messages": [("user", "clean up")]})
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert executed == ["read_file:a"]  # read_file unguarded -> ran; delete_file blocked
    blocked = next(m for m in tool_msgs if m.name == "delete_file")
    assert blocked.status == "error" and "blocked by a decision circuit" in blocked.content and "p=0.90" in blocked.content
    assert be.states[0]["tool_call"] == {"name": "delete_file", "args": {"path": "b"}}
    assert "Delete a file permanently" in be.states[0]["tool_description"]
    assert guard.policy.last.output["gates"]["block"]["value"] is True


def test_langchain_tool_guard_asks_via_interrupt_and_resumes():
    pytest.importorskip("langchain.agents")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from decision_circuits.integrations.langchain import CircuitToolGuard

    guard = CircuitToolGuard(guard_circuit(), ScriptedBackend(TABLE), gate="block")
    agent, executed = _make_agent(guard, [("send_email", {"to": "x@y"})], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    out = agent.invoke({"messages": [("user", "email them")]}, config=cfg)
    assert "__interrupt__" in out and executed == []
    req = out["__interrupt__"][0].value
    assert req["action_requests"][0]["name"] == "send_email" and "could not decide" in req["action_requests"][0]["description"]
    out = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfg)
    assert executed == ["send_email:x@y"] and out["messages"][-1].content == "done"


def test_langchain_circuit_when_predicate():
    pytest.importorskip("langchain.agents")
    from types import SimpleNamespace

    from decision_circuits.integrations.langchain import circuit_when

    when = circuit_when(guard_circuit(), ScriptedBackend(TABLE), gate="block")
    req = lambda name: SimpleNamespace(tool_call={"name": name, "args": {}, "id": f"call_{name}"}, tool=None, state={"messages": []})
    assert when(req("delete_file")) is True and when(req("send_email")) is True and when(req("read_file")) is False


def test_langchain_router_stores_results_and_picks_model():
    pytest.importorskip("langchain.agents")
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage

    from decision_circuits import argmax
    from decision_circuits.integrations.langchain import CircuitRouter

    c = Circuit()
    c.choice("tier", "How hard is this?", {"simple": None, "hard": None})
    c.gate("tier", argmax("tier", min_confidence=0.3))

    class BE:
        def answer(self, state, questions, *, model=None):
            hard = "prove" in str(state)
            return {"tier": answer_from_probabilities(questions["tier"], [0.1, 0.9] if hard else [0.9, 0.1])}

    small = _fake_model(AIMessage(content="small answered"))
    big = _fake_model(AIMessage(content="big answered"))
    router = CircuitRouter(c, BE(), gate="tier", models={"simple": small, "hard": big}, default_model=big)
    agent = create_agent(small, middleware=[router])
    out = agent.invoke({"messages": [("user", "prove the theorem")]})
    assert out["messages"][-1].content == "big answered"
    assert out["circuit"]["gates"]["tier"]["value"] == "hard"


# --- OpenAI Agents SDK -------------------------------------------------------


def test_openai_agents_guardrails():
    pytest.importorskip("agents")
    from types import SimpleNamespace

    from agents.tool_guardrails import ToolInputGuardrailData

    from decision_circuits.integrations.openai_agents import circuit_input_guardrail, circuit_tool_guardrail

    be = ScriptedBackend(TABLE)
    ig = circuit_input_guardrail(guard_circuit(), be, gate="block")
    res = asyncio.run(ig.run(agent=SimpleNamespace(name="a"), input="do the thing", context=SimpleNamespace()))
    assert res.output.tripwire_triggered is True and res.output.output_info["action"] == "block"

    tg = circuit_tool_guardrail(guard_circuit(), be, gate="block")
    ctx = SimpleNamespace(tool_name="read_file", tool_arguments='{"path": "a"}')
    out = asyncio.run(tg.run(ToolInputGuardrailData(context=ctx, agent=SimpleNamespace(name="a"))))
    assert out.behavior["type"] == "allow"
    ctx = SimpleNamespace(tool_name="delete_file", tool_arguments='{"path": "b"}')
    out = asyncio.run(tg.run(ToolInputGuardrailData(context=ctx, agent=SimpleNamespace(name="a"))))
    assert out.behavior["type"] == "reject_content" and "blocked by a decision circuit" in out.behavior["message"]
    ctx = SimpleNamespace(tool_name="send_email", tool_arguments="{}")
    out = asyncio.run(tg.run(ToolInputGuardrailData(context=ctx, agent=SimpleNamespace(name="a"))))
    assert out.behavior["type"] == "reject_content" and "human approval" in out.behavior["message"]


# --- Claude Agent SDK --------------------------------------------------------


def test_claude_agent_sdk_hook_and_permission_callback():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    from decision_circuits.integrations.claude_agent_sdk import circuit_can_use_tool, circuit_pre_tool_use

    hook = circuit_pre_tool_use(guard_circuit(), ScriptedBackend(TABLE), gate="block", tools={"delete_file", "send_email"})

    def run(name):
        return asyncio.run(hook({"hook_event_name": "PreToolUse", "tool_name": name, "tool_input": {"path": "x"}, "permission_mode": "default"}, "tu1", None))

    assert run("delete_file")["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert run("send_email")["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert run("read_file") == {}  # not in the guarded set
    assert "p=0.90" in run("delete_file")["hookSpecificOutput"]["permissionDecisionReason"]

    cb = circuit_can_use_tool(guard_circuit(), ScriptedBackend(TABLE), gate="block")
    assert isinstance(asyncio.run(cb("read_file", {}, None)), PermissionResultAllow)
    deny = asyncio.run(cb("send_email", {}, None))
    assert isinstance(deny, PermissionResultDeny) and "could not decide" in deny.message


def test_policy_bool_defaults_do_not_match_ints():
    from decision_circuits import order

    c = Circuit()
    c.score("risk", "How risky?", ["low", "mid", "high"])
    c.gate("bucket", order("risk", [0.5, 1.5]))

    class BE:
        def __init__(self, level):
            self.level = level

        def answer(self, state, questions, *, model=None):
            dist = [0.0, 0.0, 0.0]
            dist[self.level] = 1.0
            return {"risk": answer_from_probabilities(questions["risk"], dist)}

    assert CircuitPolicy(c, BE(1), gate="bucket").judge("x").action == "block"  # bucket 1 != True
    assert CircuitPolicy(c, BE(0), gate="bucket").judge("x").action == "block"  # bucket 0 != False
    assert CircuitPolicy(c, BE(0), gate="bucket", actions={0: "allow"}).judge("x").action == "allow"


def test_langchain_replay_after_interrupt_keeps_the_judgment():
    pytest.importorskip("langchain.agents")
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from decision_circuits.integrations.langchain import CircuitToolGuard

    class Flaky:
        """Uncertain on the first call, then confidently safe: a replay must not re-ask."""

        def __init__(self):
            self.n = 0

        def answer(self, state, questions, *, model=None):
            self.n += 1
            risky, auth = (0.60, 0.20) if self.n == 1 else (0.05, 0.9)
            return {"risky": {"type": "noul", "noul": risky}, "authorized": {"type": "noul", "noul": auth}}

    executed: list[str] = []
    guard = CircuitToolGuard(guard_circuit(), Flaky(), gate="block")
    model = _fake_model(AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"to": "x@y"}, "id": "c1"}]), AIMessage(content="done"))
    agent = create_agent(model, tools=_tools(executed), middleware=[guard], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}
    out = agent.invoke({"messages": [("user", "email them")]}, config=cfg)
    assert "__interrupt__" in out
    out = agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config=cfg)
    assert executed == []  # the human's rejection stands
    assert "rejected" in next(m.content for m in out["messages"] if getattr(m, "status", None) == "error")


def test_openai_tool_arguments_are_parsed():
    pytest.importorskip("agents")
    from types import SimpleNamespace

    from agents.tool_guardrails import ToolInputGuardrailData

    from decision_circuits.integrations.openai_agents import circuit_tool_guardrail

    be = ScriptedBackend(TABLE)
    tg = circuit_tool_guardrail(guard_circuit(), be, gate="block")
    asyncio.run(tg.run(ToolInputGuardrailData(context=SimpleNamespace(tool_name="read_file", tool_arguments='{"path": "a"}'), agent=SimpleNamespace(name="a"))))
    assert be.states[-1]["tool_call"]["args"] == {"path": "a"}
