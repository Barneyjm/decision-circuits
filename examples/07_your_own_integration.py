"""Plug a circuit into a framework that has no adapter yet.

Every shipped integration (LangChain, OpenAI Agents, Claude Agent SDK)
is the same twenty lines: build the state the framework hands you, ask
a `CircuitPolicy` for a judgment, map its action onto the framework's
own result type. This example does it for a made-up framework with a
`before_tool(call) -> "run" | "skip" | "confirm"` hook, using the
keyword backend from example 03 so it runs offline.

    uv run python examples/07_your_own_integration.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from decision_circuits import Circuit, Q
from decision_circuits.integrations import CircuitPolicy, tool_state

# Reuse the keyword backend from example 03 (it is just a class with `answer`).
spec = importlib.util.spec_from_file_location("ex03", Path(__file__).with_name("03_your_own_backend.py"))
assert spec and spec.loader
ex03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex03)
KeywordBackend = ex03.KeywordBackend


# --- the framework you are integrating with (pretend) ----------------------
class ToyAgent:
    """Calls `before_tool` before each tool; expects "run", "skip", or "confirm"."""

    def __init__(self, before_tool):
        self.before_tool = before_tool

    def call(self, name: str, args: dict[str, Any]) -> str:
        verdict = self.before_tool({"name": name, "args": args})
        return f"{name}({args}) -> {verdict}"


# --- the integration: this is the whole thing --------------------------------
def circuit_before_tool(policy: CircuitPolicy):
    to_framework = {"allow": "run", "block": "skip", "ask": "confirm"}

    def before_tool(call: dict[str, Any]) -> str:
        j = policy.judge(tool_state(call["name"], call["args"]))
        print(f"    [{j.action}] {j.reason}")
        return to_framework[j.action]

    return before_tool


# --- a circuit for it --------------------------------------------------------
c = Circuit()
c.noul("pii", "Do the tool arguments contain personal information?")
c.noul("angry", "Is the request hostile?")
c.gate("block", (Q("pii") | Q("angry")) >= 0.6, band=0.2, on_uncertain="escalate")

policy = CircuitPolicy(c, KeywordBackend(), gate="block", actions={"escalate": "ask"})
agent = ToyAgent(circuit_before_tool(policy))

print(agent.call("send_email", {"to": "x@example.com", "body": "your card 4412 was charged"}))
print(agent.call("send_email", {"to": "team", "body": "standup moved to 10"}))
print(agent.call("send_email", {"to": "team", "body": "this is unacceptable!"}))
