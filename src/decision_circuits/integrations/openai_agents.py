"""OpenAI Agents SDK guardrails driven by a decision circuit.

Three factories, one per guardrail kind the SDK offers:

    from agents import Agent, function_tool
    from decision_circuits.integrations.openai_agents import (
        circuit_input_guardrail, circuit_output_guardrail, circuit_tool_guardrail)

    agent = Agent(
        name="support",
        instructions="...",
        input_guardrails=[circuit_input_guardrail(c, backend, gate="off_topic")],
        output_guardrails=[circuit_output_guardrail(c2, backend, gate="leaks_pii")],
        tools=[function_tool(delete_file, tool_input_guardrails=[circuit_tool_guardrail(c3, backend, gate="block")])],
    )

Input and output guardrails trip (raise the SDK's tripwire exception)
unless the gate's action is `allow`; `output_info` carries the full
judgment for tracing. The SDK has no "ask a human" path for these, so
an uncertain gate trips too unless `actions` says otherwise.

Tool guardrails have a middle option: `block` rejects the call and
returns `reject_message` to the model as the tool result; `ask` does the
same with a message saying a human must approve, which is the closest
the SDK offers; `allow` runs the tool.

All guardrails are async and run the backend in a worker thread, so a
slow classifier never blocks the event loop.

Requires `openai-agents>=0.22` (`pip install "decision-circuits[openai-agents]"`).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

try:
    from agents import Agent, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, RunContextWrapper, input_guardrail, output_guardrail
    from agents.tool_guardrails import ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData, tool_input_guardrail
except ImportError as e:  # pragma: no cover
    raise ImportError("decision_circuits.integrations.openai_agents needs openai-agents>=0.22: pip install 'decision-circuits[openai-agents]'") from e

from decision_circuits.dsl import Circuit
from decision_circuits.integrations._policy import Action, CircuitPolicy, Judgment, tool_state
from decision_circuits.types import Backend


def _args(raw: Any) -> Any:
    """The SDK hands tool arguments as a JSON string; give circuits the dict the other adapters give."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def _info(j: Judgment) -> dict[str, Any]:
    return {"action": j.action, "reason": j.reason, **j.output}


def _run_guardrail(
    kind: str, circuit: Circuit, backend: Backend, *, gate: str, actions: Mapping[Any, Action] | None, name: str | None, **decorator_kw: Any
) -> Any:
    """Shared body of the input and output guardrails: the state is
    {"input"|"output": ..., "agent": name}; trip unless allowed."""
    policy = CircuitPolicy(circuit, backend, gate=gate, actions=actions)
    decorate = input_guardrail if kind == "input" else output_guardrail

    async def guard(ctx: RunContextWrapper[Any], agent: Agent[Any], payload: Any) -> GuardrailFunctionOutput:
        j = await asyncio.to_thread(policy.judge, {kind: payload, "agent": agent.name})
        return GuardrailFunctionOutput(output_info=_info(j), tripwire_triggered=j.action != "allow")

    return decorate(guard, name=name or f"circuit_{kind}_{gate}", **decorator_kw)


def circuit_input_guardrail(
    circuit: Circuit, backend: Backend, *, gate: str, actions: Mapping[Any, Action] | None = None, name: str | None = None, run_in_parallel: bool = True
) -> InputGuardrail[Any]:
    """Trip unless the circuit's gate allows the run input."""
    return _run_guardrail("input", circuit, backend, gate=gate, actions=actions, name=name, run_in_parallel=run_in_parallel)


def circuit_output_guardrail(
    circuit: Circuit, backend: Backend, *, gate: str, actions: Mapping[Any, Action] | None = None, name: str | None = None
) -> OutputGuardrail[Any]:
    """Trip unless the circuit's gate allows the agent's final output."""
    return _run_guardrail("output", circuit, backend, gate=gate, actions=actions, name=name)


def circuit_tool_guardrail(
    circuit: Circuit,
    backend: Backend,
    *,
    gate: str,
    actions: Mapping[Any, Action] | None = None,
    name: str | None = None,
    reject_message: str = "This tool call was blocked by a decision circuit and was not executed. {reason}",
    ask_message: str = "This tool call needs human approval before it can run; a decision circuit could not decide. {reason}",
) -> ToolInputGuardrail[Any]:
    """Guard one tool's calls: allow, reject with a message, or ask (as a reject that says so)."""
    policy = CircuitPolicy(circuit, backend, gate=gate, actions=actions)

    async def guard(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        j = await asyncio.to_thread(policy.judge, tool_state(ctx.tool_name, _args(ctx.tool_arguments), agent=data.agent.name))
        if j.action == "allow":
            return ToolGuardrailFunctionOutput.allow(output_info=_info(j))
        template = reject_message if j.action == "block" else ask_message
        return ToolGuardrailFunctionOutput.reject_content(template.format(reason=j.reason), output_info=_info(j))

    return tool_input_guardrail(guard, name=name or f"circuit_tool_{gate}")


__all__ = ["circuit_input_guardrail", "circuit_output_guardrail", "circuit_tool_guardrail"]
