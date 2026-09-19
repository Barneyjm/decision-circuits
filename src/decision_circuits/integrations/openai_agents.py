"""OpenAI Agents SDK guardrails driven by a decision circuit.

Three factories, one per guardrail kind the SDK offers:

    from agents import Agent
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
when the gate's action is `block`; `output_info` carries the full gate
results for tracing. The SDK has no "ask a human" path for these, so
an uncertain gate trips too unless `actions` says otherwise.

Tool guardrails have a middle option: `block` rejects the call and
returns `reject_message` to the model as the tool result; `ask` does the
same with a message saying a human must approve, which is the closest
the SDK offers; `allow` runs the tool.

Requires `openai-agents>=0.22` (`pip install "decision-circuits[openai-agents]"`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from agents import Agent, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, RunContextWrapper, input_guardrail, output_guardrail
    from agents.tool_guardrails import ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolInputGuardrailData, tool_input_guardrail
except ImportError as e:  # pragma: no cover
    raise ImportError("decision_circuits.integrations.openai_agents needs openai-agents>=0.22: pip install 'decision-circuits[openai-agents]'") from e

from decision_circuits.dsl import Circuit
from decision_circuits.integrations._policy import Action, RunOutput, decide, explain, tool_state
from decision_circuits.types import Backend


def _run(circuit: Circuit, backend: Backend, state: Any, model: str | None) -> RunOutput:
    answers = backend.answer(state, circuit.questions, model=model or circuit.model)
    return {"answers": answers, "gates": circuit.evaluate(answers)}


def _input_text(input: Any) -> Any:
    if isinstance(input, str):
        return input
    return [item if isinstance(item, str | Mapping) else getattr(item, "__dict__", str(item)) for item in input]


def circuit_input_guardrail(
    circuit: Any,
    backend: Any,
    *,
    gate: str,
    actions: Mapping[Any, Action] | None = None,
    model: str | None = None,
    name: str | None = None,
    run_in_parallel: bool = True,
) -> InputGuardrail[Any]:
    """Trip when the circuit's gate blocks (or is uncertain) on the run input."""

    def guard(ctx: RunContextWrapper[Any], agent: Agent[Any], input: str | list[Any]) -> GuardrailFunctionOutput:
        out = _run(circuit, backend, {"input": _input_text(input), "agent": getattr(agent, "name", None)}, model)
        action = decide(out["gates"], gate, actions)
        return GuardrailFunctionOutput(output_info={**out, "action": action, "reason": explain(out["gates"], gate)}, tripwire_triggered=action != "allow")

    guard.__name__ = name or f"circuit_input_{gate}"
    return input_guardrail(guard, name=name, run_in_parallel=run_in_parallel)


def circuit_output_guardrail(
    circuit: Circuit, backend: Backend, *, gate: str, actions: Mapping[Any, Action] | None = None, model: str | None = None, name: str | None = None
) -> OutputGuardrail[Any]:
    """Trip when the circuit's gate blocks (or is uncertain) on the agent's final output."""

    def guard(ctx: RunContextWrapper[Any], agent: Agent[Any], output: Any) -> GuardrailFunctionOutput:
        out = _run(
            circuit,
            backend,
            {
                "output": _input_text(output) if isinstance(output, str | list) else getattr(output, "__dict__", str(output)),
                "agent": getattr(agent, "name", None),
            },
            model,
        )
        action = decide(out["gates"], gate, actions)
        return GuardrailFunctionOutput(output_info={**out, "action": action, "reason": explain(out["gates"], gate)}, tripwire_triggered=action != "allow")

    guard.__name__ = name or f"circuit_output_{gate}"
    return output_guardrail(guard, name=name)


def circuit_tool_guardrail(
    circuit: Any,
    backend: Any,
    *,
    gate: str,
    actions: Mapping[Any, Action] | None = None,
    model: str | None = None,
    name: str | None = None,
    reject_message: str = "This tool call was blocked by a decision circuit and was not executed. {reason}",
    ask_message: str = "This tool call needs human approval before it can run; a decision circuit could not decide. {reason}",
) -> ToolInputGuardrail[Any]:
    """Guard one tool's calls: allow, reject with a message, or ask (as a reject that says so)."""

    def guard(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        state = tool_state(ctx.tool_name, ctx.tool_arguments, extra={"agent": getattr(data.agent, "name", None)})
        out = _run(circuit, backend, state, model)
        action = decide(out["gates"], gate, actions)
        reason = explain(out["gates"], gate)
        info = {**out, "action": action, "reason": reason}
        if action == "allow":
            return ToolGuardrailFunctionOutput.allow(output_info=info)
        template = reject_message if action == "block" else ask_message
        return ToolGuardrailFunctionOutput.reject_content(template.format(reason=reason), output_info=info)

    guard.__name__ = name or f"circuit_tool_{gate}"
    return tool_input_guardrail(guard, name=name)


__all__ = ["circuit_input_guardrail", "circuit_output_guardrail", "circuit_tool_guardrail"]
