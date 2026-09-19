"""Agent-SDK integrations. Each submodule imports its SDK on import, so
import the one you use:

    decision_circuits.integrations.langchain          CircuitToolGuard, CircuitRouter, circuit_when
    decision_circuits.integrations.openai_agents      circuit_input_guardrail, circuit_output_guardrail, circuit_tool_guardrail
    decision_circuits.integrations.claude_agent_sdk   circuit_pre_tool_use, circuit_can_use_tool

All of them are thin adapters over one object, `CircuitPolicy`, which
runs a circuit on a state and maps one gate's result to `allow`,
`block`, or `ask`. Uncertain outcomes go to a human wherever the SDK
has a way to ask. To integrate a framework that is not listed, build
its state, call `policy.judge(state)`, and map the action to the
framework's own result type; see `docs/extending.md`.
"""

from decision_circuits.integrations._policy import DEFAULT_ACTIONS, Action, CircuitPolicy, Judgment, explain, tool_state

__all__ = ["DEFAULT_ACTIONS", "Action", "CircuitPolicy", "Judgment", "explain", "tool_state"]
