"""Agent-SDK integrations. Each submodule imports its SDK on import, so
import the one you use:

    decision_circuits.integrations.langchain          CircuitToolGuard, CircuitRouter, circuit_when
    decision_circuits.integrations.openai_agents      circuit_input_guardrail, circuit_output_guardrail, circuit_tool_guardrail
    decision_circuits.integrations.claude_agent_sdk   circuit_pre_tool_use, circuit_can_use_tool

All of them share one policy: a gate result maps to `allow`, `block`, or
`ask` (`decision_circuits.integrations._policy.decide`), and the
uncertain outcomes go to a human wherever the SDK has a way to ask.
"""

from decision_circuits.integrations._policy import DEFAULT_ACTIONS, Action, GateResultDict, GateResults, RunOutput, decide, explain, tool_state

__all__ = ["DEFAULT_ACTIONS", "Action", "GateResultDict", "GateResults", "RunOutput", "decide", "explain", "tool_state"]
