"""Claude Agent SDK hooks driven by a decision circuit.

The SDK's PreToolUse hook already speaks the circuit's language: a
permission decision of `allow`, `deny`, or `ask`. A decided gate maps to
allow or deny; an uncertain gate maps to `ask`, which hands the call to
the SDK's normal permission prompt instead of guessing.

    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from decision_circuits.integrations.claude_agent_sdk import circuit_pre_tool_use, circuit_can_use_tool

    options = ClaudeAgentOptions(
        hooks={"PreToolUse": [HookMatcher(matcher="Bash|Write|Edit", hooks=[circuit_pre_tool_use(c, backend, gate="block")])]},
    )
    # or, as the permission callback:
    options = ClaudeAgentOptions(can_use_tool=circuit_can_use_tool(c, backend, gate="block"))

Both are async, as the SDK requires, and run the backend in a worker
thread so a slow classifier never blocks the event loop. The hook
returns the SDK's `hookSpecificOutput` with `permissionDecision` and a
`permissionDecisionReason` carrying the gate's probability and trace.
`can_use_tool` returns `PermissionResultAllow` / `PermissionResultDeny`;
it has no "ask", so an uncertain gate denies with an explanatory
message unless `actions` maps it to allow.

Requires `claude-agent-sdk>=0.2` (`pip install "decision-circuits[claude]"`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

try:
    from claude_agent_sdk import HookCallback, HookContext, PermissionResultAllow, PermissionResultDeny, ToolPermissionContext
    from claude_agent_sdk.types import CanUseTool, HookInput, SyncHookJSONOutput
except ImportError as e:  # pragma: no cover
    raise ImportError("decision_circuits.integrations.claude_agent_sdk needs claude-agent-sdk>=0.2: pip install 'decision-circuits[claude]'") from e

from decision_circuits.dsl import Circuit
from decision_circuits.integrations._policy import Action, CircuitPolicy, tool_state
from decision_circuits.types import Backend

PERMISSION = {"allow": "allow", "block": "deny", "ask": "ask"}


def circuit_pre_tool_use(
    circuit: Circuit, backend: Backend, *, gate: str, actions: Mapping[Any, Action] | None = None, tools: set[str] | None = None
) -> HookCallback:
    """A PreToolUse hook: `(input, tool_use_id, context) -> hook output`.
    `tools` limits which tool names are judged; others pass untouched."""
    policy = CircuitPolicy(circuit, backend, gate=gate, actions=actions)

    async def hook(input_data: HookInput, tool_use_id: str | None, context: HookContext) -> SyncHookJSONOutput:
        tool_name = input_data.get("tool_name", "")
        if tools is not None and tool_name not in tools:
            return {}
        j = await asyncio.to_thread(policy.judge, tool_state(tool_name, input_data.get("tool_input", {})))
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": PERMISSION[j.action],
                "permissionDecisionReason": f"decision circuit: {j.reason}",
            }
        }

    return hook


def circuit_can_use_tool(circuit: Circuit, backend: Backend, *, gate: str, actions: Mapping[Any, Action] | None = None) -> CanUseTool:
    """A `can_use_tool` callback: `(tool_name, tool_input, context) -> PermissionResult`."""
    policy = CircuitPolicy(circuit, backend, gate=gate, actions=actions)

    async def can_use_tool(tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext) -> PermissionResultAllow | PermissionResultDeny:
        j = await asyncio.to_thread(policy.judge, tool_state(tool_name, tool_input, description=getattr(context, "description", None)))
        if j.action == "allow":
            return PermissionResultAllow()
        why = "blocked by a decision circuit" if j.action == "block" else "a decision circuit could not decide and no human approval path is available here"
        return PermissionResultDeny(message=f"{why}: {j.reason}")

    return can_use_tool


__all__ = ["circuit_can_use_tool", "circuit_pre_tool_use"]
