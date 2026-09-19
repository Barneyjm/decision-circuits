"""A circuit as a Claude Agent SDK PreToolUse hook.

    pip install "decision-circuits[claude]"
    export ANTHROPIC_API_KEY=...       # for the agent
    export TYPESAFE_API_KEY=...        # for the circuit (or use your own backend, see 03)
    uv run python examples/06_claude_agent_sdk_hook.py

The SDK's permission decisions are allow / deny / ask, which is exactly
what a circuit produces: a decided gate allows or denies, an uncertain
one asks, and the SDK's own permission prompt takes it from there. The
reason string carries the gate's probability and trace, so the log
says why.
"""

import asyncio
import os

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query

from decision_circuits import Circuit, Q
from decision_circuits.backends import SystemOne
from decision_circuits.integrations.claude_agent_sdk import circuit_pre_tool_use

# The hook's state is {"tool_call": {"name": ..., "args": {...}}}.
c = Circuit()
c.noul(
    "destructive",
    "Would running `tool_call` delete, overwrite, or irreversibly change files or system state outside the current project?",
    true="rm, overwriting files, git push --force, package installs, anything touching $HOME or system paths",
    false="Reads, listing, builds, tests, edits inside the project",
)
c.noul("scoped", "Is `tool_call` limited to the current working directory?")
c.gate("block", (Q("destructive") & ~Q("scoped")) >= 0.5, band=0.15, on_uncertain="escalate")

hook = circuit_pre_tool_use(c, SystemOne(api_key=os.environ["TYPESAFE_API_KEY"]), gate="block", tools={"Bash", "Write", "Edit"})

options = ClaudeAgentOptions(
    hooks={"PreToolUse": [HookMatcher(matcher="Bash|Write|Edit", hooks=[hook])]},
    allowed_tools=["Bash", "Read", "Write", "Edit"],
    max_turns=4,
)


async def main() -> None:
    async for message in query(prompt="List the files here, then delete /tmp/scratch_dir if it exists.", options=options):
        print(message)


asyncio.run(main())
