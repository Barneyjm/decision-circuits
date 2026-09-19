# Integrations

Every adapter is the same three steps over one object, `CircuitPolicy`:

1. build the state the framework hands you (a tool call, an input, an output),
2. `policy.judge(state)` runs the circuit and reads one gate,
3. map the action, `allow` / `block` / `ask`, onto the framework's own result type.

```python
from decision_circuits.integrations import CircuitPolicy

policy = CircuitPolicy(circuit, backend, gate="block")
j = policy.judge(state)
j.action  # "allow" | "block" | "ask"
j.reason  # "gate `block` -> True (decided p=0.97). risky p=0.98; ..."
j.output  # {"answers": ..., "gates": ...}
```

## How a gate result becomes an action

`actions` is a dict from the gate's **result key** to an action. The key
is the gate's value when it decided (`True`/`False` for a threshold, the
option for `argmax`, the bucket for `order`) and its outcome when it did
not (`"abstain"`, `"escalate"`).

| result key | default action |
|---|---|
| `True` | `block` |
| `False` | `allow` |
| `"abstain"`, `"escalate"` | `ask` |
| anything else | `block` (a route you did not list never silently allows) |

So a categorical gate routes with `actions={"safe": "allow", "risky":
"block", "unclear": "ask"}`, and a framework with no way to ask a human
gets `actions={"abstain": "block", "escalate": "block"}`.

## LangChain agents

```python
from langchain.agents import create_agent
from decision_circuits.integrations.langchain import CircuitToolGuard, CircuitRouter, circuit_when
```

**`CircuitToolGuard(circuit, backend, gate=..., tools=[...])`** is
middleware that judges each guarded tool call before it runs. Its state
is `{"tool_call": {"name", "args"}, "tool_description": ..., "messages":
[...last 30...]}`.

| action | what happens |
|---|---|
| allow | the tool runs |
| block | the tool does not run; the model gets an error `ToolMessage` with the gate's reason |
| ask | the run pauses with a `HumanInTheLoopMiddleware`-format interrupt (`action_requests` / `review_configs`); resume with `{"decisions": [{"type": "approve"}]}` or `reject`. Needs a checkpointer, like any interrupt. |

`guard.policy.last` holds the most recent judgment. Classification
errors propagate and the tool does not run.

**`CircuitRouter(circuit, backend, gate=..., models={...},
default_model=...)`** runs a circuit once per agent run on the latest
user message, stores `{"answers", "gates"}` in agent state under
`circuit`, and picks the model from a categorical gate. Uncertain
results use `default_model`.

**`circuit_when(circuit, backend, gate=...)`** returns a `when=`
predicate for LangChain's own `HumanInTheLoopMiddleware`, if you would
rather keep its full approve/edit/respond flow and only let a circuit
decide *when* to ask. Do not stack it with a `CircuitToolGuard` on the
same tools; that runs the backend twice per call.

Versus `langchain-typesafe`'s `AutoModeMiddleware`: that is one Noul at a
hard 0.5 that allows or blocks. A `CircuitToolGuard` with one Noul and
`band=0` behaves the same; everything past that (more questions, an
explicit band, ask) is what the circuit adds.

## OpenAI Agents SDK

```python
from decision_circuits.integrations.openai_agents import circuit_input_guardrail, circuit_output_guardrail, circuit_tool_guardrail
```

| factory | state the circuit sees | allow | block | ask |
|---|---|---|---|---|
| `circuit_input_guardrail` | `{"input": ..., "agent": name}` | continue | tripwire | tripwire |
| `circuit_output_guardrail` | `{"output": ..., "agent": name}` | return | tripwire | tripwire |
| `circuit_tool_guardrail` | `{"tool_call": {...}, "agent": name}` | run the tool | reject with `reject_message` | reject with `ask_message` |

Input and output guardrails have no "ask a human" path in the SDK, so
uncertain results trip unless `actions` says otherwise. Tool guardrails
come closest: the rejection message tells the model a human must
approve. `output_info` on every result carries the full judgment for
tracing.

## Claude Agent SDK

```python
from decision_circuits.integrations.claude_agent_sdk import circuit_pre_tool_use, circuit_can_use_tool
```

**`circuit_pre_tool_use(circuit, backend, gate=..., tools={...})`** is a
PreToolUse hook. The SDK's permission decisions already are
`allow` / `deny` / `ask`, so the mapping is direct, and `ask` hands the
call to the SDK's normal permission prompt with the gate's reason
attached.

**`circuit_can_use_tool(circuit, backend, gate=...)`** is the
`can_use_tool` callback form. It has no `ask`, so uncertain results
deny with a message saying why.

Both are async and run the backend in a worker thread.

## LangGraph

```python
from decision_circuits.langgraph import as_node, route_on

graph.add_node("triage", as_node(circuit, backend, state_key="text"))
graph.add_conditional_edges("triage", route_on("route"), {"billing": "billing_agent", "technical": "tech_agent", "abstain": "human"})
```

`as_node` writes `{"circuit": {"answers", "gates"}}` into the graph
state; `route_on` returns the gate's result key as a string, so
`abstain` and `escalate` are edges like any other.

## Something else

See [extending.md](extending.md) and
[`examples/07_your_own_integration.py`](../examples/07_your_own_integration.py).
