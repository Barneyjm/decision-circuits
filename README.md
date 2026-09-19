# decision-circuits

Deterministic decision gates over calibrated model answers.

```bash
pip install decision-circuits          # no dependencies, Python 3.10+
```

## The idea in one paragraph

Ask a model small, typed questions about a piece of state and get back
probabilities, not prose. Then decide with code: thresholds, AND/OR/NOT,
argmax with a confidence floor, majority votes, verification, ordinal
buckets. That code is a **decision circuit**. The model never sees it,
so the circuit is versionable, testable offline, and auditable per item:
every result carries its probability, its outcome, and a trace. And when
the probability sits too close to a threshold to call, the gate says so
(`abstain` or `escalate`) instead of guessing.

Background: [Attaining LLM Certainty with AI Decision Circuits](https://towardsdatascience.com/attaining-llm-certainty-with-ai-decision-circuits/).

## Sixty seconds

```python
from decision_circuits import Circuit, Q, argmax

c = Circuit()
c.noul("pii", "Does this text contain personal information about a private individual?")
c.noul("angry", "Is the writer angry?")
c.choice("dept", "Which team should handle this?", {"billing": "Money, refunds", "technical": "Bugs, outages", "other": None})

c.gate("redact", Q("pii") >= 0.7, on_uncertain="escalate")
c.gate("route", argmax("dept", min_confidence=0.3))
c.gate("human", (Q("angry") | Q("pii")) >= 0.6)
```

Now something has to answer the questions. Anything with one method will do:

```python
from decision_circuits.backends import SystemOne

jev = SystemOne(api_key=TYPESAFE_API_KEY, model="jev-latest")  # TypeSafe's System One model
out = c.run(jev, "Card charged twice, refund NOW. My card ends in 4412.")

out["gates"]["redact"]
# {'value': True, 'p': 0.97, 'outcome': 'decided', 'trace': ['pii p=0.97'], ...}
out["gates"]["route"]
# {'value': 'billing', 'p': 1.0, 'confidence': 1.0, 'outcome': 'decided', ...}
```

No model handy? `c.evaluate(answers)` runs the gates on answers you
already have, and [`examples/01_first_circuit.py`](examples/01_first_circuit.py)
does exactly that with hand-written numbers.

## Where it plugs in

The point of a circuit is to sit inside an agent and make the small
decisions the agent shouldn't be trusted with. Adapters exist for the
three agent SDKs; each is a few lines over the same object.

| framework | what you get | install |
|---|---|---|
| **LangChain** agents | `CircuitToolGuard` middleware: judge each tool call, then allow, block, or pause for approval with the standard human-in-the-loop interrupt. `CircuitRouter` picks a model per run. | `[langchain]` |
| **OpenAI Agents SDK** | `circuit_input_guardrail`, `circuit_output_guardrail`, `circuit_tool_guardrail` | `[openai-agents]` |
| **Claude Agent SDK** | `circuit_pre_tool_use` hook returning allow / deny / ask; `circuit_can_use_tool` | `[claude]` |
| LangGraph | `as_node` and `route_on`: gates as conditional edges | none |

```python
from decision_circuits.integrations.langchain import CircuitToolGuard

guard = CircuitToolGuard(c, jev, gate="block", tools=[delete_file, send_email])
agent = create_agent(model, tools=[read_file, delete_file, send_email], middleware=[guard])
```

Compared with a single yes/no classifier at a fixed 0.5 (what
`langchain-typesafe`'s `AutoModeMiddleware` does), a circuit lets you
combine several questions, set the threshold and its uncertainty band
explicitly, and send the uncertain cases to a human rather than
silently allowing or blocking them. See [docs/integrations.md](docs/integrations.md).

## Backends

A backend is any object with `answer(state, questions, *, model=None)`
returning the answers map. The core package ships three and depends on
none of their SDKs; each imports its own when you construct it.

| backend | install | how it gets probabilities |
|---|---|---|
| `SystemOne(url, api_key, model)` | none | A System One server: TypeSafe's Jev, or [s1proto](https://github.com/Barneyjm/s1-proto) on open weights. Measured, calibrated. |
| `OpenAILogprobs(model, base_url=...)` | `[openai]` | Options are lettered; the next-token logprobs over the letters are the distribution. OpenAI, Fireworks, vLLM. Up to 20 options. |
| `Anthropic(model, mode="stated")` | `[anthropic]` | One tool-use call returning a probability per option. Fast; a self-report, not calibrated. |
| `Anthropic(model, mode="sampled", k=5)` | `[anthropic]` | k calls at temperature 1, one pick each; vote frequencies. |

Writing your own is a dozen lines: [`examples/03_your_own_backend.py`](examples/03_your_own_backend.py).

**Calibration is the backend's, not the package's.** Gates threshold
whatever they are handed. A System One model is trained to be
calibrated; a chat model's stated confidence usually is not. Check on a
labeled sample before trusting a threshold.

## The expression language

| form | meaning |
|---|---|
| `Q("noul_id")` | P(yes) for a yes/no question |
| `Q("choice_id")["option"]`, `Q("score_id")[level]` | probability of one option or level |
| `G("gate_id")`, `G("gate_id")["value"]` | an earlier gate's probability, or one value of a categorical gate |
| `~e` | NOT: `1 - p` |
| `a & b & c` | AND: product (independence assumed and recorded in the trace) |
| `a \| b` | OR: `1 - prod(1 - p)` |
| `e >= tau`, `e.at(tau)` | threshold with an uncertainty band; `>=` builds a node, it does not compare |
| `argmax("choice", min_confidence=...)` | top option; abstain below the floor |
| `majority("c1", "c2", "c3")` | vote across paraphrased questions |
| `verify("choice", check=Q("supported"), tau=...)` | a negative checker: escalate when the check does not support the pick |
| `order("score", [c1, c2, ...])` | bucket the expected score by cutpoints |

Every gate takes `on_uncertain="abstain" | "escalate" | "default"`
(with `default=value`) and `band=width`. `c.to_mermaid()` draws it.

## Examples

Numbered, each self-contained, in [`examples/`](examples/):

1. a circuit with hand-written answers (offline)
2. the same circuit answered by Jev
3. writing a backend (offline)
4. LangChain tool guard
5. OpenAI Agents guardrails
6. Claude Agent SDK hook
7. integrating a framework that has no adapter (offline)
8. **a refund desk**: seven questions, six gates, code-owned facts mixed with model judgments, five tickets routed to four queues (offline or Jev)
9. the refund desk answered by Claude, stated versus sampled probabilities
10. the refund desk answered through OpenAI-compatible logprobs (OpenAI, Fireworks, vLLM)

## Docs

- [docs/concepts.md](docs/concepts.md): questions, answers, gates, uncertainty, the trace
- [docs/integrations.md](docs/integrations.md): each SDK adapter and what "ask" means there
- [docs/extending.md](docs/extending.md): backends, gate ops, adapters

## Status

Alpha. The wire format is TypeSafe's `POST /v1/systemone` request plus a
`gates` block. MIT.
