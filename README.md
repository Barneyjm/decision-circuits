# decision-circuits

Deterministic decision gates over calibrated model answers.

A System One style model (TypeSafe's Jev, or an open model that speaks
the same contract) answers typed questions about a piece of state and
returns probabilities, not prose. A **decision circuit** is a set of
questions plus a set of **gates**: code that combines those
probabilities into decisions with thresholds, boolean logic, votes,
verification, and ordinal buckets, and that says out loud when it is
uncertain instead of guessing.

The model never sees the gates. That makes a circuit versionable,
testable offline, and auditable per item: every result carries its
probability, its outcome (`decided`, `abstain`, `escalate`, `default`),
and a trace of every input that fed it.

Background: [Attaining LLM Certainty with AI Decision Circuits](https://towardsdatascience.com/attaining-llm-certainty-with-ai-decision-circuits/).

```bash
pip install decision-circuits                 # no dependencies
pip install "decision-circuits[openai]"       # OpenAI-compatible logprob backend (OpenAI, Fireworks, vLLM)
pip install "decision-circuits[anthropic]"    # Claude backend
pip install "decision-circuits[langchain]"    # LangChain runnable (LangGraph needs nothing extra)
```

## Example

```python
from decision_circuits import Circuit, Q, G, argmax, order
from decision_circuits.backends import SystemOne

c = Circuit()
c.noul(
    "pii",
    "Does this text contain PII about a private individual?",
    true="Email, phone, home address, ID, or card number",
    false="No PII, or business-only details",
)
c.noul("business", "Are all identifying details about a business rather than a person?")
c.noul("angry", "Is the customer angry?")
c.choice("dept", "Which team should handle this?", {"billing": "Charges, refunds, invoices", "technical": "Bugs, outages, API", "other": "Anything else"})
c.score("urgency", "How urgent is this?", ["Low", "Medium", "High", "Critical"])

c.gate("redact", ((Q("pii") >= 0.7) & ~Q("business")) >= 0.6, on_uncertain="escalate")
c.gate("route", argmax("dept", min_confidence=0.35))
c.gate("tier", order("urgency", [1.0, 2.0, 2.6]))
c.gate("human", (Q("angry") | Q("urgency")[3]) >= 0.6, on_uncertain="escalate")
c.gate("bill_hot", (G("route")["billing"] & G("tier")[3]).at(0.5))

jev = SystemOne(api_key=TYPESAFE_API_KEY, model="jev-latest")  # standard library HTTP; no extra install
out = c.run(jev, "Card charged twice, refund NOW, my card ends in 4412.")

out["gates"]["redact"]
# {"value": True, "p": 0.89, "outcome": "decided", "trace": ["pii p=0.89"]}
out["gates"]["route"]
# {"value": "billing", "p": 1.0, "confidence": 1.0, "outcome": "decided", ...}
```

## Backends

A backend is anything with one method, `answer(state, questions, *,
model=None)`, returning the System One `answers` map (see
`decision_circuits.types`). The core package depends on nothing; each
backend imports its SDK only when you construct it.

| backend | install | how it gets probabilities |
|---|---|---|
| `SystemOne(url, api_key, model)` | none | Calls a System One server (TypeSafe's Jev, or [s1proto](https://github.com/Barneyjm/s1-proto)). Measured, calibrated distributions; the model is built for this. |
| `OpenAILogprobs(model, base_url=...)` | `[openai]` | Prefill scoring: options are lettered, the next-token logprobs over the letters are the distribution. Works with OpenAI, Fireworks, Together, vLLM. Up to 20 options. |
| `Anthropic(model, mode="stated")` | `[anthropic]` | One tool-use call returning a probability per option. Fast; the numbers are stated confidence, not calibrated. |
| `Anthropic(model, mode="sampled", k=5)` | `[anthropic]` | k tool-use calls at temperature 1, one pick each; vote frequencies with add-one smoothing. An empirical distribution at k times the cost. |

`c.run(client, url=..., headers=...)` also accepts a plain HTTP client
(httpx, requests, a FastAPI TestClient). It sends the compiled gates so
a server that evaluates circuits itself can; a server that rejects the
extra field or returns answers only gets one retry without gates and
the gates are evaluated here. `out["gates_evaluated_by"]` says which.

**Calibration is the backend's, not the package's.** Gates threshold
whatever probabilities they are given. A System One model is trained to
be calibrated; a chat model's stated confidence usually is not. Check
on a labeled sample before trusting a threshold, and prefer logprob or
sampled modes over stated ones when it matters.

## LangGraph and LangChain

Gates become conditional edges, and the uncertain outcomes are edges
like any other:

```python
from decision_circuits.langgraph import as_node, route_on

graph.add_node("triage", as_node(c, jev, state_key="text"))
graph.add_conditional_edges(
    "triage",
    route_on("route"),
    {
        "billing": "billing_agent",
        "technical": "tech_agent",
        "other": "general_agent",
        "abstain": "human",
    },
)
graph.add_conditional_edges("triage", route_on("human"), {"True": "human", "False": "continue", "escalate": "human"})
```

`as_node` writes `{"circuit": {"answers": ..., "gates": ...}}` into the
graph state. `as_runnable(c, backend)` wraps the same function as a
`RunnableLambda` when langchain-core is installed.

## The expression language

| form | meaning |
|---|---|
| `Q("noul_id")` | P(yes) for a yes/no question |
| `Q("choice_id")["option"]` | probability of one option |
| `Q("score_id")[level]` | probability of one score level |
| `G("gate_id")`, `G("gate_id")["value"]` | an earlier gate's probability, or one value of a categorical gate |
| `~e` | NOT: `1 - p` |
| `a & b & c` | AND: product (independence assumed and recorded in the trace) |
| `a \| b` | OR: `1 - prod(1 - p)` |
| `e >= tau`, `e.at(tau)` | threshold: boolean at `tau` with an uncertainty band around it |
| `argmax("choice_id", min_confidence=...)` | pick the top option; abstain below the confidence floor |
| `majority("c1", "c2", "c3")` | vote across paraphrased questions over the same options |
| `verify("choice_id", check=Q("supported"), tau=...)` | a negative checker: escalate when the check does not support the pick |
| `order("score_id", [c1, c2, ...])` | bucket the expected score by cutpoints |

An expression used as a gate without a threshold thresholds at 0.5.
`e >= tau` builds a node and does not compare; `bool(Q("x") >= 0.7)` is
always true, so use `.at(tau)` anywhere an operator would read as a
runtime comparison.

Every gate takes `on_uncertain="abstain" | "escalate" | "default"`
(with `default=value`) and `band=width`. A gate is uncertain when the
probability it acts on lies within `band` of its threshold, or when a
confidence floor fails. Uncertainty is surfaced, never silently
resolved.

## Without a server

`c.compile()` returns the flat `gates` map (the wire format).
`c.evaluate(answers)` runs the gates against answers you already have,
in the response format of any System One server. `evaluate_gates` is
the underlying function if you build the map by hand.

## Diagram

`c.to_mermaid()` renders the circuit as a Mermaid flowchart with three
columns, inputs, logic, and decisions; pass the results of a run to
color each node by outcome. Paste it into a README or render it with
`mmdc`.

```python
print(c.to_mermaid(results=out["gates"], answers=out["answers"]))
```

## Status

Alpha. Core has no dependencies; Python 3.10+. The wire format is
TypeSafe's `POST /v1/systemone` request with an added `gates` block; the reference server that evaluates gates is
[s1proto](https://github.com/Barneyjm/s1-proto). The gate semantics
(`and` as a product, `or` as noisy-or) assume independent questions;
the trace records that assumption on every result so a reviewer can
see it.
