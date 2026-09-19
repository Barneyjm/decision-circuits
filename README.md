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
pip install decision-circuits
```

## Example

```python
import httpx
from decision_circuits import Circuit, Q, G, argmax, order

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

out = c.run(
    httpx.Client(),
    "Card charged twice, refund NOW, my card ends in 4412.",
    url="https://api.typesafe.ai/v1/systemone",
    headers={"Authorization": f"Bearer {TYPESAFE_API_KEY}"},
)

out["gates"]["redact"]
# {"value": True, "p": 0.88, "outcome": "decided", "trace": [...]}
out["gates"]["route"]
# {"value": "billing", "p": 0.81, "outcome": "decided", ...}
```

`run` posts the circuit as a normal System One request. If the server
evaluates gates itself, its results are used. If it returns answers
only (TypeSafe today), the gates are evaluated on the client from the
returned probabilities. `out["gates_evaluated_by"]` says which.

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

Alpha. The wire format is TypeSafe's `POST /v1/systemone` request with
an added `gates` block; the reference server that evaluates gates is
[s1proto](https://github.com/barneyjm/s1-proto). The gate semantics
(`and` as a product, `or` as noisy-or) assume independent questions;
the trace records that assumption on every result so a reviewer can
see it.
