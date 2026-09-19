# Concepts

Four things, in the order they happen: a **state**, **questions** about
it, **answers** with probabilities, and **gates** that turn those into
decisions.

## State

Whatever you want judged. A string, a dict, a list of chat messages, a
tool call. Backends convert framework objects to JSON for you
(`to_jsonable`), so you can pass them as they are. Everything in the
state is data to the model, never instructions; the questions should
say so when the state might contain text written by someone else.

## Questions

Three kinds, matching TypeSafe's System One API.

| kind | asks | answer |
|---|---|---|
| `noul` | a yes/no question | `P(yes)` |
| `choice` | pick one of named options | a probability per option, the argmax, and a confidence |
| `score` | rate against ordered levels | a probability per level, the expected level, and a confidence |

```python
c.noul("urgent", "Does this need attention today?", true="Outage, deadline, or money at risk", false="Can wait")
c.choice("dept", "Which team?", {"billing": "Charges and refunds", "technical": "Bugs and outages"})
c.score("severity", "How bad is it?", ["Cosmetic", "Degraded", "Down"])
```

The optional descriptions (`true=`, `false=`, the option values, the
level strings) are the criteria. They matter: a model answers the
question you wrote, and "urgent" means different things to different
people. `confidence` is `1 - H(p) / log N`, so 1.0 means all the mass on
one option and 0.0 means uniform.

## Answers and backends

A **backend** answers questions. It is any object with one method:

```python
def answer(self, state, questions, *, model=None) -> dict[str, Answer]
```

The package ships a System One HTTP backend (Jev, or an open-weights
server speaking the same contract), an OpenAI-compatible logprobs
backend, and a Claude backend. The circuit does not care which one you
use, which is the point: the gates are yours and stay put while the
model behind them changes.

## Gates

A gate reads probabilities and emits a decision with a probability, an
outcome, and a trace.

```python
c.gate("rush", Q("urgent") >= 0.7)  # threshold
c.gate("redact", (Q("pii") & ~Q("business")) >= 0.6)  # AND, NOT
c.gate("human", (Q("angry") | Q("severity")[2]) >= 0.5)  # OR over a noul and one score level
c.gate("route", argmax("dept", min_confidence=0.35))  # categorical with a floor
c.gate("tier", order("severity", [0.8, 1.8]))  # ordinal buckets
c.gate("checked", verify("dept", check=Q("supported"), tau=0.8))  # a negative checker
c.gate("vote", majority("dept", "dept_paraphrase", "dept_again"))  # redundancy
c.gate("hot_bill", (G("route")["billing"] & G("tier")[2]).at(0.5))  # gates over gates
```

The arithmetic is deliberately simple and written down in the trace:
AND is a product, OR is `1 - prod(1 - p)`, NOT is `1 - p`. Both AND and
OR assume the inputs are independent, and every trace says so, because
a reviewer should see that assumption next to the number. If two
questions are obviously correlated, ask one question instead.

## Uncertainty

Every gate has a band around its threshold and a policy for what to do
inside it:

```python
c.gate("rush", Q("urgent") >= 0.7, band=0.1, on_uncertain="escalate")
```

With `p = 0.65` the gate does not return `False`. It returns
`outcome="escalate"`, `value=None`, `uncertain=True`. The policies are
`abstain` (no decision), `escalate` (someone else decides), and
`default` (a stated fallback value, with `default=...`). Uncertainty
also propagates: a gate that reads an uncertain upstream gate is itself
uncertain.

This is the part that makes circuits worth the trouble. A threshold
alone turns 0.69 into "no" with a straight face. A band turns it into
"I'm not sure," and the agent integrations turn *that* into a question
for a human.

## Results

Each gate returns:

```python
{"value": True, "p": 0.91, "confidence": None, "uncertain": False, "outcome": "decided", "trace": ["pii p=0.91"]}
```

`outcome` is one of `decided`, `abstain`, `escalate`, `default`.
`result_key(result)` gives the thing to route on: the value when
decided, otherwise the outcome. The trace is a list of strings, one per
input and operation, meant to be logged next to the decision.

## Diagrams

`c.to_mermaid()` renders the circuit as a Mermaid flowchart in three
columns: inputs, logic, decisions. Pass `results=` and `answers=` from a
run to color the nodes by outcome. It renders on GitHub and in any
Mermaid tool.
