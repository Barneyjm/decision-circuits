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

An image or a clip can be the state: `Image("receipt.png")` or
`Audio("call.wav", text="Inbound, Tuesday")` (a path, a URL, or bytes,
plus an optional caption). It is sent as `{"image": <data URI or URL>,
"text": ...}` and answered by a model built for it, chosen by name
(`circuit-vl-4b`, `circuit-audio-7b`). Text-only requests do not change.

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

## Routing: the last step

Gates say what is true; a route says what to do. `c.route` turns the gates into one action,
from a priority list:

```python
c.route(
    "action",
    [
        ("page on-call", G("urgent") & G("outage")),
        ("technical queue", G("technical")),
    ],
    otherwise="general queue",
)
```

The first rule that holds gives the action; `otherwise` when none does. In a rule a gate
counts as its decision, so `G("urgent") & G("outage")` means both decided yes. A rule too
close to call stops the route and escalates (the default `on_uncertain` for a route) rather
than falling through to a lower rule on a guess. The result's `value` is the action and its
trace says which rule held. In the diagram the route is drawn as the ladder it is: a
diamond per rule, **yes** to its action, **no** to the next rule, the last **no** to
`otherwise`, and a "too close to call" branch to a person where it stopped. After a run the
way through is drawn heavy and the branches not taken fade. `describe()` leads with it.

Routing used to be plain code over the gate results, outside the circuit (the refund desk's
`decide` still is). Inside the circuit it is versioned, traced, drawn and tested with
everything else.

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
Mermaid tool. It reads left to right as one process: the state (`state="Conversation"`
names it), the questions asked about it with their wording (`text=False` for a compact
diagram), the gates, and the action. After a run every gate is coloured by what it decided
and a yes/no question answered no greys out, wires that carried a no fade, and the way
through the route to the chosen action is drawn heavy.
Each mounted chip is its own box.

## In plain English

`c.describe()` writes the circuit as Markdown for someone who doesn't read the DSL: every
question in its own words, what each gate decides ("yes when it is at least 60% likely that
2 or more of these hold: ..."), when it holds back and what it does then, and how each chip
is wired. Pass a run's results to say what happened:

```python
out = c.run(backend, state)
print(c.describe(out["gates"], out["answers"]))
```

```
- **threat** (yes/no): Does the customer threaten legal action, a chargeback or a public complaint? → **yes 98%**
- **hot**: yes when it is at least 60% likely that the `signs` decision is yes.
  - *Between 50% and 70% it is too close to call, so it escalates to a person.*
  - Result: **yes** (100%).
```

On a chip that isn't mounted, `describe()` is its datasheet: pins, params, questions, gates.

## Chips

A chip is a sub-circuit packaged with pins, like an integrated circuit: input pins it reads,
output pins (gates) it drives, and its own questions and gates in between.

```python
from decision_circuits import Chip, G, at_least

heat = Chip("heat", inputs=["hostile", "money", "pii"], outputs=["hot"], params=["who"], version="1.0")
heat.noul("threat", "Does {who} threaten legal action, a chargeback or a public complaint?")
heat.gate("signs", at_least(2, "hostile", "money", "pii", "threat"))
heat.gate("hot", G("signs") >= 0.6, on_uncertain="escalate")
```

Inside the chip an input pin is referenced like a question. `c.mount(chip, ns, pins)` wires
it into a host circuit:

```python
pins = c.mount(heat, "customer", {"hostile": "customer_hostile", "money": G("money"), "pii": "pii"}, params={"who": "the customer"})
c.gate("supervisor", pins["hot"] >= 0.5)
```

- A pin can be wired to a host question, a host gate (`G("money")`), or one option of either
  (`Q("topic")["refund"]`).
- The chip's questions and gates are copied in as `ns.<name>` (`customer.threat`,
  `customer.hot`), so the model answers them in the same request as the host's, and a chip
  mounted twice is two independent copies.
- Params fill `{name}` placeholders in the chip's question text, once per mount. Without
  them, two copies of a chip ask the same question about the same state and get the same
  answer; with `{"who": "the customer"}` and `{"who": "the agent"}` each asks about its own
  part. Every declared param must be given.
- `mount` returns the output pins as `G` references. It refuses an unwired or unknown pin, a
  namespace already in use, and a reference inside the chip to something it does not own;
  a refused mount leaves the host unchanged.
- Chips nest: a chip can mount other chips.

Mounting compiles to the same flat gates you could have written by hand, so evaluation,
tracing and backends do not change.

**Testing a chip on its own.** `chip.evaluate(values)` runs it with no model: pins and the
chip's own questions take a number (a noul's P(yes)), a dict of option to probability (a
choice), or a wire-format answer. Test vectors ship with the chip:

```python
heat.add_test({"hostile": 0.9, "money": 0.9, "pii": 0.1, "threat": 0.2}, {"hot": True})
heat.add_test({"hostile": 0.7, "money": 0.5, "pii": 0.1, "threat": 0.3}, {"hot": "escalate"})
assert heat.test() == []  # or heat.test("cases.jsonl")
```

An expectation is what `result_key` gives: the value when decided, else the outcome.

**Sharing a chip.** `chip.to_dict()` is plain JSON with a `chip` block: name, version, pins, params,
the tests, and `requires`, the question types it asks. `Circuit.from_dict` reads it back
as a `Chip`. Ship chips in a package or a git repo like any other code. Circuits serialize
the same way, mounts included.

**Question types.** `circuit.question_types` lists what a backend must answer. `SystemOne`
pointed at TypeSafe's API refuses `multi`, `locate`, `rank` and `match` before sending,
since it answers `noul`, `choice` and `score`; pass `question_types=` to set another
server's list.
