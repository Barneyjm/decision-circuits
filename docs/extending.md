# Extending

There are three seams, and each is one small interface.

## A new backend

A backend answers questions. One method:

```python
class MyBackend:
    def answer(self, state, questions, *, model=None):
        return {qid: answer_from_probabilities(q, weights_for(q, state)) for qid, q in questions.items()}
```

`questions` is the circuit's questions map. For each one, produce a
distribution over its options (`option_keys(q)` lists them in order:
`["yes", "no"]` for a noul, the option ids for a choice, `"0".."n-1"`
for a score) and hand it to `answer_from_probabilities`, which
normalizes and fills in the argmax, expected score, and confidence.
Return the wire-format answer directly if you already have one.

Rules of the road:

- Raise on failure. A circuit never guesses; a backend that returns
  uniform distributions on error will look like uncertainty and route
  everything to a human, which hides the outage.
- Convert framework objects with `to_jsonable` before sending state
  over the wire or into a prompt. `SystemOne` does this for you.
- No base class is needed. `isinstance(x, Backend)` is structural.
- Optional: implement `answer_with_gates(state, questions, gates, *,
  model=None) -> (answers, gates_or_None)` if your server can evaluate
  circuits itself. `Circuit.run` prefers it when present.

Wrappers are backends too. Example 08 wraps a model backend with code
facts; a cache, a fallback chain, or an ensemble is the same pattern.
See [`examples/03_your_own_backend.py`](../examples/03_your_own_backend.py).

## A new integration

Every shipped adapter is a `CircuitPolicy` plus a mapping to the
framework's result type:

```python
from decision_circuits.integrations import CircuitPolicy, tool_state

policy = CircuitPolicy(circuit, backend, gate="block", actions={"escalate": "ask"})


def before_tool(call):  # whatever hook your framework offers
    j = policy.judge(tool_state(call.name, call.args))
    return {"allow": RUN, "block": SKIP, "ask": CONFIRM}[j.action]
```

`judge` returns a `Judgment` with `action`, `reason` (a sentence with
the gate's value, probability, and trace, meant for the message the
framework shows), and `output` (all answers and gate results).
`policy.last` keeps the most recent one.

Decide up front what "ask" means in your framework. If it has a
permission prompt or an interrupt, map to it. If it has nothing,
pass `actions={"abstain": "block", "escalate": "block"}` and say so in
the message, the way the OpenAI Agents tool guardrail does.

Backends block on I/O. In an async framework, call
`await asyncio.to_thread(policy.judge, state)`.

`tool_state(name, args, description=..., messages=..., **extra)` builds
the conventional state for a tool call, so circuits written for one
framework's tool guard work under another.
See [`examples/07_your_own_integration.py`](../examples/07_your_own_integration.py).

## A new gate op

Gate evaluation is `decision_circuits.gates.evaluate_gates`: a loop
over the compiled `{name: Gate}` map in declaration order, one branch
per `op`. To add one:

1. Add the op name to `OPS` and its validation to `Gate.__post_init__`.
2. Add a branch in `evaluate_gates` that reads its inputs with
   `_noul_p` (for probabilities) or straight from `answers` (for
   choice/score structure), builds a trace, and calls `_settle` with
   the value, probability, and whether the result is uncertain.
3. Add a DSL constructor in `dsl.py` returning a `Categorical` (for
   ops that read a whole choice or score) or an `Expr` subclass (for
   ops that produce a probability), and teach `Circuit._compile` to
   lower it.
4. Add a case to `render_mermaid`'s node labels.

`_settle` is where uncertainty policy lives; new ops get `abstain`,
`escalate`, and `default` for free by going through it.

## Code-owned facts

Not everything should be a model question. Dates, amounts, counts,
anything you already know: compute it and inject it as an answer with
probability 1.0 or 0.0, and the gates treat it like any other input.
Example 08 does this with a wrapper backend. It keeps the arithmetic
out of the model and puts the fact in the trace where a reviewer can
see it.
