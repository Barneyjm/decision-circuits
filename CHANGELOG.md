# Changelog

## 0.5.4 (2026-09-23)

- **Gates are always evaluated by the SDK.** `Circuit.run` sends the questions only and
  evaluates the gates itself, so the gate code that decides is the version you installed (the
  one `proofs/` proves), and a circuit decides the same way whichever backend answered it.
  Before, a System One server that knew gates evaluated them with its own copy of this
  package, so a hosted user could get an older version's gates, and new gate types fell back
  to the client after a refused request.
- Removed: `SystemOne.answer_with_gates`, the gate-support negotiation behind it, and
  `RunOutput["gates_evaluated_by"]`. Raw-HTTP callers can still send a `gates` block to a
  server that accepts one.

## 0.5.3 (2026-09-23)

Found by proving pick dominance in Lean (`proofs/`): a decided categorical gate must never read
an option it did not pick as likelier than its pick. Three readings broke it, two of them new
in 0.5.2.

- `majority` reads each value's share of the paraphrases' votes, not the mean of their
  distributions (a vote won two to one could read the loser likelier).
- `order` reads 1 for the bucket it decided and 0 elsewhere, as before 0.5.2 (summing score
  levels per bucket could read another bucket likelier, since the bucket comes from the
  expected score).
- `argmax`, `verify` and `majority` pick an argmax of the probabilities, keeping the backend's
  `choice` when it attains the maximum, instead of trusting a `choice` the probabilities do not
  support.
- `proofs/`: Lean 4 proofs of the count identities and pick dominance, checked in CI;
  `tests/test_properties.py` checks the Python against the same theorems with Hypothesis.

## 0.5.2 (2026-09-23)

Fixes. Two change results; both now match what the docs always said.

- **A categorical gate's options read their own probabilities.** `G("route")["technical"]` after `argmax`, `majority` or `verify` gave every unpicked option `1 - p`, so an option the gate did not pick could read likelier than the one it did (billing .34 picked; technical read .66). Every categorical gate now carries its distribution (`order`: each score level's probability in its bucket), and references read it.
- **A threshold inside an expression is a decision.** `(Q("pii") >= 0.9) & ~Q("biz")` used to ignore the 0.9 and multiply pii's probability; now pii counts as 1 when it passes and 0 when it does not, and a probability within the gate's band of 0.9 makes the gate uncertain. Nested thresholds (`(Q("x") >= 0.3).at(0.7)`) compile instead of raising.
- `intervene` reports an edited state the backend refuses, or an edit function that raises, as that intervention's `error`; the rest are reported as before.
- `SystemOne` marks a server as not evaluating gates only when a retry without gates succeeds; a refused question no longer downgrades it for good. A custom client's non-JSON error page (a gateway's 502) is retried like any other.
- `OpenAILogprobs` and `Anthropic` refuse multi, locate, rank and match questions by name instead of answering them as a pick-one score.
- `Circuit.run` names a question the backend returned no answer for.
- Gate names starting with `_` are reserved for generated helpers and refused.
- Mermaid: rank and match answers render; a terminal threshold's label no longer reads "≥ ≥".
- Tracing: a list of mixed types is sent as strings instead of being dropped.
- The uncertainty band is documented as it behaves: within `band` of tau, exclusive.

## 0.5.1 (2026-09-23)

- OpenTelemetry tracing, optional (`decision-circuits[otel]`): a `decision_circuits.run` span per run with the model call as a child, GenAI attributes (`gen_ai.request.model`, `gen_ai.response.id`, `gen_ai.usage.input_tokens`), an event per answer and per gate, and the gates that escalated or abstained. `intervene`/`ablate` runs nest under a `decision_circuits.intervene` span across threads. `SystemOne` propagates `traceparent`. The state is never recorded. Nothing changes without OpenTelemetry installed.

## 0.5.0 (2026-09-23)

- Questions for circuit v2 models, text states only: `Circuit.multi` (every option that applies, each with its own probability) and `Circuit.locate` (which field, list element or sentence answers it, or none).
- Gates: `at_least(k, ...)` and `count(...)` over a multi's options or a list of references (the count's whole distribution is in its result, so `G("n")[2]` is P(exactly two)); `consistent(a, b, relation)` checks two answers against each other (`same`, `complement`, `implies`). `and`/`or` accept a multi as their input.
- Interventions: `Circuit.intervene(backend, state, {name: edit})` reruns the circuit on edited states and reports which gates flipped and how every answer moved; `drop(...)`, `set_to(...)` build edits; `Circuit.ablate` removes each field or sentence in turn. Any backend.
- `answer_distributions(answer)`: every answer type as named distributions, the one reader the gates and interventions share. The multi, locate, rank and match answer formats are documented and typed in `types.py`.
- `__version__` reads the installed package's version.
- Example 13: a complaint desk on multi, locate, ask-next, the new gates, and ablation.

## 0.4.1 (2026-09-19)

- PyPI project links: documentation, source, issues, changelog, models.
- No code changes from 0.4.0.

## 0.4.0 (2026-09-19)

- `Image(...)` and `Audio(...)` media states: a path, URL, or bytes plus an optional caption, sent as `{"image"|"audio": <data URI or URL>, "text": ...}`. Text requests unchanged.
- `SystemOne` backend sends a `decision-circuits` user agent and retries 502/503/504/524 (a hosted model starting up) for up to four minutes; client timeouts count as retryable. `timeout` default is now 120 s.
- Example 12: a receipt and a recorded call through the hosted circuit-vl-4b and circuit-audio-7b.

## 0.3.0 (2026-09-19)

- `Circuit.run(backend, state)`: backends replace the HTTP client argument. `SystemOne.last_response` carries the full server response.
- Agent SDK integrations: LangChain `CircuitToolGuard` / `CircuitRouter` / `circuit_when`, OpenAI Agents guardrails, Claude Agent SDK hooks.
- Zero-dependency core; `openai`, `anthropic`, and `langchain-core` are optional extras.

## 0.2.0 (2026-09-19)

- `Backend` protocol; `SystemOne`, OpenAI-logprobs, and Anthropic backends; LangGraph adapter.

## 0.1.0 (2026-09-19)

- The DSL: `Q`, `G`, thresholds with bands, AND/OR/NOT, `argmax`, `majority`, `verify`, `order`; gates evaluated client-side or by a server that supports them; Mermaid rendering.
