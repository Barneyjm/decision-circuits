# Changelog

## 0.6.0 (2026-10-04)

- **Chips.** `Chip` is a sub-circuit with declared input and output pins. `Circuit.mount(chip,
  ns, pins)` wires it into a host: pins connect to host questions, gates or single options;
  the chip's questions and gates are copied in as `ns.<name>`; the output pins come back as
  `G` references. `params` fill `{name}` placeholders in the chip's question text per mount,
  so two copies can ask about two parts of one state. Chips nest, and a refused
  mount leaves the host unchanged. `chip.evaluate` runs one offline with shorthand pin
  values, and `add_test` / `test` keep test vectors with the chip (also from JSONL).
  `examples/14_chips.py`.
- **Parts.** `decision_circuits.parts`: `refund_risk()` (seven refund-abuse signals, three of
  them optional pins the host can answer from its own data, to hold / review / clear),
  `verified_classifier()` (any pick-one asked three ways and voted, plus "does the text answer
  it at all"), `tool_guard()` (allow / block / ask for an agent's tool call, with a strictness
  and a policy for irreversible actions the user asked for). Each ships with test vectors and a
  datasheet, and was run live in two unrelated circuits.
- A route's action reads like an option, `G("risk.action")["hold"]` (1 or 0 when it decided;
  an undecided route passes its uncertainty on), and a route output wires into a choice pin.
- Test vectors and `chip.evaluate` take `params`, so a chip with required settings can be
  tested. A dict or list setting reads as words in question text.
- **Chips plug into any circuit.** `Pin(name, type=..., options=..., ask=...)` declares what an
  input pin takes (a yes/no probability, a pick-one with the options it needs, a score), and
  `mount` refuses a wire of the wrong kind, naming the pin. A pin with `ask` is optional: wire
  it when the host knows the answer, leave it and the chip asks the model itself. `params` are
  settings with defaults: `{name}` in question text, and `P("name")` in the logic (a
  threshold, an `at_least` count, a confidence floor, a choice's options, a route's action),
  so one chip serves a lenient desk and a strict one. A chip evaluates, runs, describes and
  draws with its defaults; mounting fills in the host's.
- **Circuits and chips serialize.** `to_dict` / `Circuit.from_dict` round-trip a circuit as
  JSON (format `decision-circuits/1`), keeping gate expressions rather than compiled helpers.
  A chip's JSON carries its pins, version, tests and the question types it requires.
- `circuit.question_types` lists the question types a backend has to answer. `SystemOne`
  pointed at `api.typesafe.ai` refuses `multi`, `locate`, `rank` and `match` before the
  request, since TypeSafe answers `noul`, `choice` and `score`; `question_types=` overrides.
- **Routing.** `c.route(name, [(action, condition), ...], otherwise=...)` is the circuit's last
  step: one action from the first rule that holds. In a rule a gate counts as its decision.
  A rule too close to call stops the route and escalates instead of falling through on a guess.
  `route(...)` is the same as a gate body. It survives JSON and mounting.
- `examples/08_refund_desk.py` routes with `c.route` instead of plain code after the circuit, so
  its diagram ends in its five queues. One ticket changes queue: a decidedly hostile ticket is
  now **HUMAN (hostile)** even when an unrelated gate was unsure, where before any unsure gate
  sent everything to **HUMAN (model unsure)**. Both go to a person.
- `examples/05_openai_agents_guardrails.py` says which key is missing instead of a traceback.
- **Middleware reads routes and chips.** A route whose actions are `"allow"`, `"block"` and
  `"ask"` drives `CircuitToolGuard`, the OpenAI Agents guardrails and the Claude Agent SDK hook
  directly, with no `actions` map; an escalated route asks. A chip's output pin works as
  `gate="ns.out"`. Before, a route action named "allow" fell through to the default and blocked.
- **An uncertain input counts only when it could change the result.** An AND with a decided
  no, or an OR with a decided yes, is decided whatever its uncertain inputs turn out to be;
  likewise `at_least`. Before, any uncertain input made the gate uncertain, so a guard like
  "harmful AND NOT requested" escalated a harmless call just because "requested" was close to
  call. The trace says "the uncertain inputs cannot change it". A property test checks that a
  gate settled this way decides the same at every corner of its uncertain inputs.
- **A probability on a band's edge is outside the band**, as documented. Float error made
  0.9 against 0.8 ± 0.1 read as inside (0.9 − 0.8 is 0.0999… in floats), so it abstained or
  escalated. The same fix applies to `order` cutpoints and the `consistent` band.
- **`describe()`**: a circuit or chip in plain English, as Markdown. It covers each question in
  its own words, what each gate decides and when it holds back, and how each chip is wired.
  Given a run's results, it also says what happened. On an unmounted chip it is the datasheet.
- `to_mermaid` lays a circuit out in columns, Asked → Checks → Decide → Outcome, with every
  action stacked on the right in rule order. A rule that reads one gate branches from that
  gate (numbered), and a diamond is kept only for rules that combine. Gates and answers lead
  with their outcome ("✗ no (4%)") and thresholds read in percent. An escalation path is
  heavy amber.
- `to_mermaid` reads as one process: a state node (`state=`, None to leave it out), and a
  route drawn as its yes/no ladder. Each rule is a diamond, **yes** leads to its action and
  **no** to the next rule, and an escalation branches to a person. After a run, every gate is
  coloured by its decision, a question answered no greys out, wires that carried a no fade,
  and the way through the ladder is heavy.
  It also shows each question's wording (`text=False` for the compact form) and a
  scale answer's level name, draws each mounted chip as a box (nested chips inside their
  parent), labels a wire into a chip with its pin, shows the tau on a named threshold gate
  that feeds another gate, and leaves out the Decisions column when it would be empty.

## 0.5.6 (2026-10-04)

- **`SystemOne` retries a 529.** TypeSafe's API answers `529 Overloaded` under heavy load and
  documents it as retryable; it now waits and retries like a 503 instead of raising at once.

## 0.5.5 (2026-09-24)

- **`SystemOne` retries a rate limit.** A 429 is retried like a 502 or 503, after at least
  the `Retry-After` the server gives. A rate-limited gateway (LangSmith's default policy
  answers 429 with `retry-after: 10`) used to fail each call at once, so bulk runs lost most
  of their items.
- The `User-Agent` names the installed version (it said 0.4).

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
