# Changelog

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
