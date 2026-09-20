---
name: onboarding
description: Orientation for this repo (PyPI package decision-circuits). Load this FIRST in a new session before editing the SDK, the site, the gateway, or the deploy pipeline — it answers "what is this and what's already live" so you don't have to ask. Triggers on "what is this repo", "get me oriented", "where do I start", "what's the current state", cold starts in general, or any request touching the SDK/site/gateway/examples you don't already have context for.
---

# Orientation: decision-circuits

You're in `decision-circuits` (PyPI package, GitHub `Barneyjm/decision-circuits`). This is a small, near-zero-dependency Python SDK: a DSL for building **decision circuits** — typed questions sent to a model, calibrated probabilities back, deterministic gates (thresholds with uncertainty bands, AND/OR/NOT, argmax, majority vote, verify, order) evaluated in code, never by the model. It also owns the public marketing site and the free hosted API gateway.

**The sibling repo you also need to know about:** `~/Documents/code/s1-proto` (GitHub `Barneyjm/circuit`). That repo trains and evaluates the actual open-weights models this SDK talks to, and depends on *this* package (`decision-circuits` is on its `pyproject.toml` dependency list, not the reverse — this package has zero dependencies). If you're asked to train a model, generate benchmark data, run a GPU pod, or change how a model's forward pass works, go there instead — it has its own `onboarding` skill.

## The core idea, in one paragraph

`Circuit()` declares typed questions (`c.noul(...)`, `c.choice(...)`, `c.score(...)`) and gates over them (`c.gate("name", Q("x") >= 0.7, band=0.1, on_uncertain="escalate")`). `circuit.run(backend, state, model=...)` sends the state and questions to a backend (`SystemOne` — any server speaking `POST /v1/systemone`, including TypeSafe's Jev, our own hosted models, or a locally-run `s1proto` server), gets back calibrated probabilities, and evaluates the gates client-side (or reads server-evaluated ones) into a `Judgment`: a decided value, an uncertain escalation, or a block, each carrying its probability and a human-readable trace. The state can be plain text/JSON/chat-messages, or an `Image(...)`/`Audio(...)` media object for the vision/audio models.

## Repo layout

```
src/decision_circuits/    dsl.py (Circuit/Q/G/gates DSL), gates.py (AND/OR/NOT/threshold/argmax/majority/verify/order + evaluate_gates), types.py (Answer types, to_jsonable), media.py (Image/Audio media states), langgraph.py, backends/ (systemone.py, openai.py, anthropic.py), integrations/ (langchain.py, openai_agents.py, claude_agent_sdk.py, _policy.py shared CircuitPolicy/Judgment)
tests/                    one file per module; test_examples.py runs every examples/*.py that doesn't need a live key as an offline smoke test — "the examples are the docs," keep them running
examples/                 01-12, numbered in the order a reader should go through them: DSL basics -> real backends -> integrations -> a full business-process example (refund desk) -> the article's water-utility scenario -> images/audio through the hosted API
docs/                     concepts.md, integrations.md, extending.md
docs/index.html           the ENTIRE public site (decisioncircuits.com) — single file, inline <style> and <script>, no build step. docs/bench/items.json is the data behind the "think you're better than the model" quiz; regenerate the quiz markup+script with `uv run python tools/quiz_section.py` after editing items.json — do NOT hand-edit the quiz's generated HTML/JS block (between `// quiz:start` / `// quiz:end` comments), it gets overwritten
gateway/                  worker.js + wrangler.toml — the Cloudflare Worker at api.decisioncircuits.com: issues free API keys (KV-stored, hashed), rate-limits via Workers' rate-limit bindings (NOT KV writes — that was a deliberate fix, don't revert it), proxies to Modal
site_worker.js            the OTHER Cloudflare Worker (name "decisioncircuits", the main site deploy) — redirects every host/protocol variant to https://decisioncircuits.com, serves docs/ as static assets with a real docs/404.html
wrangler.toml             site deploy config (root); gateway/wrangler.toml is separate
CHANGELOG.md              keep current — GitHub releases are cut from its sections
.githooks/pre-commit      ruff format --check + ruff check on staged .py; core.hooksPath is set to this
```

## What's live right now (check before assuming — these change)

- **PyPI**: `pip install decision-circuits`, current version in `pyproject.toml` — check it, don't assume the number in this skill is current.
- **decisioncircuits.com**: `npx wrangler deploy` from repo root pushes `docs/`. Custom domain + www/http redirects are handled by `site_worker.js`.
- **api.decisioncircuits.com**: `npx wrangler deploy -c gateway/wrangler.toml` pushes the gateway. It proxies to whatever `MODAL_URLS` in `gateway/wrangler.toml` points at — keep that in sync with what's actually deployed on Modal (see the sibling repo's `deploy/modal_app.py`).
- **GitHub Actions**: `ci.yml` (ruff + pytest on push/PR) and `publish.yml` (PyPI on a `v*` tag push) — both should be green; check `gh run list --workflow=ci.yml --limit 1` if something feels off.

## Hard rules, not suggestions

- **Zero-dependency core.** `dependencies = []` in `pyproject.toml` is deliberate — optional integrations (`langchain`, `openai`, `anthropic`, `openai-agents`, `claude-agent-sdk`) are `dependency-groups`/extras, never core deps. Don't add a runtime dependency to the base package without a very good reason and the user's sign-off.
- **The examples are the docs and must keep running offline.** `tests/test_examples.py` executes every example that doesn't need a live API key. If you touch the DSL or a backend's interface, run the full test suite (`uv run pytest -q`), not just the file you think you changed.
- **`gh` releases + `CHANGELOG.md` + PyPI project links stay in sync** on every version bump — see the pattern in git history for 0.4.0/0.4.1 (a `CHANGELOG.md` section, a `gh release create vX.Y.Z`, `[project.urls]` in `pyproject.toml`).
- **The site is one file on purpose** (`docs/index.html`) — no build tooling, no framework. Keep it that way; don't introduce a bundler.
- **Cloudflare account is `james@southendsolutions.com` / account id `5e09cd27e2878baba3bba9f7fef6e66f`; wrangler is already logged in.** Don't re-auth unless it's actually expired.
- **Never commit a live API key.** `S1_API_KEY` (the upstream Modal auth) is a Worker secret (`wrangler secret put`), not in `wrangler.toml`. The public-facing keys the gateway issues to users are meant to be given out freely; the upstream key is not.

## How to check "what's the current state" quickly

1. `git log --oneline -20` in both this repo and `s1-proto`.
2. `cat pyproject.toml | grep version` and `gh release list --limit 3` — is PyPI/GitHub in sync with what's in `main`?
3. `curl -s https://api.decisioncircuits.com/v1/models` — which models does the live gateway actually serve right now?
4. `docs/bench/items.json` — what's currently in the quiz, and what did each model actually answer (with timing) the last time it was refreshed.
5. `docs/cold-eval.md` in the sibling `s1-proto` repo is the numbers source of truth for anything the site's scoreboard claims.

## Common task: refreshing the quiz or the family scoreboard after a model changes

1. In `s1-proto`: retrain/re-evaluate, publish to Hugging Face, update `docs/cold-eval.md` there.
2. Here: update the relevant family-card `<dl>` numbers in `docs/index.html`, and if a quiz item's answer/timing changed, re-run the item through the live `api.decisioncircuits.com` endpoint and write the result into `docs/bench/items.json`, then `uv run python tools/quiz_section.py` to regenerate the quiz section — never hand-edit the generated block.
3. `npx wrangler deploy`, commit, push. Check the live page renders (a headless screenshot at both desktop and ~390px phone width catches the two most common regressions: horizontal overflow and a card that doesn't fit).
