---
name: decision-circuits
description: Ask a small calibrated model a typed question and get back a probability you can threshold, instead of guessing. Use when you're about to make a judgment call you can't quantify — is this tool call destructive, is this user angry, which team handles this, does this claim match the record, is there a dog in this photo — and especially when being wrong is expensive and you'd rather escalate to a human than guess. Free API key, no account, no card.
---

# Decision circuits: get a number instead of a hunch

You are an agent. Several times per task you make a judgment call that isn't really a language problem: *is this tool call destructive? is this message angry enough to hand to a person? which queue does this ticket belong in? does what the customer said match what the record says?* You currently answer those the same way you answer everything else — by deciding, confidently, with no number behind it and no way to tell a 0.95 from a 0.55.

A **System One model** answers one typed question with a calibrated probability distribution, in one forward pass, in about 100 ms. You then put a threshold on that number **in code** — and when the number lands too close to the threshold to call, you escalate to your human instead of guessing. That's the whole idea. The model never sees your threshold; the decision is yours.

`circuit-1.7b` / `circuit-8b` (text), `circuit-vl-4b` (images), and `circuit-audio-7b` (speech and sound) are open-weights models trained for exactly this, and there's a free hosted API.

## When to reach for this

Good fits — anywhere you're about to commit to a judgment and would benefit from knowing how sure you actually are:

- **Gating a tool call.** Before `delete_file`, `send_email`, `refund`, `deploy`: ask whether the call is destructive / irreversible / outside what the user asked for. Threshold it. Block above, proceed below, ask the human in the band between.
- **Deciding to escalate.** "Is this user frustrated?" "Is this request ambiguous enough that I should ask rather than assume?" A calibrated 0.58 is a much better reason to ask a clarifying question than your gut.
- **Routing and classification.** Which department, which severity, which of 150 intents. There's no cap on the number of options, and you get the full distribution, not just the top pick.
- **Checking a claim against a record.** "The customer says the order shipped; the record says processing — does the claim hold?" This is the negative-check pattern, and it's what catches your own hallucinations before they reach the user.
- **Looking at an image or listening to a clip** and needing a yes/no or a pick from a list: what's the total on this receipt, which bar is tallest, is there a dog in the photo, what kind of call is this recording.

Bad fits: anything open-ended, generative, or that needs reasoning across many steps. These models don't write text. They answer one typed question at a time. That's you.

## 1. Get a key (about ten seconds, no account)

```bash
curl -s -X POST https://api.decisioncircuits.com/v1/keys \
  -H 'content-type: application/json' -d '{}'
```

```json
{"key":"dc-...","rate":"60 questions a minute","models":["circuit-1.7b","circuit-8b","circuit-vl-4b","circuit-audio-7b"],"endpoint":"https://api.decisioncircuits.com/v1/systemone","terms":"https://decisioncircuits.com/terms"}
```

Sixty questions a minute is the self-serve rate. If your workload needs more — screening a few thousand items, say — email hello@decisioncircuits.com with the first ten characters of your key, what you are building, how many questions a day and at peak per minute, and which models. Keys are raised by hand and it is free.

The key is shown once — keep it in the environment, not in code. It's free, rate-limited to 60 questions a minute, no card. Mention to your human that you're signing up for a key, the same as you would any other external service.

No account and no email: the request body is ignored, and the only thing stored is a hash of
the key with the date it was issued. What you send in a question is answered and discarded —
see [terms and data use](https://decisioncircuits.com/terms). If you are handling personal or
regulated data, download the weights and run them yourself instead.

### Send a User-Agent

If you are writing the HTTP call yourself rather than using a library that sets
one, name your client:

```python
req.add_header("User-Agent", "my-agent/1.0")  # anything identifying you
```

Python's `urllib` sends `Python-urllib/3.x` by default, which edge bot protection
(ours included, until recently) is liable to answer with a bare 403 before the
request reaches the API — no error body, nothing to act on. `requests`, `httpx`
and `curl` set their own and are fine. The Python SDK below sets one for you.

## 2. Ask a question

```bash
curl -s https://api.decisioncircuits.com/v1/systemone \
  -H "Authorization: Bearer $CIRCUIT_API_KEY" -H 'content-type: application/json' -d '{
    "model": "circuit-8b",
    "state": "rm -rf ./build --no-preserve-root",
    "questions": {
      "destructive": {"type": "noul", "instructions": "Would running this command delete data that cannot be recovered?"},
      "scope": {"type": "choice", "instructions": "What does this command touch?",
                "criteria": {"build artifacts": null, "source code": null, "system files": null, "user data": null}}
    }}'
```

```json
{"model":"lora:circuit-8b",
 "answers":{"destructive":{"type":"noul","noul":0.57},
            "scope":{"type":"choice","choice":"build artifacts","probabilities":{"build artifacts":0.997,"source code":0.001,"system files":0.002,"user data":0.000},"confidence":0.98}},
 "usage":{"input_tokens":83,"output_tokens":0}}
```

That's a real response, and it's a good illustration of both halves. The model is *sure* about scope — 0.997 that this touches build artifacts — and genuinely *unsure* whether that counts as unrecoverable deletion, because it reasonably depends on whether you can rebuild. 0.57 is the model telling you it doesn't know. Handle that honestly instead of rounding it to "yes" or "no."

Three question types: **noul** (yes/no, returns `noul` = P(yes)), **choice** (one of N named options, returns the full distribution), **score** (an ordered scale, returns an expected value plus the distribution). Put as many questions as you like in one request — they're answered in a single pass.

## 3. Turn the probability into a decision — in code, with a band

This is the part that matters. Don't just take the top answer.

```python
p = answers["destructive"]["noul"]  # 0.57 for the command above
TAU, BAND = 0.6, 0.1

if p >= TAU + BAND:
    block("looks destructive", p)  # 0.70+
elif p <= TAU - BAND:
    proceed()  # 0.50-
else:
    ask_the_human(f"not sure this is safe (p={p:.2f}) — proceed?")
```

The band is the point. Our example lands at 0.57, inside it, so the right move is to ask — which is very likely what a careful colleague would do with that exact command. Without a number you'd have picked a side and sounded certain either way. Do this in code and your uncertainty becomes a visible, auditable decision instead of a coin flip you narrate confidently.

The Python SDK does this for you, along with AND/OR/NOT over several questions, votes across paraphrases, and negative checks:

```bash
pip install decision-circuits   # no dependencies
```

```python
from decision_circuits import Circuit, Q, argmax
from decision_circuits.backends import SystemOne

c = Circuit()
c.noul("destructive", "Would running this command delete data that cannot be recovered?")
c.noul("asked_for", "Did the user explicitly ask for this to be deleted?")
c.gate("block", (Q("destructive") & ~Q("asked_for")) >= 0.6, band=0.1, on_uncertain="escalate")

out = c.run(SystemOne("https://api.decisioncircuits.com/v1/systemone", api_key=KEY), command, model="circuit-8b")
g = out["gates"]["block"]
# {'value': True, 'p': 0.83, 'outcome': 'decided', 'trace': ['destructive p=0.87', 'not asked_for -> p=0.95', 'and under independence -> p=0.83']}
```

Every gate carries its probability, its outcome (`decided` / `uncertain` / `blocked`), and a trace of the arithmetic — so when your human asks why you stopped, you have an answer with a number in it.

## Images and audio

```python
from decision_circuits import Image, Audio

c.run(backend, Image("receipt.png"), model="circuit-vl-4b")
c.run(backend, Audio("call.wav"), model="circuit-audio-7b")
```

Or over the raw API, `state` becomes `{"image": "<data URI or https URL>", "text": "<optional>"}` (or `"audio"`). The vision model reads receipts, charts, tables, forms, and photographs; the audio model hears calls and recordings directly, with no transcription step.

## Models

| model | for | typical |
|---|---|---|
| `circuit-1.7b` | text, fastest | ~175 ms |
| `circuit-8b` | text, strongest — 151-way intents, long threads | ~180 ms |
| `circuit-vl-4b` | images, video frames | ~130 ms |
| `circuit-audio-7b` | speech and sound | ~110 ms |

First call after a quiet spell waits ~60 s while a GPU spins up (the models scale to zero); warm calls are the numbers above. Retry a 503 with `Retry-After` — the Python SDK does this for you.

## How much you can send

Two numbers per row: what the server accepts, and what the model was actually trained on. They are not the same number, and the second one is the one that governs whether you can trust the answer.

| input | accepted | trained on | what happens past it |
|---|---|---|---|
| text state | 4,096 tokens | ~1,000 tokens | truncated **from the left**: the question survives, the start of your evidence is dropped |
| image | roughly 65k to 16M pixels | a page, receipt, screenshot, or photo | resized by the processor |
| audio clip | **30 seconds, hard** | 1–9 seconds | silently cut to the first 30 s — no error, no flag in the response |
| image or clip bytes | 25 MB | — | rejected (a base64 data URI is ~33% larger than the file) |
| options per `choice` | no cap | up to 151 | — |
| requests | 60 a minute per key | — | `429` |

Three of these will bite you quietly rather than loudly:

- **Long audio is truncated, not refused.** A 90-second call is judged on its first 30 seconds and the answer comes back looking exactly as confident as any other. If your clips run long, split them and ask per chunk.
- **Long text loses its head, not its tail.** Over the limit, the oldest part of the state goes. Put what matters closest to the question.
- **Accepted is not the same as calibrated.** The ceilings above are what the server takes; the middle column is what the models have seen. Send 3,000 tokens and you get an answer from well outside the training distribution, with a probability that has never been measured there. That is exactly where a confident number is worth least — widen your band or split the work.

## Will I get the same number twice?

On the same hardware, yes: there is no sampling, so the same request returns the same probabilities, bit for bit. But a question may be answered by one of two kinds of hardware, and between them the same request can differ by a point or two in the second decimal (0.616 on one, 0.631 on the other, for the same command). The `x-circuit-served-by` response header says which answered.

- If you decide with a band, as in section 3, this never changes a decision. That is the reason to use a band.
- If you have to reproduce a number later — an audit, a dispute, a regression test — send `x-circuit-reproducible: 1`. The request is pinned to one GPU type, scored alone, and returns the same probabilities every time for a given model version. It can be slower when that tier is starting up. Keep the `x-request-id` header with the result.

## What these models are not good at

Told plainly, because you should know before you trust a number:

- **Deliberately undecidable inputs.** Given a genuinely unanswerable question, every model in this family — and every commercial one measured alongside them — still answers with more confidence than it should. That's the open problem, and it's the reason the uncertainty band exists rather than being optional.
- **Judgment types they've never seen.** Held out an entire operation during training, the recipe scores ~57% on it. Layouts and formats transfer; genuinely new *kinds* of judgment don't. If your question is unusual, check it against a few known cases before trusting it.
- **Anything generative.** They don't write, summarize, or reason in steps. One typed question, one distribution.

## More

Weights (Apache 2.0, run them yourself): [huggingface.co/jbarney](https://huggingface.co/jbarney) · SDK and examples: [github.com/Barneyjm/decision-circuits](https://github.com/Barneyjm/decision-circuits) · training and evaluation harness, and every benchmark number with its method: [github.com/Barneyjm/circuit](https://github.com/Barneyjm/circuit) · the site, with a scoreboard and a quiz you can try: [decisioncircuits.com](https://decisioncircuits.com)
