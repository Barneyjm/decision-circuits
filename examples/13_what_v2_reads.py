"""What circuit v2 adds: questions that pick several options, point at the text, and choose
what to ask next; gates that count and cross-check; and a way to see what moved a decision.

    export CIRCUIT_API_KEY=dc-...        # free key from https://decisioncircuits.com/#api
    uv run python examples/13_what_v2_reads.py

A complaint desk. One request asks five questions about a customer's message:

  issues   multi    every problem the customer reports, each with its own probability
  ask      locate   which sentence says what they want, or none
  next     choice   the one follow-up question that would settle the case, or none needed
  refund   noul     do they want their money back?
  keep     noul     do they want to keep the order?

The gates are code over those numbers: how many problems (count), whether at least two
(at_least), whether "wants a refund" and "wants to keep it" contradict each other
(consistent), and whether the message says what the customer wants at all (locate's none).
Last, `ablate` removes one sentence at a time and reruns the circuit, so you see which
sentence the refund decision rests on: an intervention, not a story.

multi and locate need a circuit v2 model (circuit-1.7b v2.0 or later) and a text state.
"""

import os
import sys

from decision_circuits import Circuit, G, Q, argmax, at_least, consistent, count
from decision_circuits.backends import SystemOne

key = os.environ.get("CIRCUIT_API_KEY")
if not key:
    sys.exit("set CIRCUIT_API_KEY (get one at https://decisioncircuits.com/#api)")
api = SystemOne(os.environ.get("CIRCUIT_URL", "https://api.decisioncircuits.com/v1/systemone"), api_key=key, model="circuit-1.7b")

c = Circuit()
c.multi(
    "issues",
    "Which problems does the customer report?",
    {
        "late": "Arrived after the promised date",
        "damaged": "Broken, crushed or spoiled",
        "wrong_item": "Not what was ordered",
        "charged_wrong": "Billed the wrong amount",
    },
)
c.locate("ask", "Which sentence says what the customer wants us to do?", none="the message never says what they want")
c.choice(
    "next",
    "What should we ask the customer next, if anything, before we can resolve this?",
    {
        "order_number": "Ask for the order number",
        "photo": "Ask for a photo of the damage",
        "refund_or_replace": "Ask whether they want a refund or a replacement",
        "nothing": "Nothing: we have what we need",
    },
)
c.noul("refund", "Does the customer want their money back?")
c.noul("keep", "Does the customer want to keep the order?")

c.gate("problems", count("issues"))  # the most likely number of problems, with the whole distribution
c.gate("escalate", at_least(2, "issues", tau=0.6), band=0.1, on_uncertain="escalate")  # two or more problems: a person
c.gate("coherent", consistent("refund", "keep", relation="complement"), band=0.25, on_uncertain="escalate")  # want both? ask
c.gate("says_what", ~Q("ask")["none"] >= 0.5)  # the message states what they want
c.gate("follow_up", argmax("next", min_confidence=0.3))
c.gate("pay", (Q("refund") & G("says_what")) >= 0.6, band=0.1, on_uncertain="escalate")

message = (
    "Order 55812 arrived nine days late and the box was crushed, two of the glasses are broken. "
    "I paid for express shipping. I'd like a refund for the broken glasses and the shipping."
)
out = c.run(api, message)

a = out["answers"]
print("issues:  ", {k: round(p, 2) for k, p in a["issues"]["probabilities"].items()}, "->", a["issues"]["selected"])
print("ask:     ", [(x["text"], round(x["probability"], 2)) for x in a["ask"]["located"][:1]], "none", round(a["ask"]["none"], 2))
print("next:    ", a["next"]["choice"], round(a["next"]["probabilities"][a["next"]["choice"]], 2))
print()
for name, g in out["gates"].items():
    print(f"  {name:<10} {g['outcome']:<9} value={g['value']!r:<8} {'; '.join(g['trace'][-1:])}")

# What does the refund decision rest on? Remove each sentence and run the circuit again.
print("\nremove one sentence, run again:")
effects = c.ablate(api, message)
for seg, e in effects["effects"].items():
    moved = e["answers"]["refund"]["dp"]
    flipped = ", ".join(e["flipped"]) or "no gate flips"
    print(f"  {seg[:70]:<72} refund {moved:+.2f}   {flipped}")
