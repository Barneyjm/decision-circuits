"""A refund desk: a real process, not a yes/no.

A support ticket comes in asking for a refund. The business rules:

  * Refund automatically when the request is eligible, shows no abuse
    signals, the stated reason checks out against the order, and the
    amount is under $100.
  * Send to an agent when it is eligible but large, or the reason does
    not check out.
  * Deny when it is clearly ineligible and clean.
  * Anything the model is unsure about goes to a human, and a hostile
    customer always gets a human.

Some of those facts are the model's to judge (is the item described as
defective? is this abuse?). Some are code's (how many days since
delivery? how much money? how many prior refunds?). The circuit mixes
them: code facts are injected as answers with probability 1, so the
gates read them like any other input and the trace shows them.

    uv run python examples/08_refund_desk.py                    # offline, scripted answers
    TYPESAFE_API_KEY=... uv run python examples/08_refund_desk.py   # Jev answers the model questions
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

from decision_circuits import Answers, Circuit, G, Q, argmax, verify
from decision_circuits.backends import SystemOne

# ---------------------------------------------------------------- the circuit
c = Circuit()

# Model questions. `ticket` is the customer's message; `order` is our record.
c.noul("defective", "Does `ticket` describe the item as damaged, defective, wrong, or not as described?")
c.noul(
    "abuse",
    "Does `ticket` together with `order.prior_refunds_12mo` suggest refund abuse (repeated refunds, story inconsistent with the order, pressure tactics)?",
)
c.choice(
    "reason",
    "What is the customer's main reason for the refund?",
    {"damaged": None, "wrong_item": None, "not_received": None, "changed_mind": None, "other": None},
)
c.noul("supported", "Is the customer's stated reason consistent with `order` (dates, item, delivery status)?")
c.score("tone", "How is the customer's tone?", ["Calm", "Frustrated", "Hostile"])

# Code facts, injected below as answers with probability 1.0 (or 0.0).
c.noul("in_window", "Delivered within the last 30 days (computed).")
c.noul("small_amount", "Order under $100 (computed).")

# Gates. Each one reads probabilities and writes a decision plus a trace.
c.gate("eligible", (Q("in_window") | Q("defective")) >= 0.6, band=0.1, on_uncertain="escalate")
c.gate("risky", Q("abuse") >= 0.5, band=0.15, on_uncertain="escalate")
c.gate("reason_checked", verify("reason", check=Q("supported"), tau=0.7, min_confidence=0.3), on_uncertain="escalate")
c.gate("hostile", Q("tone")[2] >= 0.5)
c.gate("auto_refund", (G("eligible") & ~G("risky") & Q("small_amount")) >= 0.7, on_uncertain="escalate")
c.gate("route", argmax("reason", min_confidence=0.3))


# ------------------------------------------------------- code facts as answers
def facts(order: dict[str, Any], today: date) -> Answers:
    days = (today - date.fromisoformat(order["delivered_on"])).days
    return {
        "in_window": {"type": "noul", "noul": 1.0 if days <= 30 else 0.0},
        "small_amount": {"type": "noul", "noul": 1.0 if order["amount_usd"] < 100 else 0.0},
    }


class WithFacts:
    """A backend wrapper: the model answers its questions, code answers the rest."""

    def __init__(self, inner: Any, today: date):
        self.inner, self.today = inner, today

    def answer(self, state, questions, *, model=None):
        model_qs = {k: q for k, q in questions.items() if k not in ("in_window", "small_amount")}
        return {**self.inner.answer(state, model_qs, model=model), **facts(state["order"], self.today)}


# ----------------------------------------------------------- the decision
def decide(g: dict[str, Any]) -> str:
    """The last step is plain code over the gate results."""
    if any(r["outcome"] == "escalate" for r in g.values()):
        return "HUMAN (model unsure)"
    if g["hostile"]["value"]:
        return "HUMAN (hostile)"
    if g["auto_refund"]["value"] and g["reason_checked"]["outcome"] == "decided":
        return "AUTO REFUND"
    if g["eligible"]["value"] and not g["risky"]["value"]:
        return "AGENT REVIEW"
    if g["risky"]["value"]:
        return "FRAUD QUEUE"
    return "DENY (ineligible)"


# ------------------------------------------------------------- some tickets
TODAY = date(2026, 9, 19)
TICKETS = [
    (
        "T1",
        "The mug arrived cracked in two, photo attached. Refund please.",
        {"item": "ceramic mug", "amount_usd": 24.0, "delivered_on": "2026-09-14", "prior_refunds_12mo": 0},
    ),
    (
        "T2",
        "Changed my mind about the jacket, still in the bag with tags.",
        {"item": "rain jacket", "amount_usd": 180.0, "delivered_on": "2026-09-10", "prior_refunds_12mo": 1},
    ),
    (
        "T3",
        "Never got the package. Tracking says delivered but nothing here. Same as last month btw.",
        {"item": "headphones", "amount_usd": 89.0, "delivered_on": "2026-09-16", "prior_refunds_12mo": 4},
    ),
    (
        "T4",
        "I want a refund for the blender I bought in spring, it stopped working.",
        {"item": "blender", "amount_usd": 60.0, "delivered_on": "2026-04-02", "prior_refunds_12mo": 0},
    ),
    (
        "T5",
        "This is the THIRD time. Refund NOW or I'm disputing the charge and posting everywhere.",
        {"item": "phone case", "amount_usd": 15.0, "delivered_on": "2026-09-17", "prior_refunds_12mo": 0},
    ),
]

# Offline answers for the model questions, so the example runs without a key.
SCRIPT = {
    "T1": {"defective": 0.96, "abuse": 0.03, "reason": [0.9, 0.04, 0.02, 0.02, 0.02], "supported": 0.9, "tone": [0.9, 0.09, 0.01]},
    "T2": {"defective": 0.03, "abuse": 0.05, "reason": [0.02, 0.02, 0.02, 0.9, 0.04], "supported": 0.85, "tone": [0.95, 0.04, 0.01]},
    "T3": {"defective": 0.05, "abuse": 0.62, "reason": [0.02, 0.02, 0.9, 0.02, 0.04], "supported": 0.4, "tone": [0.6, 0.35, 0.05]},
    "T4": {"defective": 0.7, "abuse": 0.05, "reason": [0.8, 0.02, 0.02, 0.06, 0.1], "supported": 0.9, "tone": [0.9, 0.09, 0.01]},
    "T5": {"defective": 0.2, "abuse": 0.3, "reason": [0.2, 0.1, 0.1, 0.1, 0.5], "supported": 0.5, "tone": [0.02, 0.18, 0.8]},
}


class Scripted:
    def answer(self, state, questions, *, model=None):
        from decision_circuits import answer_from_probabilities

        s = SCRIPT[state["ticket_id"]]
        out = {}
        for qid, q in questions.items():
            v = s[qid]
            out[qid] = answer_from_probabilities(q, [v, 1 - v] if q["type"] == "noul" else v)
        return out


def run_desk(model_backend: Any, label: str, diagram: bool = True) -> None:
    """Run every ticket through the circuit with `model_backend` answering the model questions."""
    backend = WithFacts(model_backend, TODAY)
    print("answers from:", label)
    for tid, text, order in TICKETS:
        out = c.run(backend, {"ticket_id": tid, "ticket": text, "order": order})
        g = out["gates"]
        print(f"\n{tid}  ${order['amount_usd']:<6} {text[:58]!r}")
        print(
            f"     eligible={g['eligible']['value']!s:<5} risky={g['risky']['value']!s:<5} reason={g['route']['value']!s:<13} "
            f"checked={g['reason_checked']['outcome']:<9} hostile={g['hostile']['value']!s:<5} auto={g['auto_refund']['value']!s:<5}"
        )
        print(f"  => {decide(g)}")
    if diagram:
        print("\n" + c.to_mermaid(plain=True))


if __name__ == "__main__":
    key = os.environ.get("TYPESAFE_API_KEY")
    run_desk(SystemOne(api_key=key, model="jev-latest") if key else Scripted(), "Jev" if key else "scripted (offline)")
