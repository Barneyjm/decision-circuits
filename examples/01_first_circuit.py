"""A first circuit, with no model and no network.

A circuit is questions plus gates. The questions get answered by a
model (later). The gates are code. Here we hand-write the answers so
you can see exactly what the gates do with them.

    uv run python examples/01_first_circuit.py
"""

from decision_circuits import Circuit, Q, argmax

# 1. Questions: what we want a model to judge about a piece of text.
c = Circuit()
c.noul("pii", "Does this text contain personal information about a private individual?")
c.noul("angry", "Is the writer angry?")
c.choice("dept", "Which team should handle this?", {"billing": "Money, refunds, invoices", "technical": "Bugs, outages", "other": None})

# 2. Gates: code that turns probabilities into decisions.
c.gate("redact", Q("pii") >= 0.7, on_uncertain="escalate")  # a threshold with an uncertainty band
c.gate("route", argmax("dept", min_confidence=0.3))  # pick the top option, abstain if it's a toss-up
c.gate("human", (Q("angry") | Q("pii")) >= 0.6)  # OR: 1 - (1-a)(1-b)

# 3. Answers: normally a model produces these. Today we write them by hand.
answers = {
    "pii": {"type": "noul", "noul": 0.91},
    "angry": {"type": "noul", "noul": 0.35},
    "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.72, "technical": 0.20, "other": 0.08}, "confidence": 0.55},
}

results = c.evaluate(answers)
for name, r in results.items():
    print(f"{name:<7} -> {r['value']!s:<8} p={r['p']}  {r['outcome']}   trace: {'; '.join(r['trace'])}")

# 4. Change one number and watch the uncertainty handling kick in.
answers["pii"]["noul"] = 0.68  # inside the band around 0.7
print("\nwith pii=0.68:")
print("redact ->", c.evaluate(answers)["redact"])

# 5. The same circuit as a diagram (paste into any Mermaid renderer or a GitHub README).
print("\n" + c.to_mermaid(results=results, answers=answers, plain=True))
