"""The same circuit, answered by a real System One model (TypeSafe's Jev).

    export TYPESAFE_API_KEY=...
    uv run python examples/02_jev_backend.py

Nothing about the circuit changes between example 01 and this one. Only
who answers the questions. That is the whole idea: the gates are yours
and stay put; the model behind them is swappable.
"""

import os
import sys
import time

from decision_circuits import Circuit, Q, argmax
from decision_circuits.backends import SystemOne

key = os.environ.get("TYPESAFE_API_KEY")
if not key:
    sys.exit("set TYPESAFE_API_KEY (an eval key from typesafe.ai works)")

c = Circuit()
c.noul("pii", "Does this text contain personal information about a private individual?")
c.noul("angry", "Is the writer angry?")
c.choice("dept", "Which team should handle this?", {"billing": "Money, refunds, invoices", "technical": "Bugs, outages", "other": None})
c.gate("redact", Q("pii") >= 0.7, on_uncertain="escalate")
c.gate("route", argmax("dept", min_confidence=0.3))
c.gate("human", (Q("angry") | Q("pii")) >= 0.6)

jev = SystemOne(api_key=key, model="jev-latest")  # standard-library HTTP; nothing to install

for text in [
    "Card charged twice, refund NOW. My card ends in 4412 and my number is 555-0142.",
    "Is there an API for exporting reports? Thanks!",
]:
    t0 = time.perf_counter()
    out = c.run(jev, text)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n{text[:60]!r}  ({ms:.0f} ms, {out['model']})")
    for name, r in out["gates"].items():
        print(f"  {name:<7} -> {r['value']!s:<8} p={r['p']:.2f}  {r['outcome']}")
