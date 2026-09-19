"""The scenario from "Attaining LLM Certainty with AI Decision Circuits"
(Barney, Towards Data Science, May 2025), closed as a circuit.

The article classified 100 water-utility customer calls into 11 types
with two independent LLM parsers, a schema validator, and a negative
checker ("is there enough information to categorize this?"), combined
by boolean logic into high / medium / low confidence with low routed to
a human. With Claude Sonnet 3.7: one parser alone was 91% accurate; the
circuit was 87% overall but 92.5% on the 80 high-confidence calls, and
sent 2% to a human.

This example runs the same design on a System One model. The three
questions are the article's three components; the answers are
calibrated probabilities instead of parsed strings, so "confidence" is
measured rather than inferred from agreement. Both readings are
reported: the article's boolean tiers, and probability tiers at the
article's own thresholds (0.8 / 0.5). The cost model is the article's
too: $0.10 per parser run, $200 per human review, $1,000 per undetected
error, scaled to 10,000 calls.

    export TYPESAFE_API_KEY=...
    uv run python examples/11_water_utility_calls.py
    uv run python examples/11_water_utility_calls.py --url http://localhost:8901/v1/systemone   # an open-weights S1 model

Data: examples/data/water_utility_calls.json, the article's 100 calls
(github.com/Barneyjm/ai-decision-circuits).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from decision_circuits import Circuit, Q, majority, verify
from decision_circuits.backends import SystemOne

CALL_TYPES = {
    "RESTORE": "Water service was shut off (repairs, construction, non-payment) and needs to be turned back on",
    "ABATEMENT": "A property, often vacant or abandoned, with a leak or running water that needs to be stopped",
    "AMR (METERING)": "Water meter or automatic meter reading problems: blank display, wrong readings, replacement",
    "BILLING": "Charges, payment, account balance, bill amount disputes, payment plans",
    "BPCS (BROKEN PIPE)": "A broken or leaking water main or service pipe, water coming up through the street or ground",
    "BTR/O (BAD TASTE & ODOR)": "Water tastes or smells bad: chlorine, rotten eggs, metallic, discolored",
    "C/I - DEP (CAVE IN/DEPRESSION)": "A sinkhole, cave-in, or depression in the street or sidewalk",
    "CEMENT": "Restoring pavement, sidewalk, or driveway concrete after utility work",
    "CHOKED DRAIN": "A storm drain, inlet, or catch basin that is clogged and not draining",
    "CLAIMS": "Compensation for damage caused by the utility's work or a water incident",
    "COMPOST": "Compost bins, yard waste, or composting program questions",
}

c = Circuit()
# The article's primary parser: a direct classification.
c.choice("primary", "Extract the category of this customer service call to a water utility.", CALL_TYPES)
# The backup parser: the article used a chain-of-thought prompt; here the same categories under a differently framed question.
c.choice("backup", "First identify the customer's main issue or concern, then match it to the category that would handle it.", CALL_TYPES)
# The negative checker.
c.noul("enough_info", "Does this call contain enough information to categorize it into one of the water-utility call types?", true="A specific, categorizable issue is described", false="Too vague, off-topic, or missing the key detail")

c.gate("vote", majority("primary", "backup"))  # the two parsers agree, and how strongly
c.gate("checked", verify("primary", check=Q("enough_info"), tau=0.5, min_confidence=0.0), on_uncertain="escalate")


def article_tier(g: dict, a: dict) -> tuple[str | None, str, bool]:
    """The article's combiner, over the same three signals: (call_type, confidence, needs_human)."""
    primary, backup = a["primary"]["choice"], a["backup"]["choice"]
    enough = a["enough_info"]["noul"] >= 0.5
    if not enough:
        if primary == backup:
            return primary, "medium", False
        return None, "low", True
    if primary == backup:
        return primary, "high", False
    return primary, "medium", False


def probability_tier(g: dict, a: dict) -> tuple[str | None, str, bool]:
    """What a calibrated model makes possible: tier by the probability of the answer itself."""
    p = a["primary"]["probabilities"][a["primary"]["choice"]]
    if g["checked"]["outcome"] != "decided":
        return None, "low", True
    if p >= 0.8:
        return a["primary"]["choice"], "high", False
    if p >= 0.5:
        return a["primary"]["choice"], "medium", False
    return None, "low", True


def report(name: str, rows: list[tuple[str, str | None, str, bool]]) -> None:
    n = len(rows)
    correct = sum(1 for truth, pred, _, _ in rows if pred == truth)
    tiers = Counter(t for _, _, t, _ in rows)
    tier_ok = Counter(t for truth, pred, t, _ in rows if pred == truth)
    humans = sum(1 for *_, h in rows if h)
    undetected = sum(1 for truth, pred, _, h in rows if not h and pred != truth)
    scale = 10_000 / n
    cost = 0.10 * 3 * 10_000 + 200 * humans * scale + 1000 * undetected * scale  # 3 "parser runs" (questions) per call
    print(f"\n{name}")
    print(f"  overall accuracy {correct / n:.0%}   human review {humans / n:.0%}   undetected errors {undetected / n:.0%}")
    for t in ("high", "medium", "low"):
        if tiers[t]:
            print(f"  {t:<6} {tiers[t]:>3} calls, {tier_ok[t] / tiers[t]:.1%} correct")
    print(f"  article cost model per 10,000 calls: ${cost:,.0f}  (parsers ${0.10 * 3 * 10_000:,.0f}, humans ${200 * humans * scale:,.0f}, undetected ${1000 * undetected * scale:,.0f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://api.typesafe.ai/v1/systemone")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY", "x")
    if "typesafe.ai" in args.url and key == "x":
        sys.exit("set TYPESAFE_API_KEY, or pass --url for a local server")
    backend = SystemOne(args.url, api_key=key, model=args.model or ("jev-latest" if "typesafe.ai" in args.url else "s1-proto"))

    calls = json.loads((Path(__file__).with_name("data") / "water_utility_calls.json").read_text())["calls"]
    article_rows, prob_rows, single = [], [], 0
    t0 = time.perf_counter()
    for call in calls:
        out = c.run(backend, call["customer_input"])
        a, g = out["answers"], out["gates"]
        single += a["primary"]["choice"] == call["type"]
        for rows, tier in ((article_rows, article_tier), (prob_rows, probability_tier)):
            pred, conf, human = tier(g, a)
            rows.append((call["type"], pred, conf, human))
    ms = (time.perf_counter() - t0) * 1000 / len(calls)

    print(f"{len(calls)} calls, {out['model']}, {ms:.0f} ms per call for all three questions")
    print(f"single parser (primary alone): {single / len(calls):.0%}   [article, Sonnet 3.7: 91%]")
    report("article tiers (agreement + negative check)         [article: 87% overall, high 92.5% on 80, human 2%]", article_rows)
    report("probability tiers (P(answer) >= 0.8 / 0.5, verify)", prob_rows)

    wrong = [(call["customer_input"][:70], call["type"], r[1], r[2]) for call, r in zip(calls, prob_rows, strict=True) if r[1] != call["type"]]
    if wrong:
        print("\nmisses under probability tiers:")
        for text, truth, pred, conf in wrong:
            print(f"  [{conf:<6}] {truth:<32} got {pred!s:<32} {text!r}")


if __name__ == "__main__":
    main()
