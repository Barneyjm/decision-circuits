"""Write your own backend in a dozen lines.

A backend is any object with one method:

    answer(state, questions, *, model=None) -> {question_id: answer}

Return the wire-format answer for each question. The easiest way is to
produce a distribution over the question's options and let
`answer_from_probabilities` build the answer (it normalizes, picks the
argmax, computes the expected score and the confidence).

This one is a keyword heuristic, which is deliberately dumb: the point
is the contract, not the intelligence. Swap in an HTTP call, a local
model, a cache, or an ensemble; the circuit does not care.

    uv run python examples/03_your_own_backend.py
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from decision_circuits import Answers, Backend, Circuit, Q, answer_from_probabilities, argmax, option_keys

HINTS = {
    "pii": ["card", "ssn", "phone", "address", "@"],
    "angry": ["now", "!", "unacceptable", "ridiculous"],
    "billing": ["charge", "refund", "invoice", "card"],
    "technical": ["error", "bug", "api", "outage", "500"],
}


class KeywordBackend:
    """Scores each option by how many of its hint words appear in the state."""

    def answer(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Answers:
        text = str(state).lower()
        out: Answers = {}
        for qid, q in questions.items():
            keys = option_keys(q)  # ["yes","no"] for a noul; the option ids for a choice; "0".."n" for a score
            weights = []
            for k in keys:
                hints = HINTS.get(qid if q["type"] == "noul" else k, [])
                hits = sum(h in text for h in hints)
                weights.append(1.0 + 3.0 * hits if k != "no" else 1.0)
            if q["type"] == "noul":  # for a noul, "no" gets the complement of the evidence
                weights = [weights[0], 1.0 + 3.0 * (weights[0] == 1.0)]
            out[qid] = answer_from_probabilities(q, weights)
        return out


assert isinstance(KeywordBackend(), Backend)  # structural: no base class needed

if __name__ == "__main__":
    c = Circuit()
    c.noul("pii", "Does this text contain personal information?")
    c.noul("angry", "Is the writer angry?")
    c.choice("dept", "Which team?", {"billing": None, "technical": None, "other": None})
    c.gate("redact", Q("pii") >= 0.7, on_uncertain="escalate")
    c.gate("route", argmax("dept", min_confidence=0.3))

    for text in ["Card charged twice, refund NOW!", "Getting a 500 error from the API", "hello there"]:
        out = c.run(KeywordBackend(), text)
        print(f"{text!r:<40} redact={out['gates']['redact']['value']!s:<6} route={out['gates']['route']['value']!s:<10} ({out['gates']['route']['outcome']})")
