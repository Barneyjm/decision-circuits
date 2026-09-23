"""The Lean theorems in proofs/, checked against the Python on random inputs.

proofs/Proofs/Count.lean and Pick.lean prove the gate semantics over exact reals. These
properties state the same theorems about `gates.evaluate_gates` and `gates._count_dist`, which
run on floats, so a divergence between the proved specification and the code fails here.
"""

import itertools
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from decision_circuits.gates import Gate, _count_dist, evaluate_gates

prob = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
probs = st.lists(prob, min_size=0, max_size=8)
TOL = 1e-9


def brute_force(ps: list[float]) -> list[float]:
    """P(exactly j hold) by enumerating every outcome: the definition, not the algorithm."""
    out = [0.0] * (len(ps) + 1)
    for outcome in itertools.product([False, True], repeat=len(ps)):
        out[sum(outcome)] += math.prod(p if o else 1 - p for p, o in zip(ps, outcome, strict=True))
    return out


# ---- Count.lean ------------------------------------------------------------------------


@given(probs)
def test_count_dist_is_the_poisson_binomial(ps):
    assert all(math.isclose(a, b, abs_tol=TOL) for a, b in zip(_count_dist(ps), brute_force(ps), strict=True))


@given(probs)
def test_sums_to_one_and_is_nonnegative(ps):  # sum_countDist, countDist_nonneg
    d = _count_dist(ps)
    assert math.isclose(sum(d), 1.0, abs_tol=TOL) and min(d) >= -TOL


@given(probs)
def test_all_is_the_product_and_none_the_product_of_complements(ps):  # countDist_all, countDist_none
    d = _count_dist(ps)
    assert math.isclose(d[-1], math.prod(ps), abs_tol=TOL)
    assert math.isclose(d[0], math.prod(1 - p for p in ps), abs_tol=TOL)


@given(probs, st.randoms())
def test_order_does_not_matter(ps, rnd):  # countDist_perm
    qs = ps[:]
    rnd.shuffle(qs)
    assert all(math.isclose(a, b, abs_tol=TOL) for a, b in zip(_count_dist(ps), _count_dist(qs), strict=True))


def _answers(ps):
    return {f"q{i}": {"type": "noul", "noul": p} for i, p in enumerate(ps)}


@given(st.lists(prob, min_size=1, max_size=6), st.integers(min_value=0, max_value=8))
def test_at_least_gates_match_the_theorems(ps, k):  # atLeast_zero, atLeast_one, atLeast_too_many, atLeast_antitone
    refs = [f"q{i}" for i in range(len(ps))]
    r = evaluate_gates(
        {
            "k": Gate(op="at_least", inputs=refs, k=k),
            "k1": Gate(op="at_least", inputs=refs, k=k + 1),
            "or": Gate(op="or", inputs=refs),
            "and": Gate(op="and", inputs=refs),
        },
        _answers(ps),
    )
    assert r["k1"].p <= r["k"].p + TOL  # asking for more is never likelier
    if k == 0:
        assert math.isclose(r["k"].p, 1.0, abs_tol=TOL)
    if k > len(ps):
        assert r["k"].p == 0.0
    assert math.isclose(r["or"].p, 1 - math.prod(1 - p for p in ps), abs_tol=TOL)
    assert math.isclose(r["and"].p, math.prod(ps), abs_tol=TOL)


# ---- Pick.lean -------------------------------------------------------------------------


def dominates(result) -> bool:
    d = result.probabilities
    return d is not None and all(v <= d[str(result.value)] + TOL for v in d.values())


def distribution(n):
    return st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=n, max_size=n).filter(lambda xs: sum(xs) > 0)


def choice_answer(weights, reported):
    z = sum(weights)
    probs = {f"o{i}": w / z for i, w in enumerate(weights)}
    return {"type": "choice", "choice": f"o{reported}", "probabilities": probs, "confidence": 1.0}


@settings(max_examples=300)
@given(st.integers(min_value=2, max_value=5).flatmap(lambda n: st.tuples(distribution(n), st.integers(min_value=0, max_value=n - 1))))
def test_argmax_and_verify_picks_dominate_even_when_the_backend_reports_another_choice(case):
    weights, reported = case
    answers = {"c": choice_answer(weights, reported), "ok": {"type": "noul", "noul": 1.0}}
    r = evaluate_gates({"a": Gate(op="argmax", input="c"), "v": Gate(op="verify", input="c", check="ok", tau=0.5)}, answers)
    for g in (r["a"], r["v"]):
        assert g.outcome == "decided" and dominates(g)


@settings(max_examples=300)
@given(st.integers(min_value=2, max_value=4).flatmap(lambda n: st.lists(distribution(n), min_size=1, max_size=5)))
def test_majority_pick_dominates_its_vote_shares(paraphrases):
    answers = {f"p{i}": choice_answer(w, 0) for i, w in enumerate(paraphrases)}
    r = evaluate_gates({"m": Gate(op="majority", inputs=list(answers))}, answers)["m"]
    assert r.outcome != "decided" or dominates(r)
    assert math.isclose(sum(r.probabilities.values()), 1.0, abs_tol=TOL)


@given(distribution(4), st.lists(st.floats(min_value=0.0, max_value=3.0), min_size=1, max_size=3).map(sorted))
def test_order_pick_dominates(weights, cuts):
    z = sum(weights)
    probs = {str(i): w / z for i, w in enumerate(weights)}
    score = sum(i * p for i, p in enumerate(probs.values()))
    answers = {"s": {"type": "score", "score": score, "probabilities": probs, "confidence": 1.0}}
    r = evaluate_gates({"o": Gate(op="order", input="s", cutpoints=cuts, band=0.0)}, answers)["o"]
    assert dominates(r)


@given(prob, st.floats(min_value=0.05, max_value=0.95))
def test_decided_booleans_dominate(p, tau):
    r = evaluate_gates({"t": Gate(op="threshold", input="x", tau=tau, band=0.0)}, {"x": {"type": "noul", "noul": p}})["t"]
    assert dominates(r)
