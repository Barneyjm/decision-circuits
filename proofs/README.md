# Proofs

Lean 4 proofs, with Mathlib, of what the gates compute. CI checks them on every change here.

```bash
cd proofs && lake exe cache get && lake build
```

| file | proves |
|---|---|
| `Proofs/Count.lean` | The count distribution behind `at_least`, `count`, `and` and `or`: the Python loop is the recurrence of `∏ ((1 - p) + p X)`; it sums to 1 and is non-negative; `and` is the product, "none" the product of complements, `or` is 1 − P(none); `at_least 0` is 1, `at_least k` beyond the inputs is 0, and `at_least` falls as `k` rises; the order of the inputs does not matter. |
| `Proofs/Pick.lean` | Pick dominance: a decided categorical gate never reads an unpicked value likelier than its pick, for `argmax`, `verify`, `majority`, `order` and booleans. Each reading the SDK used before 0.5.3 is proved to break it, by a concrete counterexample. |

The proofs are over exact reals; the SDK runs on floats. `tests/test_properties.py` states
the same theorems about the Python and checks them on random inputs with Hypothesis (the
count distribution is also checked against brute-force enumeration of every outcome), so a
gap between the proved specification and the code fails the test suite.

Not proved, and not provable here: anything about a model. Calibration, accuracy and
robustness are measured by evals.
