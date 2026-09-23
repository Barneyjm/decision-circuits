import Mathlib

/-!
# The count distribution behind `at_least`, `count`, `and` and `or`

`gates._count_dist(ps)` returns, for independent events with probabilities `ps`, the
probability that exactly `j` of them hold (the Poisson binomial distribution). Here it is the
coefficient of `X ^ j` in `∏ ((1 - p) + p X)`. The Python builds it one event at a time;
`countDist_append` is that loop, line for line, so the theorems below are about what the
code computes (over ℝ; the code uses floats).
-/

open Polynomial

namespace DecisionCircuits

/-- `∏ ((1 - p) + p X)` over the events. -/
noncomputable def countPoly (ps : List ℝ) : ℝ[X] :=
  (ps.map fun p => C (1 - p) + C p * X).prod

/-- P(exactly `j` of the events hold). -/
noncomputable def countDist (ps : List ℝ) (j : ℕ) : ℝ :=
  (countPoly ps).coeff j

/-- P(at least `k` hold), as the `at_least` gate reads it. -/
noncomputable def atLeast (ps : List ℝ) (k : ℕ) : ℝ :=
  ∑ j ∈ Finset.Ico k (ps.length + 1), countDist ps j

lemma natDegree_factor (p : ℝ) : (C (1 - p) + C p * X).natDegree ≤ 1 := by
  compute_degree

lemma natDegree_countPoly (ps : List ℝ) : (countPoly ps).natDegree ≤ ps.length := by
  unfold countPoly
  refine (natDegree_list_prod_le _).trans ?_
  induction ps with
  | nil => simp
  | cons p ps ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    have := natDegree_factor p
    omega

/-- The Python loop: adding an event `p` gives `new[j] = old[j] (1 - p) + old[j - 1] p`. -/
theorem countDist_append (ps : List ℝ) (p : ℝ) (j : ℕ) :
    countDist (ps ++ [p]) j =
      countDist ps j * (1 - p) + (if j = 0 then 0 else countDist ps (j - 1) * p) := by
  unfold countDist countPoly
  rw [List.map_append, List.prod_append]
  simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
  rw [mul_add, coeff_add, coeff_mul_C, ← mul_assoc, mul_right_comm, coeff_mul_C]
  rcases j with _ | j
  · simp
  · simp [coeff_mul_X]

/-- The distribution sums to one. -/
theorem sum_countDist (ps : List ℝ) :
    ∑ j ∈ Finset.range (ps.length + 1), countDist ps j = 1 := by
  have h := eval_eq_sum_range' (p := countPoly ps) (n := ps.length + 1)
    (Nat.lt_succ_of_le (natDegree_countPoly ps)) 1
  simp only [one_pow, mul_one] at h
  unfold countDist
  rw [← h]
  unfold countPoly
  rw [eval_list_prod]
  simp [Function.comp_def]

/-- `and` is "all of them": P(exactly n of n) is the product. -/
theorem countDist_all (ps : List ℝ) : countDist ps ps.length = ps.prod := by
  unfold countDist countPoly
  have h := coeff_list_prod_of_natDegree_le (ps.map fun p => C (1 - p) + C p * X) 1
    (by simp only [List.mem_map]; rintro _ ⟨p, _, rfl⟩; exact natDegree_factor p)
  simp only [List.length_map, mul_one] at h
  rw [h, List.map_map]
  have hid : ((fun q : ℝ[X] => q.coeff 1) ∘ fun p : ℝ => C (1 - p) + C p * X) = id :=
    funext fun p => by simp [coeff_C, coeff_one]
  rw [hid, List.map_id]

/-- `or`'s complement: P(none of them) is the product of the complements. -/
theorem countDist_none (ps : List ℝ) : countDist ps 0 = (ps.map fun p => 1 - p).prod := by
  unfold countDist countPoly
  rw [coeff_zero_eq_eval_zero, eval_list_prod]
  simp [Function.comp_def]

/-- `or` is "at least one": 1 - P(none). -/
theorem atLeast_one (ps : List ℝ) : atLeast ps 1 = 1 - (ps.map fun p => 1 - p).prod := by
  have h := sum_countDist ps
  rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (Nat.succ_pos _)] at h
  unfold atLeast
  rw [← countDist_none]
  linarith

/-- `at_least 0` always holds. -/
theorem atLeast_zero (ps : List ℝ) : atLeast ps 0 = 1 := by
  unfold atLeast
  rw [← Finset.range_eq_Ico]
  exact sum_countDist ps

/-- `at_least k` for more events than there are never holds. -/
theorem atLeast_too_many (ps : List ℝ) (k : ℕ) (hk : ps.length < k) : atLeast ps k = 0 := by
  unfold atLeast
  rw [Finset.Ico_eq_empty (by omega), Finset.sum_empty]

/-- The order the events are listed in does not matter. -/
theorem countDist_perm {ps qs : List ℝ} (h : ps.Perm qs) (j : ℕ) :
    countDist ps j = countDist qs j := by
  unfold countDist countPoly
  rw [(h.map _).prod_eq]

/-- With probabilities in [0, 1], every count has a probability in [0, 1]. -/
theorem countDist_nonneg (ps : List ℝ) (hps : ∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1) (j : ℕ) :
    0 ≤ countDist ps j := by
  induction ps using List.reverseRecOn generalizing j with
  | nil =>
    unfold countDist countPoly
    rcases j with _ | j <;> simp [coeff_one]
  | append_singleton ps p ih =>
    have hp := hps p (by simp)
    have ih' := ih (fun q hq => hps q (by simp [hq]))
    rw [countDist_append]
    have h1 : 0 ≤ countDist ps j * (1 - p) := mul_nonneg (ih' j) (by linarith [hp.2])
    split_ifs
    · linarith
    · exact add_nonneg h1 (mul_nonneg (ih' _) hp.1)

/-- Asking for more is never likelier: `at_least k` falls as `k` rises. -/
theorem atLeast_antitone (ps : List ℝ) (hps : ∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1) {k l : ℕ} (hkl : k ≤ l) :
    atLeast ps l ≤ atLeast ps k := by
  unfold atLeast
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · exact Finset.Ico_subset_Ico hkl le_rfl
  · intro j _ _
    exact countDist_nonneg ps hps j

end DecisionCircuits
