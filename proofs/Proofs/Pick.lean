import Mathlib

/-!
# Pick dominance: a categorical gate never reads an unpicked option likelier than its pick

`G("route")["technical"]` reads a categorical gate's probability of one value. When the gate
decided `billing`, no other value may read likelier than `billing`, or a downstream gate can
route to an option the gate did not choose. `Dominates dist pick` is that property.

For each categorical gate the SDK's reading is proved to dominate, and each reading the SDK
used before is proved not to, by a concrete counterexample:

| gate | reading (0.5.3) | a reading that fails |
|---|---|---|
| argmax, verify | the choice's own distribution, with the pick an argmax of it | 1 - p for every other option (0.5.1) |
| majority | each value's share of the paraphrases' votes | the mean of their distributions (0.5.2) |
| order | 1 for the bucket the expected score falls in, 0 elsewhere | each level's mass summed per bucket (0.5.2) |
| boolean | 1 for the decided value, 0 for the other | |

Only decided results are covered: an abstain, escalate or default is not a pick.
-/

namespace DecisionCircuits

variable {ι : Type*}

/-- No value reads likelier than the pick. -/
def Dominates (dist : ι → ℝ) (pick : ι) : Prop :=
  ∀ o, dist o ≤ dist pick

/-! ## argmax and verify -/

/-- Over finitely many options there is always an argmax, and reading the distribution
itself dominates at it. The SDK picks such an argmax (keeping the backend's `choice` when it
attains the maximum, so ties go the backend's way). -/
theorem argmax_dominates [Finite ι] [Nonempty ι] (P : ι → ℝ) : ∃ pick, Dominates P pick := by
  obtain ⟨pick, h⟩ := Finite.exists_max P
  exact ⟨pick, h⟩

/-- Trusting a `choice` the distribution does not support breaks dominance, whatever the
reading: a backend reporting `choice` = 1 with P = (.6, .4) has option 0 read likelier. So the
SDK takes the argmax itself rather than the backend's word. -/
theorem trusting_choice_fails :
    ∃ (P : Fin 2 → ℝ) (choice : Fin 2), ¬ Dominates P choice := by
  refine ⟨![0.6, 0.4], 1, ?_⟩
  intro h
  have := h 0
  norm_num at this

/-- The reading before 0.5.2: the pick's probability for the pick, `1 - p` for every other
option. -/
noncomputable def oneMinusReading [DecidableEq ι] (P : ι → ℝ) (pick : ι) (o : ι) : ℝ :=
  if o = pick then P pick else 1 - P pick

/-- It breaks dominance even at a true argmax of a proper distribution: billing .34, technical
.33, other .33; billing is picked, and technical reads .66. -/
theorem oneMinusReading_fails :
    ∃ (P : Fin 3 → ℝ) (pick : Fin 3),
      (∑ o, P o) = 1 ∧ Dominates P pick ∧ ¬ Dominates (oneMinusReading P pick) pick := by
  refine ⟨![0.34, 0.33, 0.33], 0, ?_, ?_, ?_⟩
  · simp [Fin.sum_univ_three]; norm_num
  · intro o; fin_cases o <;> simp <;> norm_num
  · intro h
    have := h 1
    simp [oneMinusReading] at this
    norm_num at this

/-! ## majority -/

/-- Each value's share of the votes, one vote per paraphrase. -/
noncomputable def voteShare [DecidableEq ι] (votes : List ι) (o : ι) : ℝ :=
  (votes.count o : ℝ) / votes.length

/-- The majority winner has at least as many votes as any value, so its share dominates. -/
theorem voteShare_dominates [DecidableEq ι] (votes : List ι) (winner : ι)
    (hwin : ∀ o, votes.count o ≤ votes.count winner) : Dominates (voteShare votes) winner := by
  intro o
  unfold voteShare
  gcongr
  exact_mod_cast hwin o

/-- The 0.5.2 reading, the mean of the paraphrases' distributions, breaks it: two paraphrases
at (.51, .49) vote for option 0 and one at (.01, .99) votes for option 1. Option 0 wins the
vote two to one, and the mean reads option 1 likelier (.657 against .343). -/
theorem meanReading_fails :
    let answers : Fin 3 → Fin 2 → ℝ := ![![0.51, 0.49], ![0.51, 0.49], ![0.01, 0.99]]
    let mean : Fin 2 → ℝ := fun o => (∑ i, answers i o) / 3
    (∀ i, answers i 0 + answers i 1 = 1) ∧ ¬ Dominates mean 0 := by
  intro answers mean
  refine ⟨?_, ?_⟩
  · intro i; fin_cases i <;> simp [answers] <;> norm_num
  · intro h
    have := h 1
    simp [mean, answers, Fin.sum_univ_three] at this
    norm_num at this

/-! ## order -/

/-- Reading 1 for the bucket the gate decided and 0 elsewhere dominates, trivially. -/
theorem indicator_dominates [DecidableEq ι] (pick : ι) :
    Dominates (fun o => if o = pick then (1 : ℝ) else 0) pick := by
  intro o
  simp only
  split_ifs <;> norm_num

/-- The 0.5.2 reading breaks it. `order` buckets the *expected* score. Levels 0 and 3 with
half the mass each give an expected score of 1.5, which cutpoints (.5, 1.5, 2.5) put in
bucket 2; with those cutpoints each level lands in its own bucket, so summing level mass per
bucket reads bucket 2 at 0 and bucket 0 at .5. -/
theorem levelMassReading_fails :
    let P : Fin 4 → ℝ := ![0.5, 0, 0, 0.5]
    let expected : ℝ := ∑ i : Fin 4, ((i : ℕ) : ℝ) * P i
    expected = 1.5 ∧ (0.5 ≤ expected ∧ 1.5 ≤ expected ∧ expected < 2.5) ∧ ¬ Dominates P 2 := by
  intro P expected
  have he : expected = 1.5 := by
    simp [expected, P, Fin.sum_univ_four]; norm_num
  refine ⟨he, by rw [he]; norm_num, ?_⟩
  intro h
  have := h 0
  simp [P] at this
  norm_num at this

/-! ## boolean gates -/

/-- A decided boolean reads 1 for its value and 0 for the other. -/
theorem boolean_dominates (v : Bool) :
    Dominates (fun o : Bool => if o = v then (1 : ℝ) else 0) v :=
  indicator_dominates v

end DecisionCircuits
