import Mathlib

/-!
# Theorem B — a diagonal windowed to a diagonal, in parent coordinates

`R` is a pure coordinate-selection ("restriction") operator: its rows are
standard basis vectors of the parent (panel) space, picked out by an
injective map `σ : Fin M → Fin N` from window index to panel index.
Injectivity is essential: it is exactly the fact that `R` never reads the
same panel pixel into two different window slots, and is what rules out an
off-diagonal cross term below. Without it the claim is false (two window
rows selecting the same panel pixel would generate a nonzero off-diagonal
entry equal to `D`'s value at that shared row).

Given `D` diagonal on the window and `R` such a selection matrix, `Rᴴ * D * R`
is diagonal in the parent (panel) coordinates. -/

open Matrix

variable {N M : ℕ}

/-- `R` is a pure coordinate-selection matrix: every row is a standard basis
vector `e_{σ i}` of the parent space, for some injective `σ`. -/
def IsSelectionMatrix (R : Matrix (Fin M) (Fin N) ℂ) : Prop :=
  ∃ σ : Fin M → Fin N, Function.Injective σ ∧ ∀ i j, R i j = if j = σ i then 1 else 0

theorem restriction_of_diag_is_diag
    (D : Matrix (Fin M) (Fin M) ℂ) (hD : D.IsDiag)
    (R : Matrix (Fin M) (Fin N) ℂ) (hR : IsSelectionMatrix R) :
    (Rᴴ * D * R).IsDiag := by
  obtain ⟨σ, hσinj, hRσ⟩ := hR
  intro j j' hjj'
  show (Rᴴ * D * R) j j' = 0
  rw [Matrix.mul_apply]
  apply Finset.sum_eq_zero
  intro i _
  rw [Matrix.mul_apply, Finset.sum_mul]
  apply Finset.sum_eq_zero
  intro i' _
  rcases eq_or_ne i' i with rfl | hne
  · rw [conjTranspose_apply, hRσ i' j, hRσ i' j']
    by_cases hj : j = σ i'
    · by_cases hj' : j' = σ i'
      · exact absurd (hj.trans hj'.symm) hjj'
      · simp [hj']
    · simp [hj]
  · rw [hD hne, mul_zero, zero_mul]
