import Mathlib

/-!
# Theorem D — a zero diagonal entry of `H` kills every active `Pₖ`'s column

`H = Σ_k ρ_k Pₖᴴ Pₖ` with `ρ_k ≥ 0` real. If `H_jj = 0` then, for every `k`
with `ρ_k > 0`, `Pₖ` annihilates the `j`-th standard basis vector, i.e.
column `j` of `Pₖ` is zero.

## On the extra hypothesis this needed

The task description suggested this might need positive-semidefiniteness of
each `Pₖᴴ Pₖ` as an extra assumption. It does not: the diagonal entry
`(Pₖᴴ Pₖ)_jj` is *automatically* `‖(Pₖ)_{·j}‖² ≥ 0` for any matrix `Pₖ`,
because it is a Gram matrix diagonal entry, not an extra property to assume.
The only real hypothesis needed, beyond what the problem already states, is
`ρ_k ≥ 0` for *every* `k` (already given) — without nonnegativity for the
other terms, a single term could be zero by cancellation against a negative
one even though `Pₖ e_j ≠ 0`. With `ρ_k ≥ 0` for all `k`, the sum of
nonnegative reals argument goes through cleanly. -/

open Matrix

variable {N L : ℕ} {ι : Type*} [Fintype ι]

/-- The diagonal entry of a Gram matrix is a nonnegative real, cast into `ℂ`. -/
private theorem gram_diag_eq_ofReal_sum_normSq
    (P : Matrix (Fin L) (Fin N) ℂ) (j : Fin N) :
    (Pᴴ * P) j j = ((∑ i, Complex.normSq (P i j) : ℝ) : ℂ) := by
  rw [Matrix.mul_apply]
  push_cast
  apply Finset.sum_congr rfl
  intro i _
  rw [conjTranspose_apply, Complex.star_def, ← Complex.normSq_eq_conj_mul_self]

theorem gram_diag_zero_implies_column_zero
    (P : ι → Matrix (Fin L) (Fin N) ℂ) (ρ : ι → ℝ) (hρ : ∀ k, 0 ≤ ρ k)
    (H : Matrix (Fin N) (Fin N) ℂ) (hH : H = ∑ k, (ρ k : ℂ) • ((P k)ᴴ * P k))
    (j : Fin N) (hj : H j j = 0)
    (k₀ : ι) (hk₀ : 0 < ρ k₀) :
    (P k₀) *ᵥ (Pi.single j (1 : ℂ)) = 0 := by
  -- Reduce `H j j = 0` to a real statement about nonnegative energies.
  have hHreal : (∑ k, ρ k * ∑ i, Complex.normSq (P k i j) : ℝ) = 0 := by
    have : H j j = ((∑ k, ρ k * ∑ i, Complex.normSq (P k i j) : ℝ) : ℂ) := by
      rw [hH]
      simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul]
      push_cast
      apply Finset.sum_congr rfl
      intro k _
      rw [gram_diag_eq_ofReal_sum_normSq]
      push_cast
      ring
    rw [this] at hj
    exact_mod_cast hj
  have hnonneg : ∀ k ∈ (Finset.univ : Finset ι), 0 ≤ ρ k * ∑ i, Complex.normSq (P k i j) :=
    fun k _ => mul_nonneg (hρ k) (Finset.sum_nonneg fun i _ => Complex.normSq_nonneg _)
  have hterm : ρ k₀ * ∑ i, Complex.normSq (P k₀ i j) = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg hnonneg).1 hHreal k₀ (Finset.mem_univ k₀)
  have hsum : ∑ i, Complex.normSq (P k₀ i j) = 0 :=
    (mul_eq_zero.1 hterm).resolve_left (ne_of_gt hk₀)
  have hcol : ∀ i, P k₀ i j = 0 := by
    intro i
    have h0 : Complex.normSq (P k₀ i j) = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => Complex.normSq_nonneg _)).1 hsum i
        (Finset.mem_univ i)
    exact Complex.normSq_eq_zero.1 h0
  rw [Matrix.mulVec_single_one]
  funext i
  simpa using hcol i
