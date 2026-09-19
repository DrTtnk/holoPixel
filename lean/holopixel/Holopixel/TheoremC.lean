import Mathlib

/-!
# Theorem C — a weighted sum of diagonal matrices is diagonal

If every `H k` is diagonal and `ρ k` are scalars, then `Σ k, ρ k • H k` is
diagonal. This is the step that turns Theorem B's per-pupil diagonality into
the global claim: the accumulated normal operator `H = Σ_k ρ_k P_kᴴ P_k` of
the whole least-squares problem is exactly diagonal, term by term. -/

open Matrix

variable {N : ℕ} {ι : Type*} [Fintype ι]

theorem sum_diag_is_diag
    (H : ι → Matrix (Fin N) (Fin N) ℂ) (hH : ∀ k, (H k).IsDiag) (ρ : ι → ℂ) :
    (∑ k, ρ k • H k).IsDiag := by
  apply Finset.sum_induction (fun k => ρ k • H k) Matrix.IsDiag
  · exact fun _ _ ha hb => ha.add hb
  · exact Matrix.isDiag_zero
  · exact fun k _ => (hH k).smul (ρ k)
