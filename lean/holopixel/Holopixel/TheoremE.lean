import Mathlib

/-!
# Theorem E — the normal equations give a global minimiser

`E(u) = Σ_k ρ_k ‖P_k u − z_k‖²` is minimised by any `u₀` solving the normal
equations `H u₀ = b`, where `H = Σ_k ρ_k P_kᴴ P_k` and `b = Σ_k ρ_k P_kᴴ z_k`.

We prove this by an algebraic "complete the square" identity rather than by
calculus: for a solution `u₀` of the normal equations and any `u`,

    E(u) = E(u₀) + Σ_k ρ_k ‖P_k(u − u₀)‖² ≥ E(u₀),

because the cross term vanishes exactly when `H u₀ = b`. This needs no
differentiability machinery, only `dotProduct`/`mulVec`/`star` algebra, and it
proves the direction that actually matters for the solver: a solution of the
linear system is a global minimiser, for *free* real weights `ρ_k ≥ 0` (no
positive-definiteness of `H` is needed for this direction). We record this
honestly as a one-directional theorem; see `Holopixel/TheoremF.lean` for the
uniqueness statement in the diagonal case, proved directly rather than via a
general converse to this theorem. -/

open Matrix

variable {N L : ℕ} {ι : Type*} [Fintype ι]

/-- Squared energy `‖v‖²` of a vector, as a nonnegative real number. -/
noncomputable def sqNorm {m : ℕ} (v : Fin m → ℂ) : ℝ := ∑ i, Complex.normSq (v i)

theorem sqNorm_nonneg {m : ℕ} (v : Fin m → ℂ) : 0 ≤ sqNorm v :=
  Finset.sum_nonneg fun _ _ => Complex.normSq_nonneg _

/-- `sqNorm` agrees with the algebraic dot product `star v ⬝ᵥ v`. -/
theorem sqNorm_eq_dotProduct {m : ℕ} (v : Fin m → ℂ) : (sqNorm v : ℂ) = star v ⬝ᵥ v := by
  unfold sqNorm dotProduct
  push_cast
  apply Finset.sum_congr rfl
  intro i _
  rw [Pi.star_apply, Complex.star_def, ← Complex.normSq_eq_conj_mul_self]

/-- The weighted least-squares objective. -/
noncomputable def lsObjective (P : ι → Matrix (Fin L) (Fin N) ℂ) (z : ι → (Fin L → ℂ))
    (ρ : ι → ℝ) (u : Fin N → ℂ) : ℝ :=
  ∑ k, ρ k * sqNorm (P k *ᵥ u - z k)

/-- The normal-equations matrix `H = Σ_k ρ_k Pₖᴴ Pₖ`. -/
noncomputable def normalMatrix (P : ι → Matrix (Fin L) (Fin N) ℂ) (ρ : ι → ℝ) :
    Matrix (Fin N) (Fin N) ℂ :=
  ∑ k, (ρ k : ℂ) • ((P k)ᴴ * P k)

/-- The normal-equations right-hand side `b = Σ_k ρ_k Pₖᴴ zₖ`. -/
noncomputable def normalRhs (P : ι → Matrix (Fin L) (Fin N) ℂ) (z : ι → (Fin L → ℂ)) (ρ : ι → ℝ) :
    Fin N → ℂ :=
  ∑ k, (ρ k : ℂ) • ((P k)ᴴ *ᵥ z k)

/-- `star (Mᴴ *ᵥ v) = star v ᵥ* M`, the vector form of `Mᴴ` acting as the
adjoint of `M` with respect to the `star _ ⬝ᵥ _` pairing. -/
private theorem star_conjTranspose_mulVec {a b : ℕ} (M : Matrix (Fin a) (Fin b) ℂ)
    (v : Fin a → ℂ) : star (Mᴴ *ᵥ v) = star v ᵥ* M := by
  rw [star_mulVec, Matrix.conjTranspose_conjTranspose]

/-- The cross term in the expansion vanishes exactly when `u₀` solves the
normal equations: `Σ_k ρ_k (star (Pₖ u₀ − zₖ) ⬝ᵥ Pₖ d) = 0` for every `d`. -/
private theorem cross_term_vanishes
    (P : ι → Matrix (Fin L) (Fin N) ℂ) (z : ι → (Fin L → ℂ)) (ρ : ι → ℝ)
    (u₀ : Fin N → ℂ) (hu₀ : normalMatrix P ρ *ᵥ u₀ = normalRhs P z ρ) (d : Fin N → ℂ) :
    ∑ k, (ρ k : ℂ) * (star (P k *ᵥ u₀ - z k) ⬝ᵥ (P k *ᵥ d)) = 0 := by
  have hterm : ∀ k, (ρ k : ℂ) * (star (P k *ᵥ u₀ - z k) ⬝ᵥ (P k *ᵥ d))
      = ((ρ k : ℂ) • star ((P k)ᴴ *ᵥ (P k *ᵥ u₀ - z k))) ⬝ᵥ d := by
    intro k
    rw [dotProduct_mulVec, ← star_conjTranspose_mulVec, ← smul_eq_mul, smul_dotProduct]
  simp_rw [hterm]
  rw [← sum_dotProduct]
  have hsum : ∑ k, (ρ k : ℂ) • star ((P k)ᴴ *ᵥ (P k *ᵥ u₀ - z k)) = 0 := by
    have hstar : ∀ k, (ρ k : ℂ) • star ((P k)ᴴ *ᵥ (P k *ᵥ u₀ - z k))
        = star ((ρ k : ℂ) • ((P k)ᴴ *ᵥ (P k *ᵥ u₀ - z k))) := by
      intro k
      rw [star_smul, Complex.star_def, Complex.conj_ofReal]
    simp_rw [hstar]
    rw [← star_sum]
    have hkey : ∑ k, (ρ k : ℂ) • ((P k)ᴴ *ᵥ (P k *ᵥ u₀ - z k))
        = normalMatrix P ρ *ᵥ u₀ - normalRhs P z ρ := by
      unfold normalMatrix normalRhs
      simp only [mulVec_sub, smul_sub, mulVec_mulVec, ← smul_mulVec]
      rw [Finset.sum_sub_distrib, ← sum_mulVec]
    rw [hkey, hu₀, sub_self, star_zero]
  rw [hsum, zero_dotProduct]

/-- **Theorem E.** A solution of the normal equations `H u₀ = b` is a global
minimiser of the weighted least-squares objective `E`, for any real weights
`ρ_k ≥ 0`. -/
theorem normal_equations_minimize
    (P : ι → Matrix (Fin L) (Fin N) ℂ) (z : ι → (Fin L → ℂ)) (ρ : ι → ℝ) (hρ : ∀ k, 0 ≤ ρ k)
    (u₀ : Fin N → ℂ) (hu₀ : normalMatrix P ρ *ᵥ u₀ = normalRhs P z ρ) :
    ∀ u, lsObjective P z ρ u₀ ≤ lsObjective P z ρ u := by
  intro u
  set d : Fin N → ℂ := u - u₀ with hd
  have hexpand : ∀ k, sqNorm (P k *ᵥ u - z k)
      = sqNorm (P k *ᵥ u₀ - z k) + 2 * (star (P k *ᵥ u₀ - z k) ⬝ᵥ (P k *ᵥ d)).re
        + sqNorm (P k *ᵥ d) := by
    intro k
    set a : Fin L → ℂ := P k *ᵥ u₀ - z k with ha
    set b : Fin L → ℂ := P k *ᵥ d with hb
    have hsplit : P k *ᵥ u - z k = a + b := by
      rw [ha, hb, hd, mulVec_sub]; abel
    have hcomplex : ((sqNorm (P k *ᵥ u - z k) : ℝ) : ℂ)
        = ((sqNorm a : ℝ) : ℂ) + (star a ⬝ᵥ b) + star (star a ⬝ᵥ b)
          + ((sqNorm b : ℝ) : ℂ) := by
      rw [sqNorm_eq_dotProduct, sqNorm_eq_dotProduct, sqNorm_eq_dotProduct, hsplit,
        star_add, add_dotProduct, dotProduct_add, dotProduct_add]
      have hswap : star b ⬝ᵥ a = star (star a ⬝ᵥ b) := star_dotProduct _ _
      rw [hswap]; ring
    have hre := congrArg Complex.re hcomplex
    simp only [Complex.add_re, Complex.ofReal_re, Complex.star_def, Complex.conj_re] at hre
    linarith [hre]
  have hE : lsObjective P z ρ u
      = lsObjective P z ρ u₀
        + 2 * (∑ k, (ρ k : ℂ) * (star (P k *ᵥ u₀ - z k) ⬝ᵥ (P k *ᵥ d))).re
        + ∑ k, ρ k * sqNorm (P k *ᵥ d) := by
    unfold lsObjective
    have hre : (∑ k, (ρ k : ℂ) * (star (P k *ᵥ u₀ - z k) ⬝ᵥ (P k *ᵥ d))).re
        = ∑ k, ρ k * (star (P k *ᵥ u₀ - z k) ⬝ᵥ (P k *ᵥ d)).re := by
      rw [Complex.re_sum]
      apply Finset.sum_congr rfl
      intro k _
      rw [Complex.mul_re]
      simp
    rw [hre]
    rw [show (2 : ℝ) * ∑ k, ρ k * (star (P k *ᵥ u₀ - z k) ⬝ᵥ (P k *ᵥ d)).re
        = ∑ k, 2 * (ρ k * (star (P k *ᵥ u₀ - z k) ⬝ᵥ (P k *ᵥ d)).re) from Finset.mul_sum _ _ _]
    rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun k _ => by rw [hexpand k]; ring
  rw [hE, cross_term_vanishes P z ρ u₀ hu₀ d]
  simp only [Complex.zero_re, mul_zero]
  have hnn : 0 ≤ ∑ k, ρ k * sqNorm (P k *ᵥ d) :=
    Finset.sum_nonneg fun k _ => mul_nonneg (hρ k) (sqNorm_nonneg _)
  linarith
