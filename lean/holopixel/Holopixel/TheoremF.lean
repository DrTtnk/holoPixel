import Mathlib

/-!
# Theorem F — the diagonal solve is the unique per-coordinate minimiser

If `H = diag(d)` with every `d_j > 0`, the least-squares objective decouples
into independent one-dimensional real problems, one per coordinate `j`:

    E(u) = Σ_j [ d_j * ‖u_j‖² − 2 Re(conj(b_j) * u_j) ] + const,

each minimised uniquely at `u_j = b_j / d_j`. This file proves that scalar
statement directly: `f(x) := d * ‖x‖² − 2 Re(conj(b) x)` satisfies the
"complete the square" identity `d * f(x) = ‖d x − b‖² − ‖b‖²` (via
`Complex.normSq_sub`), which reduces minimising `f` to minimising `‖d x −
b‖² ≥ 0`, attained uniquely at `d x = b`, i.e. `x = b / d`. -/

open Complex

/-- The master identity behind Theorem F: `‖d x − b‖² = d² ‖x‖² − 2 d Re(conj(b) x) + ‖b‖²`. -/
private theorem master_identity (d : ℝ) (b x : ℂ) :
    normSq ((d : ℂ) * x - b) = d ^ 2 * normSq x - 2 * d * (star b * x).re + normSq b := by
  rw [Complex.normSq_sub, Complex.normSq_mul, Complex.normSq_ofReal]
  have hre : ((d : ℂ) * x * (starRingEnd ℂ) b).re = d * (star b * x).re := by
    have hswap : (d : ℂ) * x * (starRingEnd ℂ) b = (d : ℂ) * (star b * x) := by
      rw [Complex.star_def]; ring
    rw [hswap, Complex.re_ofReal_mul]
  rw [hre]; ring

/-- **Theorem F.** For a positive real `d` and any complex `b`, the strictly
convex real function `x ↦ d * ‖x‖² − 2 Re(conj(b) x)` is minimised over `ℂ`,
uniquely, at `x = b / d`. -/
theorem diag_coordinate_unique_minimizer (d : ℝ) (hd : 0 < d) (b : ℂ) :
    (∀ x : ℂ, d * normSq (b / (d : ℂ)) - 2 * (star b * (b / (d : ℂ))).re
        ≤ d * normSq x - 2 * (star b * x).re) ∧
      ∀ x : ℂ, d * normSq x - 2 * (star b * x).re
          = d * normSq (b / (d : ℂ)) - 2 * (star b * (b / (d : ℂ))).re → x = b / (d : ℂ) := by
  have hd0 : (d : ℂ) ≠ 0 := by exact_mod_cast hd.ne'
  have hcancel : (d : ℂ) * (b / (d : ℂ)) - b = 0 := by
    rw [mul_div_cancel₀ b hd0, sub_self]
  -- The value of `f` at the claimed minimiser `b / d`.
  have hval0 : d * normSq (b / (d : ℂ)) - 2 * (star b * (b / (d : ℂ))).re = -normSq b / d := by
    have h0 := master_identity d b (b / (d : ℂ))
    rw [hcancel, Complex.normSq_zero] at h0
    rw [eq_comm, div_eq_iff hd.ne']
    nlinarith [h0]
  refine ⟨fun x => ?_, fun x heq => ?_⟩
  · have hdf : d * (d * normSq x - 2 * (star b * x).re)
        = normSq ((d : ℂ) * x - b) - normSq b := by
      have hm := master_identity d b x; nlinarith [hm]
    have hnn : 0 ≤ normSq ((d : ℂ) * x - b) := Complex.normSq_nonneg _
    rw [hval0, div_le_iff₀ hd]
    nlinarith [hdf, hnn]
  · have hdf : d * (d * normSq x - 2 * (star b * x).re)
        = normSq ((d : ℂ) * x - b) - normSq b := by
      have hm := master_identity d b x; nlinarith [hm]
    rw [hval0] at heq
    have hz : normSq ((d : ℂ) * x - b) = 0 := by
      have hmul : d * (-normSq b / d) = -normSq b := by field_simp
      rw [heq, hmul] at hdf
      linarith [hdf]
    have hxb : (d : ℂ) * x - b = 0 := Complex.normSq_eq_zero.1 hz
    have hdx : (d : ℂ) * x = b := by linear_combination hxb
    rw [eq_div_iff hd0]
    linear_combination hdx
