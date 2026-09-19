import Mathlib

/-!
# KL Projection — the constrained minimiser both proximal steps reduce to

Both proximal closed forms in `tier2_hfh/proximal.py` (re-derived with sympy
in `tests/test_proximal.py`) reduce, after a short algebraic rewrite done in
`KLGlobalStep.lean` and `KLLocalStep.lean`, to one geometric fact: on the
circle `normSq z = a ^ 2` centred at the origin, the point closest to a
fixed external point `y` is the point on the ray from the origin through
`y`, rescaled to lie on that circle.

This is proved here in real coordinates (`z.re, z.im, y.re, y.im`), the same
representation `tests/test_local_global.py` already used (in sympy, via
Lagrange multipliers) for the non-proximal case `a=1`. Here the route is a
direct 2-D Cauchy-Schwarz / Lagrange-identity argument instead:
`(z1^2+z2^2)(y1^2+y2^2) - (z1*y1+z2*y2)^2 = (z1*y2-z2*y1)^2` (a pure `ring`
identity, `lagrange_identity_2d` below) bounds the cross term
`z.re*y.re+z.im*y.im` on the circle by `a * ‖y‖`, with equality only on the
ray through `y`. -/

open Complex

/-- Lagrange's identity in two dimensions -- pure algebra, no hypotheses. -/
private theorem lagrange_identity_2d (z1 z2 y1 y2 : ℝ) :
    (z1 ^ 2 + z2 ^ 2) * (y1 ^ 2 + y2 ^ 2) - (z1 * y1 + z2 * y2) ^ 2
      = (z1 * y2 - z2 * y1) ^ 2 := by ring

/-- `normSq` in real coordinates, as a sum of squares (`normSq_apply` gives
the product form `z.re*z.re+z.im*z.im`; squares are more convenient here). -/
theorem normSq_eq_sq (z : ℂ) : normSq z = z.re ^ 2 + z.im ^ 2 := by
  rw [normSq_apply]; ring

theorem normSq_sub_eq_sq (z y : ℂ) :
    normSq (z - y) = (z.re - y.re) ^ 2 + (z.im - y.im) ^ 2 := by
  rw [normSq_eq_sq]; simp [Complex.sub_re, Complex.sub_im]

/-- The candidate minimiser: the point on the ray through `y`, at radius `a`. -/
noncomputable def circleTarget (a : ℝ) (y : ℂ) : ℂ :=
  ⟨a * y.re / Real.sqrt (normSq y), a * y.im / Real.sqrt (normSq y)⟩

/-- `circleTarget a y` lies on the circle of radius `a`, for `y ≠ 0`. -/
theorem circleTarget_mem (a : ℝ) (y : ℂ) (hy : y ≠ 0) :
    normSq (circleTarget a y) = a ^ 2 := by
  have hy2 : 0 < normSq y := normSq_pos.mpr hy
  set ry := Real.sqrt (normSq y) with hry_def
  have hyc : y.re ^ 2 + y.im ^ 2 = ry ^ 2 := by
    rw [hry_def, Real.sq_sqrt hy2.le, normSq_eq_sq]
  have step : (circleTarget a y).re ^ 2 + (circleTarget a y).im ^ 2
      = a ^ 2 * (y.re ^ 2 + y.im ^ 2) / ry ^ 2 := by
    change (a * y.re / ry) ^ 2 + (a * y.im / ry) ^ 2
        = a ^ 2 * (y.re ^ 2 + y.im ^ 2) / ry ^ 2
    field_simp
  have hry : 0 < ry := Real.sqrt_pos.mpr hy2
  rw [normSq_eq_sq, step, hyc, mul_div_assoc, div_self (pow_ne_zero 2 hry.ne'), mul_one]

/-- **KL Projection.** For `a > 0` and `y ≠ 0`, `circleTarget a y` is the
unique minimiser of `z ↦ normSq (z - y)` over the circle `normSq z = a ^ 2`. -/
theorem circleTarget_isMinimizer (a : ℝ) (ha : 0 < a) (y : ℂ) (hy : y ≠ 0) :
    (∀ z : ℂ, normSq z = a ^ 2 →
        normSq (circleTarget a y - y) ≤ normSq (z - y)) ∧
    (∀ z : ℂ, normSq z = a ^ 2 →
        normSq (z - y) = normSq (circleTarget a y - y) → z = circleTarget a y) := by
  have hy2 : 0 < normSq y := normSq_pos.mpr hy
  set ry := Real.sqrt (normSq y) with hry_def
  have hry : 0 < ry := Real.sqrt_pos.mpr hy2
  have hyc : y.re ^ 2 + y.im ^ 2 = ry ^ 2 := by
    rw [hry_def, Real.sq_sqrt hy2.le, normSq_eq_sq]
  set w := circleTarget a y with hw_def
  have hwre : w.re = a * y.re / ry := rfl
  have hwim : w.im = a * y.im / ry := rfl
  have hwmem : normSq w = a ^ 2 := circleTarget_mem a y hy
  -- the cross term at the candidate equals a * ry exactly
  have hwcross : w.re * y.re + w.im * y.im = a * ry := by
    rw [hwre, hwim]
    have expand : a * y.re / ry * y.re + a * y.im / ry * y.im
        = a * (y.re ^ 2 + y.im ^ 2) / ry := by ring
    rw [expand, hyc]
    field_simp
  -- the 2-D Cauchy-Schwarz identity, for any z (no constraint needed yet)
  have hcs : ∀ z : ℂ, normSq z = a ^ 2 →
      (a * ry) ^ 2 - (z.re * y.re + z.im * y.im) ^ 2
        = (z.re * y.im - z.im * y.re) ^ 2 := by
    intro z hz
    have hzc : z.re ^ 2 + z.im ^ 2 = a ^ 2 := by rw [← normSq_eq_sq]; exact hz
    have hL := lagrange_identity_2d z.re z.im y.re y.im
    rw [hzc, hyc] at hL
    rw [mul_pow]
    linarith [hL]
  -- the Cauchy-Schwarz bound itself
  have hbound : ∀ z : ℂ, normSq z = a ^ 2 →
      z.re * y.re + z.im * y.im ≤ a * ry := by
    intro z hz
    have h1 := hcs z hz
    nlinarith [h1, sq_nonneg (z.re * y.im - z.im * y.re), mul_pos ha hry]
  refine ⟨fun z hz => ?_, fun z hz heq => ?_⟩
  · rw [normSq_sub_eq_sq, normSq_sub_eq_sq]
    have hzc : z.re ^ 2 + z.im ^ 2 = a ^ 2 := by rw [← normSq_eq_sq]; exact hz
    have hwc : w.re ^ 2 + w.im ^ 2 = a ^ 2 := by rw [← normSq_eq_sq]; exact hwmem
    have hb := hbound z hz
    nlinarith [hb, hzc, hwc, hwcross]
  · rw [normSq_sub_eq_sq, normSq_sub_eq_sq] at heq
    have hzc : z.re ^ 2 + z.im ^ 2 = a ^ 2 := by rw [← normSq_eq_sq]; exact hz
    have hwc : w.re ^ 2 + w.im ^ 2 = a ^ 2 := by rw [← normSq_eq_sq]; exact hwmem
    have hcross_eq : z.re * y.re + z.im * y.im = a * ry := by
      nlinarith [heq, hzc, hwc, hwcross]
    have hperp : z.re * y.im - z.im * y.re = 0 := by
      have hz0 : (z.re * y.im - z.im * y.re) ^ 2 = 0 := by
        rw [← hcs z hz, hcross_eq]; ring
      exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hz0
    -- solve the 2x2 linear system {hcross_eq, hperp} for (z.re, z.im)
    have hre_eq : z.re * ry ^ 2 = a * ry * y.re := by
      linear_combination y.re * hcross_eq + y.im * hperp - z.re * hyc
    have him_eq : z.im * ry ^ 2 = a * ry * y.im := by
      linear_combination y.im * hcross_eq - y.re * hperp - z.im * hyc
    have hre2 : z.re * ry = a * y.re :=
      mul_right_cancel₀ hry.ne' (by linear_combination hre_eq)
    have him2 : z.im * ry = a * y.im :=
      mul_right_cancel₀ hry.ne' (by linear_combination him_eq)
    apply Complex.ext
    · rw [hwre, eq_div_iff hry.ne']; exact hre2
    · rw [hwim, eq_div_iff hry.ne']; exact him2
