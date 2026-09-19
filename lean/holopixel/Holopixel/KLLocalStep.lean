import Mathlib
import Holopixel.KLProjection

/-!
# KL Local Step — the z-block proximal closed form is an exact minimiser

`tier2_hfh/proximal.py`'s `proximal_local_step` claims, per (view,pixel)
index, that

    z = a * (y + gamma*z_old) / |y + gamma*z_old|

is the exact `argmin_{|z|=a}` of

    |z - y|^2 + gamma * |z - z_old|^2

(`tests/test_proximal.py` re-derives this with sympy; this file re-derives
it again, independently, in Lean, by the same route `KLGlobalStep.lean`
uses for the u-block).

THE ARGUMENT: expanding in real coordinates gives, UNCONDITIONALLY (no
constraint on `z` needed for this identity itself),

    localObjective gamma y z_old z
      = normSq (z - target) + gamma*normSq z + normSq y + gamma*normSq z_old - normSq target

where `target = y + gamma*z_old` (`localTarget` below). On the circle
`normSq z = a^2`, the extra terms are a CONSTANT (`gamma*a^2 + ...`), so
minimising `localObjective` there is the same problem as minimising
`normSq (z - target)`, which is exactly `KLProjection.circleTarget_isMinimizer`
at radius `a`. -/

open Complex

/-- The point `y + gamma*z_old`, built directly from real components. -/
noncomputable def localTarget (gamma : ℝ) (y z_old : ℂ) : ℂ :=
  ⟨y.re + gamma * z_old.re, y.im + gamma * z_old.im⟩

theorem localTarget_re (gamma : ℝ) (y z_old : ℂ) :
    (localTarget gamma y z_old).re = y.re + gamma * z_old.re := rfl

theorem localTarget_im (gamma : ℝ) (y z_old : ℂ) :
    (localTarget gamma y z_old).im = y.im + gamma * z_old.im := rfl

/-- The z-block proximal objective, `|z-y|^2 + gamma*|z-z_old|^2`. -/
noncomputable def localObjective (gamma : ℝ) (y z_old z : ℂ) : ℝ :=
  normSq (z - y) + gamma * normSq (z - z_old)

/-- The unconditional algebraic identity behind `localObjective`: pure real
algebra, no hypothesis on `z` needed. -/
theorem localObjective_eq_general (gamma : ℝ) (y z_old z : ℂ) :
    localObjective gamma y z_old z
      = normSq (z - localTarget gamma y z_old)
        + gamma * normSq z + normSq y + gamma * normSq z_old
        - normSq (localTarget gamma y z_old) := by
  simp only [localObjective, normSq_eq_sq, Complex.sub_re, Complex.sub_im,
    localTarget_re, localTarget_im]
  ring

/-- On the circle `normSq z = a^2`, `localObjective` is `normSq (z-target)`
plus a constant that does not depend on `z`. -/
theorem localObjective_eq_normSq_sub_target (gamma a : ℝ) (y z_old z : ℂ)
    (hz : normSq z = a ^ 2) :
    localObjective gamma y z_old z
      = normSq (z - localTarget gamma y z_old)
        + (gamma * a ^ 2 + normSq y + gamma * normSq z_old
           - normSq (localTarget gamma y z_old)) := by
  rw [localObjective_eq_general, hz]; ring

/-- **KL Local Step.** For `a > 0` and `target = y + gamma*z_old ≠ 0`,
`circleTarget a target = a*target/‖target‖` (i.e. exactly
`a*(y+gamma*z_old)/|y+gamma*z_old|`) is the unique minimiser of
`localObjective` over `normSq z = a^2`. -/
theorem localStep_isMinimizer (gamma a : ℝ) (ha : 0 < a) (y z_old : ℂ)
    (htarget : localTarget gamma y z_old ≠ 0) :
    (∀ z : ℂ, normSq z = a ^ 2 →
        localObjective gamma y z_old (circleTarget a (localTarget gamma y z_old))
          ≤ localObjective gamma y z_old z) ∧
    (∀ z : ℂ, normSq z = a ^ 2 →
        localObjective gamma y z_old z
          = localObjective gamma y z_old (circleTarget a (localTarget gamma y z_old))
        → z = circleTarget a (localTarget gamma y z_old)) := by
  set target := localTarget gamma y z_old with htarget_def
  have hmem : normSq (circleTarget a target) = a ^ 2 := circleTarget_mem a target htarget
  have hmin := circleTarget_isMinimizer a ha target htarget
  refine ⟨fun z hz => ?_, fun z hz heq => ?_⟩
  · rw [localObjective_eq_normSq_sub_target gamma a y z_old z hz,
      localObjective_eq_normSq_sub_target gamma a y z_old _ hmem]
    linarith [hmin.1 z hz]
  · rw [localObjective_eq_normSq_sub_target gamma a y z_old z hz,
      localObjective_eq_normSq_sub_target gamma a y z_old _ hmem] at heq
    exact hmin.2 z hz (by linarith [heq])
