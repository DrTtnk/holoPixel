import Mathlib
import Holopixel.KLProjection

/-!
# KL Global Step — the u-block proximal closed form is an exact minimiser

`tier2_hfh/proximal.py`'s `proximal_global_step` claims, per pixel `j`, that

    u_j = (b_j + eta*u_old_j) / |b_j + eta*u_old_j|

is the exact `argmin_{|u|=1}` of

    d * |u|^2 - 2*Re(conj(b)*u) + eta * |u - u_old|^2

(`tests/test_proximal.py` re-derives this with sympy; this file re-derives
it again, independently, in Lean). The real part of `conj(b)*u` is the real
dot product `b.re*u.re + b.im*u.im` -- `globalObjective` below is stated
directly in that (equivalent, standard) form to keep the whole argument in
real coordinates, matching `KLProjection.lean`.

THE ARGUMENT: expanding both sides in real coordinates shows, UNCONDITIONALLY
(no constraint on `u` needed for this identity itself),

    globalObjective d eta b u_old u
      = normSq (u - target) + (d+eta-1)*normSq u + eta*normSq u_old - normSq target

where `target = b + eta*u_old` (`globalTarget` below). On the unit circle
`normSq u = 1`, the second term is the CONSTANT `d+eta-1`: this is
Established Fact 2 (`docs/notes_kl_convergence.md`) surviving the addition
of a proximal term -- `d` never appears in the part of the objective that
determines the minimiser. Minimising `normSq (u - target)` over that circle
is exactly `KLProjection.circleTarget_isMinimizer`, and
`circleTarget 1 target = target / |target|` is exactly the closed form
above. -/

open Complex

/-- The point `b + eta*u_old`, built directly from real components so its
`.re`/`.im` are `rfl`-computable (matching `circleTarget`'s own style). -/
noncomputable def globalTarget (eta : ℝ) (b u_old : ℂ) : ℂ :=
  ⟨b.re + eta * u_old.re, b.im + eta * u_old.im⟩

theorem globalTarget_re (eta : ℝ) (b u_old : ℂ) :
    (globalTarget eta b u_old).re = b.re + eta * u_old.re := rfl

theorem globalTarget_im (eta : ℝ) (b u_old : ℂ) :
    (globalTarget eta b u_old).im = b.im + eta * u_old.im := rfl

/-- The u-block proximal objective, `d*|u|^2 - 2*Re(conj(b)*u) + eta*|u-u_old|^2`
written via the real dot product `b.re*u.re+b.im*u.im = (conj b * u).re`. -/
noncomputable def globalObjective (d eta : ℝ) (b u_old u : ℂ) : ℝ :=
  d * normSq u - 2 * (b.re * u.re + b.im * u.im) + eta * normSq (u - u_old)

/-- The unconditional algebraic identity behind `globalObjective`: no
hypothesis on `u` is needed, it is pure real algebra. -/
theorem globalObjective_eq_general (d eta : ℝ) (b u_old u : ℂ) :
    globalObjective d eta b u_old u
      = normSq (u - globalTarget eta b u_old)
        + (d + eta - 1) * normSq u + eta * normSq u_old
        - normSq (globalTarget eta b u_old) := by
  simp only [globalObjective, normSq_eq_sq, Complex.sub_re, Complex.sub_im,
    globalTarget_re, globalTarget_im]
  ring

/-- On the unit circle, `globalObjective` is `normSq (u - target)` plus a
constant that does not depend on `u`. -/
theorem globalObjective_eq_normSq_sub_target (d eta : ℝ) (b u_old u : ℂ)
    (hu : normSq u = 1) :
    globalObjective d eta b u_old u
      = normSq (u - globalTarget eta b u_old)
        + (d + eta - 1 + eta * normSq u_old - normSq (globalTarget eta b u_old)) := by
  rw [globalObjective_eq_general, hu]; ring

/-- **KL Global Step.** For `eta > 0` and `target = b + eta*u_old ≠ 0`,
`circleTarget 1 target = target / ‖target‖` (i.e. exactly
`(b+eta*u_old)/|b+eta*u_old|`) is the unique minimiser of `globalObjective`
over `normSq u = 1`. -/
theorem globalStep_isMinimizer (d eta : ℝ) (b u_old : ℂ)
    (htarget : globalTarget eta b u_old ≠ 0) :
    (∀ u : ℂ, normSq u = 1 →
        globalObjective d eta b u_old (circleTarget 1 (globalTarget eta b u_old))
          ≤ globalObjective d eta b u_old u) ∧
    (∀ u : ℂ, normSq u = 1 →
        globalObjective d eta b u_old u
          = globalObjective d eta b u_old (circleTarget 1 (globalTarget eta b u_old))
        → u = circleTarget 1 (globalTarget eta b u_old)) := by
  set target := globalTarget eta b u_old with htarget_def
  have hmem : normSq (circleTarget 1 target) = 1 := by
    simpa using circleTarget_mem 1 target htarget
  have hmin := circleTarget_isMinimizer 1 one_pos target htarget
  refine ⟨fun u hu => ?_, fun u hu heq => ?_⟩
  · have hu' : normSq u = 1 ^ 2 := by rw [hu]; norm_num
    rw [globalObjective_eq_normSq_sub_target d eta b u_old u hu,
      globalObjective_eq_normSq_sub_target d eta b u_old _ hmem]
    linarith [hmin.1 u hu']
  · have hu' : normSq u = 1 ^ 2 := by rw [hu]; norm_num
    rw [globalObjective_eq_normSq_sub_target d eta b u_old u hu,
      globalObjective_eq_normSq_sub_target d eta b u_old _ hmem] at heq
    exact hmin.2 u hu' (by linarith [heq])
