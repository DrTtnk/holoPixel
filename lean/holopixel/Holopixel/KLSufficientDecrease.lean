import Mathlib

/-!
# KL Sufficient Decrease — the three-point inequality, in full generality

The KL convergence theorem (Attouch-Bolte-Redont-Soubeyran 2010,
arXiv:0801.1780, Theorem 8/9 -- see `docs/notes_kl_convergence.md`) needs
the sufficient-decrease inequality

    phi(x_old) - phi(x_new) >= (1/(2*lam)) * dist(x_new, x_old)^2

for each proximal block step. This file proves it in the generality it
actually holds at: for ANY set `S`, ANY function `phi : α → ℝ` on ANY
(pseudo)metric space, whenever `x_new` minimises
`phi(.) + (1/(2*lam))*dist(., x_old)^2` over `S` and `x_old ∈ S`. The
KL/semialgebraic machinery is not used anywhere in this file -- sufficient
decrease is a one-line consequence of `x_new` being a minimiser and `x_old`
being an admissible competitor for it (plug `x = x_old` into the
minimality hypothesis; `dist(x_old,x_old)=0` does the rest). This matches
the point made in the notes: sufficient decrease needs no KL property, only
that each proximal step is an EXACT minimiser of its own subproblem, which
`KLGlobalStep.lean` and `KLLocalStep.lean` establish for this solver's two
blocks. -/

/-- **KL Sufficient Decrease.** If `x_new` minimises the proximal-regularised
objective `phi(.) + (1/(2*lam))*dist(.,x_old)^2` over a set `S` containing
`x_old`, then `phi` drops from `x_old` to `x_new` by at least
`(1/(2*lam)) * dist(x_new,x_old)^2`. -/
theorem sufficient_decrease {α : Type*} [PseudoMetricSpace α] (phi : α → ℝ)
    (S : Set α) (lam : ℝ) (hlam : 0 < lam) (x_old x_new : α) (h_mem : x_old ∈ S)
    (h_min : ∀ x ∈ S, phi x_new + (1 / (2 * lam)) * dist x_new x_old ^ 2
        ≤ phi x + (1 / (2 * lam)) * dist x x_old ^ 2) :
    phi x_old - phi x_new ≥ (1 / (2 * lam)) * dist x_new x_old ^ 2 := by
  have h := h_min x_old h_mem
  simp only [dist_self] at h
  nlinarith [h]

/-- Summed over a run of iterates `x 0, x 1, ..., x n`, sufficient decrease
telescopes into a bound on the total squared path length by the total drop
in `phi` -- the fact `proximal.py`'s docstring and `tests/test_proximal.py`
call the "telescoping bound" (`step_sum <= (L0 - Linf) / c`). Stated here
for a constant per-step rate `c > 0`, which is what a FIXED `eta`/`gamma`
(no per-iteration schedule) gives. -/
theorem telescoped_step_bound {α : Type*} [PseudoMetricSpace α] (phi : α → ℝ)
    (c : ℝ) (hc : 0 < c) (x : ℕ → α) (n : ℕ)
    (hdecrease : ∀ k, phi (x k) - phi (x (k + 1)) ≥ c * dist (x (k + 1)) (x k) ^ 2) :
    c * ∑ k ∈ Finset.range n, dist (x (k + 1)) (x k) ^ 2 ≤ phi (x 0) - phi (x n) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, mul_add]
    have hstep := hdecrease n
    linarith [ih]
