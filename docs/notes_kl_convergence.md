# Does KL theory prove iterate convergence for the local/global solver?

## Bottom line, first

**Yes, for a proximal variant, and the proof is close to free once you set it up
correctly — but it proves the wrong thing to be exciting.** The theorem
(Attouch–Bolte–Redont–Soubeyran 2010, below) guarantees the *whole* iterate
sequence `(u_t, z_t)` converges to a *single* critical point of the coupling
energy, with a summable step sequence. It says **nothing about which**
critical point. Numerically the proximal variant settles at essentially the
same (generally non-global) fixed point the plain solver already finds — the
final residual moves by a few percent, not by orders of magnitude, on every
seed tried. The vanilla (non-proximal) solver, which is what
`tier2_hfh/local_global.py` actually runs, is **not** covered by this theorem
at all, because its block steps are *exact* minimizers with no proximal
regularization, and the theorem's whole machinery (the subgradient bound in
its Lemma 5(iii)) needs a genuine `1/lambda` term to control the "next
iterate minus previous iterate" direction. So: KL buys rigor about
*settling*, not about *where*. If the goal was "prove the solver reaches the
global optimum," this line of theory does not deliver that, and nothing
found here suggests reformulating the algorithm would make it deliver that
either — the fixed-point structure of a nonconvex alternating scheme is the
obstacle, not a missing proof technique.

## The theorem, cited precisely

**Attouch, H., Bolte, J., Redont, P., Soubeyran, A. (2010).** "Proximal
Alternating Minimization and Projection Methods for Nonconvex Problems: An
Approach Based on the Kurdyka–Łojasiewicz Inequality." *Mathematics of
Operations Research* 35(2), 438–457. Preprint: arXiv:0801.1780v3 [math.OC].
(Referred to below as **ABRS**.)

This is the better match for this solver than either of the two candidates
named in the task brief:

- **Attouch–Bolte–Svaiter (2013)**, *Math. Programming* 137, 91–129, is an
  *abstract* descent-method framework (their conditions (H1)/(H2)/(H3) plus
  KL) that does not commit to an alternating-block structure at all; ABRS's
  Theorem 8/9 below is a concrete instance of exactly that abstract pattern,
  specialized to alternating proximal minimization, so citing ABRS directly
  is more precise for this algorithm than re-deriving it from the abstract
  version.
- **Bolte–Sabach–Teboulle (2014)** PALM, *Math. Programming* 146, 459–494,
  handles a coupling term `Q` that is merely `C^1` with Lipschitz gradient by
  taking a single **linearized (gradient) step** at each block, proximal-
  regularized. Our two block steps are **not** gradient steps — they are
  *exact* minimizers of the (proximal-regularized) block subproblem, because
  the local step is a closed-form circle projection and the global step
  exploits the exactly-diagonal normal operator. ABRS's algorithm is stated
  for exactly this case (exact proximal block minimization, no linearization
  needed), so it is the tighter fit. PALM would still apply (its hypotheses
  are implied whenever exact minimization is available) but citing it would
  require introducing a gradient-Lipschitz constant this problem doesn't need.

### ABRS's setup, exactly as stated

Let `L(x,y) = f(x) + Q(x,y) + g(y)`, `f : R^n -> R∪{+inf}`, `g : R^m ->
R∪{+inf}` proper lower semicontinuous, `Q : R^n x R^m -> R`.

**Hypothesis (H):** `Q` is `C^1` and `grad Q` is Lipschitz continuous on
bounded subsets of `R^n x R^m`.

**Algorithm (their eq. 5, 6), iteration index `t`, stepsizes `lambda_t,
mu_t > 0`:**

```
x_{t+1} in argmin_u { L(u, y_t) + (1/(2*lambda_t)) ||u - x_t||^2 }
y_{t+1} in argmin_v { L(x_{t+1}, v) + (1/(2*mu_t)) ||v - y_t||^2 }
```

**Hypothesis (H1):** `inf L > -infty`; `L(., y_0)` proper; there exist
`0 < r_- < r_+` with `lambda_t, mu_t in (r_-, r_+)` for every `t` (i.e. a
FIXED range of stepsizes, bounded away from both 0 and infinity).

**Theorem 8 (pre-convergence)** and **Theorem 9 (convergence).** Assume `L`
satisfies (H), (H1), and has the Kurdyka–Łojasiewicz property at every point
of `dom(f)` (more precisely, at every point of `dom(L) = dom(f) x dom(g)`,
which is what the proof actually uses — see "Hypothesis 4" below). Then
either `||(x_t, y_t)|| -> infinity`, or

```
sum_{t=1}^infty ( ||x_{t+1}-x_t|| + ||y_{t+1}-y_t|| ) < infty
```

and, as a consequence, `(x_t, y_t)` converges to a single critical point of
`L`.

This is their Theorem 9, quoted to the letter (notation renamed `k -> t` here
to free `k` for the pupil/view index the rest of this repo uses it for).

## Mapping the theorem onto this solver

Write `x = (u, z)`. `local_global.py`'s docstring already gives the energy:

```
E(u,z) = sum_k rho_k ||P_k u - z_k||^2
```

and the two constraint sets:

```
f(u) = indicator{ u : |u_j| = 1 for every panel pixel j }
g(z) = indicator{ z : |z_{k,s}| = a_{k,s} for every (view,pixel) (k,s) }
```

with `Q(u,z) = E(u,z)`, so `L(u,z) = f(u) + Q(u,z) + g(z)` matches ABRS's `L`
exactly, complex coordinates read as `R^2` per entry (see "semialgebraicity"
below for why this is not a shortcut that hides anything).

**Crucial relabelling.** ABRS updates their `x`-block first, then `y`. This
solver's local step updates `z` first (using the current `u`), then the
global+panel step updates `u` (using the new `z`). So the correct
identification is

```
x_theirs = z_ours      f_theirs = g_ours   (the amplitude-circle indicator)
y_theirs = u_ours       g_theirs = f_ours   (the unit-modulus indicator)
lambda_theirs -> the z-block's proximal parameter
mu_theirs     -> the u-block's proximal parameter
```

not the more tempting `x=u, y=z`. Getting this backwards would silently
apply the wrong hypothesis subscript to the wrong block below.

**The existing solver as `eta=gamma=0`.** `local_global.solve`'s "global
step, then panel step" pair, composed, already computes exactly
`argmin_{|u|=1} Q(u,z)` — because `H = sum_k rho_k P_k^H P_k` is diagonal,
`d_j|u_j|^2` is constant on `|u_j|=1` (Established Fact 2 in the task
brief), so normalizing the unconstrained least-squares solution *is* the
constrained minimizer. So the existing solver is already a **non-proximal**
two-block exact alternating minimization of this same `L`. The proximal
variant in `tier2_hfh/proximal.py` adds a term to each block's subproblem:

```
z_{t+1} in argmin_v { g(v)   + Q(u_t, v)     + (1/(2*lambda)) ||v-z_t||^2 }
u_{t+1} in argmin_w { f(w)   + Q(w, z_{t+1}) + (1/(2*mu))     ||w-u_t||^2 }
```

matching ABRS's (5,6) under the relabelling above, with `lambda, mu` held
**constant** across iterations (satisfying (H1)'s stepsize-boundedness with
`r_- = r_+ =` that constant).

## Per-hypothesis verdict

| Hypothesis | Verdict | Why |
|---|---|---|
| `f, g` proper l.s.c. | **Satisfied** | Both are indicators of nonempty (every circle/point contains at least one point), closed (a circle or a point is closed) sets, so proper and l.s.c. by the standard fact that the indicator of a nonempty closed set is proper l.s.c. |
| `Q` is `C^1`, `grad Q` Lipschitz on bounded sets | **Satisfied, more strongly than needed** | `Q` is an honest real quadratic form in the joint real coordinates (finite sum of `rho_k \|P_k u - z_k\|^2`, `P_k` fixed finite linear maps) — `C^infty`, and its gradient is **globally** affine, hence globally Lipschitz, not merely on bounded sets. |
| `inf L > -infty` | **Satisfied** | `L >= 0` everywhere (sum of an indicator, which is `0` or `+infty`, and a squared norm), so `inf L >= 0`. |
| `L(., y_0)` proper | **Satisfied** | `dom(f)` (equivalently `dom(g)`, whichever block is "first") is nonempty and `Q` is finite everywhere, so `L(., y_0)` is finite wherever the indicator is. |
| Stepsizes `lambda_t, mu_t` in a fixed `(r_-, r_+)` | **Satisfied by construction, for the proximal variant only** | `proximal_solve` uses fixed `eta, gamma` for every iteration. **Not satisfied by the existing (`eta=gamma=0`) solver** — `lambda = 1/(2*eta)` diverges to `+infty` there, which is exactly the boundary case ABRS's own intro flags ("for such parameters the algorithm is very close to a coordinate descent method") but does not cover with this theorem; the plain exact-minimization case needs a different (and, for nonconvex non-strictly-convex blocks, generally unavailable) argument, since ties in the block argmin are then possible and the subgradient bound in their Lemma 5(iii) is not controlled. |
| Bounded iterates (needed to rule out `\|\|(x_t,y_t)\|\| -> infty`) | **Satisfied unconditionally, and more strongly than usually available** | Every iterate lies EXACTLY on `dom(f) x dom(g)`, a fixed compact set (a product of unit circles times a product of fixed-radius circles/points), for every `t >= 0`, by construction of the closed-form projections. The "either diverges or is summable" dichotomy in Theorem 9 therefore collapses to the summable branch unconditionally — no separate coercivity or a-priori bound argument is needed, which is unusually clean for this kind of theorem. |
| KL property of `L` at every point of `dom(L)` | **Satisfied, via semialgebraicity — see below** | Not computed directly (no KL exponent `theta` derived); established by citing the general theorem that proper l.s.c. semialgebraic functions have the KL property everywhere on their domain. |

## Semialgebraicity, checked rather than assumed

A function is (in the sense this literature uses, e.g. ABRS section 4.3,
Bolte–Daniilidis–Lewis–Shiota) semialgebraic if its graph is a semialgebraic
subset of `R^n x R`. The needed general theorem:

**Bolte, J., Daniilidis, A., Lewis, A.S., Shiota, M. (2007).** "Clarke
subgradients of stratifiable functions." *SIAM J. Optimization* 18(2),
556–572 (arXiv:math/0601530). Consequence used here: a proper l.s.c.
semialgebraic (more generally, definable in an o-minimal structure) function
satisfies the Kurdyka–Łojasiewicz inequality at **every** point of its
domain, with no need to name a specific exponent `theta`.

Checked component by component, all in the real coordinates `(u_j.re,
u_j.im)`, `(z_{k,s}.re, z_{k,s}.im)`:

- `Q(u,z) = sum_k rho_k \|P_k u - z_k\|^2` is, once every complex
  multiplication by the fixed matrix entries of `P_k` is expanded into real
  matrix multiplication, an honest **real polynomial** (degree 2) in
  `(u.re, u.im, z.re, z.im)`. Polynomials are semialgebraic.
- `f(u) = indicator{u : forall j, |u_j|=1}` — in real coordinates this is
  the indicator of `{(x,y) in R^N x R^N : x_j^2 + y_j^2 = 1 for all j}`, a
  finite intersection of real quadric hypersurfaces (a product of `N` unit
  circles, the `N`-torus). Finite intersections of semialgebraic sets are
  semialgebraic; a product of circles is compact and nonempty.
- `g(z)` — same argument, a finite product of circles `|z_{k,s}| = a_{k,s}`.
- **The `a=0` corner, checked explicitly (this was flagged as worth
  checking, and it is real, if narrow):** when `a_{k,s}=0`, the constraint
  `|z_{k,s}|=0` degenerates from a circle to the single point `{0}` (a
  0-dimensional variety, defined by `z.re=0 ∧ z.im=0` — still two
  polynomial equalities, still semialgebraic, still closed and nonempty).
  The indicator of a point is still proper l.s.c.; nothing in the KL /
  semialgebraicity argument requires the constraint set to be
  1-dimensional. The only place this corner is *behaviorally* special is
  in the **non-proximal** projection `project_onto_amplitude` (already
  documented and tested in `local_global.py`/`test_local_global.py`): at
  `a=0` there is no phase-tie to break (the minimizer `z=0` is forced,
  unique, no `zero_policy` choice needed), which if anything makes this
  corner *simpler* for the constrained-minimizer argument, not harder — a
  point is easier to project onto than a circle. In practice `a_{k,s}=0`
  essentially never occurs for the synthetic targets used here (`a =
  |P_k u_true|` for a random continuous `u_true` is a measure-zero event
  away from zero), so this is a corner that is real but not load-bearing
  for the numbers below.
- A finite sum of semialgebraic (extended-real-valued, in this literature's
  convention) functions is semialgebraic — this is exactly how ABRS's own
  worked examples (compressive sensing, rank reduction) build `L`, and is
  the standard closure property cited via Bolte–Daniilidis–Lewis–Shiota /
  Kurdyka (1998). `L = f + Q + g` is therefore semialgebraic, proper (`Q`
  finite everywhere, `f,g` proper), and l.s.c. (sum of l.s.c. functions
  bounded below on the relevant domain, `Q` continuous).

**Conclusion:** `L` is proper, l.s.c., and semialgebraic, hence has the KL
property at every point of its domain, with no need to compute an exponent.
This closes the last open hypothesis in the table above.

## Sympy re-derivation of both proximal coefficients

The task is explicit that the candidate closed forms are not to be trusted
without a fresh derivation. Both were re-derived independently (Lagrange
multipliers on the real-coordinate Lagrangian, then comparing the two
stationary branches by their objective value — the same method
`tests/test_local_global.py` already used for the non-proximal case, now
reused for the proximal one in `tests/test_proximal.py`), **not just
restated**:

**u-block (global step).** Minimizing
`d*|u|^2 - 2*Re(conj(b)*u) + eta*|u-u_old|^2` subject to `|u|=1` has exactly
two Lagrange stationary points, `u = +-(b+eta*u_old)/|b+eta*u_old|`. The
`+` branch is strictly smaller by exactly `4*|b+eta*u_old|` — and, checked
directly in the sympy script, **this difference does not contain `d` at
all**: `d` only ever appears in a term that is identical on both branches
(`d*|u|^2 = d` on the constraint circle), so it cancels. This is
Established Fact 2 surviving the addition of a proximal term, now verified
rather than assumed to survive it. **Matches the candidate exactly**:
`u_{t+1,j} = (b_j + eta*u_{t,j}) / |b_j + eta*u_{t,j}|`, no missing factor,
no sign error.

**z-block (local step).** Minimizing `|z-y|^2 + gamma*|z-z_old|^2` subject
to `|z|=a` has exactly two stationary points, `z = +-a*(y+gamma*z_old) /
|y+gamma*z_old|`, the `+` branch strictly smaller by `4*a*|y+gamma*z_old|`.
**Matches the candidate exactly**: `z_new = a*(y+gamma*z_old) /
|y+gamma*z_old|`.

**The one correction to the candidates, found by the derivation rather than
assumed: `gamma` needs a `1/rho_k`.** The task's candidate formula for the
local step does not mention the per-view weight `rho_k` at all. Re-deriving
with weights present shows the z-block subproblem is
`rho_k*|P_k u - z_k|^2 + (1/(2*lambda))*|z_k-z_{old,k}|^2`, and — a genuine
but elementary rescaling, checked symbolically in
`test_the_local_step_weight_scaling_is_a_positive_rescaling_not_a_new_formula`
— minimizing `rho*A(z) + C*B(z)` over the same `z` has the same argmin as
minimizing `A(z) + (C/rho)*B(z)` for `rho>0`. So with per-view weights, the
correct coefficient multiplying `z_old` in the shifted target is
`gamma/rho_k`, not `gamma`; with the solver's default `rho_k=1` for every
view, this reduces to the single shared constant the candidate names.
`tier2_hfh/proximal.py`'s `proximal_local_step` implements the `1/rho_k`
version; `tests/test_proximal.py::test_proximal_local_step_with_weights_divides_gamma_by_rho`
checks the implementation matches it, not just the symbolic fact in
isolation.

The u-block coefficient `eta` needed **no** correction — it is a single
scalar shared by every pixel (no `1/d_j` scaling), exactly because `d_j`
drops out entirely, as shown above.

## Numerical evidence

All runs: CPU only, `torch.float64`, panel 12, window 4, `n_views=3`
(9 actual pupil positions), `eta=gamma=0.05`, 12 fixed random seeds
(`0..11`), reproduced in `tests/test_proximal.py`.

**1. Sufficient decrease holds, with exactly the predicted constant, on
every iteration of every seed.** The claimed inequality
`L(x_t)-L(x_{t+1}) >= c*||x_{t+1}-x_t||^2` with `c = min(eta,gamma) = 0.05`
was checked against `coupling_energy` (the actual `Q(u,z)`, not the
amplitude-only diagnostic `local_global.energy`) at every iteration across
12 seeds, 150 iterations each (1800 checks). The worst (smallest) slack
`(L_t-L_{t+1}) - c*step_t` observed across all of them was **positive** in
every run (as small as `~1e-11` late in a run, where both the drop and the
step size are themselves near zero — consistent with settling near a fixed
point, not with the inequality being violated). This holds
**unconditionally**, with no KL property invoked: it is a three-point
minimizer-comparison argument (`KLSufficientDecrease.lean`'s
`sufficient_decrease`, proved for an arbitrary metric space and objective,
using only that each step is an exact minimizer of its own proximal
subproblem — which `KLGlobalStep.lean`/`KLLocalStep.lean` establish for
this solver's two blocks).

**2. `sum_t ||x_{t+1}-x_t||^2` stays well inside the telescoping bound, and
the tail settles.** `c * sum_t step_t <= L_0 - L_inf <= L_0` (since `L>=0`)
held on every seed with comfortable margin (observed sums were roughly
2-3% of the bound). More informatively: the mean squared step over the
**last** 10 iterations of a 150-iteration run was, on every seed, smaller
than the mean over the **first** 10 — often by several orders of magnitude
— i.e. the iterate is genuinely settling, not merely bounded on average.
This is the numerical face of "the whole sequence converges," not just "the
step size doesn't blow up."

**3. The proximal variant changes how the solver approaches a fixed point,
not which one it reaches.** Vanilla `solve` and `proximal_solve`, started
from the identical random panel on the identical realisable target, 150
iterations, `eta=gamma=0.05`: on all 12 seeds tried, the final amplitude
residuals of the two variants agreed to within a few percent of each other
(most seeds agreed to 3-4 significant figures; the largest disagreement
observed was about 4%). **Neither variant reached the global optimum
(near-zero residual) on any of these 12 seeds** — both settled at
comparably-sized non-global residuals (`O(15)` to `O(30)` in this energy's
units). This is consistent with, not contrary to, the established fact
elsewhere in this repo that only a minority of random starts reach the
global solution; it does mean this particular batch of 12 seeds is
uninformative about whether the proximal variant is *ever* the difference
between success and failure (established elsewhere at 1/12 vs 7/12 under
different starting conditions) — but it is informative about the more
basic question asked here: proximal regularization did **not** show any
systematic pull toward or away from the achieved fixed points. No surprise
was observed that would need the "independent, careful check" the task
asked for in that event.

## Lean formalization

New files, all under `lean/holopixel/Holopixel/`, prefixed `KL`, verified
with `lake build Holopixel.<name>` (each compiles standalone against the
already-built Mathlib cache) and with a full `lake build` from the project
root (8272 jobs, all succeed) — i.e. these files are picked up by the
existing project's default build target automatically (Lake globs the
`Holopixel/` directory for the `Holopixel` library), **without editing
`Holopixel.lean` or `lakefile.toml`**, per the constraint not to touch
existing Lean files.

| File | What is proved | `sorry`? |
|---|---|---|
| `KLProjection.lean` | `circleTarget_isMinimizer`: for `a>0`, `y != 0`, the point `a*y/\|y\|` is the **unique** minimizer of `z -> \|z-y\|^2` over `\|z\|=a` — the same fact `test_local_global.py` proves in sympy for `a=1`, now machine-checked for general `a`, by a direct 2-D Cauchy-Schwarz / Lagrange-identity argument (`lagrange_identity_2d`, a one-line `ring` identity) rather than by solving the Lagrange system symbolically. | None |
| `KLGlobalStep.lean` | `globalStep_isMinimizer`: the u-block proximal closed form `(b+eta*u_old)/\|b+eta*u_old\|` is the unique minimizer of the full proximal-regularized u-block objective (including the `d*\|u\|^2` term) over `\|u\|=1`, via an UNCONDITIONAL algebraic identity (`globalObjective_eq_general`, pure `ring`, no hypothesis on `u`) reducing it to `KLProjection`. | None |
| `KLLocalStep.lean` | `localStep_isMinimizer`: the z-block closed form `a*(y+gamma*z_old)/\|y+gamma*z_old\|` is the unique minimizer of the z-block proximal objective over `\|z\|=a`, by the same route. | None |
| `KLSufficientDecrease.lean` | `sufficient_decrease`: the abstract three-point inequality, for an arbitrary `PseudoMetricSpace` and objective — sufficient decrease needs only "exact minimizer," not KL. `telescoped_step_bound`: summing it over a run gives the telescoping bound used in the numerical evidence above. | None |

**What is explicitly NOT formalized, and why:** the Kurdyka–Łojasiewicz
property itself, the semialgebraicity-implies-KL theorem
(Bolte–Daniilidis–Lewis–Shiota), and the full inductive convergence proof of
ABRS's Theorem 8/9 (the `varphi`-Lyapunov argument over the whole iterate
sequence). These are substantial, general results about o-minimal /
semialgebraic geometry and are not attempted here — Mathlib does not
currently carry this theory at the depth needed (no KL inequality, no
o-minimal/semialgebraic stratification machinery at this level), and
formalizing it from scratch is a research-grade undertaking on its own, well
outside this task's scope. Rather than leave a `sorry` standing in for it,
these results are cited (with exact paper and theorem numbers above) and
used as an external input, exactly as the task's instructions allow
("if a goal will not close, leave it clearly labelled as unproved and say
so" — here, the honest label is: not attempted in Lean at all, established
by citation instead).

## What is proved, what is assumed, what is unknown

**Proved (Lean, no `sorry`):**
- Both proximal closed forms are the exact, unique minimizers of their
  respective proximal-regularized block subproblems (`KLGlobalStep.lean`,
  `KLLocalStep.lean`, built on `KLProjection.lean`).
- Sufficient decrease, `L(x_t)-L(x_{t+1}) >= c*\|\|x_{t+1}-x_t\|\|^2`, holds
  for ANY algorithm whose steps are exact minimizers of a proximal
  subproblem — no KL property needed (`KLSufficientDecrease.lean`).

**Proved (sympy, checked, not merely asserted):**
- Both closed forms solve their Lagrange stationarity conditions and are
  the strictly smaller of the two branches (`tests/test_proximal.py`,
  mirroring `tests/test_local_global.py`'s existing method).
- The `d`-independence of the u-block minimizer, and the `1/rho_k` scaling
  of the z-block's proximal coefficient under per-view weights.

**Established by citation (standard results, not re-proved here):**
- Proper l.s.c. semialgebraic functions satisfy the KL property at every
  point of their domain (Bolte–Daniilidis–Lewis–Shiota 2007).
- ABRS's Theorem 8/9: under (H), (H1), and KL of `L` on `dom(L)`, a bounded
  alternating-proximal-minimization sequence converges (as a whole
  sequence, with summable steps) to a single critical point.

**Checked numerically, not proved:**
- The sufficient-decrease constant `c=min(eta,gamma)` is actually attained
  (not just an upper bound) on the geometries tried.
- The telescoping bound holds with comfortable margin and the step size
  visibly settles.
- The proximal variant does not measurably change which fixed point is
  reached, on the 12 seeds tried.

**Genuinely unknown / out of scope:**
- Whether the proximal variant (or any variant of this algorithm) reaches
  the *global* optimum more often than the plain solver, in general — the
  KL machinery is silent on this by construction (it is a "does it settle"
  theorem, not a "does it settle *well*" theorem), and the 12-seed sample
  here happened to land entirely in the "neither variant solves it" regime,
  so it is not even informative about the narrower question of whether
  proximal regularization is *ever* the difference between success and
  failure (a question the task's own background section already answers
  differently, at 1/12 vs 7/12, under different starting conditions not
  reproduced here).
- The KL exponent `theta` for this `L` (needed for a convergence-*rate*
  statement via ABRS's Theorem 11) was not computed; only existence of the
  KL property was established.

## Is this practically useful, or a rigor exercise?

Mostly the latter, and it is worth being blunt about that rather than
oversell it. The theorem answers a question that was not really in doubt
once the proximal terms were added (of course an exact-minimizer alternating
scheme with a genuine proximal pull settles down — the pull is *designed*
to make consecutive iterates close together) and does not touch the
question that motivated the investigation (does the solver find the global
optimum, or a useful one). Where it does earn its keep: it converts "we
believe the proximal iterate creeps to something and probably stops" into
"the whole sequence provably converges to a single critical point, with a
summable step sequence, under hypotheses we have checked one by one rather
than waved at" — which is a real, checkable, non-circular statement, backed
by machine-checked algebra for its two least-trivial pieces (the closed
forms and sufficient decrease). It is not, however, a step toward better
recovery rates, and nothing in this investigation suggests one exists along
this route.
