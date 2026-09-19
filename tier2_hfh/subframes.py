"""
The local step of a local/global solver, extended to M INCOHERENT subframes.

A holographic display can show M different phase patterns per visual frame,
fast enough that the eye integrates over all of them. The subframes have no
fixed phase relation to one another (different exposures of a device that
free-runs its phase, or deliberately randomised to break speckle), so the eye
sees the mean of their INTENSITIES, not the intensity of their mean:

    I_pred = sum_m alpha_m |P u_m|^2,   sum_m alpha_m = 1,   alpha_m >= 0

`test_incoherent_sum_differs_from_coherent_sum` pins this down against the
alternative |sum_m P u_m|^2 (mutually coherent subframes), which this module
does NOT implement -- the two differ by orders of magnitude, so picking the
wrong one is not a subtle bug.

THE LOCAL STEP

In a local/global solver, the local step at each sample fixes the phases at
their current values and asks only how much amplitude each subframe should
carry. With phases fixed, `||a_m * exp(i*arg(y_m)) - y_m||^2 = (a_m - r_m)^2`
with `r_m = |y_m|`, so the M-subframe local step is a projection of the
amplitude VECTOR `r = (r_1, ..., r_M)` onto an ellipsoid:

    minimise_a   sum_m (a_m - r_m)^2
    subject to   sum_m alpha_m a_m^2 = I_target,   a_m >= 0

Fix the phases first, fix the allocation second: the two problems are
independent because the objective and the constraint above depend on `a`
only through its magnitude, never its phase.

THE DERIVATION (verified with sympy against a symbolic M=3 Lagrangian in
tests/test_subframes.py::test_sympy_confirms_the_stationarity_conditions and
test_sympy_confirms_monotonicity_of_the_multiplier_constraint -- the
comments below report what that check found, not an unchecked derivation)

Lagrangian stationarity, for FIXED lambda:

    2(a_m - r_m) + 2 lambda alpha_m a_m = 0   =>   a_m = r_m / (1 + lambda alpha_m)

For UNIFORM alpha_m = 1/M, `1 + lambda/M` is the same number for every m, so
`a` is a single scalar multiple of `r`: the projection is RADIAL, it keeps
the direction of `r` and rescales it onto the sphere of radius
sqrt(M * I_target):

    a = r * sqrt(M * I_target) / ||r||

Because r_m = |y_m| >= 0 always, this a is elementwise >= 0 automatically:
non-negativity is never a binding constraint in the uniform case, sympy
confirms the KKT stationarity point already lands in the non-negative
orthant, so there is nothing to clamp.

For NON-UNIFORM alpha, `a_m = r_m / (1 + lambda alpha_m)` no longer collapses
to one scalar, and lambda has to be found from the scalar constraint

    g(lambda) := sum_m alpha_m r_m^2 / (1 + lambda alpha_m)^2 = I_target

sympy confirms dg/dlambda = -2 sum_m alpha_m^2 r_m^2 / (1 + lambda alpha_m)^3,
which is strictly negative throughout the domain `1 + lambda alpha_m > 0 for
all m` (the domain where every a_m keeps the sign of r_m, hence stays >= 0),
i.e. for `lambda > -1 / max_m alpha_m`. On that domain g is continuous,
strictly decreasing, g(lambda) -> 0 as lambda -> infinity, and (assuming the
subframe with the largest alpha has a nonzero r there, see NON-NEGATIVITY
below) g(lambda) -> infinity as lambda approaches the domain's lower edge, so
exactly one root exists and it is found here by bisection with a bracket
that is expanded geometrically towards that edge.

NON-NEGATIVITY. Since r_m >= 0 and the root lambda satisfies
`1 + lambda alpha_m > 0` for every m by construction of the bracket, every
`a_m = r_m / (1 + lambda alpha_m)` is a non-negative number divided by a
positive one: it cannot come out negative, so the closed-form/root-find
answer already respects `a_m >= 0` without clamping. The one way this can
fail is the corner case where the subframe with the single largest alpha has
r_m EXACTLY zero while the required I_target exceeds what the rest of the
vector can supply arbitrarily close to the domain edge -- there the bracket
search cannot find a finite bracket and `project_amplitudes_weighted` raises
`RuntimeError` rather than return something wrong. This did not occur for
any of the (many, random, including some exactly-zero components) cases this
module was tested against; it is a theoretical edge documented for honesty.

WHY EQUAL ALLOCATION IS NOT OPTIMAL

Solving M independent single-frame holograms, or otherwise splitting the
target evenly (`a_m^2 = I_target / M` for every m regardless of `r`), ignores
the phases the previous outer iteration already chose. `test_...` below
shows a lopsided `r` for which the radial projection's cost is smaller than
equal-split's by several orders of magnitude: cooperating subframes beat
independent ones because the projection can leave a subframe that is already
close to right almost untouched and let the others compensate.

THE ||r|| = 0 DEGENERACY

If every subframe currently has zero amplitude at a sample, `r`'s direction
is undefined, and this is not a numerical inconvenience to paper over with an
epsilon: at r = 0 the objective `sum_m (a_m - 0)^2 = sum_m a_m^2` equals
`M * I_target` for EVERY point on the constraint sphere (uniform case), so
every feasible `a` is equally optimal and the projection is genuinely
non-unique. The policy adopted here is the symmetric point, equal amplitude
on every subframe, `a_m = sqrt(I_target)` for all m: one of the infinitely
many optimal answers, chosen because it favours no subframe. See
`project_amplitudes_uniform`.
"""

import torch


def project_amplitudes_uniform(r, target_intensity):
    """
    Radial projection for uniform subframe weights alpha_m = 1/M.

    `r`: (M, *shape) real, non-negative current amplitudes per subframe.
    `target_intensity`: (*shape) real, non-negative target intensity.
    Returns `a`: (M, *shape), minimising sum_m (a_m - r_m)^2 subject to
    (1/M) sum_m a_m^2 = target_intensity and a_m >= 0.

    a = r * sqrt(M * target_intensity) / ||r||, radial because uniform alpha
    makes every a_m the same scalar multiple of r_m (see module docstring).
    At samples where ||r|| == 0 the direction is genuinely undefined (every
    feasible point costs the same); the policy is the symmetric point,
    a_m = sqrt(target_intensity) for every m.
    """
    if torch.any(r < 0):
        raise ValueError("amplitudes must be non-negative")
    if torch.any(target_intensity < 0):
        raise ValueError("target intensity must be non-negative")

    m = r.shape[0]
    norm = r.norm(dim=0, keepdim=True)
    is_zero = norm == 0

    symmetric_direction = torch.full_like(r, float(m) ** -0.5)
    direction = torch.where(is_zero.expand_as(r), symmetric_direction, r)
    # The symmetric direction has norm exactly 1 (sum of M copies of 1/M),
    # so substituting 1 for the norm there is exact, not an approximation.
    safe_norm = torch.where(is_zero, torch.ones_like(norm), norm)

    scale = torch.sqrt(m * target_intensity).unsqueeze(0)
    return direction * scale / safe_norm


def _bracket_and_bisect(g, target, lo_bound, tol, max_expand, max_bisect):
    """
    The unique root of `g(lambda) == target` for a strictly decreasing `g`,
    on the open domain `(lo_bound, infinity)`.

    `lo_bound` is a python float shared by every sample (it depends only on
    alpha, which does not vary across samples); `target` fixes the tensor
    shape and dtype/device of the search. Both the upward expansion (`hi`)
    and the downward one (`lo`, approaching `lo_bound` geometrically since
    that boundary is where a component's denominator vanishes) are done
    elementwise with masks, because different samples need different
    directions and different numbers of doublings.
    """
    zero = torch.zeros_like(target)
    g0 = g(zero)
    need_hi = g0 > target
    need_lo = g0 < target

    lo = torch.where(need_lo, torch.full_like(target, lo_bound) * 0.5, zero)
    hi = torch.where(need_hi, torch.ones_like(target), zero)
    lo_frac = torch.where(need_lo, torch.full_like(target, 0.5), torch.zeros_like(target))

    for _ in range(max_expand):
        still_hi = need_hi & (g(hi) > target)
        still_lo = need_lo & (g(lo) < target)
        if not (still_hi.any() or still_lo.any()):
            break
        hi = torch.where(still_hi, hi * 2.0, hi)
        lo_frac = torch.where(still_lo, (lo_frac + 1.0) / 2.0, lo_frac)
        lo = torch.where(still_lo, lo_bound * lo_frac, lo)

    if bool((need_lo & (g(lo) < target)).any()):
        raise RuntimeError(
            "could not bracket the multiplier root: target intensity is not "
            "reachable while keeping every subframe amplitude non-negative "
            "(a zero-amplitude subframe would need to carry the largest "
            "weight alpha, and no finite multiplier supplies enough)")

    for _ in range(max_bisect):
        mid = (lo + hi) / 2
        go_right = g(mid) > target  # g decreasing: root is above mid
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)

    lam = (lo + hi) / 2
    residual = (g(lam) - target).abs()
    scale = target.abs().clamp_min(1.0)
    if not bool(torch.all(residual < tol * scale)):
        raise RuntimeError(
            f"multiplier bisection failed to converge: max residual "
            f"{float(residual.max()):.3e}")
    return lam


def project_amplitudes_weighted(r, alpha, target_intensity, tol=1e-10,
                                 max_expand=200, max_bisect=200):
    """
    Projection for arbitrary subframe weights alpha_m >= 0, sum_m alpha_m = 1.

    `r`: (M, *shape) non-negative current amplitudes. `alpha`: (M,) weights.
    `target_intensity`: (*shape) non-negative target.

    a_m = r_m / (1 + lambda alpha_m), lambda the unique root on
    `(-1/max(alpha), infinity)` of
    `g(lambda) = sum_m alpha_m r_m^2 / (1 + lambda alpha_m)^2 = target_intensity`
    (monotonicity proved symbolically, see module docstring and
    `tests/test_subframes.py::test_sympy_confirms_monotonicity_of_the_multiplier_constraint`).
    Raises `RuntimeError` rather than returning a wrong answer if the root
    cannot be bracketed or the bisection has not converged to `tol`.
    """
    if torch.any(r < 0):
        raise ValueError("amplitudes must be non-negative")
    if torch.any(alpha < 0):
        raise ValueError("weights must be non-negative")
    if abs(float(alpha.sum()) - 1.0) > 1e-10:
        raise ValueError(f"weights must sum to 1, got {float(alpha.sum())}")
    if torch.any(target_intensity < 0):
        raise ValueError("target intensity must be non-negative")

    shape = (r.shape[0],) + (1,) * (r.ndim - 1)
    alpha_b = alpha.to(dtype=r.dtype, device=r.device).reshape(shape)
    lo_bound = -1.0 / float(alpha.max())

    def g(lam):
        denom = 1.0 + lam.unsqueeze(0) * alpha_b
        return (alpha_b * r ** 2 / denom ** 2).sum(dim=0)

    lam = _bracket_and_bisect(g, target_intensity, lo_bound, tol, max_expand, max_bisect)
    return r / (1.0 + lam.unsqueeze(0) * alpha_b)


def project_amplitudes(r, target_intensity, alpha=None, **kwargs):
    """
    Dispatch to the closed form when `alpha` is uniform (or omitted), and to
    the root-find otherwise. Both branches solve the same problem; kept as
    two functions because the uniform case has an exact closed form worth
    using rather than a root-find that would only ever converge to it.
    """
    m = r.shape[0]
    if alpha is None or bool(torch.allclose(alpha, alpha.new_full((m,), 1.0 / m))):
        return project_amplitudes_uniform(r, target_intensity)
    return project_amplitudes_weighted(r, alpha, target_intensity, **kwargs)
