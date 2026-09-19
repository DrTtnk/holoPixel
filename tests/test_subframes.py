"""
The multi-subframe local projection: allocating a target intensity among M
mutually INCOHERENT subframes at fixed phase.

Every closed form claimed in `tier2_hfh/subframes.py` is checked here twice:
once symbolically with sympy (so the algebra is verified before it is
trusted, not after), and once numerically against `scipy.optimize.minimize`
solving the same constrained problem with no closed form assumed at all.
"""

import math

import numpy as np
import pytest
import scipy.optimize
import sympy as sp
import torch

from tier2_hfh.operators import ViewOperators
from tier2_hfh.optimise import Geometry
from tier2_hfh.subframes import (
    project_amplitudes,
    project_amplitudes_uniform,
    project_amplitudes_weighted,
)


# ---------------------------------------------------------------------------
# The derivation, verified symbolically before any of it is implemented.
# ---------------------------------------------------------------------------

def test_sympy_confirms_the_stationarity_conditions():
    """
    Lagrangian stationarity for `minimise sum (a_m-r_m)^2 s.t. sum alpha_m
    a_m^2 = I` must be `a_m = r_m / (1 + lambda alpha_m)`, for both uniform
    and non-uniform alpha. This is the equation everything else in the
    module is built from.
    """
    a1, a2, a3, r1, r2, r3, lam, I = sp.symbols("a1 a2 a3 r1 r2 r3 lam I", real=True)
    a, r = [a1, a2, a3], [r1, r2, r3]

    al1, al2, al3 = sp.symbols("al1 al2 al3", positive=True)
    alphas = [al1, al2, al3]
    L = sum((ai - ri) ** 2 for ai, ri in zip(a, r)) \
        + lam * (sum(al * ai ** 2 for al, ai in zip(alphas, a)) - I)
    stationary = [sp.diff(L, ai) for ai in a]
    solution = sp.solve(stationary, a, dict=True)[0]
    for ai, ri, al in zip(a, r, alphas):
        assert sp.simplify(solution[ai] - ri / (1 + lam * al)) == 0


def test_sympy_confirms_the_uniform_case_is_radial():
    """
    With alpha_m = 1/M every stationary a_m is the SAME scalar multiple of
    r_m -- the defining property of a radial projection -- and that scalar
    is exactly `sqrt(M * I) / ||r||`, which also satisfies the constraint
    exactly (zero residual, checked symbolically, not just to a tolerance).
    """
    M = 3
    r1, r2, r3, I = sp.symbols("r1 r2 r3 I", positive=True)
    r = [r1, r2, r3]
    norm_r = sp.sqrt(sum(ri ** 2 for ri in r))
    candidate = [ri * sp.sqrt(M * I) / norm_r for ri in r]

    implied_lambda = [sp.simplify((ri / ci - 1) * M) for ri, ci in zip(r, candidate)]
    assert sp.simplify(implied_lambda[0] - implied_lambda[1]) == 0
    assert sp.simplify(implied_lambda[0] - implied_lambda[2]) == 0

    constraint_residual = sp.simplify(
        sp.Rational(1, M) * sum(ci ** 2 for ci in candidate) - I)
    assert constraint_residual == 0


def test_sympy_confirms_monotonicity_of_the_multiplier_constraint():
    """
    g(lambda) = sum_m alpha_m r_m^2 / (1+lambda alpha_m)^2 must be strictly
    decreasing wherever every (1+lambda alpha_m) > 0, or the root-find has no
    right to assume a unique root. dg/dlambda simplifies to
    `-2 sum_m alpha_m^2 r_m^2 / (1+lambda alpha_m)^3`, which is negative
    there since every term is a positive number divided by a positive cube.
    """
    r1, r2, r3, lam = sp.symbols("r1 r2 r3 lam", real=True)
    al1, al2, al3 = sp.symbols("al1 al2 al3", positive=True)
    r, alphas = [r1, r2, r3], [al1, al2, al3]

    g = sum(al * ri ** 2 / (1 + lam * al) ** 2 for al, ri in zip(alphas, r))
    dg = sp.diff(g, lam)
    expected = -2 * sum(al ** 2 * ri ** 2 / (1 + lam * al) ** 3 for al, ri in zip(alphas, r))
    assert sp.simplify(dg - expected) == 0

    # Spot-check the sign on concrete positive numbers with the domain
    # condition (1 + lambda*alpha_i) > 0 satisfied for every i.
    subs = {al1: 0.3, al2: 0.5, al3: 0.2, r1: 1.7, r2: 0.1, r3: 4.0, lam: 0.6}
    assert all((1 + subs[lam] * subs[al]) > 0 for al in (al1, al2, al3))
    assert float(dg.subs(subs)) < 0


# ---------------------------------------------------------------------------
# Numerical cross-check against scipy, no closed form assumed.
# ---------------------------------------------------------------------------

def _scipy_projection(r, alpha, target):
    """
    Solve the same constrained problem with a generic NLP solver, no closed
    form assumed. `trust-constr` (SLSQP stalls on some of these quadratic
    programs well short of its iteration limit) with an explicit gradient
    and nonlinear constraint.
    """
    r, alpha = np.asarray(r, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    m = len(r)
    cost = lambda a: float(np.sum((a - r) ** 2))
    grad = lambda a: 2 * (a - r)
    constraint = scipy.optimize.NonlinearConstraint(
        lambda a: np.sum(alpha * a ** 2) - target, 0, 0)
    result = scipy.optimize.minimize(
        cost, x0=np.full(m, math.sqrt(target)), jac=grad, method="trust-constr",
        bounds=scipy.optimize.Bounds(0, np.inf), constraints=[constraint],
        options={"maxiter": 2000, "gtol": 1e-14, "xtol": 1e-14})
    assert result.success, result.message
    return result.x


@pytest.mark.parametrize("seed", range(20))
def test_uniform_projection_matches_scipy(seed):
    g = torch.Generator().manual_seed(seed)
    m = int(torch.randint(2, 6, (1,), generator=g))
    r = torch.rand(m, generator=g, dtype=torch.float64) * 3
    target = float(torch.rand(1, generator=g, dtype=torch.float64) * 3 + 0.01)

    mine = project_amplitudes_uniform(r, torch.tensor(target, dtype=torch.float64))
    theirs = _scipy_projection(r.tolist(), [1.0 / m] * m, target)
    assert np.allclose(mine.numpy(), theirs, atol=1e-5, rtol=1e-4), (mine, theirs)


@pytest.mark.parametrize("seed", range(20))
def test_weighted_projection_matches_scipy(seed):
    g = torch.Generator().manual_seed(seed + 1000)
    m = int(torch.randint(2, 6, (1,), generator=g))
    r = torch.rand(m, generator=g, dtype=torch.float64) * 3
    raw_alpha = torch.rand(m, generator=g, dtype=torch.float64) + 0.05
    alpha = raw_alpha / raw_alpha.sum()
    target = float(torch.rand(1, generator=g, dtype=torch.float64) * 3 + 0.01)

    mine = project_amplitudes_weighted(r, alpha, torch.tensor(target, dtype=torch.float64))
    theirs = _scipy_projection(r.tolist(), alpha.tolist(), target)
    assert np.allclose(mine.numpy(), theirs, atol=1e-5, rtol=1e-4), (mine, theirs)


def test_weighted_reduces_to_uniform_when_alpha_is_uniform():
    g = torch.Generator().manual_seed(3)
    m = 5
    r = torch.rand(m, 4, generator=g, dtype=torch.float64)
    target = torch.rand(4, generator=g, dtype=torch.float64) + 0.1
    alpha = torch.full((m,), 1.0 / m, dtype=torch.float64)

    from_uniform = project_amplitudes_uniform(r, target)
    from_weighted = project_amplitudes_weighted(r, alpha, target)
    assert torch.allclose(from_uniform, from_weighted, atol=1e-8, rtol=1e-6)


def test_dispatcher_picks_the_closed_form_for_uniform_alpha():
    g = torch.Generator().manual_seed(4)
    m = 4
    r = torch.rand(m, 3, generator=g, dtype=torch.float64)
    target = torch.rand(3, generator=g, dtype=torch.float64) + 0.1

    assert torch.equal(project_amplitudes(r, target),
                        project_amplitudes_uniform(r, target))
    alpha = torch.full((m,), 1.0 / m, dtype=torch.float64)
    assert torch.equal(project_amplitudes(r, target, alpha=alpha),
                        project_amplitudes_uniform(r, target))

    lopsided = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64)
    assert torch.equal(project_amplitudes(r, target, alpha=lopsided),
                        project_amplitudes_weighted(r, lopsided, target))


# ---------------------------------------------------------------------------
# The whole point: equal allocation is not optimal.
# ---------------------------------------------------------------------------

def test_equal_allocation_is_not_optimal():
    """
    A lopsided current amplitude vector: one subframe is already close to
    carrying the whole target, the other three are nearly dark. The radial
    projection barely disturbs that (it costs almost nothing), while
    splitting the target evenly across subframes regardless of `r` -- as
    independent single-frame holograms effectively do -- pays for ignoring
    the phases already chosen. The gap is orders of magnitude, not a rounding
    difference.
    """
    r = torch.tensor([2.0, 0.2, 0.2, 0.2], dtype=torch.float64)
    m = r.shape[0]
    target = torch.tensor(1.0, dtype=torch.float64)

    radial = project_amplitudes_uniform(r, target)
    radial_cost = float(((radial - r) ** 2).sum())

    equal_split = torch.full((m,), math.sqrt(float(target) / m), dtype=torch.float64)
    equal_cost = float(((equal_split - r) ** 2).sum())

    assert radial_cost > 0  # the target does differ from the current allocation
    assert equal_cost / radial_cost > 100, (radial_cost, equal_cost)


# ---------------------------------------------------------------------------
# The ||r|| = 0 degeneracy: no unique direction, so no epsilon is correct.
# ---------------------------------------------------------------------------

def test_zero_norm_uses_the_symmetric_allocation():
    m = 4
    r = torch.zeros(m, dtype=torch.float64)
    target = torch.tensor(2.5, dtype=torch.float64)

    a = project_amplitudes_uniform(r, target)
    assert torch.allclose(a, torch.full((m,), math.sqrt(float(target)), dtype=torch.float64),
                          atol=1e-12)
    assert abs(float((a ** 2).sum() / m) - float(target)) < 1e-12


def test_zero_norm_is_genuinely_non_unique_every_feasible_point_costs_the_same():
    """
    Proves the premise for the policy above: at r=0, EVERY point on the
    constraint sphere gives the same objective value, so picking the
    symmetric one is a choice, not a derivation.
    """
    m, target = 4, 2.5
    directions = [
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64) / math.sqrt(4 * 0.25 ** 2),
        torch.tensor([0.9, 0.1, 0.3, 0.1], dtype=torch.float64),
    ]
    costs = []
    for d in directions:
        d = d / d.norm()
        a = d * math.sqrt(m * target)
        assert abs(float((a ** 2).sum() / m) - target) < 1e-10  # feasible
        costs.append(float((a ** 2).sum()))  # r=0, so cost = sum(a^2)
    assert all(abs(c - costs[0]) < 1e-8 for c in costs)


def test_zero_norm_only_at_some_samples_in_a_batch():
    """The degenerate policy must apply per-sample, not to the whole batch."""
    r = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 3.0]], dtype=torch.float64)  # (M=3, 2)
    target = torch.tensor([4.0, 4.0], dtype=torch.float64)

    a = project_amplitudes_uniform(r, target)
    expected = torch.full((3,), math.sqrt(4.0), dtype=torch.float64)
    assert torch.allclose(a[:, 0], expected, atol=1e-12)
    assert not torch.allclose(a[:, 1], expected, atol=1e-6)
    assert abs(float((a[:, 1] ** 2).sum() / 3) - 4.0) < 1e-10


# ---------------------------------------------------------------------------
# The root-find: bracket, convergence, and failing loudly.
# ---------------------------------------------------------------------------

def test_weighted_projection_converges_to_tight_tolerance():
    g = torch.Generator().manual_seed(9)
    m = 6
    r = torch.rand(m, 50, generator=g, dtype=torch.float64) * 5
    raw = torch.rand(m, generator=g, dtype=torch.float64) + 0.05
    alpha = raw / raw.sum()
    target = torch.rand(50, generator=g, dtype=torch.float64) * 5 + 0.05

    a = project_amplitudes_weighted(r, alpha, target, tol=1e-12)
    achieved = (alpha.unsqueeze(-1) * a ** 2).sum(dim=0)
    assert torch.allclose(achieved, target, atol=1e-9, rtol=1e-8)
    assert bool((a >= 0).all())


def test_weighted_projection_raises_rather_than_guess_when_infeasible():
    """
    The subframe with the single largest alpha has r=0 there; the target is
    made far larger than the other subframes can ever supply as the
    multiplier approaches the domain edge. No finite lambda reaches it, so
    the bracket search must fail loudly instead of returning a wrong answer.
    """
    alpha = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)  # index 0 is the max
    r = torch.tensor([0.0, 0.01, 0.01], dtype=torch.float64)
    target = torch.tensor(1e6, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        project_amplitudes_weighted(r, alpha, target, max_expand=60)


def test_weighted_projection_rejects_malformed_weights():
    r = torch.rand(3, dtype=torch.float64)
    target = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        project_amplitudes_weighted(r, torch.tensor([0.5, 0.3, 0.1]), target)  # sums to 0.9
    with pytest.raises(ValueError):
        project_amplitudes_weighted(r, torch.tensor([0.5, -0.5, 1.0]), target)  # negative


def test_projection_rejects_negative_amplitudes_and_targets():
    r = torch.tensor([1.0, -0.1, 0.5], dtype=torch.float64)
    target = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        project_amplitudes_uniform(r, target)
    with pytest.raises(ValueError):
        project_amplitudes_uniform(r.abs(), torch.tensor(-1.0, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Vectorisation: the batched path must agree with a scalar reference.
# ---------------------------------------------------------------------------

def _scalar_uniform_reference(r, target):
    """Pure-python, one sample at a time -- no torch broadcasting at all."""
    m = len(r)
    norm = math.sqrt(sum(ri * ri for ri in r))
    if norm == 0.0:
        return [math.sqrt(target)] * m
    scale = math.sqrt(m * target) / norm
    return [ri * scale for ri in r]


def _scalar_weighted_reference(r, alpha, target, tol=1e-12):
    """Pure-python bisection, one sample at a time."""
    m = len(r)

    def g(lam):
        return sum(al * ri ** 2 / (1 + lam * al) ** 2 for al, ri in zip(alpha, r))

    lo_bound = -1.0 / max(alpha)
    lo, hi = lo_bound + 1e-9, 1.0
    while g(hi) > target:
        hi *= 2
    while g(lo) < target:
        lo = lo_bound + (lo - lo_bound) / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(mid) > target:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    return [ri / (1 + lam * al) for ri, al in zip(r, alpha)]


def test_vectorised_uniform_matches_scalar_loop():
    g = torch.Generator().manual_seed(21)
    m, n = 5, 30
    r = torch.rand(m, n, generator=g, dtype=torch.float64) * 4
    target = torch.rand(n, generator=g, dtype=torch.float64) * 4 + 0.01

    batched = project_amplitudes_uniform(r, target)
    for j in range(n):
        expected = _scalar_uniform_reference(r[:, j].tolist(), float(target[j]))
        assert np.allclose(batched[:, j].numpy(), expected, atol=1e-10), j


def test_vectorised_weighted_matches_scalar_loop():
    g = torch.Generator().manual_seed(22)
    m, n = 5, 25
    r = torch.rand(m, n, generator=g, dtype=torch.float64) * 4
    raw = torch.rand(m, generator=g, dtype=torch.float64) + 0.05
    alpha = raw / raw.sum()
    target = torch.rand(n, generator=g, dtype=torch.float64) * 4 + 0.01

    batched = project_amplitudes_weighted(r, alpha, target)
    for j in range(n):
        expected = _scalar_weighted_reference(
            r[:, j].tolist(), alpha.tolist(), float(target[j]))
        assert np.allclose(batched[:, j].numpy(), expected, atol=1e-8), j


def test_vectorised_over_a_2d_sample_grid():
    """The real use is one projection per retinal sample per view: (M, H, W)."""
    g = torch.Generator().manual_seed(23)
    m, h, w = 4, 6, 7
    r = torch.rand(m, h, w, generator=g, dtype=torch.float64) * 3
    target = torch.rand(h, w, generator=g, dtype=torch.float64) * 3 + 0.01

    a = project_amplitudes_uniform(r, target)
    assert a.shape == (m, h, w)
    achieved = (a ** 2).sum(dim=0) / m
    assert torch.allclose(achieved, target, atol=1e-10)


# ---------------------------------------------------------------------------
# The physics assumption: incoherent sum-of-intensities, not coherent.
# ---------------------------------------------------------------------------

def test_incoherent_sum_differs_from_coherent_sum():
    """
    This module implements `sum_m alpha_m |P u_m|^2` because the subframes
    are shown in sequence with no fixed phase relation (a free-running or
    deliberately randomised phase between exposures) and the eye's
    integration time only ever accumulates intensity, never field. The
    alternative, `|sum_m P u_m|^2`, is what a truly coherent superposition
    of the same M fields would produce, and the two are not close: with
    random relative phases the coherent sum has interference terms of the
    same order as the signal itself, so the two predictions differ by a
    large factor, not a rounding correction.
    """
    geom = Geometry(panel=16, window=8, pitch=1.524e-6, n_views=5, dtype=torch.float64)
    op = ViewOperators(geom)
    idx = torch.arange(len(geom.offsets))
    g = torch.Generator().manual_seed(31)

    m = 6
    fields = []
    for k in range(m):
        phase = torch.rand(16, 16, generator=g, dtype=torch.float64) * 2 * np.pi
        fields.append(op.forward(torch.exp(1j * phase), idx))
    stacked = torch.stack(fields)  # (M, views, W, W)

    incoherent = (stacked.abs() ** 2).mean(dim=0)
    coherent = (stacked.sum(dim=0) / m).abs() ** 2

    relative_difference = float((incoherent - coherent).abs().mean() / incoherent.mean())
    assert relative_difference > 0.5, (
        f"incoherent and coherent predictions should differ enormously, "
        f"got only {relative_difference:.3f} relative difference")
