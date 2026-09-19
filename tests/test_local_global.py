"""
The local/global (Gerchberg-Saxton / Projective-Dynamics-style) solver.

Structured like `test_operators.py`: the local projection is proved
symbolically first (sympy, not merely asserted), then each moving part is
checked against a reference (dense least squares for the global step),
then the whole cycle is checked for the behaviour actually claimed in
`tier2_hfh/local_global.py`'s docstring -- including where it is NOT
monotone, which is pinned here rather than assumed.
"""

import math

import sympy as sp
import torch

from tier2_hfh.operators import ViewOperators
from tier2_hfh.optimise import Geometry
from tier2_hfh.local_global import (
    project_onto_amplitude, panel_projection, energy,
    local_step, global_step, solve, gerchberg_saxton,
)


def geom_of(panel=12, window=4, n_views=3):
    return Geometry(panel=panel, window=window, pitch=1.524e-6, n_views=n_views,
                    dtype=torch.float64)


def _random_phase_panel(n, seed):
    g = torch.Generator().manual_seed(seed)
    phase = torch.rand(n, n, generator=g, dtype=torch.float64) * 2 * math.pi
    return torch.exp(1j * phase).to(torch.complex128)


# ---------------------------------------------------------------------------
# The local projection, proved rather than assumed.
# ---------------------------------------------------------------------------

def test_the_local_projection_formula_is_the_minimiser_by_lagrange_multipliers():
    """
    argmin_{|z|=a} |z-y|^2 = a*y/|y| for y != 0, proved symbolically.

    Treat z=(z1,z2) as real coordinates on the circle z1^2+z2^2=a^2 and y as
    a fixed point off the origin. The Lagrange stationarity conditions for
    minimising |z-y|^2 subject to that constraint,

        (z1 - y1) = lam*z1
        (z2 - y2) = lam*z2

    together with the constraint itself, are solved by sympy for (z1,z2,lam)
    with y1,y2,a left as free symbols. Solving confirms there are exactly
    two stationary points, z = +-(a/|y|)*y (not assumed -- read off the
    solver's own output), and evaluating the objective at both shows the
    "+" branch (the closed form used by `project_onto_amplitude`) is
    strictly smaller whenever a>0 and |y|>0: the difference between the two
    is 4*a*|y|, positive under exactly those conditions.
    """
    z1, z2, y1, y2, a, lam = sp.symbols("z1 z2 y1 y2 a lam", real=True)

    stationary = [
        sp.Eq(2 * (z1 - y1) - 2 * lam * z1, 0),
        sp.Eq(2 * (z2 - y2) - 2 * lam * z2, 0),
        sp.Eq(z1 ** 2 + z2 ** 2 - a ** 2, 0),
    ]
    solutions = sp.solve(stationary, [z1, z2, lam], dict=True)
    assert len(solutions) == 2, "expected exactly two Lagrange stationary points"

    r = sp.sqrt(y1 ** 2 + y2 ** 2)
    plus = next(s for s in solutions if sp.simplify(s[z1] - a * y1 / r) == 0)
    minus = next(s for s in solutions if s is not plus)
    assert sp.simplify(plus[z2] - a * y2 / r) == 0
    assert sp.simplify(minus[z1] + a * y1 / r) == 0
    assert sp.simplify(minus[z2] + a * y2 / r) == 0

    def objective(sol):
        return sp.simplify((sol[z1] - y1) ** 2 + (sol[z2] - y2) ** 2)

    d_plus, d_minus = objective(plus), objective(minus)
    assert sp.simplify(d_plus - (a - r) ** 2) == 0
    assert sp.simplify(d_minus - (a + r) ** 2) == 0
    # d_minus - d_plus = 4*a*r: strictly positive for a>0, r>0, so "plus" is
    # the strict global minimiser (there are no other critical points on the
    # compact circle for y != 0, since y1=y2=0 was excluded to reach here).
    assert sp.simplify(d_minus - d_plus - 4 * a * r) == 0


def test_project_onto_amplitude_matches_the_closed_form_for_nonzero_y():
    g = torch.Generator().manual_seed(3)
    y = (torch.randn(5, 6, generator=g, dtype=torch.float64)
         + 1j * torch.randn(5, 6, generator=g, dtype=torch.float64))
    a = 0.3 + torch.rand(5, 6, generator=g, dtype=torch.float64)
    z = project_onto_amplitude(y, a)
    assert torch.allclose(z, a.to(z.dtype) * y / y.abs(), rtol=1e-12, atol=1e-12)
    assert torch.allclose(z.abs(), a, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# The y=0 corner: an explicit, tested policy, not an epsilon.
# ---------------------------------------------------------------------------

def test_zero_policy_zero_returns_exactly_zero_at_y_equal_zero():
    y = torch.tensor([0 + 0j, 1 + 1j], dtype=torch.complex128)
    a = torch.tensor([2.0, 2.0], dtype=torch.float64)
    z = project_onto_amplitude(y, a, zero_policy="zero")
    assert z[0] == 0
    assert abs(complex(z[1]) - complex(a[1] * (1 + 1j) / abs(1 + 1j))) < 1e-12


def test_zero_policy_reference_phase_lands_exactly_on_the_circle():
    y = torch.tensor([0 + 0j], dtype=torch.complex128)
    a = torch.tensor([3.5], dtype=torch.float64)
    z = project_onto_amplitude(y, a, zero_policy="reference_phase", reference_phase=0.9)
    expected = 3.5 * complex(math.cos(0.9), math.sin(0.9))
    assert abs(complex(z[0]) - expected) < 1e-12
    assert abs(complex(z[0])) - 3.5 < 1e-12


def test_the_two_zero_policies_agree_away_from_the_zero_locus():
    g = torch.Generator().manual_seed(4)
    y = (torch.randn(20, generator=g, dtype=torch.float64)
         + 1j * torch.randn(20, generator=g, dtype=torch.float64))
    a = torch.rand(20, generator=g, dtype=torch.float64) + 0.1
    z1 = project_onto_amplitude(y, a, zero_policy="zero")
    z2 = project_onto_amplitude(y, a, zero_policy="reference_phase", reference_phase=1.234)
    assert torch.allclose(z1, z2, rtol=1e-12, atol=1e-12)


def test_unknown_zero_policy_fails_loudly():
    y = torch.tensor([1 + 1j], dtype=torch.complex128)
    try:
        project_onto_amplitude(y, torch.tensor([1.0]), zero_policy="nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_panel_projection_is_unit_amplitude_and_honours_the_same_zero_policy():
    u_star = torch.tensor([0 + 0j, 2 + 0j, 0 - 3j], dtype=torch.complex128)
    z = panel_projection(u_star, zero_policy="zero")
    assert z[0] == 0
    assert torch.allclose(z[1:].abs(), torch.ones(2, dtype=torch.float64))

    z_ref = panel_projection(u_star, zero_policy="reference_phase", reference_phase=0.0)
    assert abs(complex(z_ref[0]) - 1.0) < 1e-12          # amplitude 1, phase 0


# ---------------------------------------------------------------------------
# The global step, validated against the dense least-squares reference.
# ---------------------------------------------------------------------------

def test_global_step_matches_dense_least_squares_on_the_active_set():
    """
    solve_normal's result, called through `global_step`, must agree with
    `torch.linalg.lstsq` on the panel's active pixels. Dead pixels are
    excluded deliberately: the least-squares problem is rank-deficient
    there (no view constrains them at all) so its solution is non-unique,
    and `global_step` breaks the tie with the minimum-norm choice of 0
    while a generic LAPACK driver need not agree.
    """
    geom = geom_of(12, 4, 5)
    op = ViewOperators(geom)
    g = torch.Generator().manual_seed(5)
    z = (torch.randn(len(op.offsets), 4, 4, generator=g, dtype=torch.float64)
         + 1j * torch.randn(len(op.offsets), 4, 4, generator=g, dtype=torch.float64))

    u_star = global_step(op, z)

    from tier2_hfh.operators import dense_matrix
    P = dense_matrix(op)
    reference, *_ = torch.linalg.lstsq(P, z.reshape(-1).unsqueeze(-1))
    reference = reference.squeeze(-1)

    active = op.active.reshape(-1)
    assert int(active.sum()) < active.numel(), "this geometry should have dead pixels"
    assert torch.allclose(u_star.reshape(-1)[active], reference[active],
                          rtol=1e-8, atol=1e-8)
    assert torch.equal(u_star.reshape(-1)[~active], torch.zeros((~active).sum(),
                                                                 dtype=u_star.dtype))


def test_global_step_with_weights_matches_dense_weighted_least_squares():
    geom = geom_of(12, 4, 5)
    op = ViewOperators(geom)
    g = torch.Generator().manual_seed(6)
    z = (torch.randn(len(op.offsets), 4, 4, generator=g, dtype=torch.float64)
         + 1j * torch.randn(len(op.offsets), 4, 4, generator=g, dtype=torch.float64))
    rho = 0.2 + torch.rand(len(op.offsets), generator=g, dtype=torch.float64)

    u_star = global_step(op, z, weights=rho)

    from tier2_hfh.operators import dense_matrix
    P = dense_matrix(op)
    w = rho.repeat_interleave(4 * 4).to(torch.complex128)
    lhs = (P.conj().T * w) @ P
    rhs = (P.conj().T * w) @ z.reshape(-1)
    reference = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)

    active = op.active.reshape(-1)
    assert torch.allclose(u_star.reshape(-1)[active], reference[active],
                          rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Energy monitoring: what is actually monotone, and what is not.
# ---------------------------------------------------------------------------

def _run_realisable(panel, window, n_views, iters, seed):
    geom = geom_of(panel, window, n_views)
    op = ViewOperators(geom)
    u_true = _random_phase_panel(panel, seed)
    amplitude = op.forward(u_true).abs()
    u0 = _random_phase_panel(panel, seed + 1)
    u_final, history = solve(op, amplitude, iters=iters, u0=u0)
    return op, amplitude, u_true, u0, u_final, history


def test_the_local_and_global_steps_never_increase_the_objective():
    """
    before -> after_global is non-increasing every iteration: the global
    step is the least-squares projection of u onto the very targets whose
    distance IS the objective, so it cannot move away from them.
    """
    _, _, _, _, _, history = _run_realisable(12, 4, 5, iters=8, seed=0)
    for h in history:
        assert h["after_global"] <= h["before"] + 1e-9, h


def test_the_panel_projection_step_is_not_monotone():
    """
    after_global -> after_panel is NOT guaranteed non-increasing, and this
    test exists to pin that it is genuinely observed, not merely a
    theoretical possibility being hedged against. The panel step changes
    every pixel's amplitude back to 1 independently of what made the
    coupling objective small, so it can and does move the objective uphill.
    """
    _, _, _, _, _, history = _run_realisable(12, 4, 5, iters=8, seed=0)
    violations = [h for h in history if h["after_panel"] > h["after_global"] + 1e-9]
    assert len(violations) > 0, "expected at least one non-monotone panel step"


def test_the_overall_objective_trends_downward_across_iterations():
    """
    Despite the intra-iteration non-monotonicity above, the objective
    measured once per iteration (the "before" value, which equals the
    previous iteration's "after_panel") was observed to be monotone
    non-increasing end to end on every geometry tried during development.
    That is reported as an empirical observation, not re-derived as a
    theorem, so this test checks it holds rather than assuming it always
    must.
    """
    _, _, _, _, _, history = _run_realisable(12, 4, 5, iters=15, seed=1)
    befores = [h["before"] for h in history]
    assert all(b2 <= b1 + 1e-6 for b1, b2 in zip(befores, befores[1:]))
    assert befores[-1] < befores[0] * 0.9


# ---------------------------------------------------------------------------
# Fixed points and exact recovery: measurements, not phase masks.
# ---------------------------------------------------------------------------

def test_solve_drives_down_the_residual_on_a_realisable_target():
    """
    a_k = |P_k u_true| for a random u_true makes the problem realisable by
    construction. Recovering u_true's PHASE is not required -- global phase
    is unobservable (see `test_operators.test_global_phase_is_unobservable`)
    and phase retrieval is non-unique in general -- only that the solver,
    started from an unrelated random phase, reduces the measurement
    residual substantially.
    """
    op, amplitude, u_true, u0, u_final, history = _run_realisable(
        16, 6, 4, iters=25, seed=10)

    initial_residual = (op.forward(u0).abs() - amplitude).pow(2).sum().item()
    final_residual = (op.forward(u_final).abs() - amplitude).pow(2).sum().item()
    assert final_residual < initial_residual * 0.5, (
        f"initial {initial_residual:.3e} final {final_residual:.3e}")


def test_the_energy_function_matches_the_documented_objective():
    """L(u) as computed by `energy` must equal the definition in the docstring."""
    geom = geom_of(12, 4, 5)
    op = ViewOperators(geom)
    u = _random_phase_panel(12, seed=20)
    amplitude = 0.5 + torch.rand(len(op.offsets), 4, 4,
                                 generator=torch.Generator().manual_seed(21),
                                 dtype=torch.float64)
    got = energy(op, u, amplitude)
    y = op.forward(u)
    expected = float(((y.abs() - amplitude) ** 2).sum())
    assert abs(got - expected) < 1e-8


# ---------------------------------------------------------------------------
# The plain Gerchberg-Saxton baseline.
# ---------------------------------------------------------------------------

def test_gerchberg_saxton_runs_and_reduces_the_residual():
    geom = geom_of(12, 4, 5)
    op = ViewOperators(geom)
    u_true = _random_phase_panel(12, seed=30)
    amplitude = op.forward(u_true).abs()
    u0 = _random_phase_panel(12, seed=31)

    u_final, history = gerchberg_saxton(op, amplitude, iters=15, u0=u0)

    initial_residual = (op.forward(u0).abs() - amplitude).pow(2).sum().item()
    final_residual = (op.forward(u_final).abs() - amplitude).pow(2).sum().item()
    assert final_residual < initial_residual * 0.5, (
        f"initial {initial_residual:.3e} final {final_residual:.3e}")
    assert len(history) == 15
    assert all(math.isfinite(h) for h in history)


def test_gerchberg_saxton_keeps_the_panel_phase_only():
    geom = geom_of(12, 4, 3)
    op = ViewOperators(geom)
    u_true = _random_phase_panel(12, seed=40)
    amplitude = op.forward(u_true).abs()
    u0 = _random_phase_panel(12, seed=41)
    u_final, _ = gerchberg_saxton(op, amplitude, iters=5, u0=u0)
    assert torch.allclose(u_final.abs(), torch.ones(12, 12, dtype=torch.float64),
                          rtol=1e-10, atol=1e-10)
