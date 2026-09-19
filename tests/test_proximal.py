"""
Proximal variants of the local/global solver, for the KL convergence
question in `docs/notes_kl_convergence.md`.

Structured like `test_local_global.py`: each closed form is proved
symbolically first (sympy, not merely asserted), then checked against the
implementation, then the reduction to the non-proximal solver at
eta=gamma=0 is checked, then the numerical evidence the notes report
(sufficient decrease, the telescoping step-sum bound, and the vanilla-vs-
proximal comparison) is pinned here rather than only quoted in prose.

CPU only, torch.float64, small geometries throughout (panel 12-16, window
4-6), per the task's hard constraints.
"""

import math

import sympy as sp
import torch

# These tests run many small (panel<=16, window<=6) CPU solves; torch's
# default thread pool contends heavily on tensors this small and makes the
# suite far slower than the single-threaded path, with no effect on the
# (fully deterministic, float64) results.
torch.set_num_threads(1)

from tier2_hfh.operators import ViewOperators
from tier2_hfh.optimise import Geometry
from tier2_hfh.local_global import (
    project_onto_amplitude, global_step, panel_projection, local_step, solve,
)
from tier2_hfh.proximal import (
    proximal_global_step, proximal_local_step, coupling_energy, proximal_solve,
)


def geom_of(panel=12, window=4, n_views=3):
    return Geometry(panel=panel, window=window, pitch=1.524e-6, n_views=n_views,
                    dtype=torch.float64)


def _random_phase_panel(n, seed):
    g = torch.Generator().manual_seed(seed)
    phase = torch.rand(n, n, generator=g, dtype=torch.float64) * 2 * math.pi
    return torch.exp(1j * phase).to(torch.complex128)


def _random_complex(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g, dtype=torch.float64)
            + 1j * torch.randn(*shape, generator=g, dtype=torch.float64)).to(torch.complex128)


# ---------------------------------------------------------------------------
# Both closed forms, re-derived (not assumed) with sympy: Lagrange
# stationarity, then branch comparison, exactly as test_local_global.py
# does for the non-proximal projection.
# ---------------------------------------------------------------------------

def test_the_global_step_formula_is_the_minimiser_by_lagrange_multipliers():
    """
    argmin_{|u|=1} d*|u|^2 - 2*Re(conj(b)*u) + eta*|u-u_old|^2
        = (b+eta*u_old) / |b+eta*u_old|,

    independent of d -- because d*|u|^2 is constant on |u|=1, which this
    test also confirms directly: the difference between the two Lagrange
    branches does not involve d at all.
    """
    u1, u2, b1, b2, uo1, uo2, d, eta, lam = sp.symbols(
        "u1 u2 b1 b2 uo1 uo2 d eta lam", real=True)
    J = d * (u1**2 + u2**2) - 2 * (b1 * u1 + b2 * u2) + eta * ((u1 - uo1)**2 + (u2 - uo2)**2)
    stationary = [
        sp.Eq(sp.diff(J, u1) - lam * 2 * u1, 0),
        sp.Eq(sp.diff(J, u2) - lam * 2 * u2, 0),
        sp.Eq(u1**2 + u2**2 - 1, 0),
    ]
    solutions = sp.solve(stationary, [u1, u2, lam], dict=True)
    assert len(solutions) == 2, "expected exactly two Lagrange stationary points"

    r = sp.sqrt((b1 + eta * uo1)**2 + (b2 + eta * uo2)**2)
    plus = next(s for s in solutions if sp.simplify(s[u1] - (b1 + eta * uo1) / r) == 0)
    minus = next(s for s in solutions if s is not plus)
    assert sp.simplify(plus[u2] - (b2 + eta * uo2) / r) == 0
    assert sp.simplify(minus[u1] + (b1 + eta * uo1) / r) == 0
    assert sp.simplify(minus[u2] + (b2 + eta * uo2) / r) == 0

    def objective(sol):
        return sp.simplify(J.subs({u1: sol[u1], u2: sol[u2]}))

    diff = sp.simplify(objective(minus) - objective(plus))
    assert sp.simplify(diff - 4 * r) == 0, "the 'plus' branch must be strictly smaller"
    assert d not in diff.free_symbols, "the branch comparison must not depend on d"


def test_the_local_step_formula_is_the_minimiser_by_lagrange_multipliers():
    """argmin_{|z|=a} |z-y|^2 + gamma*|z-z_old|^2 = a*(y+gamma*z_old)/|y+gamma*z_old|."""
    z1, z2, y1, y2, zo1, zo2, gamma, a, lam = sp.symbols(
        "z1 z2 y1 y2 zo1 zo2 gamma a lam", real=True)
    K = (z1 - y1)**2 + (z2 - y2)**2 + gamma * ((z1 - zo1)**2 + (z2 - zo2)**2)
    stationary = [
        sp.Eq(sp.diff(K, z1) - lam * 2 * z1, 0),
        sp.Eq(sp.diff(K, z2) - lam * 2 * z2, 0),
        sp.Eq(z1**2 + z2**2 - a**2, 0),
    ]
    solutions = sp.solve(stationary, [z1, z2, lam], dict=True)
    assert len(solutions) == 2

    r = sp.sqrt((y1 + gamma * zo1)**2 + (y2 + gamma * zo2)**2)
    plus = next(s for s in solutions if sp.simplify(s[z1] - a * (y1 + gamma * zo1) / r) == 0)
    minus = next(s for s in solutions if s is not plus)
    assert sp.simplify(plus[z2] - a * (y2 + gamma * zo2) / r) == 0
    assert sp.simplify(minus[z1] + a * (y1 + gamma * zo1) / r) == 0
    assert sp.simplify(minus[z2] + a * (y2 + gamma * zo2) / r) == 0

    def objective(sol):
        return sp.simplify(K.subs({z1: sol[z1], z2: sol[z2]}))

    diff = sp.simplify(objective(minus) - objective(plus))
    assert sp.simplify(diff - 4 * a * r) == 0


def test_the_local_step_weight_scaling_is_a_positive_rescaling_not_a_new_formula():
    """
    `gamma/rho_k` in `proximal_local_step` for weighted views is just the
    algebraic fact argmin_z(rho*A(z) + C*B(z)) = argmin_z(A(z) + (C/rho)*B(z))
    for rho>0 -- checked symbolically for the specific A, B here rather than
    asserted as "obviously the same argmin".
    """
    z1, z2, y1, y2, zo1, zo2, gamma, rho = sp.symbols(
        "z1 z2 y1 y2 zo1 zo2 gamma rho", real=True, positive=True)
    A = (z1 - y1)**2 + (z2 - y2)**2
    B = (z1 - zo1)**2 + (z2 - zo2)**2
    weighted = sp.expand(rho * A + gamma * B)
    rescaled = sp.expand(rho * (A + (gamma / rho) * B))
    assert sp.simplify(weighted - rescaled) == 0


# ---------------------------------------------------------------------------
# The implementation matches the closed forms exactly.
# ---------------------------------------------------------------------------

def test_proximal_global_step_matches_the_closed_form():
    geom = geom_of(12, 4, 3)
    op = ViewOperators(geom)
    z = _random_complex(len(op.offsets), 4, 4, seed=1)
    u_old = _random_phase_panel(12, seed=2)
    eta = 0.3

    got = proximal_global_step(op, z, u_old, eta)

    b = op.adjoint(z)
    target = b + eta * u_old
    expected = project_onto_amplitude(target, 1.0)
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(got.abs(), torch.ones(12, 12, dtype=torch.float64),
                          rtol=1e-12, atol=1e-12)


def test_proximal_local_step_matches_the_closed_form():
    geom = geom_of(12, 4, 3)
    op = ViewOperators(geom)
    u = _random_phase_panel(12, seed=3)
    amplitude = 0.5 + torch.rand(len(op.offsets), 4, 4,
                                 generator=torch.Generator().manual_seed(4),
                                 dtype=torch.float64)
    z_old = _random_complex(len(op.offsets), 4, 4, seed=5)
    gamma = 0.4

    got = proximal_local_step(op, u, amplitude, z_old, gamma)

    y = op.forward(u)
    target = y + gamma * z_old
    expected = project_onto_amplitude(target, amplitude)
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(got.abs(), amplitude, rtol=1e-10, atol=1e-10)


def test_proximal_local_step_with_weights_divides_gamma_by_rho():
    geom = geom_of(12, 4, 3)
    op = ViewOperators(geom)
    u = _random_phase_panel(12, seed=6)
    amplitude = 0.5 + torch.rand(len(op.offsets), 4, 4,
                                 generator=torch.Generator().manual_seed(7),
                                 dtype=torch.float64)
    z_old = _random_complex(len(op.offsets), 4, 4, seed=8)
    gamma = 0.4
    rho = 0.2 + torch.rand(len(op.offsets), generator=torch.Generator().manual_seed(9),
                           dtype=torch.float64)

    got = proximal_local_step(op, u, amplitude, z_old, gamma, weights=rho)

    y = op.forward(u)
    effective_gamma = (gamma / rho)[:, None, None]
    expected = project_onto_amplitude(y + effective_gamma.to(z_old.dtype) * z_old, amplitude)
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# At eta=gamma=0 the proximal steps reduce exactly to the existing,
# non-proximal solver -- the proximal machinery adds a term, it does not
# replace the underlying projection.
# ---------------------------------------------------------------------------

def test_proximal_global_step_at_eta_zero_matches_global_and_panel_step_composed():
    geom = geom_of(12, 4, 3)
    op = ViewOperators(geom)
    z = _random_complex(len(op.offsets), 4, 4, seed=10)
    u_old = _random_phase_panel(12, seed=11)

    got = proximal_global_step(op, z, u_old, eta=0.0)
    u_star = global_step(op, z)
    expected = panel_projection(u_star)
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_proximal_local_step_at_gamma_zero_matches_local_step():
    geom = geom_of(12, 4, 3)
    op = ViewOperators(geom)
    u = _random_phase_panel(12, seed=12)
    amplitude = 0.5 + torch.rand(len(op.offsets), 4, 4,
                                 generator=torch.Generator().manual_seed(13),
                                 dtype=torch.float64)
    z_old = _random_complex(len(op.offsets), 4, 4, seed=14)

    got = proximal_local_step(op, u, amplitude, z_old, gamma=0.0)
    expected = local_step(op, u, amplitude)
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# The numerical evidence: sufficient decrease, its constant, and the
# telescoping bound on the total step length it implies.
# ---------------------------------------------------------------------------

_SEEDS = list(range(12))


def _run_proximal(seed, panel=12, window=4, n_views=3, iters=150, eta=0.05, gamma=0.05):
    geom = geom_of(panel, window, n_views)
    op = ViewOperators(geom)
    u_true = _random_phase_panel(panel, seed)
    amplitude = op.forward(u_true).abs()
    u0 = _random_phase_panel(panel, seed + 10_000)
    _, _, history = proximal_solve(op, amplitude, iters=iters, eta=eta, gamma=gamma, u0=u0)
    return history


def test_sufficient_decrease_holds_with_the_predicted_constant():
    """
    L(x_t) - L(x_{t+1}) >= c * ||x_{t+1}-x_t||^2 with c = min(eta, gamma),
    on EVERY iteration, across 12 random instances. This is a direct
    consequence of each proximal step being an exact minimiser of its own
    regularised subproblem (`KLSufficientDecrease.lean`'s
    `sufficient_decrease`, a three-point inequality that needs no KL
    property) -- checked here numerically with the actual solver rather
    than only proved abstractly.
    """
    eta, gamma = 0.05, 0.05
    c = min(eta, gamma)
    worst_slack = math.inf
    for seed in _SEEDS:
        history = _run_proximal(seed, eta=eta, gamma=gamma)
        for h in history:
            slack = (h["L"] - h["L_next"]) - c * h["step"]
            worst_slack = min(worst_slack, slack)
    assert worst_slack >= -1e-6, f"sufficient decrease violated, worst slack {worst_slack:.3e}"


def test_the_total_step_length_respects_the_telescoping_bound():
    """
    Summing sufficient decrease over a whole run gives
    c * sum_t ||x_{t+1}-x_t||^2 <= L_0 - L_final <= L_0 (since L>=0), and
    the tail of the sum (the last 10 steps) should be much smaller than the
    early average, evidence the step size is settling rather than merely
    being bounded on average.
    """
    eta, gamma = 0.05, 0.05
    c = min(eta, gamma)
    for seed in _SEEDS:
        history = _run_proximal(seed, eta=eta, gamma=gamma)
        step_sum = sum(h["step"] for h in history)
        bound = history[0]["L"] / c
        assert step_sum <= bound + 1e-6

        early_mean = sum(h["step"] for h in history[:10]) / 10
        late_mean = sum(h["step"] for h in history[-10:]) / 10
        assert late_mean <= early_mean + 1e-9, (
            "expected the step size to settle, not grow, over the run")


def test_proximal_does_not_change_which_fixed_point_is_reached_much():
    """
    Vanilla `solve` and `proximal_solve`, started from the SAME random
    panel on the SAME realisable target, are compared by their final
    amplitude residual. The notes' hypothesis is that the proximal term
    stabilises the iterate without changing where it settles; this is
    checked by requiring the two final residuals to stay within a
    generous factor of each other on every seed (neither collapses to
    near-zero while the other stays large), not by requiring them to be
    equal or by requiring either to reach the global optimum.
    """
    for seed in _SEEDS:
        geom = geom_of(12, 4, 3)
        op = ViewOperators(geom)
        u_true = _random_phase_panel(12, seed)
        amplitude = op.forward(u_true).abs()
        u0 = _random_phase_panel(12, seed + 10_000)

        u_v, _ = solve(op, amplitude, iters=150, u0=u0)
        u_p, _, _ = proximal_solve(op, amplitude, iters=150, eta=0.05, gamma=0.05, u0=u0)

        resid_v = float((op.forward(u_v).abs() - amplitude).pow(2).sum())
        resid_p = float((op.forward(u_p).abs() - amplitude).pow(2).sum())

        ratio = resid_p / resid_v if resid_v > 0 else math.inf
        assert 0.2 <= ratio <= 5.0, (
            f"seed {seed}: vanilla {resid_v:.3e} vs proximal {resid_p:.3e} "
            f"differ by more than a factor of 5 -- re-examine before trusting "
            f"'stabilises without escaping'")
