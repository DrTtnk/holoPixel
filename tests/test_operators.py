"""
The forward operator P_k, its adjoint, and the normal operator.

Everything the local/global solver rests on is here, so these tests are the
foundation: a wrong adjoint still produces plausible images while silently
invalidating every gradient and every normal-operator claim downstream.
"""

import numpy as np
import pytest
import torch

from tier2_hfh.operators import ViewOperators, dense_matrix
from tier2_hfh.optimise import Geometry, render_views


def geom_of(panel=12, window=4, n_views=3):
    return Geometry(panel=panel, window=window, pitch=1.524e-6, n_views=n_views,
                    dtype=torch.float64)


OVERLAPPING = [(12, 4, 5), (12, 4, 9), (16, 6, 6), (16, 8, 5)]
DISJOINT = [(12, 4, 3), (16, 4, 4)]


def _random_panel(n, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, n, generator=g, dtype=torch.float64)
            + 1j * torch.randn(n, n, generator=g, dtype=torch.float64))


def _random_views(k, w, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(k, w, w, generator=g, dtype=torch.float64)
            + 1j * torch.randn(k, w, w, generator=g, dtype=torch.float64))


# ---------------------------------------------------------------------------
# The adjoint. This is the test the whole subsystem hangs on.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("panel,window,n_views", OVERLAPPING + DISJOINT)
def test_adjoint_satisfies_the_dot_product_identity(panel, window, n_views):
    """<P x, y> == <x, P^H y> for arbitrary complex x and y."""
    op = ViewOperators(geom_of(panel, window, n_views))
    x = _random_panel(panel, seed=1)
    y = _random_views(len(op.offsets), window, seed=2)

    lhs = torch.vdot(op.forward(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), op.adjoint(y).reshape(-1))
    scale = max(abs(complex(lhs)), abs(complex(rhs)), 1e-300)
    assert abs(complex(lhs) - complex(rhs)) / scale < 1e-12, f"{lhs} vs {rhs}"


def test_adjoint_identity_holds_for_a_subset_of_views():
    op = ViewOperators(geom_of(16, 6, 6))
    idx = torch.tensor([0, 3, 7])
    x = _random_panel(16, seed=5)
    y = _random_views(len(idx), 6, seed=6)
    lhs = torch.vdot(op.forward(x, idx).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), op.adjoint(y, idx).reshape(-1))
    assert abs(complex(lhs) - complex(rhs)) / abs(complex(lhs)) < 1e-12


def test_a_deliberately_wrong_adjoint_is_caught():
    """
    Guard the guard. Drop the conjugate on the aperture -- harmless here since
    the aperture is real -- but also drop the W^2 factor, which is not.
    """
    op = ViewOperators(geom_of(12, 4, 5))
    x, y = _random_panel(12, seed=7), _random_views(len(op.offsets), 4, seed=8)
    good = complex(torch.vdot(x.reshape(-1), op.adjoint(y).reshape(-1)))
    bad = good / (4 * 4)
    lhs = complex(torch.vdot(op.forward(x).reshape(-1), y.reshape(-1)))
    assert abs(lhs - good) / abs(lhs) < 1e-12
    assert abs(lhs - bad) / abs(lhs) > 0.9


# ---------------------------------------------------------------------------
# Consistency with the renderer the rest of the project already uses.
# ---------------------------------------------------------------------------

def test_forward_intensity_matches_render_views():
    """
    render_views is the existing forward model and the target data was built
    against it. |P u|^2 must reproduce it exactly, or the solver would be
    solving a different problem from the one the targets describe.
    """
    geom = geom_of(16, 8, 5)
    op = ViewOperators(geom)
    g = torch.Generator().manual_seed(11)
    phase = torch.rand(16, 16, generator=g, dtype=torch.float64) * 2 * np.pi
    idx = torch.arange(len(geom.offsets))

    mine = (op.forward(torch.exp(1j * phase), idx).abs() ** 2)
    theirs = render_views(phase, geom, idx)
    assert torch.allclose(mine, theirs, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# The normal operator.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("panel,window,n_views", OVERLAPPING + DISJOINT)
def test_the_normal_operator_is_exactly_diagonal(panel, window, n_views):
    op = ViewOperators(geom_of(panel, window, n_views))
    H = dense_matrix(op, normal=True)
    off = H - torch.diag(torch.diagonal(H))
    assert float(off.abs().max()) < 1e-10, f"off-diagonal {float(off.abs().max()):.2e}"


@pytest.mark.parametrize("panel,window,n_views", OVERLAPPING + DISJOINT)
def test_the_diagonal_matches_its_closed_form(panel, window, n_views):
    """diag = W^2 * sum_k A^2, embedded at each pupil offset."""
    op = ViewOperators(geom_of(panel, window, n_views))
    H = dense_matrix(op, normal=True)
    assert torch.allclose(torch.diagonal(H).real.reshape(panel, panel),
                          op.diagonal, rtol=1e-10, atol=1e-10)
    assert float(torch.diagonal(H).imag.abs().max()) < 1e-10


def test_applying_the_normal_operator_equals_scaling_by_the_diagonal():
    """Matrix-free P^H P u must agree with the closed form, elementwise."""
    op = ViewOperators(geom_of(16, 6, 6))
    u = _random_panel(16, seed=13)
    assert torch.allclose(op.adjoint(op.forward(u)), op.diagonal * u,
                          rtol=1e-10, atol=1e-10)


def test_per_view_weights_keep_it_diagonal_and_scale_the_closed_form():
    op = ViewOperators(geom_of(12, 4, 5))
    g = torch.Generator().manual_seed(17)
    rho = 0.2 + torch.rand(len(op.offsets), generator=g, dtype=torch.float64)
    H = dense_matrix(op, normal=True, weights=rho)
    off = H - torch.diag(torch.diagonal(H))
    assert float(off.abs().max()) < 1e-10
    assert torch.allclose(torch.diagonal(H).real.reshape(12, 12),
                          op.diagonal_for(rho), rtol=1e-10, atol=1e-10)


def test_uniform_weights_reproduce_the_unweighted_diagonal():
    op = ViewOperators(geom_of(12, 4, 5))
    ones = torch.ones(len(op.offsets), dtype=torch.float64)
    assert torch.allclose(op.diagonal_for(ones), op.diagonal, rtol=1e-12)


# ---------------------------------------------------------------------------
# The structural null space.
# ---------------------------------------------------------------------------

def test_null_pixels_are_exactly_those_no_pupil_can_see():
    op = ViewOperators(geom_of(12, 4, 3))          # disjoint: many dead pixels
    assert int((~op.active).sum()) > 0, "this geometry should have dead pixels"
    for (r, c) in torch.nonzero(~op.active).tolist():
        e = torch.zeros(12, 12, dtype=torch.complex128)
        e[r, c] = 1.0
        assert float(op.forward(e).abs().max()) < 1e-12, \
            f"pixel {(r, c)} is called dead but reaches a view"


def test_active_pixels_all_reach_some_view():
    op = ViewOperators(geom_of(12, 4, 5))
    for (r, c) in torch.nonzero(op.active).tolist():
        e = torch.zeros(12, 12, dtype=torch.complex128)
        e[r, c] = 1.0
        assert float(op.forward(e).abs().max()) > 1e-12


def test_the_active_set_grows_with_the_pupil_set():
    """A pixel dead for one batch of pupils may live for the whole eyebox."""
    few = ViewOperators(geom_of(16, 6, 2))
    many = ViewOperators(geom_of(16, 6, 6))
    assert int(many.active.sum()) > int(few.active.sum())


# ---------------------------------------------------------------------------
# Conventions that are easy to get wrong and expensive to get wrong.
# ---------------------------------------------------------------------------

def test_the_dft_normalisation_constant_is_pinned():
    """torch.fft.fft2 is unnormalised, so F^H F = W^2 I. Everything scales."""
    w = 6
    eye = torch.eye(w * w, dtype=torch.complex128).reshape(w * w, w, w)
    F = torch.fft.fft2(eye).reshape(w * w, w * w).T
    assert torch.allclose(F.conj().T @ F,
                          (w * w) * torch.eye(w * w, dtype=torch.complex128),
                          atol=1e-9)


def test_dense_matrix_agrees_with_the_matrix_free_forward():
    op = ViewOperators(geom_of(12, 4, 5))
    P = dense_matrix(op)
    u = _random_panel(12, seed=19)
    assert torch.allclose(P @ u.reshape(-1), op.forward(u).reshape(-1),
                          rtol=1e-10, atol=1e-10)


def test_global_phase_is_unobservable():
    op = ViewOperators(geom_of(12, 4, 5))
    u = _random_panel(12, seed=23)
    rotated = np.exp(1j * 0.7331) * u
    assert torch.allclose(op.forward(u).abs() ** 2, op.forward(rotated).abs() ** 2,
                          rtol=1e-12, atol=1e-12)


def test_the_operator_is_linear():
    op = ViewOperators(geom_of(12, 4, 5))
    a, b = _random_panel(12, seed=29), _random_panel(12, seed=31)
    s = 0.3 - 1.7j
    assert torch.allclose(op.forward(a + s * b), op.forward(a) + s * op.forward(b),
                          rtol=1e-10, atol=1e-10)
