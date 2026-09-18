"""
Angular-spectrum propagation, checked against results derived elsewhere rather
than against itself.

The strongest test here is the Wigner shear. Section 3 of
docs/holographic_rendering_equation.md derives, and
tests/test_holographic_transport.py verifies to 5e-16, that free-space
propagation shears the Wigner distribution by exactly lam*z. That derivation
was done symbolically and has nothing to do with this implementation, so if
the two agree, both are almost certainly right.
"""

import numpy as np
import pytest
import torch

from tier2_hfh.propagate import propagate, transfer_function, wigner_shear_reference

LAM = 532e-9
PITCH = 2e-6
N = 64


def _band_limited_field(seed, n=N, keep=8):
    """Random field with no energy near the Nyquist edge, so that propagation
    does not alias and the round trip is honest."""
    rng = np.random.default_rng(seed)
    spec = np.zeros((n, n), dtype=complex)
    idx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    mask = (np.abs(idx)[:, None] <= keep) & (np.abs(idx)[None, :] <= keep)
    spec[mask] = (rng.standard_normal(mask.sum()) + 1j * rng.standard_normal(mask.sum()))
    return torch.from_numpy(np.fft.ifft2(spec))


def test_propagating_zero_distance_changes_nothing():
    u = _band_limited_field(0)
    assert torch.abs(propagate(u, PITCH, LAM, 0.0) - u).max() < 1e-12


@pytest.mark.parametrize("z", [1e-4, 1e-3, 5e-3])
def test_propagating_forward_then_back_returns_the_field(z):
    u = _band_limited_field(1)
    back = propagate(propagate(u, PITCH, LAM, z), PITCH, LAM, -z)
    assert torch.abs(back - u).max() / torch.abs(u).max() < 1e-12


@pytest.mark.parametrize("z", [1e-4, 2e-3])
def test_energy_is_conserved(z):
    u = _band_limited_field(2)
    before = (torch.abs(u) ** 2).sum()
    after = (torch.abs(propagate(u, PITCH, LAM, z)) ** 2).sum()
    assert torch.abs(after - before) / before < 1e-12


def test_propagation_composes():
    """
    Propagating z1 then z2 must equal propagating z1+z2 in one step.

    The tolerance is derived, not chosen. The transfer function's phase is
    2*pi*z/lam radians, which at z = 2 mm is about 23600; double precision
    resolves that to eps times its magnitude, around 5e-12, and no
    implementation can do better. Asserting 1e-12 here would be asserting
    something arithmetic cannot deliver.
    """
    z1, z2 = 7e-4, 1.3e-3
    u = _band_limited_field(3)
    two_steps = propagate(propagate(u, PITCH, LAM, z1), PITCH, LAM, z2)
    one_step = propagate(u, PITCH, LAM, z1 + z2)

    phase_magnitude = 2 * np.pi * (z1 + z2) / LAM
    tolerance = 10 * np.finfo(float).eps * phase_magnitude
    assert torch.abs(two_steps - one_step).max() / torch.abs(u).max() < tolerance


def test_a_plane_wave_only_acquires_a_piston_phase():
    """A normally incident plane wave stays uniform, gaining exp(2i pi z/lam)."""
    u = torch.ones((N, N), dtype=torch.complex128)
    z = 3e-4
    out = propagate(u, PITCH, LAM, z)
    expected = np.exp(2j * np.pi * z / LAM)
    assert torch.abs(out - expected).max() < 1e-10


def test_a_tilted_plane_wave_shifts_sideways_by_the_right_amount():
    """
    A single spatial frequency is a plane wave at sin(theta) = lam*fx. Over a
    distance z it must translate by z*tan(theta), which is the shear the
    rendering equation predicts.
    """
    k = 3
    x = np.arange(N) * PITCH
    fx = np.fft.fftfreq(N, d=PITCH)[k]
    u = torch.from_numpy(np.exp(2j * np.pi * fx * x)[None, :].repeat(N, 0))

    sin_t = LAM * fx
    z = 2e-3
    shift = z * sin_t / np.sqrt(1 - sin_t ** 2)          # z * tan(theta)

    out = propagate(u, PITCH, LAM, z).numpy()
    ref = np.exp(2j * np.pi * fx * (x - shift))[None, :].repeat(N, 0)
    ref = ref * (out[0, 0] / ref[0, 0])                  # ignore the piston
    assert np.abs(out - ref).max() < 1e-8


@pytest.mark.parametrize("z", [5e-4, 2e-3])
def test_the_transfer_function_matches_the_wigner_shear_derived_separately(z):
    """
    The phase of the transfer function must be the paraxial shear that section
    3 of the rendering equation derives symbolically: a linear phase in the
    frequency variable whose slope is lam*z. Checked on the paraxial form,
    since that is the regime in which the shear is exact.
    """
    h = transfer_function((N, N), PITCH, LAM, z, paraxial=True).numpy()
    fx = np.fft.fftfreq(N, d=PITCH)

    measured = np.unwrap(np.angle(h[0, :]))
    quadratic = -np.pi * LAM * z * fx ** 2 + 2 * np.pi * z / LAM
    assert np.ptp(np.abs(np.unwrap(measured - quadratic))) < 1e-8

    shear = wigner_shear_reference(N, PITCH, LAM, z)
    assert np.allclose(shear, LAM * z * fx / PITCH)


def test_evanescent_components_are_discarded_not_amplified():
    """
    Beyond fx = 1/lam the square root turns imaginary. Keeping those terms
    makes the operator blow up exponentially; they must be zeroed.
    """
    fine = 0.2e-6                                    # pitch below lam/2
    h = transfer_function((N, N), fine, LAM, 1e-3)
    assert torch.abs(h).max() <= 1.0 + 1e-12
    fy, fx = np.meshgrid(np.fft.fftfreq(N, d=fine), np.fft.fftfreq(N, d=fine), indexing="ij")
    evanescent = (fy ** 2 + fx ** 2) >= 1.0 / LAM ** 2
    assert evanescent.any()
    assert np.abs(h.numpy()[evanescent]).max() == 0.0


def test_gradients_flow_through_propagation():
    """The optimiser differentiates through this, so it must be autograd-safe."""
    phase = torch.zeros((N, N), dtype=torch.float64, requires_grad=True)
    field = torch.exp(1j * phase)
    out = propagate(field, PITCH, LAM, 1e-3)
    (torch.abs(out) ** 2).sum().backward()
    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
