"""
Scalar wave propagation, panel to eye, in torch so it is differentiable.

The angular spectrum method. A field is decomposed into plane waves, each
acquires the phase it earns over distance z, and the field is reassembled:

    U_z = F^-1 { F{U} * exp(2i pi z sqrt(1/lam^2 - fx^2 - fy^2)) }

This is exact for scalar fields, unlike the Fresnel kernel, which is its
paraxial expansion. Section 11 of docs/holographic_rendering_equation.md notes
our field of view is not comfortably paraxial -- tan and sin differ by 18% at
32 degrees -- so the exact transfer function is used and the paraxial one is
offered only for comparison.

Evanescent components, where fx^2 + fy^2 > 1/lam^2, do not propagate. They are
zeroed rather than allowed to grow exponentially.

The connection to the rest of the repository: section 3 of the rendering
equation proves free-space propagation shears the Wigner distribution by
exactly lam*z, verified there to 5e-16. `tests/test_propagation.py` checks
this implementation against that same shear, so the two derivations have to
agree or one of them is wrong.
"""

import numpy as np
import torch


def frequency_grid(shape, pitch, device=None, dtype=torch.float64):
    """Spatial frequency coordinates in cycles per metre."""
    ny, nx = shape
    fy = torch.fft.fftfreq(ny, d=pitch, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(nx, d=pitch, device=device, dtype=dtype)
    return torch.meshgrid(fy, fx, indexing="ij")


def transfer_function(shape, pitch, wavelength, z, device=None, paraxial=False,
                      dtype=torch.float64):
    """
    Angular-spectrum transfer function for a propagation of z metres.
    Evanescent orders are set to zero, so the operator never amplifies.
    """
    fy, fx = frequency_grid(shape, pitch, device=device, dtype=dtype)
    f2 = fy ** 2 + fx ** 2
    inv_lam2 = 1.0 / wavelength ** 2

    if paraxial:
        phase = 2 * np.pi * z * (1.0 / wavelength - 0.5 * wavelength * f2)
        return torch.exp(1j * phase)

    propagating = f2 < inv_lam2
    kz = torch.sqrt(torch.clamp(inv_lam2 - f2, min=0.0))
    return torch.where(propagating,
                       torch.exp(2j * np.pi * z * kz),
                       torch.zeros((), dtype=torch.complex128, device=device))


def propagate(field, pitch, wavelength, z, paraxial=False):
    """
    Propagate a complex field by z metres. Accepts a leading batch axis.
    Negative z propagates backwards and is the exact inverse wherever no
    evanescent components were discarded.
    """
    h = transfer_function(field.shape[-2:], pitch, wavelength, z,
                          device=field.device, paraxial=paraxial)
    return torch.fft.ifft2(torch.fft.fft2(field) * h)


def wigner_shear_reference(n, pitch, wavelength, z):
    """
    How far the Wigner distribution shears, in samples, for the frequency
    bins of an n-sample grid. Section 3 of the rendering equation:
    W_z(x, u) = W_0(x - lam z u, u). Used by the tests to check this module
    against a result derived independently.
    """
    u = np.fft.fftfreq(n, d=pitch)
    return wavelength * z * u / pitch
