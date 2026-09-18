"""
What an eye at a given pupil position sees of the panel.

A view is formed in three steps: propagate the panel field to the pupil plane,
admit only the light falling inside the pupil, and let the eye's lens form an
image on the retina. A lens in its focal plane performs a Fourier transform,
so the retinal field is the transform of the pupil-plane field restricted to
the aperture.

Moving the pupil across the eyebox therefore selects different parts of the
panel's angular spectrum, and that is exactly what makes different views
different. No depth is involved anywhere: the target is a set of intensities
and so is the output.

Accommodation is a quadratic phase added inside the pupil. An eye focused at
distance d applies exp(-i pi r^2 / (lam d)), so sweeping d sweeps focus and
lets the reconstruction be scored on whether it blurs correctly, not only on
whether one plane is sharp.
"""

import numpy as np
import torch

from tier2_hfh.propagate import propagate


def pupil_mask(shape, pitch, diameter, centre, device=None, dtype=torch.float64):
    """Circular aperture of the given diameter, centred at `centre` metres."""
    ny, nx = shape
    y = (torch.arange(ny, device=device, dtype=dtype) - ny / 2) * pitch - centre[0]
    x = (torch.arange(nx, device=device, dtype=dtype) - nx / 2) * pitch - centre[1]
    r2 = y[:, None] ** 2 + x[None, :] ** 2
    return (r2 <= (diameter / 2) ** 2).to(dtype)


def defocus_phase(shape, pitch, wavelength, focus_distance, centre,
                  device=None, dtype=torch.float64):
    """
    Quadratic phase for an eye accommodated at `focus_distance`. An infinite
    distance is a relaxed eye and contributes nothing.
    """
    if not np.isfinite(focus_distance):
        return torch.ones(shape, dtype=torch.complex128, device=device)
    ny, nx = shape
    y = (torch.arange(ny, device=device, dtype=dtype) - ny / 2) * pitch - centre[0]
    x = (torch.arange(nx, device=device, dtype=dtype) - nx / 2) * pitch - centre[1]
    r2 = y[:, None] ** 2 + x[None, :] ** 2
    return torch.exp(-1j * np.pi * r2 / (wavelength * focus_distance))


def retinal_intensity(panel_field, pitch, wavelength, eye_distance,
                      pupil_diameter, pupil_centre, focus_distance=np.inf):
    """
    Intensity an eye sees, for one pupil position. `pupil_centre` is (y, x) in
    metres from the optical axis. Returns a real image, fftshifted so the
    optical axis is at the centre.
    """
    at_pupil = propagate(panel_field, pitch, wavelength, eye_distance)
    shape = panel_field.shape[-2:]
    aperture = pupil_mask(shape, pitch, pupil_diameter, pupil_centre,
                          device=panel_field.device)
    lens = defocus_phase(shape, pitch, wavelength, focus_distance, pupil_centre,
                         device=panel_field.device)
    retinal = torch.fft.fftshift(torch.fft.fft2(at_pupil * aperture * lens),
                                 dim=(-2, -1))
    return torch.abs(retinal) ** 2


def pupil_positions(n_views, eyebox, device=None):
    """
    The (y, x) pupil centres matching an n_views x n_views light field taken
    over an eyebox of the given width. Returned in the same row-major order as
    the rendered light field, so view (v, u) corresponds to index v*n + u.
    """
    p = torch.linspace(-eyebox / 2, eyebox / 2, n_views, dtype=torch.float64,
                       device=device)
    return torch.stack(torch.meshgrid(p, p, indexing="ij"), dim=-1).reshape(-1, 2)
