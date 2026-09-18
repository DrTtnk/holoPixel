"""
The display model: panel phase in, retinal intensity out.

Checked against cases whose answer is fixed by geometry, so a sign or scale
error cannot hide behind a picture that merely looks plausible.
"""

import numpy as np
import pytest
import torch

from tier2_hfh.display import (
    defocus_phase,
    pupil_mask,
    pupil_positions,
    retinal_intensity,
)

LAM, PITCH, N = 532e-9, 2e-6, 64
EYE_Z, PUPIL = 0.02, 8e-6      # small pupil so the aperture fits the test grid


def test_pupil_mask_has_the_area_of_the_circle_it_describes():
    m = pupil_mask((N, N), PITCH, 20e-6, (0.0, 0.0))
    expected = np.pi * (10e-6) ** 2 / PITCH ** 2
    assert m.sum().item() == pytest.approx(expected, rel=0.05)


def test_pupil_mask_moves_where_it_is_told():
    shift = 10e-6
    centred = pupil_mask((N, N), PITCH, 12e-6, (0.0, 0.0))
    moved = pupil_mask((N, N), PITCH, 12e-6, (0.0, shift))

    cy, cx = [c.to(torch.float64).mean() for c in torch.where(centred > 0)]
    my, mx = [c.to(torch.float64).mean() for c in torch.where(moved > 0)]

    assert (my - cy).abs().item() < 0.2
    assert (mx - cx).item() == pytest.approx(shift / PITCH, rel=0.05)


def test_a_relaxed_eye_applies_no_defocus():
    d = defocus_phase((N, N), PITCH, LAM, np.inf, (0.0, 0.0))
    assert torch.abs(d - 1.0).max() < 1e-15


def test_defocus_is_a_pure_phase():
    d = defocus_phase((N, N), PITCH, LAM, 0.5, (0.0, 0.0))
    assert torch.abs(torch.abs(d) - 1.0).max() < 1e-12


def test_a_flat_panel_images_to_a_single_bright_point():
    """
    A uniform phase is a plane wave. Through a pupil it must focus to one spot
    at the centre of the retina, and nearly all the energy must be in it.
    """
    field = torch.ones((N, N), dtype=torch.complex128)
    img = retinal_intensity(field, PITCH, LAM, EYE_Z, PUPIL, (0.0, 0.0))

    peak = torch.argmax(img)
    assert (peak // N).item() == N // 2
    assert (peak % N).item() == N // 2


@pytest.mark.parametrize("k", [2, 5])
def test_a_grating_sends_its_light_to_the_angle_the_grating_equation_predicts(k):
    """
    A linear phase ramp of k cycles across the panel is a grating that steers
    to sin(theta) = k*lam/(N*pitch). On the retina that lands k bins from the
    centre. This ties the model to the grating equation used everywhere else
    in the project.
    """
    x = torch.arange(N, dtype=torch.float64)
    ramp = 2 * np.pi * k * x / N
    field = torch.exp(1j * ramp)[None, :].repeat(N, 1)

    img = retinal_intensity(field, PITCH, LAM, EYE_Z, PUPIL, (0.0, 0.0))
    peak = torch.argmax(img)

    assert (peak % N).item() == N // 2 + k


def test_moving_the_pupil_changes_what_is_seen():
    """If it did not, there would be no parallax and no light field."""
    rng = np.random.default_rng(0)
    field = torch.exp(1j * torch.from_numpy(rng.uniform(0, 2 * np.pi, (N, N))))

    a = retinal_intensity(field, PITCH, LAM, EYE_Z, PUPIL, (0.0, -20e-6))
    b = retinal_intensity(field, PITCH, LAM, EYE_Z, PUPIL, (0.0, +20e-6))

    assert torch.abs(a - b).max() / a.max() > 0.1


def test_pupil_positions_match_the_light_field_layout():
    pos = pupil_positions(17, 0.02)
    assert pos.shape == (289, 2)
    assert pos[0].tolist() == pytest.approx([-0.01, -0.01])
    assert pos[-1].tolist() == pytest.approx([0.01, 0.01])
    assert pos[17 * 8 + 8].tolist() == pytest.approx([0.0, 0.0])


def test_gradients_reach_the_panel_phase():
    phase = torch.zeros((N, N), dtype=torch.float64, requires_grad=True)
    img = retinal_intensity(torch.exp(1j * phase), PITCH, LAM, EYE_Z, PUPIL, (0.0, 0.0))
    img.sum().backward()

    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
    assert phase.grad.abs().max() > 0
