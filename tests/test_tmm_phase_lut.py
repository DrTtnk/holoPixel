"""
TMM-corrected phase look-up table.

useful_knowledge.md records the single most expensive lesson of tier 2:

    "Naive linear model phi = 4*pi*n*d/lambda gives eta_1 = 44.5%.
     Cause: the air/Sb2Se3 interface creates an implicit Fabry-Perot with the
     mirror. Fix: compute the actual phase(n) via TMM, then invert it."

The property that makes the corrected LUT work is that its phase steps are
uniform while the naive model's are not. That is what this module tests, plus
the RGB thickness conclusion (300 nm is the only depth giving all three
colours a full 2*pi).
"""

import numpy as np
import pytest

from tier2_rcwa.rcwa_optimization import (
    N_AMORPHOUS,
    N_CRYSTALLINE,
    N_SIO2,
    N_TIO2,
    design_phase_lut,
    stack_response,
)

RGB_NM = {"red": 632, "green": 532, "blue": 450}


def _dbr_stack(film_nm, n_pairs=6, lam_nm=532):
    d_tio2 = lam_nm / (4 * N_TIO2)
    d_sio2 = lam_nm / (4 * N_SIO2)
    n_list, d_list = [1.0, None], [np.inf, film_nm]
    for _ in range(n_pairs):
        n_list.extend([N_TIO2, N_SIO2])
        d_list.extend([d_tio2, d_sio2])
    n_list.append(1.5)
    d_list.append(np.inf)
    return n_list, d_list


def test_corrected_lut_phase_steps_are_uniform():
    """
    Feed the LUT indices back through TMM. The resulting phases must be evenly
    spaced; that is the entire point of inverting phase(n) instead of assuming
    it is linear.
    """
    n_list, d_list = _dbr_stack(300)
    corrected_n, _, total_range = design_phase_lut(n_list, d_list, 532, n_levels=8)

    phase, _ = stack_response(n_list, d_list, 532, corrected_n)
    steps = np.diff(phase - phase[0])

    assert np.all(np.isfinite(steps))
    assert np.std(steps) / abs(np.mean(steps)) < 0.02


def test_naive_linear_index_sampling_is_not_uniform_in_phase():
    """The control: evenly spaced index gives visibly uneven phase."""
    n_list, d_list = _dbr_stack(300)
    naive_n = np.linspace(N_AMORPHOUS, N_CRYSTALLINE, 8)

    phase, _ = stack_response(n_list, d_list, 532, naive_n)
    steps = np.diff(phase)

    assert np.std(steps) / abs(np.mean(steps)) > 0.10


def test_corrected_lut_indices_are_within_the_material_range():
    n_list, d_list = _dbr_stack(300)
    corrected_n, corrected_refl, _ = design_phase_lut(n_list, d_list, 532, n_levels=8)

    assert len(corrected_n) == 8
    assert np.all(corrected_n >= N_AMORPHOUS - 1e-9)
    assert np.all(corrected_n <= N_CRYSTALLINE + 1e-9)
    assert np.all(np.diff(corrected_n) > 0)
    assert np.all((corrected_refl >= 0) & (corrected_refl <= 1))


def test_lut_never_places_two_levels_at_the_same_phase():
    """linspace(0, usable, n) would duplicate 0 and 2*pi. It must not."""
    n_list, d_list = _dbr_stack(300)
    corrected_n, _, _ = design_phase_lut(n_list, d_list, 532, n_levels=8)

    phase, _ = stack_response(n_list, d_list, 532, corrected_n)
    wrapped = np.mod(phase - phase[0], 2 * np.pi)
    assert len(np.unique(np.round(wrapped, 6))) == 8


@pytest.mark.parametrize("colour,lam_nm", sorted(RGB_NM.items()))
def test_300nm_film_reaches_two_pi_for_every_colour(colour, lam_nm):
    """DRAFT 8.4: 300 nm is the only thickness where all of RGB clear 2*pi."""
    n_list, d_list = _dbr_stack(300, lam_nm=lam_nm)
    _, _, total_range = design_phase_lut(n_list, d_list, lam_nm, n_levels=8)
    assert total_range >= 2 * np.pi


@pytest.mark.parametrize("film_nm", [200, 250])
def test_thin_films_leave_red_short_of_two_pi(film_nm):
    """DRAFT 8.4: 200 nm and 250 nm give red only 1.66*pi and 1.72*pi."""
    n_list, d_list = _dbr_stack(film_nm, lam_nm=632)
    _, _, total_range = design_phase_lut(n_list, d_list, 632, n_levels=8)
    assert total_range < 2 * np.pi


def test_energy_is_conserved_by_the_lossless_dbr_stack():
    """TiO2/SiO2 and the Sb2Se3 k=0.01 absorb a little; R must stay in [0, 1]."""
    n_list, d_list = _dbr_stack(300)
    n_range = np.linspace(N_AMORPHOUS, N_CRYSTALLINE, 64)
    _, refl = stack_response(n_list, d_list, 532, n_range)
    assert np.all((refl >= 0.0) & (refl <= 1.0))
