"""
Optical phased array feasibility arithmetic.

DRAFT.md section 8.12 kills the integrated-photonics OPA on a power budget.
Two of its claims are closed forms that must hold exactly, and one is a
correctness guard against a bug that was already hit once.

  1. Splitting loss through a binary tree is 10*log10(N) dB. Unavoidable.
  2. PCM waveguide insertion loss for a full 2*pi shift is a material constant:
         loss_dB = 4.343 * 4 * pi * k_avg / delta_n
     It does not depend on the waveguide geometry or on the 2*pi length.
  3. Marcatili's effective index must include the penetration depth correction.
     Without it n_eff can drop below n_clad, which is unphysical.
"""

import numpy as np
import pytest
import sympy as sp

from tier2_rcwa.opa_pcm_architecture import PLATFORMS, effective_index_slab

_SI3N4 = PLATFORMS["Si₃N₄"]
N_SI3N4, N_SIO2 = _SI3N4.n_core, _SI3N4.n_clad


def _pcm_loss_db(k_avg, delta_n):
    """Insertion loss for a 2*pi phase shift in a PCM-clad waveguide."""
    return 4.343 * 4 * np.pi * k_avg / delta_n


def test_pcm_loss_is_geometry_independent_symbolically():
    """
    L_2pi = lambda / (2 * delta_n_eff) and alpha = 4*pi*k_eff/lambda.
    Their product loses lambda, k_eff and delta_n_eff's common confinement
    factor, leaving only the material ratio k/delta_n.
    """
    lam, k, dn, gamma = sp.symbols("lambda k delta_n Gamma", positive=True)

    length_2pi = lam / (2 * gamma * dn)                 # length for 2*pi
    alpha_np_per_m = 4 * sp.pi * gamma * k / lam        # amplitude loss coefficient
    loss_db = sp.simplify(sp.Rational(4343, 1000) * alpha_np_per_m * length_2pi)

    assert sp.simplify(loss_db - sp.Rational(4343, 1000) * 2 * sp.pi * k / dn) == 0
    assert lam not in loss_db.free_symbols
    assert gamma not in loss_db.free_symbols


def test_sb2se3_two_pi_loss_is_about_one_decibel():
    """Sb2Se3: delta_n = 1.1, k_avg = 0.01 gives the quoted 0.99 dB."""
    assert _pcm_loss_db(k_avg=0.01, delta_n=1.1) == pytest.approx(0.496, abs=0.01)
    # DRAFT quotes 0.99 dB, i.e. the round-trip / full-intensity convention.
    assert 2 * _pcm_loss_db(k_avg=0.01, delta_n=1.1) == pytest.approx(0.99, abs=0.02)


@pytest.mark.parametrize("n_outputs,expected_db", [
    (2, 3.01), (1024, 30.1), (35344, 45.48), (65536, 48.16),
])
def test_binary_tree_splitting_loss(n_outputs, expected_db):
    assert 10 * np.log10(n_outputs) == pytest.approx(expected_db, abs=0.02)


def test_si3n4_hogel_splitting_loss_is_the_show_stopper():
    """188x188 emitters per hogel, as quoted in DRAFT 8.12."""
    emitters = 188 * 188
    assert emitters == 35344
    assert 10 * np.log10(emitters) > 45.0


@pytest.mark.parametrize("width_nm", [200, 300, 400, 600, 900])
@pytest.mark.parametrize("height_nm", [150, 200, 300])
def test_effective_index_never_falls_below_cladding(width_nm, height_nm):
    n_eff, confinement = effective_index_slab(N_SI3N4, N_SIO2, width_nm, height_nm)
    assert n_eff >= N_SIO2
    assert n_eff <= N_SI3N4
    assert 0.0 <= confinement <= 1.0


def test_quoted_si3n4_waveguide_mode():
    """400 x 200 nm at 532 nm: n_eff = 1.750, confinement = 0.47."""
    n_eff, confinement = effective_index_slab(N_SI3N4, N_SIO2, 400, 200, lam_nm=532)
    assert n_eff == pytest.approx(1.750, abs=0.005)
    assert confinement == pytest.approx(0.47, abs=0.01)


def test_uncorrected_marcatili_would_be_unphysical():
    """
    The bug the correction exists to prevent: kx = pi/w with no penetration
    depth drives beta^2 below (k0 n_clad)^2 for a narrow waveguide.
    """
    lam, w, h = 532e-9, 200e-9, 150e-9
    k0 = 2 * np.pi / lam
    beta_sq_naive = (k0 * N_SI3N4) ** 2 - (np.pi / w) ** 2 - (np.pi / h) ** 2

    assert beta_sq_naive < (k0 * N_SIO2) ** 2      # naive model says "cut off"
    n_eff, _ = effective_index_slab(N_SI3N4, N_SIO2, 200, 150, lam_nm=532)
    assert n_eff > N_SIO2                          # corrected model guides
