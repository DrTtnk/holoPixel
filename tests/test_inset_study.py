"""Foveal-inset study: the dispersion model and the residuals it optimises."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = (Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
        / "remapper_designs" / "foveal_inset")
sys.path.insert(0, str(HERE))

import inset_study as ins  # noqa: E402


@pytest.mark.parametrize("material", ["PMMA", "PC"])
def test_cauchy_index_reproduces_nd_and_the_abbe_number(material):
    nd, v = ins.MATERIALS[material]
    assert ins.index(material, 0.5876) == pytest.approx(nd, abs=1e-12)
    assert (nd - 1.0) / (ins.index(material, 0.4861) - ins.index(material, 0.6563)) == pytest.approx(v, rel=1e-12)


def test_a_thick_plano_convex_singlet_has_its_paraxial_focal_length():
    """Residual machinery on a known lens: R = 26 mm PMMA, convex front, flat
    back: f = R / (n - 1) (the flat back adds no power), and on the back focal
    plane, f - t / n behind the lens, every ray at angle theta lands at f tan
    theta, so the traced local focal length at the axis is f."""
    lay = ins.layout(["PMMA"])
    radius = 26.0
    x = np.zeros(lay["size"])
    n_g = ins.index("PMMA", 0.530)
    f = radius / (n_g - 1.0)
    x[0], x[1], x[2] = 18.0, 4.0, f - 4.0 / n_g
    x[3] = ins.R0_MM**2 / (2 * radius)                                   # front s2
    _, info = ins.residuals(torch.tensor(x, dtype=torch.float64), lay)
    assert float(info["focal"][0]) == pytest.approx(f, rel=1e-4)


def test_the_achromat_seed_has_the_target_power_and_no_first_order_colour():
    """Thin-lens check of seed_radii for PMMA + PC: total power 1 / 52.3 mm and
    sum(phi_i / V_i) = 0 (the achromat condition)."""
    r1, r2, r3, r4 = ins.seed_radii(["PMMA", "PC"])
    (n1, v1), (n2, v2) = ins.MATERIALS["PMMA"], ins.MATERIALS["PC"]
    phi1 = (n1 - 1.0) * (1.0 / r1 - 1.0 / r2)
    phi2 = (n2 - 1.0) * (1.0 / r3 - 1.0 / r4)
    assert phi1 + phi2 == pytest.approx(1.0 / 52.3, rel=1e-12)
    assert phi1 / v1 + phi2 / v2 == pytest.approx(0.0, abs=1e-15)
