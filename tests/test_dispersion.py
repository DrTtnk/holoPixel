"""Per-channel refractive indices for narrow-band primaries: each element keeps
its design index at the d line and takes its material's dispersion around it."""
import sys
from pathlib import Path

import numpy as np
import pytest
from optiland.materials import Material

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import dispersion as dp  # noqa: E402


def _n(name, ref, um):
    m = Material(name, reference=ref) if ref else Material(name)
    return float(np.asarray(m.n(um)).ravel()[0])


@pytest.mark.parametrize("material, name, ref", [("nlasf46b", "N-LASF46B", "schott"), ("silica", "SiO2", "malitson")])
def test_the_design_index_holds_at_d_and_the_power_follows_the_material(material, name, ref):
    """n'(lambda) - 1 = (n_base - 1) (n(lambda) - 1) / (n(d) - 1): the element's power
    scales with its material's (n - 1), so its focal shift is the material's."""
    base = 1.9 if material == "nlasf46b" else 1.5
    got = dp.channel_indices(base, material)
    nd = _n(name, ref, dp.D_LINE_UM)
    for n_c, um in zip(got, dp.PRIMARIES_UM):
        assert n_c - 1.0 == pytest.approx((base - 1.0) * (_n(name, ref, um) - 1.0) / (nd - 1.0), rel=1e-12)
    assert dp.channel_indices(base, material, (dp.D_LINE_UM,) * 3) == pytest.approx((base,) * 3, rel=1e-12)


def test_blue_bends_most():
    r, g, b = dp.channel_indices(1.9, "nlasf46b")
    assert r < g < b
