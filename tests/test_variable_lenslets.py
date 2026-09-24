"""The retina-matched variable-focal lenslet array, as the searches and the
exporters use it."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import foveation_target as ft  # noqa: E402
import variable_lenslets as vl  # noqa: E402


def test_a_flipped_array_is_the_mirror_image_in_v():
    u, v = np.array([0.0, 3000.0, -5000.0, 8000.0]), np.array([0.0, 4000.0, -6000.0, 2000.0])
    assert vl.vertex_height_um(u, v, -1) == pytest.approx(vl.vertex_height_um(u, -v, 1), rel=1e-12)
    assert vl.focal_of_position(-1)(u, v) == pytest.approx(ft.lenslet_focal_um(u, -v), rel=1e-12)
    with pytest.raises(ValueError):
        vl.focal_of_position(0)


def test_the_bowl_table_is_the_vertex_surface_relative_to_the_panel_centre():
    gu, gv, table = vl.bowl_table(1)
    i, j = np.argmin(np.abs(gu - 3.0)), np.argmin(np.abs(gv + 2.0))
    expected = 1e-3 * (vl.vertex_height_um(gu[i] * 1e3, gv[j] * 1e3, 1) - vl.vertex_height_um(0.0, 0.0, 1))
    assert table[i, j] == pytest.approx(expected, rel=1e-12)
    assert np.all(np.isfinite(table))
    assert gu[0] < -ft.HALF_PANEL_MM and gu[-1] > ft.HALF_PANEL_MM     # covers the panel


def test_every_lens_vertex_lies_on_the_vertex_surface():
    """The array built by build() has its lens vertices on vertex_height_um (checked
    on a sample of lenses; the full array is large)."""
    arr = vl.build(1, subdivisions=2)
    rng = np.random.default_rng(0)
    k = rng.choice(len(arr.mesh.centres), 2000, replace=False)
    c = arr.mesh.centres[k]
    assert arr.centre_height_um[k] == pytest.approx(vl.vertex_height_um(c[:, 0], c[:, 1], 1), abs=1e-9)


def test_a_72_um_array_has_a_quarter_of_the_lenses_on_its_own_vertex_surface():
    a36, a72 = vl.build(1), vl.build(1, pitch_um=72.0)
    assert len(a72.mesh.centres) == pytest.approx(len(a36.mesh.centres) / 4.0, rel=0.03)
    c = a72.mesh.centres
    assert a72.centre_height_um == pytest.approx(vl.vertex_height_um(c[:, 0], c[:, 1], 1, pitch_um=72.0), abs=1e-9)
    assert a72.focal_um == pytest.approx(vl.focal_of_position(1, pitch_um=72.0)(c[:, 0], c[:, 1]), rel=1e-12)
