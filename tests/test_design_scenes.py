"""The multi-design display file: per-channel design copies, and the ST-map that
tells each panel pixel which content direction it must show."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import design_scenes as ds  # noqa: E402
import dispersion as dp  # noqa: E402


def _views(tmp_path, direction, per_view):
    np.save(tmp_path / "direction.npy", np.asarray(direction, dtype=np.float32))
    for k, pix in enumerate(per_view):
        np.save(tmp_path / f"pix_{k}.npy", np.asarray(pix, dtype=np.int32))


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_the_stmap_is_each_pixels_mean_direction_over_the_pupil(tmp_path):
    """Pixel 3 is reached from two views in two directions: its ST-map entry is
    the field angle of their mean; pixel 5 once; every other pixel unseen."""
    a, b, c = _unit([math.tan(math.radians(2.0)), 1.0, 0.0]), _unit([math.tan(math.radians(4.0)), 1.0, 0.0]), \
        _unit([0.0, 1.0, math.tan(math.radians(-7.0))])
    _views(tmp_path, [[a, b, c]], [[[3, -1, 5]], [[-1, 3, -1]]])
    st, seen = ds.stmap([tmp_path], panel_pixels=4)
    mean = _unit(a + b)                                              # directions are stored float32: 1e-6 deg
    assert st.shape == (4, 4, 2) and seen.shape == (4, 4)
    assert st[0, 3] == pytest.approx([math.degrees(math.atan2(mean[0], mean[1])), 0.0], abs=1e-6)
    assert st[1, 1] == pytest.approx([0.0, -7.0], abs=1e-6)                # pixel 5: row 1, column 1
    assert seen.sum() == 2 and seen[0, 3] and seen[1, 1]


def test_a_channel_design_carries_that_channels_glass_index(tmp_path):
    design = tmp_path / "design"
    design.mkdir()
    np.savez(design / "remapper.npz", n_surfaces=2, surf0_kind="mirror", surf0_verts=np.zeros((3, 3)),
             surf0_faces=np.array([[0, 1, 2]]), surf1_kind="glass", surf1_verts=np.zeros((3, 3)),
             surf1_faces=np.array([[0, 1, 2]]), surf1_index=1.9)
    (design / "design.json").write_text(json.dumps({"remapper_npz": "remapper.npz", "x": 1}))
    for c in range(3):
        out = ds.channel_design(design, tmp_path / f"ch{c}", c)
        r = np.load(out / "remapper.npz")
        assert float(r["surf1_index"]) == pytest.approx(dp.channel_indices(1.9, ds.LENS_MATERIAL)[c], rel=1e-12)
        assert "surf0_index" not in r
        assert json.loads((out / "design.json").read_text()) == {"remapper_npz": "remapper.npz", "x": 1}
    dispersive = np.load(ds.dispersive_remapper(design, tmp_path / "disp.npz"))
    assert tuple(dispersive["surf1_index"]) == pytest.approx(dp.channel_indices(1.9, ds.LENS_MATERIAL), rel=1e-12)
