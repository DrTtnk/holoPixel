"""The foveal chart is a ruler: each grating patch has the stated period in
arcmin, along the stated axis."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import hmd_encoded as he  # noqa: E402


@pytest.fixture(scope="module")
def chart():
    return he.foveal_chart()


@pytest.mark.parametrize("patch", he.FOVEAL_PATCHES)
def test_every_grating_has_its_period(chart, patch):
    period_arcmin, axis, (cx, cz) = patch
    n = chart.shape[0]
    px_per_deg = n / (2 * he.FOVEAL_HALF_DEG)
    half = he.PATCH_DEG / 2 * 0.8
    to_px = lambda deg: int(round((deg + he.FOVEAL_HALF_DEG) * px_per_deg))  # noqa: E731
    block = chart[to_px(cz - half):to_px(cz + half), to_px(cx - half):to_px(cx + half)].mean(-1)
    profile = block.mean(0) if axis == "x" else block.mean(1)            # bars across the given axis
    pad = 32 * len(profile)                                              # a few cycles only: interpolate the peak
    spectrum = np.abs(np.fft.rfft(profile - profile.mean(), n=pad))
    freq = np.fft.rfftfreq(pad, d=1.0 / px_per_deg)                      # cycles per degree
    assert 60.0 / freq[np.argmax(spectrum)] == pytest.approx(period_arcmin, rel=0.08)
