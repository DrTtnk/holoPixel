"""Merit of a coaxial remapper, measured with optiland's own operands.

Per field angle theta (object at infinity, stop at the eye):
  height  chief-ray image height minus the target r(theta) (sign fixed by the seed)
  spot    RMS spot radius on the MLA plane (hexapolar pupil sampling)
  tilt    chief-ray angle at the MLA plane; it must stay well inside the lenslet
          acceptance, because the pupil cone already fills about +/- 0.35 rad
          of it at the field edge
Units: mm and degrees. The lens pitch is 0.030 mm, so a spot of 0.005 mm is
the goal and 0.3 mm is a failed design.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from optiland.optimization.operand import operand_registry as ops

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import foveation_target_radial as ft  # noqa: E402

import coaxial_optiland as co  # noqa: E402

MAX_FIELD_DEG = 41.6852
FIELDS_DEG = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 38.0, MAX_FIELD_DEG]
SPOT_RINGS = 6


def image_surface(lens):
    return lens.surfaces.num_surfaces - 1


def target_height_mm(theta_deg, sign):
    return sign * float(ft.panel_radius_mm(math.radians(theta_deg)))


def measure(lens, sign, fields=FIELDS_DEG):
    img = image_surface(lens)
    rows = []
    for theta in fields:
        hy = theta / MAX_FIELD_DEG
        h = float(ops.get("real_y_intercept")(lens, img, 0.0, hy, 0.0, 0.0, co.WAVELENGTH_UM))
        n = float(ops.get("real_N")(lens, img, 0.0, hy, 0.0, 0.0, co.WAVELENGTH_UM))
        spot = float(ops.get("rms_spot_size")(lens, img, 0.0, hy, SPOT_RINGS, co.WAVELENGTH_UM))
        rows.append({"theta_deg": theta, "height_mm": h, "target_mm": target_height_mm(theta, sign),
                     "height_err_mm": h - target_height_mm(theta, sign), "spot_rms_mm": spot,
                     "tilt_deg": math.degrees(math.acos(min(1.0, abs(n))))})
    return rows


def table(rows):
    lines = [f"{'theta':>7} {'height':>9} {'target':>9} {'err':>9} {'spot_rms':>9} {'tilt':>7}"]
    for r in rows:
        lines.append(f"{r['theta_deg']:7.2f} {r['height_mm']:9.4f} {r['target_mm']:9.4f} "
                     f"{r['height_err_mm']:9.4f} {r['spot_rms_mm']:9.4f} {r['tilt_deg']:7.2f}")
    spots = np.array([r["spot_rms_mm"] for r in rows])
    errs = np.array([abs(r["height_err_mm"]) for r in rows])
    lines.append(f"spot rms: median {np.median(spots):.4f} max {spots.max():.4f} mm | "
                 f"|height err| max {errs.max():.4f} mm | tilt max {max(r['tilt_deg'] for r in rows):.2f} deg")
    return "\n".join(lines)
