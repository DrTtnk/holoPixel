"""The one place the screen hardware is defined.

Panel: 1.03 inch micro-OLED (OLEDoS), 2560 x 2560, active area 18.432 x 18.432 mm,
7.2 um pixel pitch, up to 1800 cd/m^2 (displaymodule.com product page).

Lenslets: flat-top hexagons 36 um flat to flat (5 pixels). The foveation
target (foveation_target) follows the retina along every azimuth (temporal
pitch on both horizontal sides) and is stretched anamorphically over the whole
square panel; the display then samples the 70 x 45 deg field at 0.67x to
1.96x the true retinal pitch (below 1: finer than the retina). About
300,000 lenses tile the panel.
"""
import math
import os
import re

import numpy as np

PANEL_PIXELS = 2560
PIXEL_UM = 7.2
PANEL_MM = PANEL_PIXELS * PIXEL_UM * 1e-3

LENS_PITCH_UM = 36.0                      # flat to flat = neighbour spacing
LENS_SIDE_UM = LENS_PITCH_UM / math.sqrt(3.0)
LENS_INDEX = 1.5
LENS_MIN_THICKNESS_UM = 10.0

PUPIL_DIAMETER_MM = 4.0


def _field_half_deg(text):
    """ "<width>x<height>" full field in degrees -> (half width, half height)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", text)
    if m is None or float(m[1]) <= 0.0 or float(m[2]) <= 0.0:
        raise ValueError(f"HOLOPIXEL_FIELD_DEG must be <width>x<height> in degrees, like 100x80, not {text!r}")
    return float(m[1]) / 2.0, float(m[2]) / 2.0


# The target field of view (tangent-plane half-angles, deg): the single source for
# the foveation map, the searches, the exporters and the evaluator. Designs and
# results record it and are refused under another one.
FIELD_HALF_DEG = _field_half_deg(os.environ.get("HOLOPIXEL_FIELD_DEG", "70x45"))
# The field's outline: "rect" (the whole <width>x<height> box) or "ellipse" (the
# ellipse inscribed in it; the panel then carries a black mask outside the
# ellipse's image). A design records it in FIELD_DEG, after the size.
FIELD_SHAPE = os.environ.get("HOLOPIXEL_FIELD_SHAPE", "rect")
if FIELD_SHAPE not in ("rect", "ellipse"):
    raise ValueError(f"HOLOPIXEL_FIELD_SHAPE must be rect or ellipse, not {FIELD_SHAPE!r}")
FIELD_DEG = [2.0 * FIELD_HALF_DEG[0], 2.0 * FIELD_HALF_DEG[1]] + ([] if FIELD_SHAPE == "rect" else [FIELD_SHAPE])


def in_field(tx_deg, tz_deg):
    """True for the directions (tangent-plane angles, deg) inside the field."""
    x, z = np.asarray(tx_deg, float) / FIELD_HALF_DEG[0], np.asarray(tz_deg, float) / FIELD_HALF_DEG[1]
    if FIELD_SHAPE == "rect":
        return (np.abs(x) <= 1.0) & (np.abs(z) <= 1.0)
    return x**2 + z**2 <= 1.0 + 1e-12


def field_boundary_deg(n):
    """(tx, tz) deg of n points along each edge of the rectangle, or of 4 n points
    round the ellipse."""
    hx, hz = FIELD_HALF_DEG
    if FIELD_SHAPE == "rect":
        s = np.linspace(-1.0, 1.0, n)
        one = np.ones_like(s)
        return (np.concatenate([s * hx, s * hx, one * hx, -one * hx]),
                np.concatenate([one * hz, -one * hz, s * hz, s * hz]))
    a = np.linspace(0.0, 2.0 * np.pi, 4 * n, endpoint=False)
    return hx * np.cos(a), hz * np.sin(a)
