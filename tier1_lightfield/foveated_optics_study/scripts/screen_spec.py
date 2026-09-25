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
FIELD_DEG = [2.0 * FIELD_HALF_DEG[0], 2.0 * FIELD_HALF_DEG[1]]
