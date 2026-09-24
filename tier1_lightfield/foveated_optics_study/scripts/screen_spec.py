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

PANEL_PIXELS = 2560
PIXEL_UM = 7.2
PANEL_MM = PANEL_PIXELS * PIXEL_UM * 1e-3

LENS_PITCH_UM = 36.0                      # flat to flat = neighbour spacing
LENS_SIDE_UM = LENS_PITCH_UM / math.sqrt(3.0)
LENS_INDEX = 1.5
LENS_MIN_THICKNESS_UM = 10.0

PUPIL_DIAMETER_MM = 4.0
