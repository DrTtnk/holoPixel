"""The one place the screen hardware is defined.

Panel: 1.03 inch micro-OLED (OLEDoS), 2560 x 2560, active area 18.432 x 18.432 mm,
7.2 um pixel pitch, up to 1800 cd/m^2 (displaymodule.com product page).

Lenslets: flat-top hexagons 36 um flat to flat (5 pixels), so at the field
edge, where the R = 6 target's local focal length is about 12.7 mm, one lens
spans 36 um / 12.7 mm = 2.8 mrad = 9.7 arcmin, matching the Watson retinal
pitch there (about 10 arcmin at 35 deg). About 300,000 lenses, against the
~340,000 retina-matched samples for 70 x 45 deg.
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
