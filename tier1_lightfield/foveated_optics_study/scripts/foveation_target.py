"""R = 6 radial foveation target for the remapper.

A lens at panel radius r must appear at field eccentricity theta, with
dr/dtheta = F(theta), the local focal length. F follows the Watson (temporal
meridian) midget-RGC pitch, F = F0 * p(0) / p(theta), capped at F0 / R, and F0
is set so that theta = 35 deg (the horizontal field edge) lands on the panel
edge (screen_spec). The mapping is radial about the optical axis and keeps the
azimuth, so the 70 x 45 deg field fits on the square panel.
"""
from __future__ import annotations

import numpy as np

import screen_spec as spec
from retina_model import one_mosaic_spacing_deg, square_equivalent_pitch_deg

RATIO = 6.0
HALF_PANEL_MM = spec.PANEL_MM / 2.0
EDGE_RAD = np.radians(35.0)
LENS_NEIGHBOUR_MM = spec.LENS_PITCH_UM * 1e-3

_TH = np.linspace(0.0, np.radians(60.0), 60001)
_P0 = one_mosaic_spacing_deg(0.0, "temporal")
_S = np.maximum(_P0 / one_mosaic_spacing_deg(np.degrees(_TH), "temporal"), 1.0 / RATIO)
_R_UNIT = np.concatenate([[0.0], np.cumsum(0.5 * (_S[1:] + _S[:-1]) * np.diff(_TH))])
F0_MM = HALF_PANEL_MM / np.interp(EDGE_RAD, _TH, _R_UNIT)


def local_focal_mm(theta_rad):
    th = np.asarray(theta_rad, dtype=float)
    s = np.maximum(_P0 / one_mosaic_spacing_deg(np.degrees(th), "temporal"), 1.0 / RATIO)
    return F0_MM * s


def panel_radius_mm(theta_rad):
    return F0_MM * np.interp(theta_rad, _TH, _R_UNIT)


def eccentricity_rad(radius_mm):
    return np.interp(np.asarray(radius_mm) / F0_MM, _R_UNIT, _TH)


def target_pitch_rad(theta_rad):
    """Angular spacing between neighbouring lens centres at this eccentricity."""
    return LENS_NEIGHBOUR_MM / local_focal_mm(theta_rad)


def retina_pitch_rad(theta_x_rad, theta_z_rad):
    """Watson square-equivalent midget-RGC pitch in the direction (theta_x, theta_z)."""
    return np.radians(square_equivalent_pitch_deg(np.degrees(theta_x_rad), np.degrees(theta_z_rad)))


def blur_tolerance_rad(theta_x_rad, theta_z_rad):
    """Blur the eye cannot see: the larger of the retina's own pitch there and
    the display's sampling pitch there (detail finer than either is gone anyway)."""
    ecc = np.arctan(np.hypot(np.tan(theta_x_rad), np.tan(theta_z_rad)))
    return np.maximum(retina_pitch_rad(theta_x_rad, theta_z_rad), target_pitch_rad(ecc))


def field_to_panel_mm(theta_x, theta_y):
    """Field direction (tangent-plane angles about +Y) -> panel (x, y) in mm."""
    tx, ty = np.asarray(theta_x, float), np.asarray(theta_y, float)
    ecc = np.arctan(np.hypot(np.tan(tx), np.tan(ty)))
    az = np.arctan2(np.tan(ty), np.tan(tx))
    r = panel_radius_mm(ecc)
    return r * np.cos(az), r * np.sin(az)
