"""FROZEN radial foveation target (2026-09-24), kept for the historical agent
designs (remapper_designs/prism, remapper_designs/coaxial) and the
uniform-lenslet coaxial candidates. New work uses foveation_target.

Retina-matched radial foveation target for the remapper.

A lens at panel radius r must appear at field eccentricity theta, with
dr/dtheta = F(theta), the local focal length. F follows the Watson (temporal
meridian) midget-RGC pitch everywhere, F = F0 * p(0) / p(theta), and F0 is set
so that theta = 35 deg (the horizontal field edge) lands on the panel edge
(screen_spec). The display then samples the field at one fixed multiple of the
retinal pitch (SAMPLING_OVER_RETINA, about 2 on this panel). The mapping is
radial about the optical axis and keeps the azimuth, so the 70 x 45 deg field
fits on the square panel.
"""
from __future__ import annotations

import numpy as np

import screen_spec as spec
from retina_model import one_mosaic_spacing_deg, square_equivalent_pitch_deg

HALF_PANEL_MM = spec.PANEL_MM / 2.0
EDGE_RAD = np.radians(35.0)
LENS_NEIGHBOUR_MM = spec.LENS_PITCH_UM * 1e-3

_TH = np.linspace(0.0, np.radians(60.0), 60001)
_P0 = one_mosaic_spacing_deg(0.0, "temporal")
_S = _P0 / one_mosaic_spacing_deg(np.degrees(_TH), "temporal")
_R_UNIT = np.concatenate([[0.0], np.cumsum(0.5 * (_S[1:] + _S[:-1]) * np.diff(_TH))])
F0_MM = HALF_PANEL_MM / np.interp(EDGE_RAD, _TH, _R_UNIT)
SAMPLING_OVER_RETINA = LENS_NEIGHBOUR_MM / F0_MM / np.radians(_P0)
WAVELENGTH_MM = 0.55e-3
# RMS radius (sqrt(2) sigma) of the least-squares Gaussian fit to the Airy pattern:
# the blur metric is an RMS radius, and the Airy pattern's own diverges.
DIFFRACTION_RMS_RAD = np.sqrt(2.0) * 0.42 * WAVELENGTH_MM / spec.PUPIL_DIAMETER_MM


def local_focal_mm(theta_rad):
    th = np.asarray(theta_rad, dtype=float)
    return F0_MM * _P0 / one_mosaic_spacing_deg(np.degrees(th), "temporal")


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
    """Blur the eye cannot see: the retina's own pitch there, but never below
    the RMS diffraction blur of the design pupil, which no optics can beat."""
    return np.maximum(retina_pitch_rad(theta_x_rad, theta_z_rad), DIFFRACTION_RMS_RAD)


def lenslet_focal_um():
    """Lenslet focal length at which the pupil's image under a lenslet (f D / F)
    just fills one lens pitch at the field edge, where F is smallest."""
    return spec.LENS_PITCH_UM * float(local_focal_mm(EDGE_RAD)) / spec.PUPIL_DIAMETER_MM


def field_to_panel_mm(theta_x, theta_y):
    """Field direction (tangent-plane angles about +Y) -> panel (x, y) in mm."""
    tx, ty = np.asarray(theta_x, float), np.asarray(theta_y, float)
    ecc = np.arctan(np.hypot(np.tan(tx), np.tan(ty)))
    az = np.arctan2(np.tan(ty), np.tan(tx))
    r = panel_radius_mm(ecc)
    return r * np.cos(az), r * np.sin(az)


# The steepest usable plano-convex lenslet: its sphere must reach past the hex
# corner (and past the panel-border lattice triangles, up to ~1.1 x side).
LENSLET_MIN_RADIUS_OVER_SIDE = 1.15


def lenslet_focal_of_radius_um(r_um):
    """Variable-focal lenslet array: the lens at panel radius r gets
    f = pitch F(theta(r)) / D, so the pupil's image fills one lens pitch
    everywhere (5 x 5 views per lens), floored at the steepest usable lens."""
    wanted = spec.LENS_PITCH_UM * local_focal_mm(eccentricity_rad(np.asarray(r_um) * 1e-3)) / spec.PUPIL_DIAMETER_MM
    floor = LENSLET_MIN_RADIUS_OVER_SIDE * spec.LENS_SIDE_UM / (spec.LENS_INDEX - 1.0)
    return np.maximum(wanted, floor)
