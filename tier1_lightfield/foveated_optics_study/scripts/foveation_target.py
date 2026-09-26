"""Retina-matched, left-right symmetric foveation target for the remapper.

Directions are tangent-plane angles (theta_x, theta_z) about the eye's axis, or
polar (eccentricity theta, azimuth phi) with tan theta_x = tan theta cos phi,
tan theta_z = tan theta sin phi. Along every azimuth the (unscaled) panel
radius grows as

    R(theta, phi) = integral_0^theta p(0) / p(t, phi) dt,

p the Watson square-equivalent midget-RGC pitch with the temporal pitch on
both horizontal sides (so one design serves both eyes and the plane-symmetric
optics can realise it) and the superior and inferior pitches as they are. Two
constant scales and a vertical offset,

    u = GX R cos phi,   v = GZ R sin phi + V0,

stretch the 70 x 45 deg field over the whole square panel: its left and right
edges on the panel's, its top and bottom on the panel's. The map is
anamorphic (GX != GZ), so its local Jacobian is anisotropic; local_focal_mm is
its areal scale sqrt(det J).

Lenslets (lenslet_focal_um): a pixel sees the part of the pupil of side
d = pixel F / f. Each such view is diffraction-blurred to ~0.59 lambda / d RMS,
which must not exceed the retinal pitch, and the pupil's image under a lens
must fit in it (at most 5 pixels along the map's tighter axis), so the views
per lens follow the retina: about one at the fovea, 5 x 5 from ~2 deg out.
"""
from __future__ import annotations

import numpy as np

import screen_spec as spec
from retina_model import one_mosaic_spacing_deg

HALF_PANEL_MM = spec.PANEL_MM / 2.0
FIELD_HALF_DEG = spec.FIELD_HALF_DEG
LENS_NEIGHBOUR_MM = spec.LENS_PITCH_UM * 1e-3
WAVELENGTH_MM = 0.55e-3
# RMS radius (sqrt(2) sigma) of the least-squares Gaussian fit to the Airy pattern:
# the blur metric is an RMS radius, and the Airy pattern's own diverges.
DIFFRACTION_RMS_RAD = np.sqrt(2.0) * 0.42 * WAVELENGTH_MM / spec.PUPIL_DIAMETER_MM
# The steepest usable plano-convex lenslet: its sphere must reach past the hex
# corner (and past the panel-border lattice triangles, up to ~1.1 x side).
LENSLET_MIN_RADIUS_OVER_SIDE = 1.15


def lenslet_floor_um(pitch_um):
    """Focal length of the steepest usable lens of a hex array of this pitch."""
    return LENSLET_MIN_RADIUS_OVER_SIDE * pitch_um / np.sqrt(3.0) / (spec.LENS_INDEX - 1.0)


LENSLET_FLOOR_UM = lenslet_floor_um(spec.LENS_PITCH_UM)


def _square_pitch_deg(x_deg, y_deg, horizontal):
    """Watson square-equivalent pitch (retina_model.spacing_xy_deg), with the
    horizontal meridian chosen by the caller."""
    x, y = np.asarray(x_deg, dtype=float), np.asarray(y_deg, dtype=float)
    r = np.hypot(x, y)
    rr = np.where(r > 0.0, r, 1.0)
    sh = horizontal(x, r)
    sv = np.where(y >= 0, one_mosaic_spacing_deg(r, "superior"), one_mosaic_spacing_deg(r, "inferior"))
    s = np.sqrt((x / rr) ** 2 * sh**2 + (y / rr) ** 2 * sv**2)
    return np.sqrt(3.0) / 2.0 * np.where(r > 0.0, s, one_mosaic_spacing_deg(0.0, "temporal"))


def retina_pitch_rad(theta_x_rad, theta_z_rad):
    """The true (left-right asymmetric) Watson pitch, positive theta_x temporal."""
    return np.radians(_square_pitch_deg(np.degrees(theta_x_rad), np.degrees(theta_z_rad),
                                        lambda x, r: np.where(x >= 0, one_mosaic_spacing_deg(r, "temporal"),
                                                              one_mosaic_spacing_deg(r, "nasal"))))


def design_retina_pitch_rad(theta_x_rad, theta_z_rad):
    """The pitch the map follows: temporal on both horizontal sides."""
    return np.radians(_square_pitch_deg(np.degrees(theta_x_rad), np.degrees(theta_z_rad),
                                        lambda x, r: one_mosaic_spacing_deg(r, "temporal")))


def blur_tolerance_rad(theta_x_rad, theta_z_rad):
    """Blur the eye cannot see: the retina's own pitch there, but never below
    the RMS diffraction blur of the design pupil, which no optics can beat."""
    return np.maximum(retina_pitch_rad(theta_x_rad, theta_z_rad), DIFFRACTION_RMS_RAD)


# R(theta, phi) on a polar table; theta in radians, phi over the full circle
_TH = np.radians(np.linspace(0.0, 70.0, 7001))
_PHI = np.radians(np.linspace(0.0, 360.0, 1441))
_P0 = float(design_retina_pitch_rad(0.0, 0.0))
_RATE = _P0 / design_retina_pitch_rad(_TH[None, :] * np.cos(_PHI)[:, None], _TH[None, :] * np.sin(_PHI)[:, None])
_R = np.concatenate([np.zeros((len(_PHI), 1)),
                     np.cumsum(0.5 * (_RATE[:, 1:] + _RATE[:, :-1]) * np.diff(_TH)[None, :], axis=1)], axis=1)


def _polar(theta_x, theta_z):
    tx, tz = np.tan(np.asarray(theta_x, float)), np.tan(np.asarray(theta_z, float))
    return np.arctan(np.hypot(tx, tz)), np.mod(np.arctan2(tz, tx), 2 * np.pi)


def _r_unit(theta, phi):
    """Bilinear interpolation of R on the polar table."""
    if np.any(theta > _TH[-1]):
        raise ValueError("direction beyond the tabulated eccentricity")
    fi, fj = phi / (_PHI[1] - _PHI[0]), theta / (_TH[1] - _TH[0])
    i0 = np.clip(np.floor(fi).astype(int), 0, len(_PHI) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(_TH) - 2)
    wi, wj = fi - i0, fj - j0
    return ((1 - wi) * ((1 - wj) * _R[i0, j0] + wj * _R[i0, j0 + 1])
            + wi * ((1 - wj) * _R[i0 + 1, j0] + wj * _R[i0 + 1, j0 + 1]))


def _unscaled(theta_x, theta_z):
    th, ph = _polar(theta_x, theta_z)
    r = _r_unit(th, ph)
    return r * np.cos(ph), r * np.sin(ph)


def _boundary():
    tx, tz = spec.field_boundary_deg(4001)
    return _unscaled(np.radians(tx), np.radians(tz))


_BX, _BZ = _boundary()
GX_MM = HALF_PANEL_MM / np.abs(_BX).max()
GZ_MM = spec.PANEL_MM / (_BZ.max() - _BZ.min())
V0_MM = -0.5 * (_BZ.max() + _BZ.min()) * GZ_MM


def field_to_panel_mm(theta_x, theta_z):
    """Field direction (tangent-plane angles, positive theta_z up) -> panel (u, v) mm."""
    x, z = _unscaled(theta_x, theta_z)
    return GX_MM * x, GZ_MM * z + V0_MM


# inverse: theta as a function of R / R_max on each azimuth row (each row its own range)
_RN = np.linspace(0.0, 1.0, 7001)
_TH_OF_RN = np.stack([np.interp(_RN * _R[i, -1], _R[i], _TH) for i in range(len(_PHI))])


def panel_to_field_rad(u_mm, v_mm):
    """Panel point -> field direction (theta_x, theta_z), the inverse of field_to_panel_mm."""
    x, z = np.asarray(u_mm, float) / GX_MM, (np.asarray(v_mm, float) - V0_MM) / GZ_MM
    r, ph = np.hypot(x, z), np.mod(np.arctan2(z, x), 2 * np.pi)
    fi = ph / (_PHI[1] - _PHI[0])
    i0 = np.clip(np.floor(fi).astype(int), 0, len(_PHI) - 2)
    w = fi - i0

    def row(i):
        rn = r / _R[i, -1]
        if np.any(rn > 1.0):
            raise ValueError("panel point beyond the tabulated field")
        fr = rn / (_RN[1] - _RN[0])
        k0 = np.clip(np.floor(fr).astype(int), 0, len(_RN) - 2)
        wr = fr - k0
        return (1 - wr) * _TH_OF_RN[i, k0] + wr * _TH_OF_RN[i, k0 + 1]

    th = (1 - w) * row(i0) + w * row(i0 + 1)
    t = np.tan(th)
    return np.arctan(t * np.cos(ph)), np.arctan(t * np.sin(ph))


def jacobian_mm_per_rad(theta_x, theta_z, h=5e-4):
    """d(u, v) / d(theta_x, theta_z), (..., 2, 2) with columns d/dtheta_x, d/dtheta_z.
    The step spans several cells of the polar table (1.7e-4 rad): a smaller one
    differentiates the bilinear interpolant cell by cell and makes J jagged."""
    tx, tz = np.asarray(theta_x, float), np.asarray(theta_z, float)
    ux1, vx1 = field_to_panel_mm(tx + h, tz)
    ux0, vx0 = field_to_panel_mm(tx - h, tz)
    uz1, vz1 = field_to_panel_mm(tx, tz + h)
    uz0, vz0 = field_to_panel_mm(tx, tz - h)
    return np.stack([np.stack([(ux1 - ux0) / (2 * h), (uz1 - uz0) / (2 * h)], -1),
                     np.stack([(vx1 - vx0) / (2 * h), (vz1 - vz0) / (2 * h)], -1)], -2)


def local_focal_mm(theta_x, theta_z):
    """The map's areal scale, sqrt(det J), mm per rad."""
    return np.sqrt(np.abs(np.linalg.det(jacobian_mm_per_rad(theta_x, theta_z))))


def target_pitch_rad(theta_x, theta_z):
    """Angular spacing between neighbouring lens centres in this direction."""
    return LENS_NEIGHBOUR_MM / local_focal_mm(theta_x, theta_z)


def mapped(u_mm, v_mm):
    """True where a panel point serves a direction of the tabulated field (the
    anamorphic map leaves the panel corners beyond it)."""
    x, z = np.asarray(u_mm, float) / GX_MM, (np.asarray(v_mm, float) - V0_MM) / GZ_MM
    r, ph = np.hypot(x, z), np.mod(np.arctan2(z, x), 2 * np.pi)
    i = np.clip(np.round(ph / (_PHI[1] - _PHI[0])).astype(int), 0, len(_PHI) - 1)
    return r <= 0.999 * np.minimum(_R[np.clip(i - 1, 0, len(_PHI) - 1), -1], _R[np.clip(i + 1, 0, len(_PHI) - 1), -1])


def lenslet_focal_um(u_um, v_um, pitch_um=spec.LENS_PITCH_UM):
    """Variable-focal lenslet array: the focal length of the lens at panel (u, v),
    set so its views follow the retina (module docstring), floored at the
    steepest usable lens. A lens beyond the mapped field serves no direction
    and is a floor lens. pitch_um: the array's lens pitch (pitch / pixel pixels
    per lens)."""
    u, v = np.asarray(u_um, float) * 1e-3, np.asarray(v_um, float) * 1e-3
    ok = mapped(u, v)
    floor = lenslet_floor_um(pitch_um)
    f = np.full(np.broadcast(u, v).shape, floor)
    if np.any(ok):
        uu, vv = np.broadcast_to(u, f.shape)[ok], np.broadcast_to(v, f.shape)[ok]
        tx, tz = panel_to_field_rad(uu, vv)
        J = jacobian_mm_per_rad(tx, tz)
        F = np.sqrt(np.abs(np.linalg.det(J)))
        s_min = np.linalg.svd(J, compute_uv=False)[..., -1]
        d_diff = DIFFRACTION_RMS_RAD * spec.PUPIL_DIAMETER_MM / design_retina_pitch_rad(tx, tz)
        d_lens = spec.PUPIL_DIAMETER_MM * F / (pitch_um / spec.PIXEL_UM * s_min)
        d = np.minimum(spec.PUPIL_DIAMETER_MM, np.maximum(d_diff, d_lens))
        f[ok] = np.maximum(spec.PIXEL_UM * F / d, floor)
    return f[()] if f.ndim == 0 else f


def _sampling_range():
    tx, tz = np.meshgrid(np.radians(np.linspace(-FIELD_HALF_DEG[0], FIELD_HALF_DEG[0], 61)),
                         np.radians(np.linspace(-FIELD_HALF_DEG[1], FIELD_HALF_DEG[1], 61)))
    keep = spec.in_field(np.degrees(tx), np.degrees(tz))
    tx, tz = tx[keep], tz[keep]
    ratio = target_pitch_rad(tx.ravel(), tz.ravel()) / retina_pitch_rad(tx.ravel(), tz.ravel())
    return float(ratio.min()), float(ratio.max())


SAMPLING_OVER_RETINA_RANGE = _sampling_range()
