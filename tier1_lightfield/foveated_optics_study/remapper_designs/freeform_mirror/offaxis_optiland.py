"""Off-axis (plane-symmetric) remapper as an optiland model: the reference for
offaxis_tracer.py.

Frame: the eye pupil is the stop, a disc of radius PUPIL_RADIUS_MM at z = 0;
z is the eye's forward axis, y is up, x is to the side. Every surface has an
absolute position (0, y, z) and a tilt rx about the x axis (optiland's
convention: local = Rx(-rx) (p - origin)).

A prescription is plain JSON:
  {"surfaces": [{"y_mm": y, "z_mm": z, "rx_deg": a, "radius_mm": R, "conic": k,
                 "xy": [[C00, C01, ...], [C10, ...], ...],
                 "mirror": bool, "index_after": n}, ...]}
with sag = sphere(R, k) + sum C_ij x^i y^j (optiland's XY polynomial) in the
surface's local frame. The last surface is the image surface (the lenslet
array's face). index_after is the medium after the surface; a mirror keeps
the medium it is in.
"""
from __future__ import annotations

import math

import numpy as np
import optiland.backend as be
from optiland import optic
from optiland.materials import IdealMaterial
from optiland.rays import RealRays

PUPIL_RADIUS_MM = 2.0
WAVELENGTH_UM = 0.55


def field_direction(theta_x_deg, theta_y_deg):
    """Direction of a field point, optiland's angle-field convention: (tan x, tan y, 1)."""
    d = np.stack([np.tan(np.radians(theta_x_deg)), np.tan(np.radians(theta_y_deg)),
                  np.ones_like(np.asarray(theta_x_deg, dtype=float))], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def build(rx):
    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=be.inf, z=-be.inf)
    lens.surfaces.add(index=1, radius=be.inf, z=0.0, is_stop=True)
    for k, s in enumerate(rx["surfaces"], start=2):
        material = "mirror" if s["mirror"] else (IdealMaterial(n=s["index_after"]) if s["index_after"] != 1.0
                                                else "air")
        lens.surfaces.add(index=k, surface_type="polynomial", radius=s["radius_mm"], conic=s["conic"],
                          coefficients=s["xy"], y=s["y_mm"], z=s["z_mm"], rx=math.radians(s["rx_deg"]),
                          material=material)
    lens.set_aperture(aperture_type="EPD", value=2.0 * PUPIL_RADIUS_MM)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def trace(rx, fields_deg, pupil):
    """fields_deg (F, 2) as (theta_x, theta_y); pupil (P, 2) normalised. Returns
    the optiland Optic after tracing F x P rays launched on the pupil plane."""
    lens = build(rx)
    d = field_direction(fields_deg[:, 0], fields_deg[:, 1])
    F, P = len(fields_deg), len(pupil)
    dd = np.repeat(d, P, axis=0)
    xy = np.tile(pupil * PUPIL_RADIUS_MM, (F, 1))
    rays = RealRays(x=xy[:, 0] - dd[:, 0] / dd[:, 2], y=xy[:, 1] - dd[:, 1] / dd[:, 2], z=-np.ones(F * P),
                    L=dd[:, 0], M=dd[:, 1], N=dd[:, 2], intensity=np.ones(F * P),
                    wavelength=np.full(F * P, WAVELENGTH_UM))
    lens.surfaces.trace(rays)
    return lens
