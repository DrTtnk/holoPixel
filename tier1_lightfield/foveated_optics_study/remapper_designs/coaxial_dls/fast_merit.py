"""One-trace residual vector for the coaxial remapper, for scipy least_squares.

Parameter vector x:
    [eye_relief, (t_glass, gap_after) per element, (s2, k, b4, b6, b8) per surface]
Shape terms are sags in mm at the normalisation radius R0 (lens-design practice):
s2 = c R0^2 / 2 (paraxial sagitta, c = 1/R), b_2i = a_2i R0^(2i). Raw r^8
coefficients would make a finite-difference step of 1e-8 move the rim by
hundreds of mm and ruin the Jacobian. The prescription is rebuilt from x on
every call: one source of truth.

Residuals, all in mm-equivalent units:
  landing   every pupil ray's image-plane miss from the field's target point
            (x = 0, y = r(theta)): the real-ray aberration merit, which carries
            mapping AND focus, per ray, so the Jacobian sees every aberration
  chief     the chief ray's height miss, weighted up (the mapping itself)
  tilt      chief-ray angle at the MLA above TILT_MAX_DEG (one-sided)
  lost      a ray that did not reach the image (TIR, missed surface): LOST_MM
  shape     glass thinner than MIN_EDGE_MM, or air gap below MIN_AIR_MM, at the
            radius the rays actually use there (one-sided)
"""
from __future__ import annotations

import math

import numpy as np

import coaxial_optiland as co
import merit

# A tilted chief ray at the MLA is not crosstalk (the pupil cone may land under
# the neighbouring lens, see lf_evaluate.metrics); only steep incidence on the
# lenslets themselves is capped.
TILT_MAX_DEG = 30.0
MAX_RAY_RADIUS_MM = 200.0
MIN_EDGE_MM, MIN_AIR_MM = 1.0, 0.3
LOST_MM = 5.0
W_CHIEF, W_TILT_PER_DEG, W_SHAPE = 3.0, 0.05, 2.0
RINGS = ((0.5, 6), (1.0, 12))  # (normalised pupil radius, points): meridional and skew rays


def pupil_samples():
    pts = [(0.0, 0.0)]
    for r, n in RINGS:
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
            pts.append((r * np.cos(a), r * np.sin(a)))
    return np.array(pts)


R0_MM = 20.0


def to_vector(rx):
    x = [rx["eye_relief_mm"]]
    for el in rx["elements"]:
        x += [el["thickness_mm"], el["gap_after_mm"]]
    for el in rx["elements"]:
        for side in ("front", "back"):
            s = el[side]
            c = 0.0 if math.isinf(s["radius_mm"]) else 1.0 / s["radius_mm"]
            coef = list(s["coefficients"]) + [0.0] * (4 - len(s["coefficients"]))
            x += [c * R0_MM**2 / 2, s["conic"], coef[1] * R0_MM**4, coef[2] * R0_MM**6, coef[3] * R0_MM**8]
    return np.array(x, dtype=float)


def to_prescription(x, indices):
    n = len(indices)
    x = np.asarray(x, dtype=float)
    rx = {"eye_relief_mm": float(x[0]), "elements": []}
    shape = x[1 + 2 * n:].reshape(n, 2, 5)
    for i, index in enumerate(indices):
        el = {"index": float(index), "thickness_mm": float(x[1 + 2 * i]), "gap_after_mm": float(x[2 + 2 * i])}
        for j, side in enumerate(("front", "back")):
            s2, k, b4, b6, b8 = shape[i, j]
            c = 2 * s2 / R0_MM**2
            el[side] = {"radius_mm": math.inf if c == 0.0 else 1.0 / c, "conic": float(k),
                        "coefficients": [0.0, float(b4 / R0_MM**4), float(b6 / R0_MM**6), float(b8 / R0_MM**8)]}
        rx["elements"].append(el)
    return rx


def trace(rx, fields_deg):
    lens = co.build(rx, [0.0, merit.MAX_FIELD_DEG])
    pupil = pupil_samples()
    hy = np.repeat(np.asarray(fields_deg) / merit.MAX_FIELD_DEG, len(pupil))
    px = np.tile(pupil[:, 0], len(fields_deg))
    py = np.tile(pupil[:, 1], len(fields_deg))
    lens.trace_generic(0.0 * hy, hy, px, py, co.WAVELENGTH_UM)
    return lens, pupil


def residuals(x, indices, fields_deg=tuple(merit.FIELDS_DEG), sign=1.0):
    rx = to_prescription(x, indices)
    lens, pupil = trace(rx, fields_deg)
    img = merit.image_surface(lens)
    nf, npu = len(fields_deg), len(pupil)
    X = np.asarray(lens.surfaces.x[img]).reshape(nf, npu)
    Y = np.asarray(lens.surfaces.y[img]).reshape(nf, npu)
    N = np.asarray(lens.surfaces.N[img]).reshape(nf, npu)
    alive = (np.asarray(lens.surfaces.intensity[img]).reshape(nf, npu) > 0) & np.isfinite(X) & np.isfinite(Y)
    target = np.array([merit.target_height_mm(t, sign) for t in fields_deg])[:, None]

    land = np.where(alive, np.hypot(X, Y - target), LOST_MM) / math.sqrt(npu)
    chief = W_CHIEF * np.where(alive[:, 0], Y[:, 0] - target[:, 0], LOST_MM)
    tilt_deg = np.degrees(np.arccos(np.clip(np.abs(np.where(alive[:, 0], N[:, 0], 1.0)), 0.0, 1.0)))
    tilt = W_TILT_PER_DEG * np.maximum(tilt_deg - TILT_MAX_DEG, 0.0)
    return np.concatenate([land.ravel(), chief, tilt, shape_residuals(lens, img)])


def used_radius(lens, surface):
    y = np.asarray(lens.surfaces.y[surface])
    x = np.asarray(lens.surfaces.x[surface])
    r = np.hypot(x, y)
    r = r[np.isfinite(r)]
    return float(r.max()) if r.size else 0.0


def shape_residuals(lens, img):
    """Edge thickness of every glass and every air gap, at the radius used. A
    ray that ran off to a non-physical radius scores as lost geometry."""
    out = []
    for s in range(2, img - 1):
        r = 1.05 * max(used_radius(lens, s), used_radius(lens, s + 1))
        if r > MAX_RAY_RADIUS_MM:
            out.append(W_SHAPE * LOST_MM)
            continue
        t = float(lens.surfaces.get_thickness(s)[0])
        edge = t + float(lens.surfaces[s + 1].geometry.sag(r, 0.0)) - float(lens.surfaces[s].geometry.sag(r, 0.0))
        floor = MIN_EDGE_MM if s % 2 == 0 else MIN_AIR_MM
        out.append(W_SHAPE * max(floor - edge, 0.0) if np.isfinite(edge) else W_SHAPE * LOST_MM)
    return np.array(out)


def spot_table(x, indices, fields_deg=tuple(merit.FIELDS_DEG), sign=1.0):
    lens = co.build(to_prescription(x, indices), [0.0, merit.MAX_FIELD_DEG])
    return merit.table(merit.measure(lens, sign, fields=list(fields_deg)))
