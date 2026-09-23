"""Coaxial remapper as an optiland sequential model.

The eye pupil is the stop, `eye_relief_mm` in front of the first surface; the
object is at infinity (field angles are directions seen from the eye); the
image plane is the MLA's flat face. Traced this way (eye -> panel) the
remapper is a front-stop wide-angle lens whose image heights must follow
foveation_target.panel_radius_mm(theta). Monochrome, ideal (constant-index)
glasses, to match the Cycles model.

A prescription is plain JSON:
  {"eye_relief_mm": float,
   "elements": [{"index": n, "thickness_mm": t, "gap_after_mm": g,
                 "front": {"radius_mm": R, "conic": k, "coefficients": [C1, C2, ...]},
                 "back":  {...}}, ...]}
with optiland's even-asphere convention: sag = sphere(R, k) + sum_i C_i r^(2i)
(coefficients[0] multiplies r^2). The last element's gap_after_mm ends on the
image plane.
"""
from __future__ import annotations

import math

import optiland.backend as be
from optiland import optic
from optiland.materials import IdealMaterial

PUPIL_Y_MM = -3.6
PUPIL_RADIUS_MM = 2.0
WAVELENGTH_UM = 0.55


def _radius(c):
    return math.inf if c == 0.0 else 1.0 / c


def prescription_from_agent(params):
    """Convert the torch coaxial design (remapper_designs/coaxial) to JSON."""
    import torch
    with torch.no_grad():
        elements, image_y = params.build(60.0)
    out, prev_back = [], None
    for el in elements:
        f, b = el.front, el.back
        if prev_back is not None:
            out[-1]["gap_after_mm"] = float(f.vertex_y) - prev_back
        out.append({
            "index": float(el.index),
            "thickness_mm": float(b.vertex_y) - float(f.vertex_y),
            "front": {"radius_mm": _radius(float(f.c)), "conic": float(f.k),
                      "coefficients": [0.0, float(f.a4), float(f.a6), float(f.a8)]},
            "back": {"radius_mm": _radius(float(b.c)), "conic": float(b.k),
                     "coefficients": [0.0, float(b.a4), float(b.a6), float(b.a8)]},
        })
        prev_back = float(b.vertex_y)
    out[-1]["gap_after_mm"] = float(image_y) - prev_back
    return {"eye_relief_mm": float(elements[0].front.vertex_y) - PUPIL_Y_MM, "elements": out}


def build(rx, fields_deg):
    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=be.inf, thickness=rx["eye_relief_mm"], is_stop=True)
    k = 2
    for el in rx["elements"]:
        for side, thickness, material in (("front", el["thickness_mm"], IdealMaterial(n=el["index"])),
                                          ("back", el["gap_after_mm"], "air")):
            s = el[side]
            lens.surfaces.add(index=k, surface_type="even_asphere", radius=s["radius_mm"], conic=s["conic"],
                              coefficients=list(s["coefficients"]), thickness=thickness, material=material)
            k += 1
    lens.surfaces.add(index=k)
    lens.set_aperture(aperture_type="EPD", value=2.0 * PUPIL_RADIUS_MM)
    lens.fields.set_type(field_type="angle")
    for theta in fields_deg:
        lens.fields.add(y=theta)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens
