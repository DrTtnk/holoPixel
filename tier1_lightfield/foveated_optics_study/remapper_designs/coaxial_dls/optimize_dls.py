"""Damped-least-squares optimisation of the coaxial remapper with optiland.

    python optimize_dls.py <in_prescription.json> <out_prescription.json> <stage>

stage "shape":   curvatures, conics and every spacing (eye relief >= 20 mm)
stage "asphere": the same plus r^4, r^6, r^8 coefficients

Operands per field (merit.FIELDS_DEG): chief-ray image height on target r(theta),
RMS spot on the MLA plane towards zero, and chief-ray tilt at the MLA kept below
TILT_MAX_DEG (a one-sided bound, so it only acts when violated).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from optiland import optimization as opt

import coaxial_optiland as co
import merit

TILT_MAX_DEG = 8.0
W_HEIGHT, W_SPOT, W_TILT = 1.0, 1.0, 1.0
EYE_RELIEF_MIN_MM = 20.5


def prescription_from_optic(lens, template):
    """Read the optimised surfaces back into the JSON prescription."""
    rx = json.loads(json.dumps(template))
    rx["eye_relief_mm"] = float(lens.surfaces.get_thickness(1)[0])
    k = 2
    for el in rx["elements"]:
        for side in ("front", "back"):
            geom = lens.surfaces[k].geometry
            el[side]["radius_mm"] = float(geom.radius)
            el[side]["conic"] = float(geom.k)
            el[side]["coefficients"] = [float(c) for c in geom.coefficients]
            k += 1
        el["thickness_mm"] = float(lens.surfaces.get_thickness(k - 2)[0])
        el["gap_after_mm"] = float(lens.surfaces.get_thickness(k - 1)[0])
    return rx


def build_problem(lens, sign, stage):
    img = merit.image_surface(lens)
    p = opt.OptimizationProblem()
    for theta in merit.FIELDS_DEG:
        hy = theta / merit.MAX_FIELD_DEG
        chief = {"optic": lens, "surface_number": img, "Hx": 0.0, "Hy": hy, "Px": 0.0, "Py": 0.0,
                 "wavelength": co.WAVELENGTH_UM}
        p.add_operand(operand_type="real_y_intercept", target=merit.target_height_mm(theta, sign),
                      weight=W_HEIGHT, input_data=chief)
        p.add_operand(operand_type="rms_spot_size", target=0.0, weight=W_SPOT,
                      input_data={"optic": lens, "surface_number": img, "Hx": 0.0, "Hy": hy,
                                  "num_rays": merit.SPOT_RINGS, "wavelength": co.WAVELENGTH_UM})
        if theta > 0:
            p.add_operand(operand_type="real_N", min_val=math.cos(math.radians(TILT_MAX_DEG)),
                          weight=W_TILT, input_data=chief)
    p.add_variable(lens, "thickness", surface_number=1, min_val=EYE_RELIEF_MIN_MM, max_val=40.0)
    for s in range(2, img):
        p.add_variable(lens, "reciprocal_radius", surface_number=s)
        p.add_variable(lens, "conic", surface_number=s, min_val=-20.0, max_val=20.0)
        glass = (s % 2 == 0)
        p.add_variable(lens, "thickness", surface_number=s,
                       min_val=2.0 if glass else 0.5, max_val=30.0 if glass else 60.0)
        if stage == "asphere":
            for c in (1, 2, 3):
                p.add_variable(lens, "asphere_coeff", surface_number=s, coeff_number=c)
    return p


def main():
    src, dst, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    rx = json.loads(src.read_text())
    lens = co.build(rx, [0.0, merit.MAX_FIELD_DEG])
    sign = 1.0
    print("BEFORE\n" + merit.table(merit.measure(lens, sign)), flush=True)
    problem = build_problem(lens, sign, stage)
    print(f"operands {len(problem.operands)}, variables {len(problem.variables)}, "
          f"merit before {float(problem.sum_squared()):.6f}", flush=True)
    opt.LeastSquares(problem).optimize(maxiter=400, method_choice="trf", tol=1e-9)
    print(f"merit after {float(problem.sum_squared()):.6f}", flush=True)
    print("AFTER\n" + merit.table(merit.measure(lens, sign)), flush=True)
    dst.write_text(json.dumps(prescription_from_optic(lens, rx), indent=1))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
