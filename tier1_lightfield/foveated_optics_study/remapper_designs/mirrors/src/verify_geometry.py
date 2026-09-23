"""Standalone check of eye relief / clearance against the exported
design.json + remapper.npz, using EXACTLY the evaluator's own geometry()
logic (imported, not reimplemented), so failures show up before the
1-3 minute Blender run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import mla_design as mla  # noqa: E402
import mla_mesh  # noqa: E402
import lf_evaluate as ev  # noqa: E402

DESIGN_DIR = Path(__file__).resolve().parents[1]


def main():
    design = json.loads((DESIGN_DIR / "design.json").read_text())
    remapper = DESIGN_DIR / design["remapper_npz"]
    ev.validate_surfaces(remapper)
    focal = float(design["focal_um"])
    radius = mla.radius_for_focal_length_um(focal, ev.INDEX)
    panel_um = 2044 * ev.PIXEL_UM
    mesh = mla_mesh.build(panel_um=panel_um, side_um=ev.SIDE_UM, radius_um=radius,
                          min_thickness_um=ev.MIN_THICKNESS_UM, subdivisions=3)
    gap = mla.back_focal_gap_um(radius, ev.INDEX, mesh.centre_thickness)
    geom = ev.geometry(design, remapper, mesh, gap)
    print(json.dumps(geom, indent=1))
    print("eye_relief >= 20:", geom["eye_relief_mm"] >= 20.0)
    print("clearance >= 15:", geom["clearance_mm"] >= 15.0)
    print(f"gap_um={gap:.2f} radius_um={radius:.2f} centre_thickness_um={mesh.centre_thickness:.2f}")


if __name__ == "__main__":
    main()
