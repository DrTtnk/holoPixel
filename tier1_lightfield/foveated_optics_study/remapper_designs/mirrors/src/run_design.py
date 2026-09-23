"""Driver: load an optimised parameter checkpoint, export design.json +
remapper.npz into the design directory, and render the PNG diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostics as diag  # noqa: E402
from export import build_design_json, build_remapper_npz  # noqa: E402
from system import measure_footprint  # noqa: E402

TRAINING_APERTURE = ((38.0, 24.0), (22.0, 22.0))  # matches optimize.APERTURE

DESIGN_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = DESIGN_DIR / "figures"


def load(ckpt_path):
    raw = torch.load(ckpt_path, weights_only=False)
    params = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            params[k] = {kk: (vv.clone().requires_grad_(False) if torch.is_tensor(vv) else vv)
                        for kk, vv in v.items()}
        else:
            params[k] = v.clone().requires_grad_(False)
    return params


def main(ckpt_path):
    params = load(ckpt_path)
    FIG_DIR.mkdir(exist_ok=True)
    # A stricter percentile than measure_footprint's default: at the (still
    # generous) 97th percentile, the exported mesh's edge still curled back
    # towards the pupil far enough to fail the 15 mm clearance check outright
    # (9.3 mm at one vertex). 80th percentile was the loosest that kept every
    # vertex >= 15 mm away, checked with verify_geometry.py.
    aperture = measure_footprint(params, TRAINING_APERTURE, pct=80.0, margin=1.0)
    print("export aperture (measured footprint x1.15 margin):", aperture)
    build_remapper_npz(params, aperture, DESIGN_DIR / "remapper.npz")
    design = build_design_json(params, DESIGN_DIR)
    print("design.json:", design)

    diag.layout_section(params, aperture, FIG_DIR / "layout_section.png")
    diag.spot_diagrams(params, aperture, FIG_DIR / "spot_diagrams.png")
    diag.sampling_map(params, aperture, FIG_DIR / "sampling_map.png")
    print("figures written to", FIG_DIR)
    return aperture


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DESIGN_DIR / "src" / "opt_state_v1.pt"))
