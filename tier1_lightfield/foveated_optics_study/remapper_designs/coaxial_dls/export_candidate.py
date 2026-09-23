"""Export one GPU-search candidate to the shared evaluator contract (lf_evaluate.py).

    python export_candidate.py <best_*.json> <out_dir> [--rank 0] [--focal-um 114]

Each lens becomes a closed glass solid of revolution with exact loop normals.
The lenslet array sits on the candidate's image surface (its vertex there, flat
side to the panel), so the panel is one back focal gap behind it.
Lenslet focal length default: the pupil cone at the field edge
(F_edge = 12.7 mm, 4 mm pupil) just fills one 36 um lens pitch on the panel,
f = pitch * F_edge / pupil = 114 um.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

import mla_design as mla  # noqa: E402
import screen_spec as spec  # noqa: E402
import show_candidates as sc  # noqa: E402

N_RADIAL, N_AZIMUTH = 256, 512   # a facet follows the true normal to half an azimuth step, 0.35 deg
PANEL_BASIS = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]   # panel faces the eye along -y


def export(candidate_json, out_dir, rank=0, focal_um=114.0, device="cuda"):
    entry = json.loads(Path(candidate_json).read_text())[rank]
    solids, batch = sc.candidate_solids(entry, torch.device(device), n_r=N_RADIAL, n_phi=N_AZIMUTH)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {"n_surfaces": len(solids)}
    for k, s in enumerate(solids):
        data.update({f"surf{k}_kind": "glass", f"surf{k}_index": s["index"],
                     f"surf{k}_verts": sc.to_world(s["verts"], 0.0), f"surf{k}_faces": s["faces"],
                     f"surf{k}_normals": s["normals"] @ sc.TRACER_TO_WORLD})
    np.savez(out / "remapper.npz", **data)
    radius = mla.radius_for_focal_length_um(focal_um, spec.LENS_INDEX)
    centre_thickness_um = spec.LENS_MIN_THICKNESS_UM + float(mla.sag_um(spec.LENS_SIDE_UM, radius))
    vertex_y = sc.PUPIL_Y + float(batch.z[0, -1])
    design = {"focal_um": focal_um, "remapper_npz": "remapper.npz",
              "panel_pose": {"origin_mm": [0.0, vertex_y + centre_thickness_um * 1e-3, 0.0], "basis": PANEL_BASIS},
              "source": {"candidate": str(Path(candidate_json).name), "rank": rank,
                         "spot_in_tolerance_per_field": entry["spot_in_tolerance_per_field"],
                         "fields_deg": entry["fields_deg"]}}
    (out / "design.json").write_text(json.dumps(design, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_json")
    ap.add_argument("out_dir")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--focal-um", type=float, default=114.0)
    args = ap.parse_args()
    print(export(args.candidate_json, args.out_dir, args.rank, args.focal_um))


if __name__ == "__main__":
    main()
