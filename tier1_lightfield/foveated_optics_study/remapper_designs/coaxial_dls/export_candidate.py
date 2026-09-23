"""Export one GPU-search candidate to the shared evaluator contract (lf_evaluate.py).

    python export_candidate.py <best_*.json> <out_dir> [--rank 0] [--focal-um F]

Each lens becomes a closed glass solid of revolution with exact loop normals.
The lenslet array sits on the candidate's image surface (its vertex there, flat
side to the panel), so the panel is one back focal gap behind it.
Lenslet focal length default: foveation_target.lenslet_focal_um(), at which
the pupil cone at the field edge just fills one lens pitch on the panel.
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

import foveation_target as ft  # noqa: E402
import mla_design as mla  # noqa: E402
import mla_mesh  # noqa: E402
import screen_spec as spec  # noqa: E402
import gpu_search as gs  # noqa: E402
import show_candidates as sc  # noqa: E402

N_RADIAL, N_AZIMUTH = 256, 512   # a facet follows the true normal to half an azimuth step, 0.35 deg
PANEL_BASIS = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]   # panel faces the eye along -y


def export(candidate_json, out_dir, rank=0, focal_um=ft.lenslet_focal_um(), device="cuda"):
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
    vertex_y = sc.PUPIL_Y + float(batch.z[0, -1])
    if entry["layout"]["lenslets"] == "variable_retina":
        # the variable-focal array: the centre lens's vertex on the image surface, the bowl follows
        centre_um = float(mla_mesh.variable_vertex_profile_um(0.0, **gs._PROFILE))
        lenslets = {"lenslets": "variable_retina"}
    else:
        radius = mla.radius_for_focal_length_um(focal_um, spec.LENS_INDEX)
        centre_um = spec.LENS_MIN_THICKNESS_UM + float(mla.sag_um(spec.LENS_SIDE_UM, radius))
        lenslets = {"focal_um": focal_um}
    design = {**lenslets, "remapper_npz": "remapper.npz",
              "panel_pose": {"origin_mm": [0.0, vertex_y + centre_um * 1e-3, 0.0], "basis": PANEL_BASIS},
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
    ap.add_argument("--focal-um", type=float, default=ft.lenslet_focal_um())
    args = ap.parse_args()
    print(export(args.candidate_json, args.out_dir, args.rank, args.focal_um))


if __name__ == "__main__":
    main()
