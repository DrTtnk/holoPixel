"""Writes design.json + remapper.npz from an optimized params.npz (pose,
coeff, index), per lf_evaluate.py's export contract: one glass surface, the
whole watertight prism solid, S2's faces flagged mirror_faces=True.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_export as me  # noqa: E402
from design import Design  # noqa: E402

DTYPE = me.DTYPE
FOCAL_UM = 43.0
GAP_MM = 3.0
HERE = Path(__file__).resolve().parent


def build(params_path, out_dir=HERE, nx=61, nu=61, focal_um=FOCAL_UM, gap_mm=GAP_MM):
    z = np.load(params_path)
    design = Design(index=float(z["index"]), requires_grad=False)
    design.pose = torch.tensor(z["pose"], dtype=DTYPE)
    design.coeff = torch.tensor(z["coeff"], dtype=DTYPE)

    verts, faces, loop_normals, mirror_faces = me.build_solid(design, nx=nx, nu=nu)
    origin, basis = design.panel_frame(gap_mm)
    origin, basis = origin.detach().numpy(), basis.detach().numpy()
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-9) or np.linalg.det(basis) < 0:
        raise RuntimeError("panel_pose basis is not a proper right-handed rotation")

    out_dir = Path(out_dir)
    np.savez(out_dir / "remapper.npz", n_surfaces=1,
            surf0_verts=verts.astype(np.float64), surf0_faces=faces.astype(np.int64),
            surf0_normals=loop_normals.astype(np.float64), surf0_kind="glass",
            surf0_index=design.index, surf0_mirror_faces=mirror_faces)
    design_json = {"focal_um": focal_um,
                   "panel_pose": {"origin_mm": origin.tolist(), "basis": basis.tolist()},
                   "remapper_npz": "remapper.npz"}
    design_json["field_deg"] = [70.0, 45.0]  # searched for the 70 x 45 deg field (lf_evaluate refuses another)
    (out_dir / "design.json").write_text(json.dumps(design_json, indent=1))
    print(f"wrote {out_dir/'design.json'} and {out_dir/'remapper.npz'}: "
         f"{len(verts)} verts, {len(faces)} faces, {int(mirror_faces.sum())} mirror faces")
    return design, verts, faces, loop_normals, mirror_faces


if __name__ == "__main__":
    params = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "params.npz")
    build(params)
