"""Export one pancake design to the shared evaluator contract (lf_evaluate.py).

    python export_pancake.py <best_pancake_*.json> <out_dir> [--rank 0]

  polariser  the flat reflective polariser, a sheet over its footprint
  glass      lens 1, its eye-facing faces an ideal half-mirror
             (surf_half_mirror_faces), then any correctors as plain glass
  absorber   a black plate around the polariser, so nothing reaches the
             panel except through the fold
The lenslet array is the variable-focal one, its vertex surface on the
design's image surface. The panel frame must be right-handed with w towards
the light; in the pancake the light arrives along +z, so w = -z and the
panel's v runs along -y of the tracer frame: the array is built with the
vertical flip that keeps each lens under the field it serves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import export_fold as ef
import fold_search as fs
import offaxis_tracer as ot
import pancake_search as ps  # noqa: F401  (registers the family)
import variable_lenslets as vl

PLATE_HALF_MM = 40.0


def _plate_with_hole(z, half_hole_x, half_hole_y):
    """Square annulus in the plane z (tracer frame): outer half-width PLATE_HALF_MM."""
    o, hx, hy = PLATE_HALF_MM, half_hole_x, half_hole_y
    outer = [(-o, -o), (o, -o), (o, o), (-o, o)]
    inner = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    verts = np.array([(x, y, z) for x, y in outer + inner])
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces += [(i, j, 4 + j), (i, 4 + j, 4 + i)]
    return verts, np.array(faces)


def export(best_json, out_dir, rank=0, device="cuda"):
    entry = json.loads(Path(best_json).read_text())[rank]
    dev = torch.device(device)
    lay = fs.layout(entry["n_el"], entry["flip_u"], entry["flip_v"], family="pancake")
    x = torch.tensor([entry["x"]], dtype=torch.float64, device=dev)
    idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=dev)
    batch = fs.to_batch(x, idx, lay)
    ctx = fs.context(dev)
    with torch.no_grad():
        _, d_img, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    if not bool(alive.all()):
        raise ValueError("the design loses rays; it cannot be exported")
    if not bool((d_img[0, ..., 2] > 0).all()):
        raise ValueError("light must reach the image surface along +z")
    pts = diag["points"][0].cpu().numpy().reshape(batch.z.shape[1], -1, 3)
    pts = np.concatenate([pts, pts * np.array([-1.0, 1.0, 1.0])], axis=1)   # both halves of the field
    batch = ot.Batch(*(t.detach().cpu() for t in batch[:7]), mirror=batch.mirror)

    data = {}
    gx, gy = ef._grid(batch, 1, pts[1])
    vp, npol = ef._surface(batch, 1, gx, gy)
    fp = ef._quads(ef.GRID, ef.GRID)
    surfaces = [("polariser", vp, fp, ef._orient(vp, fp, npol), None, None)]
    v, f, n = ef._solid(batch, 2, 3, np.concatenate([pts[0], pts[2]]), pts[3])
    front = np.zeros(len(f), dtype=bool)
    front[:2 * (ef.GRID - 1) ** 2] = True                                   # _solid lists the front faces first
    surfaces.append(("glass", v, f, n, entry["indices"][0], front))
    for e in range(entry["n_el"] - 1):
        v, f, n = ef._solid(batch, 4 + 2 * e, 5 + 2 * e, pts[4 + 2 * e], pts[5 + 2 * e])
        surfaces.append(("glass", v, f, n, entry["indices"][e + 1], None))
    for k, (kind, v, f, n, index, hm_faces) in enumerate(surfaces):
        data.update({f"surf{k}_kind": kind, f"surf{k}_verts": ef.to_world(v), f"surf{k}_faces": f,
                     f"surf{k}_normals": n @ ef.TRACER_TO_WORLD})
        if index is not None:
            data[f"surf{k}_index"] = index
        if hm_faces is not None:
            data[f"surf{k}_half_mirror_faces"] = hm_faces
    loc = ef._local(batch, 1, pts[1])
    bv, bf = _plate_with_hole(float(batch.z[0, 1]) - 0.05, np.abs(loc[:, 0]).max() + ef.MARGIN_MM,
                              np.abs(loc[:, 1]).max() + ef.MARGIN_MM)
    k = len(surfaces)
    data.update({f"surf{k}_kind": "absorber", f"surf{k}_verts": ef.to_world(bv), f"surf{k}_faces": bf})
    data["n_surfaces"] = k + 1
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "remapper.npz", **data)

    s_img = batch.z.shape[1] - 1
    o, axes = ef._frame(batch, s_img)
    w = -axes[2]                                                           # towards the light
    u = axes[0]
    v = np.cross(w, u)
    array_flip = int(lay["flip_v"] * np.sign(v @ axes[1]))                 # panel v along -y here
    basis = np.stack([u, v, w]) @ ef.TRACER_TO_WORLD
    origin = ef.to_world(o) - float(vl.vertex_height_um(0.0, 0.0, array_flip)) * 1e-3 * basis[2]
    design = {"lenslets": fs.LENSLETS, "lenslet_flip_v": array_flip, "remapper_npz": "remapper.npz",
              "panel_pose": {"origin_mm": origin.tolist(), "basis": basis.tolist()},
              "source": {"design": Path(best_json).name, "rank": rank, "material": entry["material"],
                         "family": "pancake"}}
    (out / "design.json").write_text(json.dumps(design, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best_json")
    ap.add_argument("out_dir")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    print(export(args.best_json, args.out_dir, args.rank))


if __name__ == "__main__":
    main()
