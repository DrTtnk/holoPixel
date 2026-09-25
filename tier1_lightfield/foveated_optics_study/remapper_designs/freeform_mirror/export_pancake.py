"""Export one pancake design to the shared evaluator contract (lf_evaluate.py).

    python export_pancake.py <best_pancake_*.json> <out_dir> [--rank 0]

  polariser  the flat reflective polariser, a sheet over its footprint
  glass      lens 1 (round), its eye-facing faces an ideal half-mirror
             (surf_half_mirror_faces), then any correctors as plain glass
  absorber   black plates around the polariser and around lens 1, and the
             lenses' edges blackened, so nothing reaches the panel except
             through the fold
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


def _plate_with_round_hole(z, cy, radius, segments=256):
    """Square plate in the plane z (tracer frame), outer half-width PLATE_HALF_MM,
    with a polygonal hole about (0, cy) whose rim stays inside the radius."""
    phi = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    inner = np.column_stack([radius * np.cos(phi), cy + radius * np.sin(phi)])
    scale = PLATE_HALF_MM / np.abs(np.column_stack([np.cos(phi), np.sin(phi)])).max(1)
    outer = np.column_stack([scale * np.cos(phi), cy + scale * np.sin(phi)])
    verts = np.column_stack([np.concatenate([outer, inner]), np.full(2 * segments, z)])
    i = np.arange(segments)
    j = (i + 1) % segments
    faces = np.concatenate([np.stack([i, j, segments + j], 1), np.stack([i, segments + j, segments + i], 1)])
    return verts, faces


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
    lay = fs.entry_layout(entry, "pancake")
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
    batch = ef.cpu_batch(batch)

    data = {}
    gx, gy = ef._grid(batch, 1, pts[1])
    vp, npol, _ = ef._surface(batch, 1, gx, gy)
    fp = ef._quads(ef.GRID, ef.GRID)
    surfaces = [("polariser", vp, fp, ef._orient(vp, fp, npol), None, None, None)]
    # the half-mirror's footprint is wide, the back's small: _solid clamps each
    # surface flat beyond its own hits; the lens must stay behind the polariser
    z_pol, z_hm = float(batch.z[0, 1]), float(batch.z[0, 2])
    v, f, n = ef._solid(batch, 2, 3, np.concatenate([pts[0], pts[2]]), pts[3], disc=True,
                        front_floor=z_pol + fs.MIN_AIR_MM / 5.0 - z_hm)
    if float(v[:, 2].min()) <= float(batch.z[0, 1]):
        raise ValueError("lens 1 reaches the polariser plane")
    n_face = 2 * (ef.GRID - 1) ** 2                                        # _solid: front faces, back faces, walls
    front = np.zeros(len(f), dtype=bool)
    front[:n_face] = True
    walls = np.zeros(len(f), dtype=bool)
    walls[2 * n_face:] = True
    lens_back_z = float(v[:, 2].max())
    surfaces.append(("glass", v, f, n, entry["indices"][0], front, walls))
    for e in range(entry["n_el"] - 1):
        v, f, n = ef._solid(batch, 4 + 2 * e, 5 + 2 * e, pts[4 + 2 * e], pts[5 + 2 * e], disc=True)
        walls = np.zeros(len(f), dtype=bool)
        walls[2 * n_face:] = True
        surfaces.append(("glass", v, f, n, entry["indices"][e + 1], None, walls))
    for k, (kind, v, f, n, index, hm_faces, wall_faces) in enumerate(surfaces):
        data.update({f"surf{k}_kind": kind, f"surf{k}_verts": ef.to_world(v), f"surf{k}_faces": f,
                     f"surf{k}_normals": n @ ef.TRACER_TO_WORLD})
        if index is not None:
            data[f"surf{k}_index"] = index
        if hm_faces is not None:
            data[f"surf{k}_half_mirror_faces"] = hm_faces
        if wall_faces is not None:
            data[f"surf{k}_absorber_faces"] = wall_faces                   # blackened lens edges
    loc = ef._local(batch, 1, pts[1])
    bv, bf = _plate_with_hole(float(batch.z[0, 1]) - 0.05, np.abs(loc[:, 0]).max() + ef.MARGIN_MM,
                              np.abs(loc[:, 1]).max() + ef.MARGIN_MM)
    k = len(surfaces)
    data.update({f"surf{k}_kind": "absorber", f"surf{k}_verts": ef.to_world(bv), f"surf{k}_faces": bf})
    # a second plate just behind lens 1, around it: nothing passes beside the lens
    rim = surfaces[1][1][:ef.GRID, :2]                                    # one side of the front grid: on the circle
    cy = float(surfaces[1][1][ef.GRID ** 2 // 2, 1])
    lens_r = float(np.hypot(rim[:, 0], rim[:, 1] - cy).min())
    bv2, bf2 = _plate_with_round_hole(lens_back_z + 0.05, cy, lens_r - 0.5 * ef.MARGIN_MM)
    data.update({f"surf{k + 1}_kind": "absorber", f"surf{k + 1}_verts": ef.to_world(bv2), f"surf{k + 1}_faces": bf2})
    data["n_surfaces"] = k + 2
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
              "field_deg": entry["field_deg"],
              "panel_pose": {"origin_mm": origin.tolist(), "basis": basis.tolist()},
              "source": {"design": Path(best_json).name, "rank": rank, "material": entry["material"],
                         "family": "pancake"}}
    (out / "design.json").write_text(json.dumps(design, indent=1))
    np.savez(out / "rays.npz", **ef.fans(entry, dev, family="pancake"))
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
