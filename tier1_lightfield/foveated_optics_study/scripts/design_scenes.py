"""One Blender file with a display scene per remapper design.

Each scene holds the design's remapper and lenslet array with dispersive glass
(narrow-band R, G, B: dispersion.py), its panel, and pinhole cameras at the
pupil centre. The panel does not show a baked image: it shows content through
per-channel ST-maps (a UV map stored as an image). Each panel pixel holds the
field direction it must show, the mean direction of the rays that reach it over
the whole pupil, calibrated in Cycles separately for each channel's indices, so
the image at the pupil is right in every colour, lateral colour included.
Content textures can be swapped in the file without calibrating again.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import dispersion as dp
import hmd_encoded as he
import lf_evaluate as ev
import lf_pipeline as lp
import mla_mesh
import screen_spec as spec

LENS_MATERIAL = "nlasf46b"        # every remapper glass
MLA_MATERIAL = "silica"


def _field_deg(d):
    tx, tz = ev._field_deg(d)
    return np.stack([tx, tz], -1)


def stmap(views_dirs, panel_pixels):
    """(N, N, 2) field direction (theta_x, theta_z) in degrees that each panel
    pixel must show: the field angle of the mean direction of the camera rays
    that reach it, over every pupil view of every views folder (a wide set and
    a fine foveal set sample the same pixels). Row j is along +v, column i along
    +u. Also (N, N) bool: the pixel is seen at all."""
    n2 = panel_pixels**2
    total, count = np.zeros((n2, 3)), np.zeros(n2)
    for views_dir in views_dirs:
        views_dir = Path(views_dir)
        d = np.load(views_dir / "direction.npy").astype(np.float64).reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        k = 0
        while (views_dir / f"pix_{k}.npy").exists():
            pix = np.load(views_dir / f"pix_{k}.npy").ravel()
            ok = pix >= 0
            for c in range(3):
                total[:, c] += np.bincount(pix[ok], weights=d[ok, c], minlength=n2)
            count += np.bincount(pix[ok], minlength=n2)
            k += 1
        if k == 0:
            raise FileNotFoundError(f"no views in {views_dir}")
    seen = count > 0
    st = np.zeros((n2, 2))
    st[seen] = _field_deg(total[seen])
    return st.reshape(panel_pixels, panel_pixels, 2), seen.reshape(panel_pixels, panel_pixels)


def _indexed(remapper, index_of):
    r = dict(np.load(remapper))
    for k in range(int(r["n_surfaces"])):
        if f"surf{k}_index" in r:
            r[f"surf{k}_index"] = np.asarray(index_of(float(r[f"surf{k}_index"])))
    return r


def channel_design(design_dir, out_dir, channel):
    """A copy of the design whose remapper glass takes LENS_MATERIAL's index for
    one channel (0, 1, 2: R, G, B), each glass scaled from its design index."""
    design_dir, out_dir = Path(design_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    design = json.loads((design_dir / "design.json").read_text())
    shutil.copy(design_dir / "design.json", out_dir / "design.json")
    r = _indexed(design_dir / design["remapper_npz"], lambda n: dp.channel_indices(n, LENS_MATERIAL)[channel])
    np.savez(out_dir / design["remapper_npz"], **r)
    return out_dir


def dispersive_remapper(design_dir, out_npz):
    """The design's remapper with every glass index given per channel (R, G, B)."""
    design_dir = Path(design_dir)
    design = json.loads((design_dir / "design.json").read_text())
    r = _indexed(design_dir / design["remapper_npz"], lambda n: dp.channel_indices(n, LENS_MATERIAL))
    np.savez(out_npz, **r)
    return Path(out_npz)


def mla_indices():
    return dp.channel_indices(ev.INDEX, MLA_MATERIAL)


FOVEA_RES, FOVEA_FOV_DEG = 1200, 12.0          # 0.6 arcmin per view pixel over the fovea
WIDE_RES, WIDE_FOV_DEG = 2048, 76.0
DISPLAY_SAMPLES = 64


def _export(family, best_json, design_dir, rank):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remapper_designs" / "freeform_mirror"))
    if family == "fold":
        import export_fold as ex
    elif family == "pancake":
        import export_pancake as ex
    else:
        raise ValueError(f"unknown family {family!r}")
    return ex.export(best_json, design_dir, rank)


def channel_stmap(design_dir, work, channel):
    """Calibrate one channel in Cycles (wide and foveal pupil views at that
    channel's indices) and return its ST-map as (N, N, 3): u, v over
    +/- WIDE_HALF_DEG, and seen. The views and meshes are deleted afterwards."""
    work = Path(work)
    ch = channel_design(design_dir, work / "design", channel)
    n_mla = mla_indices()[channel]
    ev.render_views(ch, work / "wide", resolution=WIDE_RES, fov_deg=WIDE_FOV_DEG, mla_index=n_mla)
    ev.render_views(ch, work / "fovea", resolution=FOVEA_RES, fov_deg=FOVEA_FOV_DEG, mla_index=n_mla)
    st, seen = stmap([work / "wide" / "views", work / "fovea" / "views"], spec.PANEL_PIXELS)
    half = he.HALF_DEG
    out = np.concatenate([(st + half) / (2 * half), seen[..., None]], -1).astype(np.float32)
    for sub in ("wide", "fovea"):
        shutil.rmtree(work / sub / "views")
        (work / sub / "mla_mesh.npz").unlink()
    return out


def calibrate_scene(name, family, best_json, out_dir, rank=0):
    """Export the design and calibrate its three channels: out_dir/name/design
    and out_dir/name/stmap_{R,G,B}.npy."""
    base = Path(out_dir).resolve() / name
    design_dir = _export(family, best_json, base / "design", rank)
    for c in range(3):
        np.save(base / f"stmap_{'RGB'[c]}.npy", channel_stmap(design_dir, base / f"ch{'RGB'[c]}", c))
    return base


def save_scene(name, out_dir):
    """Save the display scene of a calibrated design (dispersive glass, ST-map
    panel) as out_dir/name/name.blend."""
    base = Path(out_dir).resolve() / name
    design_dir = base / "design"
    content = Path(out_dir).resolve() / "content"
    content.mkdir(exist_ok=True)
    if not (content / "wide.npy").exists():
        np.save(content / "wide.npy", he.angle_chart().astype(np.float32))
        np.save(content / "fovea.npy", he.foveal_chart().astype(np.float32))
    design = json.loads((design_dir / "design.json").read_text())
    mesh, gap, _ = ev.lenslet_array(design, spec.PANEL_MM * 1e3, 2)
    np.savez(base / "mla_mesh.npz", verts=mesh.verts, faces=mesh.faces, loop_normals=mesh.loop_normals,
             face_lens=mesh.face_lens, face_wall=mesh.face_lens == mla_mesh.WALL)
    remapper = dispersive_remapper(design_dir, base / "remapper_dispersive.npz")
    cfg = {"mode": "build", "index": list(mla_indices()), "mla_npz": str(base / "mla_mesh.npz"),
           "panel_pose": design["panel_pose"], "gap_um": gap, "panel_pixels": spec.PANEL_PIXELS,
           "pixel_um": spec.PIXEL_UM, "camera": {"resolution": WIDE_RES, "fov_deg": WIDE_FOV_DEG},
           "views_mm": [[0.0, 0.0]], "tmp_dir": str(base / "exr"), "remapper_npz": str(remapper),
           "panel_stmap": {"stmaps": [str(base / f"stmap_{c}.npy") for c in "RGB"],
                           "wide_npy": str(content / "wide.npy"), "wide_half_deg": he.HALF_DEG,
                           "fovea_npy": str(content / "fovea.npy"), "fovea_half_deg": he.FOVEAL_HALF_DEG},
           "save_blend": str(base / f"{name}.blend"), "out_npz": str(base / "build.npz")}
    lp.run_blender(cfg, base)
    (base / "mla_mesh.npz").unlink()
    return base / f"{name}.blend"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    scene = sub.add_parser("scene", help="export, calibrate the three channels, save the scene")
    scene.add_argument("name")
    scene.add_argument("family", choices=("fold", "pancake"))
    scene.add_argument("best_json")
    scene.add_argument("out_dir")
    scene.add_argument("--rank", type=int, default=0)
    blend = sub.add_parser("blend", help="save the scene again from its existing calibration")
    blend.add_argument("name")
    blend.add_argument("out_dir")
    args = ap.parse_args()
    if args.command == "scene":
        calibrate_scene(args.name, args.family, args.best_json, args.out_dir, args.rank)
    print(save_scene(args.name, args.out_dir))


if __name__ == "__main__":
    main()
