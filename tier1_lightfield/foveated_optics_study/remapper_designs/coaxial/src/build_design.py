"""Turn an optimised Params state dict into design.json + remapper.npz,
exactly as lf_evaluate.py's export contract requires. Also runs the shared
evaluator's own validate_surfaces() locally (fast, no Blender) before
anything is handed to the real Cycles evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import optics as op  # noqa: E402
import mesh_export as me  # noqa: E402
import design_opt as d  # noqa: E402
import lf_evaluate as ev  # noqa: E402

DTYPE = op.DTYPE
MARGIN_MM = 3.0


def element_aperture_mm(vertex_y, max_ecc_deg=d.EDGE_ECC_DEG):
    return (vertex_y - d.PUPIL_Y) * np.tan(np.radians(max_ecc_deg + 3.0)) + d.PUPIL_R + MARGIN_MM


def _domain_limit_mm(surf, domain_margin=0.9):
    """The largest r at which the even-asphere's sphere-term sqrt domain is
    still comfortably valid (see optics.py's _safe_sphere_r safety clamp,
    which freezes -- flattens -- the sag beyond this, a fine safety valve
    for a transient Newton overshoot during ray tracing but not a shape a
    meshed, exported solid should ever actually use)."""
    u = float((1.0 + surf.k) * surf.c**2)
    if u <= 0:
        return float("inf")
    return float(np.sqrt(domain_margin / u))


def safe_aperture_mm(front, back, fov_aperture, min_thickness_mm=2.0, n=400):
    """The optimiser's loss never sees an unused radius (see NOTES.md: no
    ray from the real pupil/field grid needs the full field-of-view-derived
    aperture at every element), so front and back can cross -- go to
    negative edge thickness -- or run past the asphere's own valid domain,
    beyond the radius any ray actually uses. Clip the MESHED aperture to the
    smallest of: the FOV aperture, the positive-thickness limit, and each
    surface's own domain limit -- so the exported solid is a physically
    valid, genuinely-curved cap everywhere it is meshed. NOTES.md records
    where this leaves less clear aperture than the FOV formula would like."""
    limit = min(fov_aperture, _domain_limit_mm(front), _domain_limit_mm(back))
    r = torch.linspace(0.0, limit, n, dtype=DTYPE)
    thickness = (back.vertex_y + op.sag(r, back)) - (front.vertex_y + op.sag(r, front))
    ok = (thickness.detach().numpy() >= min_thickness_mm)
    first_bad = np.argmax(~ok) if not ok.all() else n
    if first_bad == 0:
        raise ValueError("element has non-positive thickness even on axis")
    return float(r[max(first_bad - 2, 1)])


def export(params: "d.Params", out_dir: Path, focal_um=43.0, n_r=48, n_phi=96, max_ecc_deg=d.EDGE_ECC_DEG):
    out_dir.mkdir(parents=True, exist_ok=True)
    big_aperture = element_aperture_mm(d.Y0, max_ecc_deg) + 20.0  # generous, only used for the domain-safety clamp
    with torch.no_grad():
        elements, image_y = params.build(big_aperture)

    data = {"n_surfaces": len(elements) * 2}
    surf_k = 0
    total_vol = 0.0
    for el in elements:
        front_y = float(el.front.vertex_y)
        back_y = float(el.back.vertex_y)
        fov_aperture = element_aperture_mm(front_y, max_ecc_deg)
        aperture = safe_aperture_mm(el.front, el.back, fov_aperture)
        # build_element_solid auto-corrects the two possible global
        # orientations (see mesh_export.orient_outward / NOTES.md), so this
        # is a check that the exported solid is genuinely valid, not a retry.
        verts, faces, normals = me.build_element_solid(el.front, el.back, aperture, n_r=n_r, n_phi=n_phi)
        vol = me.check_closed_and_outward(verts, faces)
        if aperture < fov_aperture - 1.0:
            print(f"  element {surf_k}: aperture clipped to {aperture:.2f} mm "
                 f"(FOV would want {fov_aperture:.2f} mm)")
        total_vol += vol
        data[f"surf{surf_k}_kind"] = "glass"
        data[f"surf{surf_k}_index"] = float(el.index)
        data[f"surf{surf_k}_verts"] = verts
        data[f"surf{surf_k}_faces"] = faces
        data[f"surf{surf_k}_normals"] = normals
        surf_k += 1
    data["n_surfaces"] = surf_k  # one closed solid per element (front+back+rim merged)

    remapper_path = out_dir / "remapper.npz"
    np.savez(remapper_path, **data)
    ev.validate_surfaces(remapper_path)  # fast local check, no Blender

    design = {
        "focal_um": focal_um,
        "panel_pose": {"origin_mm": [0.0, float(image_y), 0.0],
                       "basis": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]},
        "remapper_npz": "remapper.npz",
    }
    (out_dir / "design.json").write_text(json.dumps(design, indent=1))
    print(f"exported {surf_k} glass elements, image_y={float(image_y):.4f} mm, total glass volume {total_vol:.1f} mm^3")
    return design, remapper_path


def prescription_summary(params: "d.Params", aperture_mm):
    with torch.no_grad():
        elements, image_y = params.build(aperture_mm)
    lines = [f"image plane (MLA flat face) y = {float(image_y):.4f} mm"]
    for i, el in enumerate(elements):
        f, b = el.front, el.back
        lines.append(
            f"element {i}: index={el.index:.4f}  "
            f"front(y={float(f.vertex_y):.4f}, R={1.0/float(f.c) if float(f.c) else float('inf'):.3f}, "
            f"k={float(f.k):.3f}, a4={float(f.a4):.3e}, a6={float(f.a6):.3e})  "
            f"back(y={float(b.vertex_y):.4f}, R={1.0/float(b.c) if float(b.c) else float('inf'):.3f}, "
            f"k={float(b.k):.3f}, a4={float(b.a4):.3e}, a6={float(b.a6):.3e})  "
            f"center_thickness={float(b.vertex_y) - float(f.vertex_y):.4f} mm"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent
    params = d.Params(d.N_ELEMENTS)
    params.load_state_dict(torch.load(ckpt))
    aperture_mm = element_aperture_mm(d.Y0) + 20.0
    print(prescription_summary(params, aperture_mm))
    export(params, out)
