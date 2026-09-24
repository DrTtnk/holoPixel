"""Export an inset_study design to the evaluator/headset contract (plain display,
no lenslets: "lenslets": "none"), for viewing with scripts/hmd_view.py.

    python export_inset.py <inset_report.json> <name> <out_dir>

Cycles glass has one index, so the lenses get their nd; colour is judged in
inset_study.py, not in the render.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coaxial_dls"))

import gpu_tracer as gt  # noqa: E402
import inset_study as ins  # noqa: E402
import show_candidates as sc  # noqa: E402

PANEL_BASIS = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]   # panel faces the eye along -y


def export(report_json, name, out_dir):
    entry = json.loads(Path(report_json).read_text())[name]
    mats = {"singlet_PMMA": ["PMMA"], "doublet_PC_PMMA": ["PC", "PMMA"], "doublet_PMMA_PC": ["PMMA", "PC"]}[name]
    lay = ins.layout(mats)
    x = torch.tensor(entry["x"], dtype=torch.float64)
    b = ins.to_batches(x, lay)[1]
    nd = [ins.MATERIALS[m][0] for m in mats]
    fields = torch.tensor(ins.FIELDS_DEG, dtype=torch.float64)
    with torch.no_grad():
        _, _, alive, diag = gt.trace(b, fields, ins.pupil_samples(), diagnostics=True)
    if not bool(alive.all()):
        raise ValueError("the design loses rays")
    used = diag["radius"][0].amax(dim=(1, 2)).numpy()
    z, c, k, a = (t[0].numpy() for t in (b.z, b.c, b.k, b.a))
    data = {"n_surfaces": len(mats)}
    for e in range(len(mats)):
        s0, s1 = 2 * e, 2 * e + 1
        v, f, n = sc.lathe(z[s0], z[s1], (c[s0], k[s0], a[s0]), (c[s1], k[s1], a[s1]),
                           max(used[s0], used[s1]) + 1.0, 256, 512)
        data.update({f"surf{e}_kind": "glass", f"surf{e}_index": nd[e], f"surf{e}_verts": sc.to_world(v, 0.0),
                     f"surf{e}_faces": f, f"surf{e}_normals": n @ sc.TRACER_TO_WORLD})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "remapper.npz", **data)
    design = {"lenslets": "none", "remapper_npz": "remapper.npz",
              "panel_pose": {"origin_mm": [0.0, sc.PUPIL_Y + float(z[-1]), 0.0], "basis": PANEL_BASIS},
              "source": {"report": Path(report_json).name, "design": name}}
    (out / "design.json").write_text(json.dumps(design, indent=1))
    return out


if __name__ == "__main__":
    print(export(*sys.argv[1:4]))
