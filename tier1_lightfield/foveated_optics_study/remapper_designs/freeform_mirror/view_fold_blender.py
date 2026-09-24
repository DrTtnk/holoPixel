"""Eye, remapper optics, panel and ray fans of one exported fold or pancake
design, for eyeballing; the side view is framed on the optics. Run with
    blender -b --factory-startup --python view_fold_blender.py -- <design_dir> <out.blend>
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
sys.path.insert(0, str(REPO / "blender"))
import eye_and_screen as eye  # noqa: E402
import lf_blender as lb  # noqa: E402
import screen_spec as spec  # noqa: E402

FIELD_COLOURS = [(1.0, 0.2, 0.1), (1.0, 0.7, 0.1), (0.9, 0.9, 0.2), (0.2, 0.9, 0.3), (0.2, 0.6, 1.0),
                 (0.8, 0.3, 1.0), (1.0, 1.0, 1.0)]


def emissive(name, rgb, strength=3.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Emission Color"].default_value = (*rgb, 1.0)
    b.inputs["Emission Strength"].default_value = strength
    m.diffuse_color = (*rgb, 1.0)
    return m


def polyline(name, pts, mat, radius=0.05):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (*q, 1.0)
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)


def side_view(png, points):
    """Orthographic side view (from -X, looking along +X) with Workbench, framed on
    the points (world y, z) with a 10 % margin."""
    lo, hi = points[:, 1:].min(0), points[:, 1:].max(0)
    centre, size = 0.5 * (lo + hi), hi - lo
    cam = bpy.data.objects.new("SIDE", bpy.data.cameras.new("SIDE"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 1.1 * max(size[0], size[1] * 4.0 / 3.0)
    cam.data.clip_end = 1000.0
    cam.location = (-200.0, centre[0], centre[1])
    cam.rotation_euler = (1.5708, 0.0, -1.5708)
    bpy.context.scene.collection.objects.link(cam)
    sc = bpy.context.scene
    sc.camera = cam
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.color_type = "MATERIAL"
    sc.display.shading.show_xray = True                                 # rays inside and between the optics show
    sc.display.shading.xray_alpha = 0.25
    sc.render.resolution_x, sc.render.resolution_y = 1600, 1200
    sc.render.filepath = str(png.resolve())
    bpy.ops.render.render(write_still=True)


def main():
    design_dir, out = (Path(a) for a in sys.argv[sys.argv.index("--") + 1:])
    design = json.loads((design_dir / "design.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    u = bpy.context.scene.unit_settings
    u.system, u.scale_length, u.length_unit = "METRIC", 0.001, "MILLIMETERS"
    eye.build_eye()
    for ob in lb.add_remapper({"remapper_npz": str(design_dir / design["remapper_npz"])}):
        ob.data.materials[0].diffuse_color = (0.6, 0.8, 1.0, 0.4)
    o, basis = np.asarray(design["panel_pose"]["origin_mm"]), np.asarray(design["panel_pose"]["basis"])
    h = 0.5 * spec.PANEL_MM
    corners = [o + np.array(c) @ basis for c in ((-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0))]
    me = bpy.data.meshes.new("PANEL")
    me.from_pydata([tuple(c) for c in corners], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("PANEL", me)
    ob.data.materials.append(emissive("panel", (0.1, 0.1, 0.15), 0.5))
    bpy.context.scene.collection.objects.link(ob)
    rays = np.load(design_dir / "rays.npz")
    mats = [emissive(f"field_{i}", c) for i, c in enumerate(FIELD_COLOURS)]
    for f in range(rays["paths"].shape[0]):
        for p in range(rays["paths"].shape[1]):
            if rays["alive"][f, p]:
                polyline(f"ray_{f}_{p}", rays["paths"][f, p], mats[f % len(mats)])
    live = rays["paths"][rays["alive"]].reshape(-1, 3)
    side_view(out.with_suffix(".png"), np.concatenate([live, np.asarray(corners)]))
    bpy.ops.wm.save_as_mainfile(filepath=str(out.resolve()))
    print("VIEW_DONE")


main()
