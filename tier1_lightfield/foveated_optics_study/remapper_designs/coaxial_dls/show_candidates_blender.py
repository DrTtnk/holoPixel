"""Blender side of show_candidates.py: builds the side-by-side comparison scene."""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "blender"))
import eye_and_screen as eye  # noqa: E402

FIELD_COLOURS = [(1.0, 0.2, 0.1), (1.0, 0.7, 0.1), (0.2, 0.9, 0.3), (0.2, 0.6, 1.0), (0.8, 0.3, 1.0)]
PANEL_MM = 18.432


def material(name, rgb, emit=0.0, transmission=0.0, rough=0.2, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Transmission Weight"].default_value = transmission
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Strength"].default_value = emit
    b.inputs["Alpha"].default_value = alpha
    m.diffuse_color = (*rgb, alpha)
    return m


def mesh_obj(name, verts, faces, mat, col):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    col.objects.link(ob)
    return ob


def polyline(name, pts, mat, col, radius=0.03):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (*q, 1.0)
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    col.objects.link(ob)


def main():
    spec = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    u = bpy.context.scene.unit_settings
    u.system, u.scale_length, u.length_unit = "METRIC", 0.001, "MILLIMETERS"
    glass = material("resin", (0.75, 0.88, 1.0), transmission=0.9, rough=0.05, alpha=0.35)
    panel_mat = material("panel", (0.05, 0.05, 0.08), emit=0.3)
    ray_mats = [material(f"field_{i}", c, emit=3.0) for i, c in enumerate(FIELD_COLOURS)]
    for ci, cand in enumerate(spec["candidates"]):
        col = bpy.data.collections.new(f"CANDIDATE_{ci}")
        bpy.context.scene.collection.children.link(col)
        before = set(bpy.data.objects)
        eye.build_eye()
        for ob in set(bpy.data.objects) - before:
            ob.location.x += cand["offset_x"]
            for owner in list(ob.users_collection):
                owner.objects.unlink(ob)
            col.objects.link(ob)
        for si, s in enumerate(cand["solids"]):
            mesh_obj(f"C{ci}_lens{si}_n{s['index']:.2f}", s["verts"], s["faces"], glass, col)
        h = PANEL_MM / 2
        x0, y = cand["offset_x"], cand["panel_y"]
        mesh_obj(f"C{ci}_panel", [(x0 - h, y, -h), (x0 + h, y, -h), (x0 + h, y, h), (x0 - h, y, h)],
                 [(0, 1, 2, 3)], panel_mat, col)
        for ri, r in enumerate(cand["rays"]):
            polyline(f"C{ci}_ray{ri}", r["points"], ray_mats[r["field"]], col)
        bpy.ops.object.text_add(location=(x0 - 20, -10, 30))
        t = bpy.context.object
        t.data.body = cand["label"]
        t.data.size = 3.0
        t.rotation_euler = (1.5708, 0, 0)
        for owner in list(t.users_collection):
            owner.objects.unlink(t)
        col.objects.link(t)
    bpy.ops.wm.save_as_mainfile(filepath=spec["out_blend"])
    print("SHOW_DONE")


main()
