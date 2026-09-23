"""Viewport-only visualisation layer for a built light-field scene. Runs inside
Blender on a file saved by lf_blender.py in `build` mode:

    blender -b <scene.blend> --factory-startup --python lf_viz.py -- viz.json

Everything added here lives in the VIZ collection, which never renders, so
pupil-camera renders of the saved file stay physically unchanged:
  - ray fans from a few pupil points through a row of lenses to the pixel each
    one lights, using the verified direct-view law: pixel offset = -(f/L)(c - p)
  - a marker at each pupil view position
  - an orthographic side camera (OVERVIEW) framing the eye and the screen
  - the virtual 3D content, if any (VIZ/VIRTUAL_CONTENT): what the encoded
    light field represents, placed where it appears to be
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np


def emission(name, rgb, strength=2.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def polyline(name, points, radius, mat, col):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, q in zip(spline.points, points):
        p.co = (*q, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def main():
    cfg = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    col = bpy.data.collections.new("VIZ")
    bpy.context.scene.collection.children.link(col)
    col.hide_render = True

    f_mm, L = cfg["focal_um"] * 1e-3, cfg["eye_relief_mm"]
    pupil_y, y_vertex, y_panel = cfg["pupil_y_mm"], cfg["y_vertex_mm"], cfg["y_panel_mm"]
    colours = [(1.0, 0.25, 0.1), (0.1, 0.9, 0.3), (0.2, 0.45, 1.0)]
    for k, (p, rgb) in enumerate(zip(cfg["fan_pupil_mm"], colours)):
        mat = emission(f"ray_{k}", rgb)
        for i, c in enumerate(cfg["fan_lenses_mm"]):
            c, p = np.asarray(c), np.asarray(p)
            pixel = c - (f_mm / L) * (c - p)
            pts = [(p[0], pupil_y, p[1]), (c[0], y_vertex, c[1]), (pixel[0], y_panel, pixel[1])]
            polyline(f"ray_{k}_{i}", pts, cfg["ray_radius_mm"], mat, col)

    marker = emission("pupil_view", (1.0, 0.85, 0.2), 4.0)
    for k, (x, z) in enumerate(cfg["views_mm"]):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.06, location=(x, pupil_y - 0.05, z), segments=12,
                                             ring_count=6)
        obj = bpy.context.object
        obj.name = f"view_{k:02d}"
        obj.data.materials.append(marker)
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        col.objects.link(obj)

    if cfg.get("content"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import lf_blender
        virtual = bpy.data.collections.new("VIRTUAL_CONTENT")
        col.children.link(virtual)
        lf_blender.add_content(cfg["content"], virtual)

    cam = bpy.data.objects.new("OVERVIEW", bpy.data.cameras.new("OVERVIEW"))
    col.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = cfg["overview_span_mm"]
    cam.location = (60.0, 0.5 * (cfg["overview_y_range_mm"][0] + cfg["overview_y_range_mm"][1]), 0.0)
    cam.rotation_euler = (math.pi / 2, 0.0, math.pi / 2)
    cam.data.clip_end = 1000.0

    bpy.ops.wm.save_as_mainfile(filepath=cfg["out_blend"])
    print(f"LF_VIZ_DONE {cfg['out_blend']}")


main()
