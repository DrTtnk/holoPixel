"""Append the per-design display scenes (design_scenes.py) into one .blend.

    blender -b --factory-startup --python combine_scenes.py -- <out.blend> <name>=<scene.blend> ...

Each scene keeps its remapper, lenslet array, ST-map panel and wide pinhole
camera, and gains a foveal pinhole camera (FOVEA_FOV_DEG) at the same pupil
centre. Scenes whose lenslet meshes are identical share one mesh block, and the
content textures, the same in every scene, are shared too.
"""
import sys

import bpy
import numpy as np

FOVEA_FOV_DEG = 12.0
SAMPLES = 64


def _mesh_key(mesh):
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    return len(mesh.vertices), len(mesh.polygons), co.tobytes().__hash__()


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    out, pairs = args[0], [a.split("=", 1) for a in args[1:]]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    empty = bpy.context.scene
    for name, path in pairs:
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            if len(src.scenes) != 1:
                raise ValueError(f"{path}: expected one scene, found {len(src.scenes)}")
            dst.scenes = list(src.scenes)
        scene = dst.scenes[0]
        scene.name = name
        wide = scene.camera
        wide.name = f"{name}_WIDE_CAMERA"
        fovea = wide.copy()
        fovea.data = wide.data.copy()
        fovea.name = f"{name}_FOVEA_CAMERA"
        fovea.data.angle = np.radians(FOVEA_FOV_DEG)
        scene.collection.objects.link(fovea)
        scene.cycles.samples = SAMPLES
        scene.cycles.filter_width = 1.0
        for obj in scene.objects:
            if obj.type == "MESH" and obj.name.startswith(("MLA", "PANEL", "REMAP")):
                obj.name = f"{name}_{obj.name}"
        for img in bpy.data.images:
            if img.name.startswith("STMAP_"):
                img.name = f"{name}_{img.name}"
    bpy.data.scenes.remove(empty)
    shared = {}
    for mesh in list(bpy.data.meshes):
        if not any(o.data == mesh and o.name.endswith("_MLA") for o in bpy.data.objects):
            continue
        key = _mesh_key(mesh)
        if key in shared:
            mesh.user_remap(shared[key])
            bpy.data.meshes.remove(mesh)
        else:
            shared[key] = mesh
    for prefix in ("CONTENT_wide", "CONTENT_fovea"):
        imgs = sorted((i for i in bpy.data.images if i.name.split(".")[0] == prefix), key=lambda i: i.name)
        for img in imgs[1:]:
            img.user_remap(imgs[0])
            bpy.data.images.remove(img)
    print(f"COMBINED {len(bpy.data.scenes)} scenes, {len(shared)} lenslet meshes, "
          f"{sum(1 for i in bpy.data.images if i.name.startswith('CONTENT'))} content images")
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print("COMBINE_DONE", out)


main()
