"""Append the per-design display scenes (design_scenes.py) into one .blend.

    blender -b --factory-startup --python combine_scenes.py -- <out.blend> <name>=<scene.blend> ...

Each scene keeps its remapper, lenslet array and ST-map panel. Its camera
becomes the eye: a fisheye over about the monocular field of view (EYE_FOV_DEG
across, EYE_RES), with a PUPIL_MM pupil; a second, foveal camera (FOVEA_FOV_DEG,
perspective) has the same pupil. Both focus at infinity; set the focus distance
to look at the depth scene's sphere (0.35 m) or cube (0.7 m). The panel's
CONTENT_SWITCH value picks the resolution charts (0) or the depth scene (1).
Scenes whose lenslet meshes are identical share one mesh block, and the
content textures, the same in every scene, are shared too.
"""
import sys

import bpy
import numpy as np

FOVEA_FOV_DEG = 12.0
EYE_FOV_DEG = 160.0               # monocular field: ~160 deg across, ~130 deg high
EYE_RES = (1600, 1300)            # 0.1 deg per pixel on the fisheye
PUPIL_MM = 4.0
FOCUS_MM = 1e6                    # infinity
SAMPLES = 64


def _pupil(cam):
    """A PUPIL_MM aperture focused at FOCUS_MM. Cycles: radius [scene units] =
    lens[mm] * 1e-3 / (2 * fstop), whatever the unit scale (useful_knowledge.md)."""
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = cam.data.lens * 1e-3 / PUPIL_MM
    cam.data.dof.focus_distance = FOCUS_MM
    cam.data.dof.aperture_blades = 0


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
        eye = scene.camera
        eye.name = f"{name}_EYE_CAMERA"
        fovea = eye.copy()
        fovea.data = eye.data.copy()
        fovea.name = f"{name}_FOVEA_CAMERA"
        fovea.data.angle = np.radians(FOVEA_FOV_DEG)
        scene.collection.objects.link(fovea)
        eye.data.type = "PANO"
        eye.data.panorama_type = "FISHEYE_EQUIDISTANT"
        eye.data.fisheye_fov = np.radians(EYE_FOV_DEG)
        for cam in (eye, fovea):
            _pupil(cam)
        scene.render.resolution_x, scene.render.resolution_y = EYE_RES
        scene.cycles.samples = SAMPLES
        scene.cycles.filter_width = 1.0
        for obj in scene.objects:
            if obj.type == "MESH" and obj.name.startswith(("MLA", "PANEL", "REMAP")):
                obj.name = f"{name}_{obj.name}"
        for img in bpy.data.images:
            if img.name.startswith("STMAP_"):
                img.name = f"{name}_{img.name}"
            elif img.name.split(".")[0] == "CONTENT_scene_panel":
                img.name = f"{name}_SCENE_PANEL"          # per design: the depth scene encoded for its optics
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
