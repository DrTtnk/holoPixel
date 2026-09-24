"""Blender side of the physical light-field screen model. Runs inside Blender:

    blender -b --factory-startup --python lf_blender.py -- config.json

World frame is the schematic eye's (blender/eye_and_screen.py): millimetres,
optical axis +Y, corneal apex at the origin, pupil centre at y = PUPIL_Y. The
eye is built for reference only and never renders. Cameras are ideal pinholes
(or a thin-lens aperture) placed in the pupil plane, looking along +Y.

Panel frame (um): u, v span the panel, w points from the panel towards the
optics. The flat MLA face is w = 0; the emitting panel is gap_um behind it
(w = -gap) and emits from its front (+w) face only. `panel_pose` places this
frame in the world: world = origin_mm + (uvw_um * 1e-3) @ basis, with basis
rows the world directions of u, v, w.

Optional remapper surfaces (world mm) sit between the MLA and the eye:
  glass     closed solid of the given index: lossless refraction, and total
            internal reflection where it occurs (no Fresnel partial loss).
            surf{k}_mirror_faces (one bool per face) mirror-coats those faces;
            surf{k}_half_mirror_faces makes those faces an ideal pancake
            half-mirror: a mirror on the ray's first specular event, the glass
            from then on.
            surf{k}_absorber_faces blackens those faces (a lens's edge).
  mirror    perfect specular reflector (both sides)
  polariser ideal pancake reflective polariser: transparent, except a mirror
            on the ray's second specular event (after the half-mirror).
            Cycles has no polarisation; switching on the ray's glossy-bounce
            count gives exactly the ideal pancake path (no ghosts).
  absorber  opaque black (baffles, housings)
Elements must be separated by air: Cycles has no nested-dielectric priority.

Modes:
  calibrate  panel emits its pixel index (R = i, G = j, B = 1); per view, the
             panel pixel each camera ray lands on, plus each view's world ray
             directions. Cameras use a lens shift to frame the panel.
  evaluate   per view, saved to disk: the panel pixel each camera ray reaches
             (panel emits (1, i + 1, j + 1)), and the lens it enters (second
             render, MLA opaque, emitting lens + 1 from its face attribute).
             Cameras keep a fixed orientation, so one direction render serves
             every view.
  display    panel emits the given image; renders each view.
  build      panel emits the given image; saves the scene (save_blend), no render.
  target     no screen: renders the virtual 3D `content` through the same pupil
             cameras, giving the target colour of every calibration ray.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "blender"))
import eye_and_screen as eye  # noqa: E402


def setup(cfg):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    u = sc.unit_settings
    u.system, u.scale_length, u.length_unit = "METRIC", 0.001, "MILLIMETERS"
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    if not any(d.use for d in prefs.devices):
        raise RuntimeError("no OptiX device")
    c = sc.cycles
    c.device = "GPU"
    c.use_adaptive_sampling = False
    c.use_denoising = False
    c.seed = 0
    c.max_bounces = 64
    c.transmission_bounces = 64
    c.glossy_bounces = 64
    c.diffuse_bounces = 0
    c.volume_bounces = 0
    c.transparent_max_bounces = 8
    # Russian roulette would randomly end 1-sample calibration paths after a few
    # bounces; start it only after the bounce limit, i.e. never.
    c.min_light_bounces = 64
    c.min_transparent_bounces = 8
    c.caustics_reflective = True
    c.caustics_refractive = True
    c.blur_glossy = 0.0
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 0.0
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.resolution_x = sc.render.resolution_y = cfg["camera"]["resolution"]
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.exr_codec = "ZIP"
    world = bpy.data.worlds.new("black")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    sc.world = world


def add_eye_reference():
    before = set(bpy.data.objects)
    eye.build_eye()
    col = bpy.data.collections.new("EYE_REFERENCE")
    bpy.context.scene.collection.children.link(col)
    for obj in set(bpy.data.objects) - before:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        col.objects.link(obj)
    col.hide_render = True


def pose_matrix(cfg):
    basis = np.asarray(cfg["panel_pose"]["basis"], dtype=np.float64)
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-9) or np.linalg.det(basis) < 0:
        raise ValueError(f"panel_pose basis is not a proper rotation:\n{basis}")
    return np.asarray(cfg["panel_pose"]["origin_mm"], dtype=np.float64), basis


def panel_to_world(uvw_um, cfg):
    origin, basis = pose_matrix(cfg)
    return origin + (np.asarray(uvw_um, dtype=np.float64) * 1e-3) @ basis


def link(name, verts, faces, loop_normals, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    if len(mesh.polygons) != len(faces):
        raise RuntimeError(f"{name}: validate removed faces: {len(faces)} -> {len(mesh.polygons)}")
    if loop_normals is not None:
        mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
        mesh.normals_split_custom_set(np.asarray(loop_normals).tolist())
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.materials.append(material)
    return obj


def node_material(name, build):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(build(nt), out.inputs["Surface"])
    return mat


def _sharp_mirror(nt):
    b = nt.nodes.new("ShaderNodeBsdfGlossy")
    b.distribution = "GGX"
    b.inputs["Roughness"].default_value = 0.0
    b.inputs["Color"].default_value = (1, 1, 1, 1)
    return b


def _glass_closure(nt, index):
    """Deterministic lossless dielectric: pure refraction, or pure reflection
    under total internal reflection. Cycles' Refraction BSDF alone drops TIR
    paths, and the Glass BSDF picks Fresnel reflection at random, which would
    corrupt a 1-sample pixel-ID calibration."""
    refr = nt.nodes.new("ShaderNodeBsdfRefraction")
    refr.inputs["IOR"].default_value = index
    refr.inputs["Roughness"].default_value = 0.0
    refr.inputs["Color"].default_value = (1, 1, 1, 1)
    fresnel = nt.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = index
    tir = nt.nodes.new("ShaderNodeMath")
    tir.operation = "GREATER_THAN"
    tir.inputs[1].default_value = 0.9999
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fresnel.outputs["Fac"], tir.inputs[0])
    nt.links.new(tir.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(refr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(_sharp_mirror(nt).outputs["BSDF"], mix.inputs[2])
    return mix.outputs["Shader"]


def glass(name, index):
    return node_material(name, lambda nt: _glass_closure(nt, index))


def mirror(name):
    return node_material(name, lambda nt: _sharp_mirror(nt).outputs["BSDF"])


def _glossy_depth_is(nt, n):
    """1 where the ray has had exactly n glossy (specular) bounces, else 0."""
    lp = nt.nodes.new("ShaderNodeLightPath")
    cmp = nt.nodes.new("ShaderNodeMath")
    cmp.operation = "COMPARE"
    cmp.inputs[1].default_value = float(n)
    cmp.inputs[2].default_value = 0.5
    nt.links.new(lp.outputs["Glossy Depth"], cmp.inputs[0])
    return cmp.outputs["Value"]


def polariser(name):
    """Ideal pancake reflective polariser: a mirror on the second specular event,
    transparent otherwise."""
    def build(nt):
        mix = nt.nodes.new("ShaderNodeMixShader")
        see_through = nt.nodes.new("ShaderNodeBsdfTransparent")
        see_through.inputs["Color"].default_value = (1, 1, 1, 1)
        nt.links.new(_glossy_depth_is(nt, 1), mix.inputs["Fac"])
        nt.links.new(see_through.outputs["BSDF"], mix.inputs[1])
        nt.links.new(_sharp_mirror(nt).outputs["BSDF"], mix.inputs[2])
        return mix.outputs["Shader"]
    return node_material(name, build)


def half_mirror(name, index):
    """Ideal pancake half-mirror on a glass face: a mirror on the first specular
    event, the lossless glass afterwards."""
    def build(nt):
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(_glossy_depth_is(nt, 0), mix.inputs["Fac"])
        nt.links.new(_glass_closure(nt, index), mix.inputs[1])
        nt.links.new(_sharp_mirror(nt).outputs["BSDF"], mix.inputs[2])
        return mix.outputs["Shader"]
    return node_material(name, build)


def absorber(name):
    def build(nt):
        e = nt.nodes.new("ShaderNodeEmission")
        e.inputs["Strength"].default_value = 0.0
        return e.outputs["Emission"]
    return node_material(name, build)


def add_mla(cfg):
    m = np.load(cfg["mla_npz"])
    _, basis = pose_matrix(cfg)
    obj = link("MLA", panel_to_world(m["verts"], cfg), m["faces"], m["loop_normals"] @ basis,
               glass("MLA_glass", cfg["index"]))
    if "face_lens" in m:
        attr = obj.data.attributes.new("lens", "FLOAT", "FACE")
        attr.data.foreach_set("value", m["face_lens"].astype(np.float32))
    return obj


def lens_id_material():
    """Opaque: emits (lens + 1) of the face hit, 0 on walls and floor (lens -1)."""
    def build(nt):
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_type = "GEOMETRY"
        a.attribute_name = "lens"
        add = nt.nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        add.inputs[1].default_value = 1.0
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(a.outputs["Fac"], add.inputs[0])
        nt.links.new(add.outputs["Value"], em.inputs["Color"])
        return em.outputs["Emission"]
    return node_material("MLA_lens_id", build)


def add_remapper(cfg):
    if not cfg.get("remapper_npz"):
        return []
    r = np.load(cfg["remapper_npz"])
    objs = []
    for k in range(int(r["n_surfaces"])):
        kind = str(r[f"surf{k}_kind"])
        if kind == "glass":
            mat = glass(f"REMAP{k}_glass", float(r[f"surf{k}_index"]))
        elif kind == "mirror":
            mat = mirror(f"REMAP{k}_mirror")
        elif kind == "absorber":
            mat = absorber(f"REMAP{k}_absorber")
        elif kind == "polariser":
            mat = polariser(f"REMAP{k}_polariser")
        else:
            raise ValueError(f"surface {k}: unknown kind {kind!r}")
        normals = r[f"surf{k}_normals"] if f"surf{k}_normals" in r else None
        obj = link(f"REMAP{k}_{kind}", r[f"surf{k}_verts"], r[f"surf{k}_faces"], normals, mat)
        slot = np.zeros(len(obj.data.polygons), dtype=np.int32)
        for key, coating in (("mirror_faces", lambda: mirror(f"REMAP{k}_coating")),
                             ("half_mirror_faces", lambda: half_mirror(f"REMAP{k}_half_mirror",
                                                                       float(r[f"surf{k}_index"]))),
                             ("absorber_faces", lambda: absorber(f"REMAP{k}_blackened"))):
            if f"surf{k}_{key}" not in r:
                continue
            coated = np.asarray(r[f"surf{k}_{key}"], dtype=bool)
            if kind != "glass" or len(coated) != len(obj.data.polygons) or np.any(slot[coated] != 0):
                raise ValueError(f"surface {k}: {key} needs a glass solid and one flag per face, "
                                 "and a face takes one coating")
            obj.data.materials.append(coating())
            slot[coated] = len(obj.data.materials) - 1
        obj.data.polygons.foreach_set("material_index", slot)
        objs.append(obj)
    return objs


def add_panel(cfg, rgb):
    """rgb: (N, N, 3) float, row j along +v, column i along +u. Emits along +w only."""
    n = cfg["panel_pixels"]
    half = 0.5 * n * cfg["pixel_um"]
    w = -cfg["gap_um"]
    corners = panel_to_world([[-half, -half, w], [half, -half, w], [half, half, w], [-half, half, w]], cfg)
    mesh = bpy.data.meshes.new("PANEL")
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    uv = mesh.uv_layers.new(name="uv")
    for loop, (s, t) in zip(mesh.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[loop.index].uv = (s, t)
    obj = bpy.data.objects.new("PANEL", mesh)
    bpy.context.scene.collection.objects.link(obj)
    img = bpy.data.images.new("panel_pixels", n, n, alpha=True, float_buffer=True)
    # Changing the colorspace of a generated image rebuilds (and zeroes) its
    # buffer, so it must be set before the pixels are written.
    img.colorspace_settings.name = "Non-Color"
    rgba = np.concatenate([rgb.astype(np.float32), np.ones((n, n, 1), np.float32)], axis=2).ravel()
    img.pixels.foreach_set(rgba)
    img.pack()
    back = np.empty(rgba.size, np.float32)
    img.pixels.foreach_get(back)
    if not np.array_equal(back, rgba):
        raise RuntimeError("panel image pixels did not survive being written")

    def build(nt):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = "Closest"
        tex.extension = "CLIP"
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        dark = nt.nodes.new("ShaderNodeEmission")
        dark.inputs["Strength"].default_value = 0.0
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
        nt.links.new(em.outputs["Emission"], mix.inputs[1])
        nt.links.new(dark.outputs["Emission"], mix.inputs[2])
        return mix.outputs["Shader"]

    mesh.materials.append(node_material("PANEL_emission", build))
    return obj


def add_content(spec, collection=None):
    """Virtual 3D content: emissive objects with an object-space checker, so the
    target light field is deterministic (no lighting noise at 1 sample)."""
    objs = []
    for k, item in enumerate(spec):
        kind = item["type"]
        if kind == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=item["radius"], location=item["center"],
                                                 segments=96, ring_count=48)
        elif kind == "cube":
            bpy.ops.mesh.primitive_cube_add(size=item["size"], location=item["center"],
                                            rotation=item.get("rotation", (0, 0, 0)))
        elif kind == "plane":
            bpy.ops.mesh.primitive_plane_add(size=item["size"], location=item["center"],
                                             rotation=item.get("rotation", (0, 0, 0)))
        else:
            raise ValueError(f"content {k}: unknown type {kind!r}")
        obj = bpy.context.object
        obj.name = f"CONTENT_{k}_{kind}"
        if kind == "sphere":
            obj.data.polygons.foreach_set("use_smooth", np.ones(len(obj.data.polygons), dtype=bool))

        def build(nt, item=item):
            tc = nt.nodes.new("ShaderNodeTexCoord")
            ch = nt.nodes.new("ShaderNodeTexChecker")
            ch.inputs["Scale"].default_value = item["checker_scale"]
            ch.inputs["Color1"].default_value = (*item["color"], 1.0)
            ch.inputs["Color2"].default_value = (*[0.25 * c for c in item["color"]], 1.0)
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            nt.links.new(tc.outputs["Object"], ch.inputs["Vector"])
            nt.links.new(ch.outputs["Color"], em.inputs["Color"])
            return em.outputs["Emission"]

        obj.data.materials.append(node_material(f"CONTENT_{k}", build))
        if collection is not None:
            for owner in list(obj.users_collection):
                owner.objects.unlink(obj)
            collection.objects.link(obj)
        objs.append(obj)
    return objs


def target(cfg, cam, tmp):
    """Render the virtual content alone through the same pupil cameras as calibration."""
    sc = bpy.context.scene
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    images = []
    for k, view in enumerate(cfg["views_mm"]):
        place(cam, view, cfg["camera"]["aim_distance_mm"])
        images.append(render(tmp / f"target_{k}.exr")[:, :, :3])
    return {"images": np.stack(images)}


def add_camera(cfg):
    cam = bpy.data.objects.new("PUPIL_CAMERA", bpy.data.cameras.new("PUPIL_CAMERA"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle = math.radians(cfg["camera"]["fov_deg"])
    cam.data.clip_start = 0.01
    cam.data.clip_end = 1000.0
    return cam


def place(cam, view_mm, aim_distance_mm):
    """Pupil point (x, z), looking along +Y. With aim_distance_mm, an off-axis
    lens shift keeps the axis point at that distance centred in frame."""
    cam.location = (view_mm[0], eye.PUPIL_Y, view_mm[1])
    if aim_distance_mm is None:
        cam.data.shift_x = cam.data.shift_y = 0.0
        return
    width = 2.0 * math.tan(cam.data.angle / 2.0) * aim_distance_mm
    cam.data.shift_x = -view_mm[0] / width
    cam.data.shift_y = -view_mm[1] / width


def render(path):
    sc = bpy.context.scene
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path))
    res = sc.render.resolution_x
    a = np.empty(res * res * 4, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    Path(path).unlink()
    return a.reshape(res, res, 4)


def direction_world():
    nt = bpy.context.scene.world.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Generated"], nt.nodes["Background"].inputs["Color"])


def hide_all_meshes():
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = True


def calibrate(cfg, cam, tmp):
    sc = bpy.context.scene
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    aim = cfg["camera"]["aim_distance_mm"]
    ids = []
    for k, view in enumerate(cfg["views_mm"]):
        place(cam, view, aim)
        a = render(tmp / f"id_{k}.exr")
        valid = a[:, :, 2] > 0.5
        i = np.where(valid, np.rint(a[:, :, 0] / np.maximum(a[:, :, 2], 1e-9)), -1).astype(np.int32)
        j = np.where(valid, np.rint(a[:, :, 1] / np.maximum(a[:, :, 2], 1e-9)), -1).astype(np.int32)
        ids.append(np.stack([i, j], axis=-1))
    hide_all_meshes()
    direction_world()
    directions = []
    for k, view in enumerate(cfg["views_mm"]):
        place(cam, view, aim)
        directions.append(render(tmp / f"direction_{k}.exr")[:, :, :3])
    return {"ids": np.stack(ids), "direction": np.stack(directions)}


def _decode(channel, what, k):
    valid = channel > 0.5
    worst = float(np.max(np.abs(channel[valid] - np.rint(channel[valid])))) if valid.any() else 0.0
    if worst > 1e-3:
        raise RuntimeError(f"view {k}: {what} ids are not integers (worst {worst}); radiance was scaled")
    return np.where(valid, np.rint(channel) - 1, -1).astype(np.int32)


def evaluate(cfg, cam, out_dir):
    """Per view: the panel pixel each camera ray reaches (pix_k, flat j*N+i), and
    the lens it enters (entered_k), read from a second render in which the MLA
    is opaque and emits its own lens id."""
    sc = bpy.context.scene
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    n = cfg["panel_pixels"]
    mla = bpy.data.objects["MLA"]
    glass_mat, id_mat = mla.data.materials[0], lens_id_material()
    for k, view in enumerate(cfg["views_mm"]):
        place(cam, view, None)
        a = render(out_dir / f"eval_{k}.exr")
        i, j = _decode(a[:, :, 1], "pixel column", k), _decode(a[:, :, 2], "pixel row", k)
        np.save(out_dir / f"pix_{k}.npy", np.where((i >= 0) & (j >= 0), j * n + i, -1).astype(np.int32))
        mla.data.materials[0] = id_mat
        a = render(out_dir / f"entered_{k}.exr")
        np.save(out_dir / f"entered_{k}.npy", _decode(a[:, :, 0], "entered lens", k))
        mla.data.materials[0] = glass_mat
    hide_all_meshes()
    direction_world()
    place(cam, cfg["views_mm"][0], None)
    np.save(out_dir / "direction.npy", render(out_dir / "direction.exr")[:, :, :3])
    return {"views_mm": np.asarray(cfg["views_mm"])}


def display(cfg, cam, tmp):
    sc = bpy.context.scene
    d = cfg["display"]
    sc.cycles.samples = d["samples"]
    sc.cycles.filter_width = d["filter_width_px"]
    cam.data.dof.use_dof = d["aperture_radius_mm"] > 0
    if cam.data.dof.use_dof:
        # Cycles: radius [scene units] = lens[mm] * 1e-3 / (2 * fstop), independent of
        # the unit scale (measured, see useful_knowledge.md).
        cam.data.dof.aperture_fstop = cam.data.lens * 1e-3 / (2.0 * d["aperture_radius_mm"])
        cam.data.dof.focus_distance = d["focus_distance_mm"]
        cam.data.dof.aperture_blades = 0
    images = []
    for k, view in enumerate(cfg["views_mm"]):
        place(cam, view, cfg["camera"].get("aim_distance_mm"))
        images.append(render(tmp / f"display_{k}.exr")[:, :, :3])
    return {"images": np.stack(images)}


def main():
    cfg = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    tmp = Path(cfg["tmp_dir"])
    tmp.mkdir(parents=True, exist_ok=True)
    setup(cfg)
    if cfg["mode"] == "target":
        add_content(cfg["content"])
        result = target(cfg, add_camera(cfg), tmp)
        np.savez_compressed(cfg["out_npz"], **result)
        print(f"LF_BLENDER_DONE {cfg['mode']} {cfg['out_npz']}")
        return
    add_eye_reference()
    if "mla_npz" in cfg:                  # a plain display (no lenslets) has none
        add_mla(cfg)
    add_remapper(cfg)
    n = cfg["panel_pixels"]
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    if cfg["mode"] == "calibrate":
        rgb = np.stack([i, j, np.ones_like(i)], axis=-1).astype(np.float32)
    elif cfg["mode"] == "evaluate":
        rgb = np.stack([np.ones_like(i), i + 1, j + 1], axis=-1).astype(np.float32)
    else:  # display, build
        rgb = np.load(cfg["panel_image_npy"])
    add_panel(cfg, rgb)
    cam = add_camera(cfg)
    if cfg.get("save_blend"):
        bpy.ops.wm.save_as_mainfile(filepath=cfg["save_blend"])
    if cfg["mode"] == "calibrate":
        result = calibrate(cfg, cam, tmp)
    elif cfg["mode"] == "evaluate":
        result = evaluate(cfg, cam, tmp)
    elif cfg["mode"] == "build":
        result = {"saved": np.array(cfg["save_blend"])}
    else:
        result = display(cfg, cam, tmp)
    np.savez_compressed(cfg["out_npz"], **result)
    print(f"LF_BLENDER_DONE {cfg['mode']} {cfg['out_npz']}")


if __name__ == "__main__":
    main()
