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
             (panel emits (0, i + 1, j + 1)), and the lens it enters (second
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
    sc.render.use_persistent_data = True         # keep the BVH between the many renders of one scene
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
    c.tile_size = max(2048, cfg["camera"]["resolution"])      # one tile: 3.3 s a view, not 4.5 s in four
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.exr_codec = "NONE"      # read back at once and deleted: compressing only costs time
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


def glass_index(value):
    """A glass index as stored: one value, or three (R, G, B) for dispersive glass."""
    a = np.asarray(value, dtype=np.float64).ravel()
    if a.size == 1:
        return float(a[0])
    if a.size == 3:
        return tuple(float(x) for x in a)
    raise ValueError(f"a glass index is one value or three (R, G, B), not {a.size}")


def _glass_closure(nt, index):
    """One index: a deterministic lossless dielectric (_single_glass). Three
    indices (R, G, B): dispersive glass, still one deterministic closure, whose
    index is picked by the view layer's "channel" property (0, 1, 2): the scene
    renders one view layer per channel and the compositor keeps channel c of
    layer c (colour_layers)."""
    if not isinstance(index, tuple):
        return _single_glass(nt, index)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "VIEW_LAYER"
    attr.attribute_name = "channel"
    ior = index[0]
    for c in (1, 2):
        ior = _math(nt, "MULTIPLY_ADD", _math(nt, "COMPARE", attr.outputs["Fac"], float(c), 0.5),
                    index[c] - index[0], ior)
    return _single_glass(nt, ior)


def _single_glass(nt, index):
    """Deterministic lossless dielectric: pure refraction, or pure reflection
    under total internal reflection. Cycles' Refraction BSDF alone drops TIR
    paths, and the Glass BSDF picks Fresnel reflection at random, which would
    corrupt a 1-sample pixel-ID calibration. index: a value or a socket."""
    refr = nt.nodes.new("ShaderNodeBsdfRefraction")
    refr.inputs["Roughness"].default_value = 0.0
    refr.inputs["Color"].default_value = (1, 1, 1, 1)
    fresnel = nt.nodes.new("ShaderNodeFresnel")
    for socket in (refr.inputs["IOR"], fresnel.inputs["IOR"]):
        if isinstance(index, float):
            socket.default_value = index
        else:
            nt.links.new(index, socket)
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
               glass("MLA_glass", glass_index(cfg["index"])))
    if "face_lens" in m:
        attr = obj.data.attributes.new("lens", "FLOAT", "FACE")
        attr.data.foreach_set("value", m["face_lens"].astype(np.float32))
        # the steps between lenses are black (a black matrix); slot 0 stays the
        # lens glass, which evaluate() swaps for the lens-id material
        obj.data.materials.append(absorber("MLA_wall"))
        slot = m["face_wall"].astype(np.int32)
        obj.data.polygons.foreach_set("material_index", slot)
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
    mat = node_material("MLA_lens_id", build)
    # seen by camera rays only: as a light, each of the ~10 M lenslet faces would
    # enter a light tree that Cycles rebuilds for every render (~20 s each)
    mat.cycles.emission_sampling = "NONE"
    return mat


def add_remapper(cfg):
    if not cfg.get("remapper_npz"):
        return []
    r = np.load(cfg["remapper_npz"])
    objs = []
    for k in range(int(r["n_surfaces"])):
        kind = str(r[f"surf{k}_kind"])
        if kind == "glass":
            mat = glass(f"REMAP{k}_glass", glass_index(r[f"surf{k}_index"]))
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
                                                                       glass_index(r[f"surf{k}_index"]))),
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


def float_image(name, rgb):
    """A packed float image (Non-Color) from (H, W, 3) values, row 0 at the bottom."""
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    # Changing the colorspace of a generated image rebuilds (and zeroes) its
    # buffer, so it must be set before the pixels are written.
    img.colorspace_settings.name = "Non-Color"
    rgba = np.concatenate([rgb.astype(np.float32), np.ones((h, w, 1), np.float32)], axis=2).ravel()
    img.pixels.foreach_set(rgba)
    img.pack()
    back = np.empty(rgba.size, np.float32)
    img.pixels.foreach_get(back)
    if not np.array_equal(back, rgba):
        raise RuntimeError(f"image {name} pixels did not survive being written")
    return img


def _texture(nt, img, vector, interpolation):
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = interpolation
    tex.extension = "CLIP"
    nt.links.new(vector, tex.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tex.outputs["Color"], sep.inputs["Color"])
    return sep.outputs


def _math(nt, op, a, b=None, c=None):
    node = nt.nodes.new("ShaderNodeMath")
    node.operation = op
    for socket, value in zip(node.inputs, (a, b, c)):
        if value is None:
            continue
        if isinstance(value, float):
            socket.default_value = value
        else:
            nt.links.new(value, socket)
    return node.outputs["Value"]


def image_colour(img):
    """Panel colour: the image itself, one texel per panel pixel."""
    def colour(nt):
        uv = nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = "Closest"
        tex.extension = "CLIP"
        nt.links.new(uv, tex.inputs["Vector"])
        return tex.outputs["Color"]
    return colour


def stmap_colour(st):
    """Panel colour through per-channel ST-maps. stmaps[c] holds, per panel pixel,
    (u, v, seen): the content coordinate of the field direction the pixel must
    show in channel c, over +/- wide_half_deg. Inside +/- fovea_half_deg the
    fovea texture (finer) is shown instead of the wide one. With st["scene"]
    ({"npy", "switch"}), a Value node CONTENT_SWITCH mixes in a baked panel
    image (one texel per panel pixel): 0 the ST-map content, 1 the image."""
    stmaps = [float_image(f"STMAP_{c}", np.load(path)) for c, path in zip("RGB", st["stmaps"])]
    wide, fovea = float_image("CONTENT_wide", np.load(st["wide_npy"])), float_image("CONTENT_fovea",
                                                                                    np.load(st["fovea_npy"]))
    scene = float_image("CONTENT_scene_panel", np.load(st["scene"]["npy"])) if "scene" in st else None
    k = st["wide_half_deg"] / st["fovea_half_deg"]
    edge = 0.5 / k                                                     # the fovea's half-width in wide coordinates

    def colour(nt):
        uv = nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]
        channels = []
        for c, img in enumerate(stmaps):
            u, v, seen = _texture(nt, img, uv, "Closest")[:3]
            xy = nt.nodes.new("ShaderNodeCombineXYZ")
            nt.links.new(u, xy.inputs["X"])
            nt.links.new(v, xy.inputs["Y"])
            w = _texture(nt, wide, xy.outputs["Vector"], "Linear")[c]
            fxy = nt.nodes.new("ShaderNodeCombineXYZ")
            nt.links.new(_math(nt, "MULTIPLY_ADD", _math(nt, "SUBTRACT", u, 0.5), k, 0.5), fxy.inputs["X"])
            nt.links.new(_math(nt, "MULTIPLY_ADD", _math(nt, "SUBTRACT", v, 0.5), k, 0.5), fxy.inputs["Y"])
            f = _texture(nt, fovea, fxy.outputs["Vector"], "Linear")[c]
            inside = _math(nt, "MULTIPLY", _math(nt, "LESS_THAN", _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", u, 0.5)), edge),
                           _math(nt, "LESS_THAN", _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", v, 0.5)), edge))
            mixed = _math(nt, "ADD", w, _math(nt, "MULTIPLY", inside, _math(nt, "SUBTRACT", f, w)))
            channels.append(_math(nt, "MULTIPLY", mixed, seen))
        rgb = nt.nodes.new("ShaderNodeCombineColor")
        for socket, value in zip(rgb.inputs, channels):
            nt.links.new(value, socket)
        if scene is None:
            return rgb.outputs["Color"]
        switch = nt.nodes.new("ShaderNodeValue")
        switch.name = switch.label = "CONTENT_SWITCH"
        switch.outputs[0].default_value = float(st["scene"]["switch"])
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = scene
        tex.interpolation = "Closest"
        tex.extension = "CLIP"
        nt.links.new(uv, tex.inputs["Vector"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(switch.outputs[0], mix.inputs["Factor"])
        nt.links.new(rgb.outputs["Color"], mix.inputs["A"])
        nt.links.new(tex.outputs["Color"], mix.inputs["B"])
        return mix.outputs["Result"]
    return colour


def add_panel(cfg, colour):
    """colour(nt) -> the panel's colour socket, row j along +v, column i along +u
    of the UV square. Emits along +w only."""
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

    def build(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(colour(nt), em.inputs["Color"])
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


def target_views(cfg, cam, tmp):
    """The virtual content alone along the evaluate cameras' rays (no lens shift):
    tmp/target_k.npy per view, (res, res, 3), row 0 at the bottom."""
    sc = bpy.context.scene
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    for k, view in enumerate(cfg["views_mm"]):
        place(cam, view, None)
        np.save(tmp / f"target_{k}.npy", render(tmp / f"target_{k}.exr")[:, :, :3])
    return {"n_views": np.array(len(cfg["views_mm"]))}


def add_camera(cfg):
    cam = bpy.data.objects.new("PUPIL_CAMERA", bpy.data.cameras.new("PUPIL_CAMERA"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle = math.radians(cfg["camera"]["fov_deg"])
    cam.data.clip_start = 0.01
    cam.data.clip_end = 1e5                      # content out to 100 m
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


MULTIVIEW_BATCH = 16    # views per render call: ~1 s of fixed cost per call is shared; 16 views are ~3 GB of result


def view_cameras(cam, views_mm):
    """One camera per pupil view (<cam>_<k>) and one multi-view render view each
    (suffix _<k>). Blender picks a view's camera by swapping the view suffix at
    the end of the ACTIVE camera's name, and silently keeps the active camera
    when that name ends in no view suffix: the first view camera is made active."""
    sc = bpy.context.scene
    cams = []
    for k, view in enumerate(views_mm):
        c = bpy.data.objects.new(f"{cam.name}_{k}", cam.data)
        sc.collection.objects.link(c)
        c.rotation_euler = cam.rotation_euler
        place(c, view, None)
        cams.append(c)
    sc.camera = cams[0]
    sc.render.use_multiview = True
    sc.render.views_format = "MULTIVIEW"
    for v in sc.render.views:
        v.use = False
    views = [sc.render.views.new(f"view{k}") for k in range(len(views_mm))]
    for k, v in enumerate(views):
        v.camera_suffix = f"_{k}"
    sc.render.image_settings.views_format = "INDIVIDUAL"
    return views


def render_views(views, stem, out_dir):
    """(k, RGBA image) of every view, MULTIVIEW_BATCH views per render call."""
    sc = bpy.context.scene
    res = sc.render.resolution_x
    for start in range(0, len(views), MULTIVIEW_BATCH):
        batch = range(start, min(start + MULTIVIEW_BATCH, len(views)))
        for k, v in enumerate(views):
            v.use = k in batch
        sc.render.filepath = str(out_dir / f"{stem}.exr")
        bpy.ops.render.render(write_still=True)
        for k in batch:
            path = out_dir / f"{stem}_{k}.exr"                            # the view's suffix
            img = bpy.data.images.load(str(path))
            a = np.empty(res * res * 4, np.float32)
            img.pixels.foreach_get(a)
            bpy.data.images.remove(img)
            path.unlink()
            yield k, a.reshape(res, res, 4)


def evaluate(cfg, cam, out_dir):
    """Per view: the panel pixel each camera ray reaches (pix_k, flat j*N+i), and
    the lens it enters (entered_k), read from a second render in which the MLA
    is opaque and emits its own lens id. All pixel renders first, then all lens
    renders: the material changes once, not twice per view. The views render as
    multi-view batches (view_cameras), the same pixels as one render per view."""
    sc = bpy.context.scene
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    n = cfg["panel_pixels"]
    mla = bpy.data.objects["MLA"]
    views = view_cameras(cam, cfg["views_mm"])
    for k, a in render_views(views, "eval", out_dir):
        i, j = _decode(a[:, :, 1], "pixel column", k), _decode(a[:, :, 2], "pixel row", k)
        np.save(out_dir / f"pix_{k}.npy", np.where((i >= 0) & (j >= 0), j * n + i, -1).astype(np.int32))
    mla.data.materials[0] = lens_id_material()
    for k, a in render_views(views, "entered", out_dir):
        np.save(out_dir / f"entered_{k}.npy", _decode(a[:, :, 0], "entered lens", k))
    sc.render.use_multiview = False
    sc.camera = cam
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


def colour_layers():
    """Three view layers R, G, B with "channel" 0, 1, 2 (read by dispersive glass),
    and a compositor that keeps channel c of layer c."""
    sc = bpy.context.scene
    layers = [sc.view_layers[0], sc.view_layers.new("G"), sc.view_layers.new("B")]
    layers[0].name = "R"
    ng = bpy.data.node_groups.new("combine_channels", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    out = ng.nodes.new("NodeGroupOutput")
    comb = ng.nodes.new("CompositorNodeCombineColor")
    for c, vl in enumerate(layers):
        vl["channel"] = float(c)
        rl = ng.nodes.new("CompositorNodeRLayers")
        rl.layer = vl.name
        sep = ng.nodes.new("CompositorNodeSeparateColor")
        ng.links.new(rl.outputs["Image"], sep.inputs["Image"])
        ng.links.new(sep.outputs[c], comb.inputs[c])
    ng.links.new(comb.outputs["Image"], out.inputs[0])
    sc.compositing_node_group = ng


def _any_dispersive(cfg):
    indices = [cfg["index"]]
    if cfg.get("remapper_npz"):
        r = np.load(cfg["remapper_npz"])
        indices += [r[f"surf{k}_index"] for k in range(int(r["n_surfaces"])) if f"surf{k}_index" in r]
    return any(isinstance(glass_index(i), tuple) for i in indices)


def main():
    cfg = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    tmp = Path(cfg["tmp_dir"])
    tmp.mkdir(parents=True, exist_ok=True)
    setup(cfg)
    if cfg["mode"] in ("target", "target_views"):
        add_content(cfg["content"])
        result = (target if cfg["mode"] == "target" else target_views)(cfg, add_camera(cfg), tmp)
        np.savez_compressed(cfg["out_npz"], **result)
        print(f"LF_BLENDER_DONE {cfg['mode']} {cfg['out_npz']}")
        return
    if cfg["mode"] in ("calibrate", "evaluate") and _any_dispersive(cfg):
        raise ValueError(f"{cfg['mode']} renders pixel ids along one path per camera ray: "
                         "a dispersive (three-index) glass picks one at random")
    if _any_dispersive(cfg):
        colour_layers()
    add_eye_reference()
    if "mla_npz" in cfg:                  # a plain display (no lenslets) has none
        add_mla(cfg)
    add_remapper(cfg)
    n = cfg["panel_pixels"]
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    if cfg["mode"] == "calibrate":
        rgb = np.stack([i, j, np.ones_like(i)], axis=-1).astype(np.float32)
    elif cfg["mode"] == "evaluate":
        # red 0: in the lens-id render (MLA opaque, emitting lens + 1) a ray that
        # reaches the panel through no lens reads 0, i.e. no lens, not lens 0
        rgb = np.stack([np.zeros_like(i), i + 1, j + 1], axis=-1).astype(np.float32)
    else:  # display, build
        rgb = None
    if rgb is None and "panel_stmap" in cfg:
        colour = stmap_colour(cfg["panel_stmap"])
    else:
        colour = image_colour(float_image("panel_pixels", np.load(cfg["panel_image_npy"]) if rgb is None else rgb))
    add_panel(cfg, colour)
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
