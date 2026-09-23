"""
Eye and holographic screen, to scale, as a half-section.

Built headless into a fresh file, so nothing already open in Blender is
touched. One Blender unit is one millimetre and the scene unit scale matches,
so the N-panel reads real dimensions.

The optical axis is Y: the cornea faces +Y, the screen sits in front of it at
+Y, and the scene would be beyond that; the retina is at negative Y. Geometry
is generated parametrically rather than cut with booleans, so passing --half
yields a clean section (the x >= 0 side) with no boolean to go wrong.

Anatomy, standard schematic eye, corneal apex at the origin:

    corneal apex            y =   0.0
    entrance pupil          y =  -3.6 mm   anterior chamber depth
    centre of rotation      y = -13.5 mm   what the eye pivots about
    retina                  y = -24.2 mm   axial length
    eyeball radius              12.0 mm
    corneal radius               7.8 mm over an 11.7 mm aperture
    pupil                        4.0 mm    bright-light average, 2-8 mm range

Screen, from this project's derivations:

    eye relief                  20.0 mm    pupil to screen
    field of view              100 deg
    cap radius                  20.0 mm    centred ON THE PUPIL, so every point
                                           faces the eye and a pixel need only
                                           cover the eyebox, not the field of
                                           view -- worth 27x in pixel count
    cap aperture                30.6 mm
    cap sagitta                  7.1 mm
    flat equivalent             47.7 mm    1.56x wider, and further out
"""

import math
import sys

import bpy
import bmesh

EYE_R, CORNEA_R, CORNEA_APERTURE = 12.0, 7.8, 11.7
PUPIL_Y, PUPIL_D, ROTATION_Y, AXIAL = -3.6, 4.0, -13.5, -24.2
RELIEF, FOV_DEG, SUBSTRATE, EYEBOX = 20.0, 100.0, 0.7, 10.0

HALF = math.radians(FOV_DEG / 2)
CAP_APERTURE = 2 * RELIEF * math.sin(HALF)
CAP_SAGITTA = RELIEF * (1 - math.cos(HALF))
FLAT_D = 2 * RELIEF * math.tan(HALF)
CAP_AREA = 2 * math.pi * RELIEF ** 2 * (1 - math.cos(HALF))


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    u = bpy.context.scene.unit_settings
    u.system, u.scale_length, u.length_unit = "METRIC", 0.001, "MILLIMETERS"


def mat(name, colour, alpha=1.0, rough=0.4, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*colour, 1.0)
    b.inputs["Roughness"].default_value = rough
    if emit:
        b.inputs["Emission Color"].default_value = (*colour, 1.0)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        b.inputs["Transmission Weight"].default_value = 1.0 - alpha
    return m


FULL = "--half" not in sys.argv     # a whole solid unless a section is asked for


def half_cap(name, radius, theta0, theta1, centre_y, facing, material,
             rings=64, segments=64, thickness=0.0):
    """
    A spherical zone about the Y axis. `theta` is measured from the apex;
    `facing` is +1 for a cap opening towards +Y. With --half only the x >= 0
    side is generated, which exposes the interior for a section drawing.
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    grid = []
    for i in range(rings + 1):
        th = theta0 + (theta1 - theta0) * i / rings
        row = []
        for j in range(segments + 1):
            ph = (2 * math.pi if FULL else math.pi) * j / segments
            row.append(bm.verts.new((
                radius * math.sin(th) * math.sin(ph),
                centre_y + facing * radius * math.cos(th),
                radius * math.sin(th) * math.cos(ph),
            )))
        grid.append(row)
    for i in range(rings):
        for j in range(segments):
            bm.faces.new((grid[i][j], grid[i][j + 1],
                          grid[i + 1][j + 1], grid[i + 1][j]))
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    if thickness:
        m = obj.modifiers.new("shell", "SOLIDIFY")
        m.thickness, m.offset = thickness, 1.0
    obj.data.materials.append(material)
    return obj


def half_annulus(name, inner, outer, y, material, segments=64):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    rows = []
    span = 2 * math.pi if FULL else math.pi
    for r in (inner, outer):
        rows.append([bm.verts.new((r * math.sin(span * j / segments), y,
                                   r * math.cos(span * j / segments)))
                     for j in range(segments + 1)])
    for j in range(segments):
        bm.faces.new((rows[0][j], rows[0][j + 1], rows[1][j + 1], rows[1][j]))
    bm.normal_update(); bm.to_mesh(mesh); bm.free()
    obj.data.materials.append(material)
    return obj


def build():
    reset()
    build_eye()
    build_screen()


def build_eye():
    """The schematic eye alone, into the current scene, with no reset."""
    cornea_half = math.asin((CORNEA_APERTURE / 2) / CORNEA_R)
    # where the cornea meets the sclera, as an angle on the eyeball
    sclera_start = math.asin(min(1.0, (CORNEA_APERTURE / 2) / EYE_R))
    eye_centre_y = -CORNEA_R * math.cos(cornea_half) - \
        math.sqrt(max(EYE_R ** 2 - (CORNEA_APERTURE / 2) ** 2, 0)) + \
        CORNEA_R * math.cos(cornea_half)
    eye_centre_y = -math.sqrt(EYE_R ** 2 - (CORNEA_APERTURE / 2) ** 2) - \
        (CORNEA_R * math.cos(cornea_half) - CORNEA_R) - 0.0

    half_cap("sclera", EYE_R, sclera_start, math.pi, eye_centre_y, +1,
             mat("sclera", (0.94, 0.93, 0.91), rough=0.35))
    half_cap("cornea", CORNEA_R, 0.0, cornea_half, -CORNEA_R, +1,
             mat("cornea", (0.75, 0.87, 0.96), alpha=0.22, rough=0.03))
    half_annulus("iris", PUPIL_D / 2, CORNEA_APERTURE / 2, PUPIL_Y,
                 mat("iris", (0.24, 0.40, 0.55), rough=0.6))
    half_annulus("pupil_4mm", 0.0, PUPIL_D / 2, PUPIL_Y - 0.03,
                 mat("pupil", (0.01, 0.01, 0.02), rough=0.95))

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=(0, ROTATION_Y, 0))
    r = bpy.context.object
    r.name = "centre_of_rotation"
    r.data.materials.append(mat("rot", (1.0, 0.3, 0.08), emit=6.0))


def build_screen():
    half_annulus("eyebox_10mm", EYEBOX / 2 - 0.15, EYEBOX / 2, PUPIL_Y + 0.4,
                 mat("eyebox", (1.0, 0.85, 0.2), emit=4.0))

    half_cap("holo_screen_spherical", RELIEF, 0.0, HALF, PUPIL_Y, +1,
             mat("screen", (0.30, 0.62, 0.95), alpha=0.28, rough=0.08),
             thickness=SUBSTRATE)
    half_annulus("holo_screen_flat_alternative", 0.0, FLAT_D / 2,
                 PUPIL_Y + RELIEF,
                 mat("flat", (0.95, 0.42, 0.20), alpha=0.10, rough=0.3))


def report():
    w = 30
    print("=" * 70)
    print("EYE AND HOLOGRAPHIC SCREEN — half-section, millimetres")
    print("=" * 70)
    for k, v in (("corneal apex", 0.0), ("entrance pupil", PUPIL_Y),
                 ("centre of rotation", ROTATION_Y), ("retina", AXIAL)):
        print(f"  {k:<{w}} y = {v:8.2f}")
    print(f"  {'eyeball diameter':<{w}}     {2*EYE_R:8.2f}")
    print(f"  {'pupil diameter':<{w}}     {PUPIL_D:8.2f}")
    print(f"  {'eyebox diameter':<{w}}     {EYEBOX:8.2f}")
    print()
    print(f"  {'eye relief, pupil to screen':<{w}}     {RELIEF:8.2f}")
    print(f"  {'field of view':<{w}}     {FOV_DEG:8.1f} deg")
    print(f"  {'cap aperture':<{w}}     {CAP_APERTURE:8.2f}")
    print(f"  {'cap sagitta':<{w}}     {CAP_SAGITTA:8.2f}")
    print(f"  {'cap area':<{w}}     {CAP_AREA:8.1f} mm2")
    print(f"  {'screen apex at':<{w}} y = {PUPIL_Y + RELIEF:8.2f}")
    print(f"  {'nearest screen edge at':<{w}} y = "
          f"{PUPIL_Y + RELIEF*math.cos(HALF):8.2f}")
    print(f"  {'clearance, cornea to edge':<{w}}     "
          f"{PUPIL_Y + RELIEF*math.cos(HALF):8.2f}")
    print()
    print(f"  {'FLAT equivalent diameter':<{w}}     {FLAT_D:8.2f}  "
          f"({FLAT_D/CAP_APERTURE:.2f}x wider)")
    print("=" * 70)


if __name__ == "__main__":
    build()
    report()
    out = sys.argv[sys.argv.index("--out") + 1]
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print(f"saved {out}")
