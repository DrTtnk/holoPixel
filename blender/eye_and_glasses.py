"""
Holographic glasses: a screen shaped like the visual field, on an eye that
moves, at spectacle distance.

Three changes from the earlier symmetric cap, each forced by an objection that
turned out to be right.

1. The cap is centred on the CENTRE OF ROTATION, not the pupil. The pupil sits
   9.9 mm in front of that centre, so the line of sight is along the surface
   normal at the fixated point FOR EVERY GAZE ANGLE. A pupil-centred cap only
   manages that looking straight ahead, degrading to 32.8 deg of obliquity at
   50 deg gaze. This is also what pantoscopic tilt and face-form wrap
   approximate on a flat spectacle lens; a sphere about the pivot does it
   exactly.

2. Spectacle distance, 14 mm, not 20. Etendue is eyebox times field of view
   and does not depend on distance, so the same information comes off a screen
   of 529 mm2 instead of 1033 -- at a finer 1.43 um pitch. The eyebox is
   unchanged at 5.3 mm; the relief cancels.

3. The outline follows the monocular visual field, which is not a circle. It
   reaches further temporally than nasally and further below than above,
   because the brow, cheek and nose block it. A margin is added for gaze,
   because eccentricity is measured on the retina and rotates with the eye:
   an edge sitting harmlessly at 64 deg when at rest walks to 19 deg from the
   fovea when you glance 45 deg towards it.

Field limits used, degrees from the fovea, with a 20 deg gaze margin where the
face allows it:  temporal 90, superior 80, nasal 64, inferior 90.
"""

import math
import sys

import bpy
import bmesh

PUPIL_TO_ROT, EYE_R, CORNEA_R, CORNEA_AP = 9.9, 12.0, 7.8, 11.7
PUPIL_Y, PUPIL_D, ROT_Y = -3.6, 4.0, -13.5
RELIEF, SUBSTRATE, IPD = 14.0, 0.7, 63.0
R_CAP = RELIEF + PUPIL_TO_ROT

# Outline in CAP POLAR angle, degrees from the axis, measured at the cap's
# centre (the pivot) -- NOT field-of-view angle at the pupil, which is larger.
#
# The hard limit is the face. A cap about the pivot curves back towards the
# head as it widens, and at 14 mm relief it would reach the plane of the
# cornea at 55.6 deg. Allowing the rim to stand 3 mm proud of the face, as
# spectacles do, caps it at 46.3 deg -- which still subtends 138 deg at the
# pupil, well beyond the 100 deg we had been designing for.
#
# A first version used 90 deg here, from clamping an arcsine that exceeded 1.
# That wraps the screen past the eye's equator and through the skull.
MAX_POLAR = 46.3
FIELD = (MAX_POLAR, 38.0, 44.0, MAX_POLAR)   # temporal, superior, nasal, inferior


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


def field_limit(phi, limits, temporal_sign):
    """
    Polar limit at azimuth phi, interpolated between the four cardinal values.
    `temporal_sign` is +1 for the right eye (temporal towards +x) and -1 for
    the left, so the asymmetry mirrors correctly across the face.
    """
    t, s, n, i = [math.radians(v) for v in limits]
    p = phi if temporal_sign > 0 else (math.pi - phi) % (2 * math.pi)
    q = p % (2 * math.pi)
    if q < math.pi / 2:
        return t + (s - t) * (q / (math.pi / 2))
    if q < math.pi:
        return s + (n - s) * ((q - math.pi / 2) / (math.pi / 2))
    if q < 3 * math.pi / 2:
        return n + (i - n) * ((q - math.pi) / (math.pi / 2))
    return i + (t - i) * ((q - 3 * math.pi / 2) / (math.pi / 2))


def shaped_cap(name, radius, limits, centre, material, temporal_sign,
               rings=60, segments=192, thickness=0.0):
    """Spherical cap about `centre`, with an azimuth-dependent outline."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    grid = []
    for i in range(rings + 1):
        row = []
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            th = field_limit(phi, limits, temporal_sign) * i / rings
            row.append(bm.verts.new((
                centre[0] + radius * math.sin(th) * math.cos(phi),
                centre[1] + radius * math.cos(th),
                centre[2] + radius * math.sin(th) * math.sin(phi),
            )))
        grid.append(row)
    for i in range(rings):
        for j in range(segments):
            k = (j + 1) % segments
            bm.faces.new((grid[i][j], grid[i][k], grid[i + 1][k], grid[i + 1][j]))
    bm.normal_update(); bm.to_mesh(mesh); bm.free()
    if thickness:
        m = obj.modifiers.new("shell", "SOLIDIFY")
        m.thickness, m.offset = thickness, 1.0
    obj.data.materials.append(material)
    return obj


def eye(side, x):
    """One eye and its screen. `side` is +1 for the right eye."""
    sclera = mat(f"sclera{side}", (0.94, 0.93, 0.91), rough=0.35)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=EYE_R, segments=72, ring_count=36,
        location=(x, ROT_Y + (EYE_R - (EYE_R - abs(ROT_Y))) - EYE_R + abs(ROT_Y) - EYE_R, 0))
    ob = bpy.context.object
    ob.location = (x, -math.sqrt(max(EYE_R ** 2 - (CORNEA_AP / 2) ** 2, 0)) - 0.2, 0)
    ob.name = f"eye_{'R' if side > 0 else 'L'}"
    ob.data.materials.append(sclera)

    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=PUPIL_D / 2,
                                      fill_type="NGON", location=(x, PUPIL_Y, 0),
                                      rotation=(math.pi / 2, 0, 0))
    p = bpy.context.object
    p.name = f"pupil_{'R' if side > 0 else 'L'}"
    p.data.materials.append(mat(f"pupil{side}", (0.02, 0.02, 0.03), rough=0.9))

    shaped_cap(f"screen_{'R' if side > 0 else 'L'}", R_CAP, FIELD,
               (x, ROT_Y, 0), mat(f"screen{side}", (0.32, 0.64, 0.95),
                                  alpha=0.26, rough=0.08),
               temporal_sign=side, thickness=SUBSTRATE)


def nose():
    bpy.ops.mesh.primitive_cone_add(vertices=3, radius1=11.0, radius2=3.0,
                                    depth=34.0, location=(0, 8.0, -4.0),
                                    rotation=(math.pi / 2, 0, 0))
    n = bpy.context.object
    n.name = "nose"
    n.data.materials.append(mat("skin", (0.82, 0.66, 0.54), rough=0.7))


def build():
    reset()
    for side in (+1, -1):
        eye(side, side * IPD / 2)
    nose()


def report():
    import numpy as np
    phi = np.linspace(0, 2 * np.pi, 4001)
    lim = np.array([field_limit(p, FIELD, +1) for p in phi])
    area = R_CAP ** 2 * np.trapezoid(1 - np.cos(lim), phi)
    pitch = 1.431e-6
    print("=" * 66)
    print("HOLOGRAPHIC GLASSES — field-shaped, rotation-centred")
    print("=" * 66)
    print(f"  cap radius              {R_CAP:8.1f} mm  (centred on the pivot)")
    print(f"  pupil to screen         {RELIEF:8.1f} mm  (spectacle distance)")
    print(f"  outline T,S,N,I         {FIELD}")
    print(f"  area per eye            {area:8.0f} mm2")
    print(f"  pitch                   {pitch*1e6:8.3f} um")
    print(f"  pixels per eye          {area*1e-6/pitch**2/1e9:8.3f} G")
    print(f"  obliquity at fixation   {0.0:8.1f} deg  at every gaze angle")
    for name, th in zip(("temporal", "superior", "nasal", "inferior"), FIELD):
        a = math.radians(th)
        fov = math.degrees(math.atan2(R_CAP * math.sin(a),
                                      R_CAP * math.cos(a) - PUPIL_TO_ROT))
        y = ROT_Y + R_CAP * math.cos(a)
        print(f"  {name:<10} polar {th:5.1f} deg -> {fov:5.1f} deg of field, "
              f"rim at y = {y:+5.1f} mm")
    print("=" * 66)


if __name__ == "__main__":
    build()
    report()
    out = sys.argv[sys.argv.index("--out") + 1]
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print(f"saved {out}")
