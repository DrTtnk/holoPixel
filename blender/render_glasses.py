"""Render the field-shaped holographic glasses."""
import math, os, sys
import bpy
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--outdir") + 1]
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = int(os.environ.get("SAMPLES", 64))
scene.cycles.use_denoising = True
scene.cycles.device = "CPU"
scene.render.resolution_x, scene.render.resolution_y = 1500, 1000
scene.world = bpy.data.worlds.new("w")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.06, 0.06, 0.08, 1)

for name, loc, e in (("key", (90, -120, 90), 900000), ("fill", (-110, -60, 20), 350000),
                     ("rim", (0, 110, 60), 500000)):
    l = bpy.data.lights.new(name, "POINT"); l.energy = e
    o = bpy.data.objects.new(name, l); o.location = loc
    bpy.context.collection.objects.link(o)

cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 55
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam); scene.camera = cam
TARGET = Vector((0, 2, 0))

VIEWS = {"front":        (0, 215, 10),
         "three_quarter":(150, 130, 70),
         "from_above":   (0, 20, 210),
         "side":         (210, 6, 8)}

only = os.environ.get("ONLY")
for name, pos in VIEWS.items():
    if only and name != only:
        continue
    cam.location = pos
    cam.rotation_euler = (TARGET - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{OUT}/glasses_{name}.png"
    bpy.ops.render.render(write_still=True)
    print(f"rendered {name}")
