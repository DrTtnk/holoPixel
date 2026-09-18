"""Render the eye-and-screen model from several angles."""
import math, sys
import bpy
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--outdir") + 1]

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = int(__import__('os').environ.get('SAMPLES', 96))
scene.cycles.use_denoising = True
scene.cycles.device = "CPU"
scene.render.resolution_x = 1400
scene.render.resolution_y = 1000
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("w")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.07, 1)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

for name, loc, energy in (("key", (60, -80, 60), 400000),
                          ("fill", (-70, -40, 20), 150000),
                          ("rim", (0, 60, 40), 200000)):
    light = bpy.data.lights.new(name, "POINT")
    light.energy = energy
    ob = bpy.data.objects.new(name, light)
    ob.location = loc
    bpy.context.collection.objects.link(ob)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 52
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

TARGET = Vector((0, 2.0, 0))

def look_from(pos):
    cam.location = pos
    d = TARGET - Vector(pos)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

# The geometry is a half cut on the YZ plane, so a camera on -X sees the
# section face; the optical axis (-Y to +Y) then runs across the frame.
VIEWS = {
    "section":       (-92, 2, 0),
    "three_quarter": (-64, -30, 44),
    "from_scene":    (-14, 78, 12),
    "grazing":       (-46, 30, 10),
}

only = __import__("os").environ.get("ONLY")
for name, pos in VIEWS.items():
    if only and name != only:
        continue
    look_from(pos)
    scene.render.filepath = f"{OUT}/eye_{name}.png"
    bpy.ops.render.render(write_still=True)
    print(f"rendered {name}")
