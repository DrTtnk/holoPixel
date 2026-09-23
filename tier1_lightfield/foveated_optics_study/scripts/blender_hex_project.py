"""Blender scene starter for the global-remapper and tiled-freeform hex variants."""
from __future__ import annotations
import math
import bpy

PANEL_MM = 8.176
HEX_SIDE_UM = 17.37
GLOBAL_Z_MM = 0.8
TILED_Z_MM = 0.0
PUPIL_Z_MM = 20.0
PUPIL_DIAMETER_MM = 4.0

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def ensure_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def material(name, rgba, metallic=0.0, roughness=0.4):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = rgba
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    return m

def create_hex_mesh(name, radius_mm, depth_mm=0.025):
    verts = []
    for z in (-depth_mm/2, depth_mm/2):
        for i in range(6):
            a = math.radians(30 + 60*i)
            verts.append((radius_mm*math.cos(a), radius_mm*math.sin(a), z))
    faces = [tuple(range(6)), tuple(range(6,12))]
    for i in range(6):
        j = (i+1) % 6
        faces.append((i,j,6+j,6+i))
    mesh = bpy.data.meshes.new(name+'Mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh

def hex_centers(panel_mm, side_mm):
    dx = 1.5*side_mm
    dy = math.sqrt(3)*side_mm
    y = -panel_mm/2 + side_mm
    row = 0
    while y <= panel_mm/2 - side_mm:
        offset = 0.0 if row % 2 == 0 else 0.75*side_mm
        x = -panel_mm/2 + side_mm + offset
        while x <= panel_mm/2 - side_mm:
            yield x, y
            x += dx
        y += dy
        row += 1

def field_angles(x_mm, y_mm):
    # Visualization placeholder only. Replace with optimized table import.
    xn = 2*x_mm/PANEL_MM
    yn = 2*y_mm/PANEL_MM
    thx = 35.0*math.copysign(abs(xn)**1.65, xn) if xn else 0.0
    thy = 22.5*math.copysign(abs(yn)**1.65, yn) if yn else 0.0
    return thx, thy

def build_hex_variant(collection_name, tiled_freeform=False, z_mm=0.0):
    col = ensure_collection(collection_name)
    side_mm = HEX_SIDE_UM*1e-3
    mesh = create_hex_mesh(collection_name+'Tile', side_mm*0.96)
    rgba = (0.18,0.5,0.9,1.0) if not tiled_freeform else (0.9,0.35,0.18,1.0)
    mat = material(collection_name+'Material', rgba, metallic=0.1)
    for idx,(x,y) in enumerate(hex_centers(PANEL_MM, side_mm)):
        obj = bpy.data.objects.new(f'{collection_name}_{idx:05d}', mesh)
        col.objects.link(obj)
        obj.location = (x,y,z_mm)
        if tiled_freeform:
            thx, thy = field_angles(x,y)
            obj.rotation_euler[1] = math.radians(thx/2)
            obj.rotation_euler[0] = -math.radians(thy/2)
        obj.data.materials.append(mat)

def main():
    clear_scene()
    build_hex_variant('GLOBAL_REMAPPER_VARIANT', False, GLOBAL_Z_MM)
    build_hex_variant('TILED_FREEFORM_VARIANT', True, TILED_Z_MM)
    bpy.ops.mesh.primitive_cube_add(location=(0,0,12.0), scale=(2.8,2.8,0.08))
    bpy.context.object.name = 'STEERING_MIRROR'
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=PUPIL_DIAMETER_MM/2, fill_type='NGON', location=(0,0,PUPIL_Z_MM))
    bpy.context.object.name = 'PUPIL_PLANE'
    print('Generated HoloPixel foveated hex optics scene. Replace placeholder mapping with optimizer output.')

if __name__ == '__main__':
    main()