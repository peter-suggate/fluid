"""Prepare the two posed CC0 figures for opaque, editable signed-distance sampling.
Run in Blender with -- <terrace-host|terrace-reader> <original downloaded GLB>.
Source URLs and authors are in assets/pool-neighborhood/CREDITS.md.
"""
from pathlib import Path
import sys
import bpy
import bmesh
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc

root = Path(__file__).resolve().parents[2]
asset, source = sys.argv[sys.argv.index('--') + 1:]
assert asset in ('terrace-host', 'terrace-reader')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.import_scene.gltf(filepath=source)
for obj in list(bpy.context.scene.objects):
    if obj.type == 'MESH' and ('parcel' in obj.name.lower() or 'teeth' in obj.name.lower()):
        bpy.data.objects.remove(obj, do_unlink=True)
# Transfer alpha-cut hair and brows into the editable surface. Treating the
# transparent margins of the source cards as solid makes the hair a helmet.
for obj in list(bpy.context.scene.objects):
    if obj.type != 'MESH' or len(obj.data.materials) != 1:
        continue
    mat = obj.data.materials[0]
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if not bsdf or not bsdf.inputs['Alpha'].is_linked:
        continue
    color = bsdf.inputs['Base Color']
    if not color.is_linked:
        continue
    image = color.links[0].from_node
    if image.type != 'TEX_IMAGE' or not image.image:
        continue
    image = image.image
    width, height = image.size
    pixels = np.empty(width*height*4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    opacity = pixels.reshape(height,width,4)[:,:,3]
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.subdivide_edges(bm,edges=list(bm.edges),cuts=3,use_grid_fill=True)
    uv = bm.loops.layers.uv.active
    def visible(face):
        u = sum(loop[uv].uv.x for loop in face.loops)/len(face.loops)
        v = sum(loop[uv].uv.y for loop in face.loops)/len(face.loops)
        return opacity[int((v%1)*height), int((u%1)*width)] >= (.35 if 'eye' in obj.name else .28)
    bmesh.ops.delete(bm,geom=[face for face in bm.faces if not visible(face)],context='FACES')
    bm.to_mesh(obj.data)
    bm.free()
parts = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
bpy.ops.object.select_all(action='DESELECT')
for obj in parts:
    obj.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
original = bpy.context.object
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
mesh = original.data
mesh.calc_loop_triangles()
verts = [v.co.copy() for v in mesh.vertices]
tris = [list(t.vertices) for t in mesh.loop_triangles]
uv = mesh.uv_layers.active.data
corner_uvs = [[uv[i].uv.copy() for i in t.loops] for t in mesh.loop_triangles]
slots = [t.material_index for t in mesh.loop_triangles]
bvh = BVHTree.FromPolygons(verts, tris, all_triangles=True)
# Garment hems and the hidden-body cutouts are open in the distribution mesh.
# Give them a physical back before fusing overlapping garments, hair and skin.
solid = original.modifiers.new('Inside garment lining', 'SOLIDIFY')
solid.thickness = .012
solid.offset = -1
bpy.ops.object.modifier_apply(modifier=solid.name)
original.data.remesh_voxel_size = .006
original.data.use_remesh_preserve_attributes = False
bpy.ops.object.voxel_remesh()
decimate = original.modifiers.new('Adaptive silhouette', 'DECIMATE')
decimate.ratio = .48
bpy.ops.object.modifier_apply(modifier=decimate.name)
mesh = original.data
target_uv = mesh.uv_layers.new(name='Source colour')
for poly in mesh.polygons:
    _, _, index, _ = bvh.find_nearest(poly.center)
    poly.material_index = slots[index]
    poly.use_smooth = True
    triangle = [verts[v] for v in tris[index]]
    a, b, c = corner_uvs[index]
    for li in poly.loop_indices:
        co = mesh.vertices[mesh.loops[li].vertex_index].co
        weights = poly_3d_calc(triangle, co)
        target_uv.data[li].uv = a * weights[0] + b * weights[1] + c * weights[2]
for mat in mesh.materials:
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    for socket in ('Alpha',):
        for link in list(bsdf.inputs[socket].links):
            mat.node_tree.links.remove(link)
    bsdf.inputs['Alpha'].default_value = 1
    if bsdf.inputs['Base Color'].is_linked:
        tex = bsdf.inputs['Base Color'].links[0].from_node
        if tex.type == 'TEX_IMAGE' and tex.image:
            img = tex.image
            if max(img.size) > 1024:
                factor=1024/max(img.size)
                img.scale(round(img.size[0]*factor), round(img.size[1]*factor))
                img.pack()
    bsdf.inputs['Roughness'].default_value = .65
original.name = asset
bpy.ops.export_scene.gltf(filepath=str(root/'public/assets/pool-neighborhood'/f'{asset}.glb'),
                          use_selection=True, export_format='GLB', export_yup=True)
bpy.data.orphans_purge(do_recursive=True)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(root/'assets/pool-neighborhood'/f'{asset}.blend'), compress=True)
