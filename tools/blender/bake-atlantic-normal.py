"""Bake the authored Cycles basalt bump into a portable GLB tangent normal map.

Run after export-atlantic-lighthouse.py, before bake_mesh_volume.py. The normal
map changes shading only; the physical terrain mesh is preserved exactly.
"""
import argparse
from pathlib import Path
import sys

import bpy

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=ROOT/'assets/atlantic-lighthouse/storm-lighthouse.blend')
parser.add_argument('--size',type=int,default=2048)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
asset=ROOT/'public/assets/atlantic-lighthouse/atlantic-lighthouse-rock.glb'
bpy.ops.import_scene.gltf(filepath=str(asset))
objects=[o for o in bpy.context.selected_objects if o.type=='MESH']
assert len(objects)==1
ob=objects[0];bpy.context.view_layer.objects.active=ob
materials=list(ob.data.materials);indices=[p.material_index for p in ob.data.polygons]
with bpy.data.libraries.load(str(args.source),link=False) as (source,destination):
    destination.materials=[name for name in source.materials if name.startswith('Wet Atlantic basalt')]
assert len(destination.materials)==1
source_material=destination.materials[0]
ob.data.materials.clear();ob.data.materials.append(source_material)
for face in ob.data.polygons:face.material_index=0
bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=1.15,island_margin=.005)
bpy.ops.object.mode_set(mode='OBJECT')
image=bpy.data.images.new('Atlantic basalt normal',width=args.size,height=args.size,alpha=False)
image.colorspace_settings.name='Non-Color'
target=source_material.node_tree.nodes.new('ShaderNodeTexImage');target.image=image
source_material.node_tree.nodes.active=target
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=1
scene.render.bake.margin=8;scene.render.bake.use_clear=True
bpy.ops.object.bake(type='NORMAL',normal_space='TANGENT')
image.filepath_raw=str(ROOT/'assets/atlantic-lighthouse/basalt-normal.png');image.file_format='PNG';image.save()
ob.data.materials.clear()
for material in materials:
    ob.data.materials.append(material)
    nodes,links=material.node_tree.nodes,material.node_tree.links
    tex=nodes.new('ShaderNodeTexImage');tex.image=image
    normal=nodes.new('ShaderNodeNormalMap');normal.inputs['Strength'].default_value=1
    links.new(tex.outputs['Color'],normal.inputs['Color']);links.new(normal.outputs['Normal'],nodes.get('Principled BSDF').inputs['Normal'])
for face,index in zip(ob.data.polygons,indices):face.material_index=index
bpy.ops.object.select_all(action='DESELECT');ob.select_set(True);bpy.context.view_layer.objects.active=ob
bpy.ops.export_scene.gltf(filepath=str(asset),export_format='GLB',use_selection=True,export_apply=True,
    export_yup=True,export_normals=True,export_materials='EXPORT')
print('BAKED portable tangent normals',asset,flush=True)
