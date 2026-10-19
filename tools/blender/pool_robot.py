"""Blender --background --python tools/blender/pool_robot.py

BUB: an original poolside droid. Ellipsoid parts are the editable Blender
model AND the runtime analytic mesh recipe, keeping silhouette/materials in
sync without loading another renderer. Coordinates in the recipe are Y-up,
centred on the one-metre character, facing +Z.
"""
import bpy
import json
from pathlib import Path
from mathutils import Vector
ROOT = Path(__file__).resolve().parents[2]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
colors = {'porcelain': (.86,.82,.68), 'lagoon': (.025,.32,.34),
          'rubber': (.018,.027,.038), 'lens': (.25,.85,1),
          'coral': (.95,.20,.10), 'steel': (.32,.38,.40)}
mats = {}
for name, rgb in colors.items():
    m = bpy.data.materials.new(name); m.use_nodes = True
    m.diffuse_color = (*rgb,1)
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*rgb,1)
    bsdf.inputs['Roughness'].default_value = .32
    bsdf.inputs['Metallic'].default_value = .65 if name == 'steel' else .08
    mats[name] = m
parts = []
def part(name, center, radius, material):
    x,y,z = center; rx,ry,rz = radius
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=24, location=(x,-z,y))
    obj = bpy.context.object; obj.name = name; obj.scale = (rx,rz,ry)
    obj.data.materials.append(mats[material])
    for polygon in obj.data.polygons: polygon.use_smooth = True
    obj['runtime_primitive'] = 'ellipsoid'
    parts.append({'name':name,'center':center,'radius':radius,'color':colors[material]})
part('rubber bumper', (0,-.34,0), (.33,.12,.28), 'rubber')
part('pear shaped enamel chassis', (0,-.13,0), (.32,.29,.27), 'porcelain')
part('lagoon belly plate', (0,-.12,.236), (.205,.18,.055), 'lagoon')
part('neck swivel', (0,.13,0), (.23,.045,.22), 'steel')
part('oversized round head', (0,.265,0), (.31,.195,.275), 'porcelain')
part('dark friendly visor', (0,.285,.232), (.245,.105,.065), 'rubber')
for side in [-1,1]:
    part('bright eye '+str(side), (side*.103,.30,.293), (.048,.052,.021), 'lens')
    part('eye glint '+str(side), (side*.09,.32,.31), (.015,.017,.008), 'porcelain')
    part('wheel '+str(side), (side*.255,-.405,-.015), (.10,.095,.16), 'rubber')
    part('hub '+str(side), (side*.335,-.405,-.015), (.021,.052,.065), 'lagoon')
    part('little arm '+str(side), (side*.335,-.08,0), (.045,.12,.08), 'lagoon')
part('smile', (0,.226,.294), (.055,.014,.012), 'porcelain')
part('power button', (0,-.07,.288), (.035,.035,.012), 'coral')
for x in [-.07,0,.07]:
    part('belly indicator '+str(x), (x,-.19,.283), (.016,.012,.012), 'lens')
part('antenna', (.17,.454,-.03), (.015,.03,.015), 'steel')
part('antenna light', (.17,.484,-.03), (.025,.016,.025), 'coral')
# The recipe is extracted from the authored Blender objects, not a separate
# runtime reconstruction. Applying edits to transforms/materials here changes
# the GLB and the runtime together.
for record in parts:
    obj = bpy.data.objects[record['name']]
    record['center'] = [round(obj.location.x,6),round(obj.location.z,6),round(-obj.location.y,6)]
    record['radius'] = [round(obj.scale.x,6),round(obj.scale.z,6),round(obj.scale.y,6)]
    record['color'] = [round(v,6) for v in obj.active_material.diffuse_color[:3]]
(ROOT/'lib/interaction/features/robot/bub.generated.json').write_text(json.dumps({'parts':parts},indent=2)+'\n')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(filepath=str(ROOT/'public/assets/characters/bub.glb'), export_format='GLB', use_selection=True, export_yup=True)
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'assets/characters/bub.blend'))
# Studio preview is saved separately from the clean editable asset.
bpy.ops.mesh.primitive_plane_add(size=200, location=(0,0,-.502))
floor=bpy.context.object
floor.data.materials.append(mats['lagoon'])
world=bpy.data.worlds.new('Studio'); world.use_nodes=True
world.node_tree.nodes['Background'].inputs[0].default_value=(.45,.58,.65,1)
world.node_tree.nodes['Background'].inputs[1].default_value=.4
scene.world=world
for loc,power,size in [((2,-3,4),400,3),((-3,-1,2),250,2),((1,3,3),450,2)]:
    bpy.ops.object.light_add(type='AREA',location=loc)
    light=bpy.context.object; light.data.energy=power; light.data.shape='DISK'; light.data.size=size
    light.rotation_euler=(-light.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(1.35,-2.5,1.05))
scene.camera=bpy.context.object
scene.camera.rotation_euler=(Vector((0,0,0))-scene.camera.location).to_track_quat('-Z','Y').to_euler()
scene.camera.data.type='ORTHO'; scene.camera.data.ortho_scale=1.55
scene.render.engine='CYCLES'; scene.cycles.samples=32; scene.cycles.use_denoising=True
scene.render.resolution_x=900; scene.render.resolution_y=900; scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX'
scene.render.filepath=str(ROOT/'public/assets/characters/bub-preview.png')
bpy.ops.render.render(write_still=True)
