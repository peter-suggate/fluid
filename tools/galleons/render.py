"""Render a captured galleon frame in Cycles. Open the packed broadside.blend first."""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

capture, sky, image = map(Path, sys.argv[sys.argv.index("--") + 1:])
frame = json.loads((capture / "frame.json").read_text())
scene = bpy.context.scene
source = bpy.data.collections.new("Galleon source")
for obj in list(scene.objects):
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    source.objects.link(obj)
waterline = frame["container"]["height_m"] * frame["container"]["fillFraction"]
for ship in frame["galleons"]:
    obj = bpy.data.objects.new(ship["id"], None)
    scene.collection.objects.link(obj)
    obj.instance_type = 'COLLECTION'
    obj.instance_collection = source
    obj.location = (ship["x"], -ship["z"], waterline)
    obj.rotation_euler.z = math.pi if ship["heading"] < 0 else 0

# Interpolate the topmost water-to-air crossing in each native vertex column.
nx, ny, nz = frame["dims"]
phi = np.fromfile(capture / "water.f32", dtype="<f4").reshape(nz, ny, frame["rowBytes"] // 4)[:, :, :nx]
wet = phi[:, :-1] <= 0
crossings = wet & (phi[:, 1:] > 0)
z, x = np.indices((nz, nx))
y = ny - 2 - crossings[:, ::-1].argmax(axis=1)
lo, hi = phi[z, y, x], phi[z, y + 1, x]
height = frame["h"] * (y + np.divide(-lo, hi - lo, out=np.zeros_like(lo), where=hi != lo))
valid = crossings.any(axis=1)
points = np.stack((x * frame["h"] - frame["container"]["width_m"] / 2,
                   frame["container"]["depth_m"] / 2 - z * frame["h"], height), axis=-1).reshape(-1, 3)
faces = [(i, i + nx, i + nx + 1, i + 1) for zz, xx in np.argwhere(valid[:-1,:-1] & valid[1:,:-1] & valid[1:,1:] & valid[:-1,1:])
         for i in [int(xx + nx * zz)]]
mesh = bpy.data.meshes.new("Captured FLIP interface")
mesh.from_pydata(points.tolist(), [], faces)
mesh.update()
ocean = bpy.data.objects.new("Ocean", mesh)
scene.collection.objects.link(ocean)
for face in mesh.polygons:
    face.use_smooth = True
def material(name, rgb, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bs = mat.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*rgb, 1)
    bs.inputs['Roughness'].default_value = roughness
    return mat, bs

sea, surface = material("Atlantic water", (.012, .088, .105), .10)
surface.inputs["IOR"].default_value = 1.333
surface.inputs["Transmission Weight"].default_value = .30
mesh.materials.append(sea)
nodes, links = sea.node_tree.nodes, sea.node_tree.links
noise = nodes.new('ShaderNodeTexNoise')
coords = nodes.new('ShaderNodeTexCoord')
links.new(coords.outputs['Object'], noise.inputs['Vector'])
noise.inputs['Scale'].default_value = 1.15
noise.inputs['Detail'].default_value = 3.5
bump = nodes.new('ShaderNodeBump')
bump.inputs['Strength'].default_value = .31
bump.inputs['Distance'].default_value = .10
links.new(noise.outputs['Fac'], bump.inputs['Height'])
links.new(bump.outputs['Normal'], surface.inputs['Normal'])

def point(p):
    return Vector((p['x'], -p['z'], p['y']))

iron, _ = material("Cast iron", (.026, .022, .018), .44)
for shot in frame["shots"]:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=shot['radius'], location=point(shot['position']))
    bpy.context.object.data.materials.append(iron)

# Attenuating powder smoke is a photographic effect in this still.
for index, shot in enumerate(frame["shots"]):
    age = frame["time"] - shot["start"]
    if not 0 <= age < 1.3:
        continue
    heading = point(shot['velocity']).normalized()
    smoke = bpy.data.materials.new('Powder smoke')
    smoke.use_nodes = True
    nodes, links = smoke.node_tree.nodes, smoke.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    volume = nodes.new('ShaderNodeVolumePrincipled')
    volume.inputs['Color'].default_value = (.74, .69, .59, 1)
    tex = nodes.new('ShaderNodeTexCoord')
    grain = nodes.new('ShaderNodeTexNoise')
    links.new(tex.outputs['Generated'], grain.inputs['Vector'])
    grain.inputs['Scale'].default_value = 3.4
    grain.inputs['Detail'].default_value = 4
    mask = nodes.new('ShaderNodeVectorMath')
    mask.operation = 'DISTANCE'
    links.new(tex.outputs['Generated'], mask.inputs[0])
    mask.inputs[1].default_value = (.5, .5, .5)
    edge = nodes.new('ShaderNodeMath')
    edge.operation = 'SUBTRACT'
    links.new(grain.outputs['Fac'], edge.inputs[0])
    links.new(mask.outputs['Value'], edge.inputs[1])
    opacity = nodes.new('ShaderNodeMath')
    opacity.operation = 'MULTIPLY'
    opacity.inputs[1].default_value = 2.1
    links.new(edge.outputs[0], opacity.inputs[0])
    links.new(opacity.outputs[0], volume.inputs['Density'])
    links.new(volume.outputs['Volume'], output.inputs['Volume'])
    for k in range(4):
        pos = point(shot['muzzle']) + heading * (.65 + k * (.75 + age))
        pos.z += age * (.6 + .78*k)
        pos.y += .42*math.sin(index + k*1.7)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=pos)
        puff = bpy.context.object
        radius = .48 + age * 1.5 + .18*k
        puff.scale = (1.6*radius, radius, .94*radius)
        puff.data.materials.append(smoke)

scene.world = bpy.data.worlds.new('Clouded Atlantic sky')
scene.world.use_nodes = True
nodes, links = scene.world.node_tree.nodes, scene.world.node_tree.links
env = nodes.new('ShaderNodeTexEnvironment')
env.image = bpy.data.images.load(str(sky.resolve()))
links.new(env.outputs['Color'], nodes['Background'].inputs['Color'])
nodes['Background'].inputs['Strength'].default_value = .70
bpy.ops.object.light_add(type='SUN')
sun = bpy.context.object
sun.rotation_euler = Vector((.55, .48, -.68)).to_track_quat('-Z', 'Y').to_euler()
sun.data.energy = 2.0
sun.data.color = (1, .84, .66)
sun.data.angle = .035
bpy.ops.object.camera_add(location=(37, -62, 13))
cam = bpy.context.object
cam.rotation_euler = (Vector((0, 0, 20.5)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
cam.data.lens = 29.5
cam.data.clip_end = 1000
scene.camera = cam
scene.render.engine = 'CYCLES'
scene.cycles.samples = 48
scene.cycles.use_denoising = True
scene.render.resolution_x, scene.render.resolution_y = 1440, 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(image.resolve())
bpy.ops.render.render(write_still=True)
