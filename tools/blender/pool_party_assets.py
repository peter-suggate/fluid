"""Reproducible, metre-scale pool furniture. Run with Blender --background --python.

Outputs editable source, individual GLBs, and a Cycles art-direction preview.
Blender uses Z up; glTF and the volume baker both export (x, z, -y).
"""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, default=64)
parser.add_argument('--no-render', action='store_true')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
OUT = ROOT / 'public/assets/pool-party'
SOURCE = ROOT / 'assets/pool-party'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for c in list(bpy.data.collections):
    if c.name != 'Collection':
        bpy.data.collections.remove(c)
scene = bpy.context.scene
bpy.context.preferences.filepaths.save_version = 0
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1
rng = np.random.default_rng(20261018)


def texture(name, rgb, style):
    size = 512
    y, x = np.mgrid[:size, :size] / size
    noise = rng.normal(0, 1, (size, size))
    if style == 'wood':
        grain = np.sin(2 * math.pi * (x * 47 + .45 * np.sin(y * 8) + .12 * np.sin(y * 31)))
        variation = .82 + .12 * grain + .03 * noise + .09 * np.sin(x * 21 + y * 3)
    elif style == 'fabric':
        variation = .94 + .025 * noise + .045 * np.sin(x * math.pi * 256) * np.sin(y * math.pi * 256)
    else:
        variation = .96 + .02 * noise
    pixels = np.ones((size, size, 4), dtype=np.float32)
    pixels[:, :, :3] = np.clip(np.array(rgb)[None, None, :] * variation[:, :, None], 0, 1)
    img = bpy.data.images.new(name, width=size, height=size)
    img.pixels.foreach_set(pixels.ravel())
    img.filepath_raw = str(OUT / (name + '.png'))
    img.file_format = 'PNG'
    img.save()
    img.pack()
    return img


def material(name, rgb, roughness=.5, metallic=0, style=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (*rgb, 1)
    p = mat.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*rgb, 1)
    p.inputs['Roughness'].default_value = roughness
    p.inputs['Metallic'].default_value = metallic
    if style:
        t = mat.node_tree.nodes.new('ShaderNodeTexImage')
        t.image = texture(name, rgb, style)
        mat.node_tree.links.new(t.outputs['Color'], p.inputs['Base Color'])
    return mat


teak = material('oiled-teak', (.48, .27, .115), .36, style='wood')
canvas = material('ivory-canvas', (.88, .84, .72), .85, style='fabric')
teal = material('lagoon-linen', (.055, .36, .37), .82, style='fabric')
coral = material('terracotta-linen', (.67, .19, .105), .84, style='fabric')
steel = material('satin-stainless', (.56, .60, .62), .26, .95)
dark = material('charcoal-rubber', (.022, .028, .029), .72)
enamel = material('lagoon-enamel', (.025, .29, .32), .24, .12)
white = material('warm-white-polymer', (.82, .83, .77), .32)
ceramic = material('glazed-ivory', (.89, .83, .68), .19)
lemon = material('lemon-rind', (.95, .57, .018), .48)
stone = material('travertine', (.62, .54, .41), .67, style='stone')
active = None


def collection(name):
    global active
    active = bpy.data.collections.new(name)
    scene.collection.children.link(active)
    return active


def finish(obj, name, mat, bevel=0, smooth=True):
    obj.name = name
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    active.objects.link(obj)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new('Machined / sewn edge', 'BEVEL')
        mod.width = bevel
        mod.segments = 4
    if smooth and obj.type == 'MESH':
        for poly in obj.data.polygons:
            poly.use_smooth = True
        mod = obj.modifiers.new('Face-weighted normals', 'WEIGHTED_NORMAL')
        mod.keep_sharp = True
        mod.weight = 50
    return obj


def box(name, at, size, mat, bevel=.012, rotation=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=at)
    obj = bpy.context.object
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.rotation_euler.x = rotation
    return finish(obj, name, mat, bevel)


def rod(name, a, b, radius, mat, vertices=32):
    a, b = Vector(a), Vector(b)
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=(b-a).length, location=(a+b)/2)
    obj = bpy.context.object
    obj.rotation_euler = (b-a).to_track_quat('Z', 'Y').to_euler()
    return finish(obj, name, mat, min(radius*.18, .005))


def sphere(name, at, scale, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=24, location=at)
    obj = bpy.context.object
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, mat)


def tube(name, points, radius, mat, closed=False):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    curve.use_fill_caps = True
    spline = curve.splines.new('POLY')
    spline.points.add(len(points)-1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1)
    spline.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, curve)
    active.objects.link(obj)
    curve.materials.append(mat)
    return obj


def piping(name, at, width, length, radius, mat, tilt=0):
    pts = []
    for cx, cy, start in [(width/2-radius, length/2-radius, 0),
                           (-width/2+radius, length/2-radius, 90),
                           (-width/2+radius, -length/2+radius, 180),
                           (width/2-radius, -length/2+radius, 270)]:
        for i in range(9):
            a = math.radians(start+i*90/8)
            x, y = cx+radius*math.cos(a), cy+radius*math.sin(a)
            pts.append((at[0]+x, at[1]+y*math.cos(tilt), at[2]+y*math.sin(tilt)))
    tube(name, pts, .004, mat, True)


assets = {}
collection('teak-lounger')
assets[active.name] = active
for side in [-1, 1]:
    box('teak-side-rail', (side*.365, 0, .34), (.065, 2.02, .10), teak)
    for end in [-1, 1]:
        box('tapered-leg', (side*.32, end*.77, .17), (.075, .095, .34), teak)
        box('rubber-foot', (side*.32, end*.77, .018), (.08, .10, .036), dark, .009)
    rod('recline-pivot', (side*.34, -.30, .43), (side*.40, -.30, .43), .026, steel)
for i in range(18):
    box('spaced-teak-slat', (0, -.94+i*.108, .365), (.70, .084, .035), teak, .008)
box('seat-cushion', (0, .32, .445), (.69, 1.28, .13), teal, .055)
piping('seat-welt', (0, .32, .474), .68, 1.27, .07, canvas)
tilt = -.52
box('reclined-back-cushion', (0, -.64, .635), (.69, .79, .14), teal, .06, tilt)
piping('back-welt', (0, -.64, .68), .68, .78, .07, canvas, tilt)
box('head-pillow', (0, -.85, .815), (.48, .25, .13), canvas, .06, tilt)
for side in [-1, 1]:
    rod('back-support', (side*.29, -.85, .37), (side*.29, -.89, .79), .022, steel)
box('folded-pool-towel', (0, .64, .535), (.57, .39, .05), canvas, .022)
for y in [.49, .53, .75, .79]:
    box('towel-woven-stripe', (0, y, .561), (.55, .012, .002), coral, .0008)

collection('garden-parasol')
assets[active.name] = active
rod('weighted-base', (0, 0, .015), (0, 0, .14), .34, stone, 64)
rod('teak-mast', (0, 0, .12), (0, 0, 2.85), .038, teak)
rod('runner', (0, 0, 1.76), (0, 0, 1.91), .058, steel)
segments, radial = 128, 14
verts, faces = [], []
for j in range(radial+1):
    r = .035+(1.62-.035)*j/radial
    for i in range(segments):
        a = 2*math.pi*i/segments
        sag = .055*math.sin(4*a)**2*(r/1.62)**1.3
        z = 2.83-.38*(r/1.62)**.85-sag
        verts.append((r*math.cos(a), r*math.sin(a), z))
for j in range(radial):
    for i in range(segments):
        a=j*segments+i; b=j*segments+(i+1)%segments
        faces.append((a, b, b+segments, a+segments))
mesh = bpy.data.meshes.new('tailored-eight-panel-canopy')
mesh.from_pydata(verts, [], faces)
mesh.update()
obj = bpy.data.objects.new('tailored-eight-panel-canopy', mesh)
active.objects.link(obj)
mesh.materials.append(canvas)
for p in mesh.polygons:
    p.use_smooth = True
# Consistent UVs survive GLB; shader nodes do not need procedural evaluation.
uv = mesh.uv_layers.new(name='UVMap')
for p in mesh.polygons:
    for li in p.loop_indices:
        co=mesh.vertices[mesh.loops[li].vertex_index].co
        uv.data[li].uv=(co.x/3.3+.5, co.y/3.3+.5)
mod = obj.modifiers.new('Double-sided cloth thickness', 'SOLIDIFY')
mod.thickness = .012
for i in range(8):
    a = i*math.pi/4
    pts=[]
    for j in range(17):
        r=.04+1.58*j/16
        pts.append((r*math.cos(a), r*math.sin(a), 2.81-.38*(r/1.62)**.85))
    tube('canopy-rib', pts, .009, steel)
    rod('spreader', (0, 0, 1.84), (.85*math.cos(a), .85*math.sin(a), 2.58), .012, steel)
    sphere('rib-end-cap', pts[-1], (.018, .018, .025), teak)
sphere('finial', (0, 0, 2.88), (.055, .055, .075), teak)

collection('insulated-cooler')
assets[active.name] = active
box('rounded-insulated-body', (0, 0, .28), (.76, .51, .49), enamel, .075)
box('moulded-base', (0, 0, .055), (.70, .46, .09), dark, .035)
box('lid-gasket', (0, 0, .522), (.77, .52, .025), dark, .012)
box('rounded-insulated-lid', (0, 0, .565), (.80, .55, .085), white, .037)
box('lid-inset', (0, 0, .607), (.59, .37, .012), white, .006)
for side in [-1, 1]:
    box('latch', (side*.22, -.266, .46), (.055, .027, .12), steel, .009)
    rod('hinge', (side*.22-.06, .266, .524), (side*.22+.06, .266, .524), .018, steel)
    tube('carry-handle', [(side*.39, -.13, .42), (side*.46, -.13, .40),
                         (side*.46, -.13, .26), (side*.46, .13, .26),
                         (side*.46, .13, .40), (side*.39, .13, .42)], .016, steel)
    rod('handle-grip', (side*.46, -.09, .26), (side*.46, .09, .26), .024, dark)
rod('drain-cap', (.34, -.25, .12), (.34, -.28, .12), .023, white)


def vessel(name, radius, height, at, mat):
    # Closed lathed cross section: underside, exterior, rim, interior, floor.
    profile = [(0, .006), (radius*.86, .006), (radius, .025),
               (radius*1.05, height-.015), (radius*1.02, height),
               (radius*.90, height), (radius*.86, .035), (0, .035)]
    vertices, faces = [], []
    n=64
    for r, z in profile:
        for i in range(n):
            a=2*math.pi*i/n
            vertices.append((at[0]+r*math.cos(a), at[1]+r*math.sin(a), at[2]+z))
    for j in range(len(profile)-1):
        for i in range(n):
            a=j*n+i; b=j*n+(i+1)%n
            faces.append((a,b,b+n,a+n))
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    obj=bpy.data.objects.new(name,mesh)
    active.objects.link(obj)
    mesh.materials.append(mat)
    # Weld the duplicated points at the axis so the vessel is manifold.
    bpy.context.view_layer.objects.active=obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=.00001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)
    for p in mesh.polygons: p.use_smooth=True
    return obj


collection('refreshment-set')
assets[active.name] = active
rod('teak-serving-tray', (0, 0, .012), (0, 0, .04), .34, teak, 96)
tube('raised-tray-rim', [(.325*math.cos(i*math.pi/64), .325*math.sin(i*math.pi/64), .052)
                       for i in range(128)], .012, teak, True)
vessel('ceramic-pitcher', .095, .285, (.10, .08, .04), ceramic)
tube('pitcher-handle', [(.18+.085*math.sin(i*math.pi/32), .08, .205+.085*math.cos(i*math.pi/32))
                        for i in range(33)], .015, ceramic)
for x,y in [(-.14,.06),(-.08,-.18),(.15,-.17)]:
    vessel('ceramic-tumbler', .045, .105, (x,y,.04), ceramic)
sphere('fresh-lemon', (-.09,.22,.08), (.055,.038,.037), lemon)

# Export assets at their origin before laying out the preview.
bpy.ops.object.select_all(action='DESELECT')
manifest = json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else {'version':1, 'units':'metres', 'upAxis':'Y', 'assets':[]}
manifest['assets'] = [a for a in manifest['assets'] if a['id'] not in assets]
for name, c in assets.items():
    for obj in c.objects: obj.select_set(True)
    bpy.context.view_layer.objects.active=next(iter(c.objects))
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')), export_format='GLB',
                              use_selection=True, export_apply=True, export_yup=True,
                              export_normals=True, export_texcoords=True, export_materials='EXPORT')
    manifest['assets'].append({'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz',
                               'sourceCollection':name})
    bpy.ops.object.select_all(action='DESELECT')
(OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'pool-party-assets.blend'))

# The preview is a separate arrangement; the source assets remain local-space.
for name, c in assets.items():
    delta = {'teak-lounger':(-.5,.05,0), 'garden-parasol':(.7,1.7,0),
             'insulated-cooler':(1.05,-.35,0),'refreshment-set':(1.10,.55,.48)}[name]
    for obj in c.objects: obj.location += Vector(delta)
collection('preview-stage')
box('side-table', (1.10,.55,.43), (.8,.8,.1), teak, .035)
for x in [.80,1.40]:
    for y in [.25,.85]: box('table-leg',(x,y,.20),(.045,.045,.40),teak)
for x in range(-6,7):
    for y in range(-1,7):
        box('travertine-paver',(x*.75,y*.75,-.07),(.744,.744,.14),stone,.008)
poolmat=material('preview-pool-blue',(.025,.24,.27),.13)
box('pool-water', (0,-3.2,-.13), (12,3.1,.08),poolmat,.01)
for x in range(-8,9):
    box('pool-coping',(x*.75,-1.5,-.035),(.744,.38,.15),canvas,.02)
world=bpy.data.worlds.new('warm-afternoon-sky')
world.use_nodes=True
scene.world=world
nodes=world.node_tree.nodes
sky=nodes.new('ShaderNodeTexSky')
sky.sky_type='MULTIPLE_SCATTERING'
sky.sun_elevation=math.radians(38)
sky.sun_rotation=math.radians(125)
world.node_tree.links.new(sky.outputs['Color'],nodes.get('Background').inputs['Color'])
nodes.get('Background').inputs['Strength'].default_value=.35
bpy.ops.object.camera_add(location=(5.1,-7.6,4.3))
camera=bpy.context.object
camera.rotation_euler=(Vector((.25,.45,1.0))-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.lens=48
scene.camera=camera
scene.render.engine='CYCLES'
scene.cycles.samples=args.samples
scene.cycles.use_denoising=True
scene.render.resolution_x=1440
scene.render.resolution_y=1080
scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX'
scene.view_settings.exposure=-.8
scene.render.image_settings.file_format='PNG'
scene.render.filepath=str(OUT/'pool-party-assets-preview.png')
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'pool-party-preview.blend'))
if not args.no_render: bpy.ops.render.render(write_still=True)
