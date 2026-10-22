"""Build the editable Stillwater Grove environment and render it with Cycles.

All distances are metres, Z up. Requires Blender 4.0+; no downloads or add-ons.
Run: blender -b --python-exit-code 1 --python tools/blender/forest_waterfall.py
     -- --width 2400 --samples 128
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector, noise

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
SOURCE = ROOT / 'assets/forest-waterfall'
OUT = ROOT / 'artifacts/forest-waterfall'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--width', type=int, default=2400)
parser.add_argument('--samples', type=int, default=128)
parser.add_argument('--no-render', action='store_true')
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
SOURCE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
random.seed(20261022)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene['title'] = 'Stillwater Grove'
scene['description'] = 'A broad, secluded forest pool below a nineteen-metre cascade; procedural botanical geometry and layered mossy stone.'
scene['water_mode'] = 'Art-directed still geometry, not a baked fluid simulation.'


def collection(name):
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    return col


geo = collection('01 | Gorge · strata, boulders and submerged bed')
detail = collection('02 | Forest floor · roots, moss, litter and wood')
stage = collection('90 | Cameras and daylight')


def mesh(name, verts, faces, mat, col=geo, smooth=True):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.materials.append(mat)
    data.update()
    obj = bpy.data.objects.new(name, data)
    col.objects.link(obj)
    for p in data.polygons:
        p.use_smooth = smooth
    return obj


def value_noise(x, y, z=0):
    return noise.noise_vector(Vector((x, y, z)))[0]


def cliff_edge(x):
    return 8.65 + .65 * math.sin(x * .33) + .28 * math.sin(x * 1.4)


def cliff_top(x, y=12):
    return 19.4 + 1.4 * math.sin(x * .19) + .65 * value_noise(x * .35, y * .19)


def terrain_height(x, y):
    r = math.sqrt((x / 10.5)**2 + ((y + 2) / 10)**2)
    ground = .28 + .25*math.sin(x*.6)*math.sin(y*.5) + .35*value_noise(x*.36,y*.36)
    ground += max(0, abs(x)-10) * .16 + max(0,-y-14)*.05
    if r < 1.04:
        ground = -1.85 + 1.95 * min(1, (r / 1.04)**5) + .07*value_noise(x,y)
    edge = cliff_edge(x)
    if y > edge:
        t = min(1, (y-edge)/1.2)
        ground = ground*(1-t) + cliff_top(x,y)*t
        ground += max(0,y-18)*.08
        if y>32:
            fade=min(1,(y-32)/30)
            ground += fade*(6+7*math.sin(x*.041+y*.024)**2)+max(0,y-60)*.11
    return ground


def natural_material(name, low, high, moss=False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*high,1)
    mat.use_nodes = True
    ns, ls = mat.node_tree.nodes, mat.node_tree.links
    p = ns.get('Principled BSDF')
    p.inputs['Roughness'].default_value=.78
    tex=ns.new('ShaderNodeTexCoord')
    n=ns.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value=1.5
    n.inputs['Detail'].default_value=5.5; n.inputs['Roughness'].default_value=.73
    ls.new(tex.outputs['Object'],n.inputs['Vector'])
    ramp=ns.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position=.17; ramp.color_ramp.elements[0].color=(*low,1)
    ramp.color_ramp.elements[1].position=.81; ramp.color_ramp.elements[1].color=(*high,1)
    ls.new(n.outputs['Fac'],ramp.inputs[0])
    ls.new(ramp.outputs[0],p.inputs['Base Color'])
    fine=ns.new('ShaderNodeTexNoise'); fine.inputs['Scale'].default_value=37
    fine.inputs['Detail'].default_value=3
    ls.new(tex.outputs['Object'],fine.inputs['Vector'])
    bump=ns.new('ShaderNodeBump'); bump.inputs['Strength'].default_value=.5
    bump.inputs['Distance'].default_value=.075
    ls.new(fine.outputs['Fac'],bump.inputs['Height'])
    broad=ns.new('ShaderNodeBump'); broad.inputs['Strength'].default_value=.55
    broad.inputs['Distance'].default_value=.22
    ls.new(n.outputs['Fac'],broad.inputs['Height']); ls.new(bump.outputs[0],broad.inputs['Normal'])
    ls.new(broad.outputs[0],p.inputs['Normal'])
    if moss:
        g=ns.new('ShaderNodeNewGeometry'); sep=ns.new('ShaderNodeSeparateXYZ')
        ls.new(g.outputs['Normal'],sep.inputs[0])
        add=ns.new('ShaderNodeMath'); add.operation='ADD'
        ls.new(sep.outputs['Z'],add.inputs[0]); ls.new(n.outputs['Fac'],add.inputs[1])
        mask=ns.new('ShaderNodeMapRange'); mask.inputs['From Min'].default_value=.6
        mask.inputs['From Max'].default_value=1.1
        ls.new(add.outputs[0],mask.inputs[0])
        greens=ns.new('ShaderNodeValToRGB')
        greens.color_ramp.elements[0].color=(.015,.038,.006,1)
        greens.color_ramp.elements[1].color=(.18,.26,.035,1)
        ls.new(fine.outputs['Fac'],greens.inputs[0])
        mix=ns.new('ShaderNodeMixRGB')
        ls.new(mask.outputs[0],mix.inputs[0]); ls.new(ramp.outputs[0],mix.inputs[1])
        ls.new(greens.outputs[0],mix.inputs[2]); ls.new(mix.outputs[0],p.inputs['Base Color'])
    return mat


rock_mat=natural_material('Basalt & greywacke | wet mineral grain, upward moss',(.016,.022,.022),(.17,.19,.16),True)
bed_mat=natural_material('Submerged river gravel | warm mineral bed',(.045,.064,.046),(.24,.28,.19))
earth_mat=natural_material('Forest humus | moss, loam and fine relief',(.011,.021,.006),(.10,.15,.025),True)
wood_mat=natural_material('Fallen timber | aged chestnut heartwood',(.019,.012,.007),(.12,.075,.026))

# Mineral fractures sit below the moss colour layer and break the smooth
# shading of the eroded stone into fine, irregular planes.
ns,ls=rock_mat.node_tree.nodes,rock_mat.node_tree.links
p=ns.get('Principled BSDF');coord=ns.new('ShaderNodeTexCoord')
crack=ns.new('ShaderNodeTexVoronoi');crack.feature='DISTANCE_TO_EDGE'
crack.inputs['Scale'].default_value=3.2
ls.new(coord.outputs['Object'],crack.inputs['Vector'])
b=ns.new('ShaderNodeBump');b.inputs['Strength'].default_value=.45;b.inputs['Distance'].default_value=.055
ls.new(crack.outputs['Distance'],b.inputs['Height'])
old=p.inputs['Normal'].links[0].from_socket
ls.new(old,b.inputs['Normal']);ls.new(b.outputs['Normal'],p.inputs['Normal'])

# Continuous land/basin mesh; cliff facade is a separate connected rock surface.
verts=[]; faces=[]; nx=200; ny=205
for j in range(ny+1):
    y=-29+j*79/ny
    for i in range(nx+1):
        x=-43+i*86/nx
        verts.append((x,y,terrain_height(x,y)))
for j in range(ny):
    for i in range(nx):
        a=j*(nx+1)+i; faces.append((a,a+1,a+nx+2,a+nx+1))
land=mesh('Continuous basin, wooded banks and high plateau',verts,faces,earth_mat)
land.data.materials.append(bed_mat)
for p in land.data.polygons:
    c=p.center
    if (c.x/10.5)**2+((c.y+2)/10)**2<1.10: p.material_index=1

# Continue the forest into rising country, so the wide camera never sees the
# edge of a finite stage behind the upper canopy.
verts=[];faces=[];nx=180;ny=100
for j in range(ny+1):
    y=34+j*126/ny
    for i in range(nx+1):
        x=-130+i*260/nx
        verts.append((x,y,terrain_height(x,y)-.12))
for j in range(ny):
    for i in range(nx):
        a=j*(nx+1)+i;faces.append((a,a+1,a+nx+2,a+nx+1))
mesh('Distant wooded ridgeline | continuous forest backdrop',verts,faces,earth_mat)

# Vertical fracture plane. Broken horizontal beds and varying flutes avoid a
# regular wall of repeated rocks; the waterfall occupies the central gap.
verts=[];faces=[];nx=180;nz=74
for j in range(nz+1):
    t=j/nz
    for i in range(nx+1):
        x=-30+60*i/nx; z=t*cliff_top(x)
        step=.22*math.sin(z*3.9)+.19*math.sin(z*8.4+x*.12)
        flute=.34*math.sin(x*2.5)+.22*math.sin(x*6.3)
        y=cliff_edge(x)+.12+step+flute+.34*value_noise(x*.8,z*.6,8)
        verts.append((x,y,z-.25))
for j in range(nz):
    for i in range(nx):
        a=j*(nx+1)+i;faces.append((a,a+1,a+nx+2,a+nx+1))
wall=mesh('Weathered ravine | fractured vertical beds',verts,faces,rock_mat)


def boulder(name,loc,scale,seed,mat=rock_mat,sub=3):
    rng=random.Random(seed)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=sub,radius=1,location=loc)
    o=bpy.context.object;o.name=name
    for c in list(o.users_collection): c.objects.unlink(o)
    geo.objects.link(o)
    for v in o.data.vertices:
        p=v.co
        # Broad angular facets mixed with smaller eroded deviations.
        f=1+.26*value_noise(p.x*2+seed,p.y*2,p.z*2)+.095*value_noise(p.x*6,p.y*6+seed,p.z*6)
        for k in range(3):
            p[k]=math.copysign(abs(p[k])**.60,p[k])*scale[k]*f
    o.rotation_euler=(rng.uniform(-.17,.17),rng.uniform(-.20,.20),rng.uniform(-.5,.5))
    o.data.materials.append(mat)
    for p in o.data.polygons:p.use_smooth=True
    return o


for i in range(72):
    x=random.uniform(-26,26); z=random.uniform(1,18)
    if abs(x+1)<3.2: continue
    sy=random.uniform(.5,1.7)
    boulder('Cliff buttress · mossy exposed strata',(x,cliff_edge(x)+.35,z),
            (random.uniform(.8,2.7),sy,random.uniform(1,3.3)),i)

for i,(loc,scale) in enumerate([
    ((-1,8.7,18.3),(3.3,1.1,.65)),
    ((-1,7.4,11.65),(2.8,1.5,.65)),
    ((-.5,7.1,4.6),(2.65,1.5,.75)),
    ((-3.6,7.5,7.7),(1.3,1.0,2.6)),
    ((2.8,7.7,14.2),(1.4,1.1,2.5)),
]): boulder('Cascade ledge · exposed wet greywacke',loc,scale,200+i)

# Lumpy asymmetrical shoreline, with a few open shallow coves.
for i in range(94):
    a=random.uniform(0,2*math.pi)
    rr=random.uniform(.98,1.13)
    x=10.5*math.cos(a)*rr;y=-2+10*math.sin(a)*rr
    if y>6.2 and abs(x)<3.5:continue
    r=random.uniform(.35,1.15)
    z=max(-.18,terrain_height(x,y))
    boulder('Shore boulder · moss above the waterline',(x,y,z-.1),(r*1.3,r*.8,r*.7),300+i)
for i in range(115):
    a=random.uniform(0,6.28);r=random.uniform(.40,.99)
    x=10.5*r*math.cos(a); y=-2+10*r*math.sin(a)
    s=random.uniform(.10,.40)
    boulder('Rounded submerged river stone',(x,y,terrain_height(x,y)+.08),(s,s*.7,s*.45),500+i,bed_mat,2)
for i,(x,y,s) in enumerate([(-7,-9,1.7),(8,-7,1.4),(-9,1,2.1),(7.8,4,1.8),(-5,5.6,1.5),(3.8,4.6,1.2)]):
    boulder('Hero waterside moss stone',(x,y,.26),(s,s*.75,s*.67),800+i)


def branch(name,pts,radius,mat=wood_mat):
    cu=bpy.data.curves.new(name,'CURVE');cu.dimensions='3D'
    cu.resolution_u=8;cu.bevel_depth=radius;cu.bevel_resolution=3
    sp=cu.splines.new('BEZIER');sp.bezier_points.add(len(pts)-1)
    for i,(p,co) in enumerate(zip(sp.bezier_points,pts)):
        p.co=co;p.handle_left_type='AUTO';p.handle_right_type='AUTO';p.radius=1-i/(len(pts)+1)
    ob=bpy.data.objects.new(name,cu);detail.objects.link(ob);cu.materials.append(mat)
    return ob


branch('Old fallen tree along west bank',[(-14,-6,1.8),(-12,-3,1.3),(-10,0,1.2),(-8,3,1.0)],.46)
for i in range(7):
    x=-13+i*.65;y=-5+i*1.1
    branch('Broken limb on fallen trunk',[(x,y,1.4),(x-1.1,y+.5,1.8),(x-1.9,y+.7,2.0)],.13)
for i in range(24):
    x=random.choice([-1,1])*random.uniform(10.5,19);y=random.uniform(-10,8)
    z=terrain_height(x,y)+.06
    branch('Exposed trailing bank root',[(x,y,z+.2),(x+.6,y-.5,z+.12),(x+1.5,y-1,z)],.045)

# A thin, irregular scattering of fallen leaves breaks up the shaded soil.
litter_mats=[natural_material('Forest litter | '+name,tuple(c*.4 for c in color),color)
             for name,color in [('beech ochre',(.18,.105,.025)),('aged russet',(.10,.041,.012)),('olive',(.065,.085,.016))]]
lv=[];lf=[];li=[]
for i in range(3100):
    x=random.uniform(-27,27);y=random.uniform(-23,18)
    if (x/10.9)**2+((y+2)/10.4)**2<1.05 or y>cliff_edge(x)-1:continue
    z=terrain_height(x,y)+.018;a=random.uniform(0,math.tau);length=random.uniform(.07,.20)
    axis=Vector((math.cos(a),math.sin(a),0));side=Vector((-axis.y,axis.x,0))
    c=Vector((x,y,z));k=len(lv)
    for u,v,h in [(-.5,0,0),(0,-.22,0),(0,0,.035),(.5,0,.018),(0,.22,0)]:
        lv.append(tuple(c+axis*u*length+side*v*length+Vector((0,0,h))))
    for f in [(0,1,2),(1,3,2),(3,4,2),(4,0,2)]:lf.append(tuple(k+j for j in f));li.append(i%3)
leaves=mesh('Fallen leaves | curled ochre, russet and olive litter',lv,lf,litter_mats[0],detail)
for m in litter_mats[1:]:leaves.data.materials.append(m)
for p,i in zip(leaves.data.polygons,li):p.material_index=i

print('Building water ...',flush=True)
from waterfall_water import build_water
build_water()
print('Building foliage ...',flush=True)
from waterfall_foliage import build_foliage
botanical_inventory=build_foliage(terrain_height, cliff_edge=cliff_edge, cliff_top=cliff_top)

# A light atmospheric veil gives scale, separate from the localized impact mist.
mist=bpy.data.materials.new('Cool ravine air | subtle distance scattering')
mist.use_nodes=True;ns=mist.node_tree.nodes;ns.clear()
out=ns.new('ShaderNodeOutputMaterial');vol=ns.new('ShaderNodeVolumePrincipled')
vol.inputs['Density'].default_value=.0006
vol.inputs['Color'].default_value=(.70,.82,.87,1)
vol.inputs['Anisotropy'].default_value=.30
mist.node_tree.links.new(vol.outputs['Volume'],out.inputs['Volume'])
bpy.ops.mesh.primitive_cube_add(size=1,location=(0,12,18))
air=bpy.context.object;air.name='Ravine air · subtle aerial perspective';air.scale=(100,100,65)
air.data.materials.append(mist)
for c in list(air.users_collection):c.objects.unlink(air)
stage.objects.link(air);air.display_type='WIRE'

world=bpy.data.worlds.new('Soft blue sky through canopy');scene.world=world;world.use_nodes=True
ns=world.node_tree.nodes;bg=ns.get('Background')
sky=ns.new('ShaderNodeTexSky')
sky_types={item.identifier for item in sky.bl_rna.properties['sky_type'].enum_items}
sky.sky_type='MULTIPLE_SCATTERING' if 'MULTIPLE_SCATTERING' in sky_types else 'NISHITA'
sky.sun_elevation=math.radians(40)
sky.sun_rotation=math.radians(140);sky.sun_disc=False;sky.air_density=1.1
world.node_tree.links.new(sky.outputs['Color'],bg.inputs['Color']);bg.inputs['Strength'].default_value=.12


def light(name,kind,loc,energy,color,size,target):
    d=bpy.data.lights.new(name,kind);d.energy=energy;d.color=color
    if kind=='AREA':d.shape='DISK';d.size=size
    if kind=='SUN':d.angle=size
    o=bpy.data.objects.new(name,d);stage.objects.link(o);o.location=loc
    o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
    return o


light('Late morning · warm canopy shafts','SUN',(-30,-35,45),3.7,(1,.86,.66),.055,(0,5,1))
light('Open sky above the pool','AREA',(0,-5,24),1800,(.71,.85,1),18,(0,4,3))
light('Soft forest opening · camera fill','AREA',(8,-20,16),1700,(.80,.91,1),16,(0,5,8))


def camera(name,loc,target,lens):
    d=bpy.data.cameras.new(name);o=bpy.data.objects.new(name,d);stage.objects.link(o)
    o.location=loc;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
    d.lens=lens;d.clip_end=350
    d.dof.use_dof=False
    return o


scene.camera=camera('01 · Stillwater Grove | wide hero',(25,-49,23),(-1,5,10),36)
camera('02 · Waterline | intimate cascade',(9,-17,4.5),(-1,6,8),32)
camera('03 · Elevated environment overview',(34,-40,32),(0,6,8),39)
scene.render.engine='CYCLES'
scene.cycles.samples=args.samples
scene.cycles.use_denoising=True
scene.cycles.adaptive_threshold=.018
scene.cycles.max_bounces=9
scene.cycles.transmission_bounces=6
scene.cycles.transparent_max_bounces=8
scene.cycles.volume_bounces=1
scene.cycles.sample_clamp_indirect=4
scene.render.resolution_x=args.width
scene.render.resolution_y=round(args.width*2/3)
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.render.image_settings.color_mode='RGB'
scene.render.image_settings.color_depth='8'
scene.render.filepath=str(OUT/'stillwater-grove.png')
scene.view_settings.view_transform='AgX'
scene.view_settings.look='AgX - Medium High Contrast'
scene.view_settings.exposure=.30
try:
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='METAL';prefs.get_devices()
    gpu=False
    for d in prefs.devices:
        d.use=d.type=='METAL'
        gpu=gpu or d.use
    if gpu:scene.cycles.device='GPU'
except Exception as exc:print('Using CPU Cycles:',exc)

# Make the saved project pleasant to open: camera framing, material preview
# available without running the authoring script, organized named collections.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_perspective='CAMERA'
            area.spaces.active.clip_end=350
bpy.ops.object.select_all(action='DESELECT')
scene.camera.select_set(True);bpy.context.view_layer.objects.active=scene.camera
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'stillwater-grove.blend'))
stats={
    'title':'Stillwater Grove','seed':20261022,
    'objects':len(scene.objects),'mesh_datablocks':len(bpy.data.meshes),
    'unique_vertices':sum(len(m.vertices) for m in bpy.data.meshes),
    'unique_polygons':sum(len(m.polygons) for m in bpy.data.meshes),
    'materials':len(bpy.data.materials),'cameras':3,
    'botanical_inventory':botanical_inventory,
    'render_engine':'Cycles','device':scene.cycles.device,
    'width':scene.render.resolution_x,'height':scene.render.resolution_y,
    'samples':args.samples,'source':'assets/forest-waterfall/stillwater-grove.blend',
    'water':'Modeled still geometry; no time-dependent fluid cache',
}
(OUT/'scene-manifest.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2),flush=True)
if not args.no_render:
    bpy.ops.render.render(write_still=True)
print('Stillwater Grove complete.',flush=True)
