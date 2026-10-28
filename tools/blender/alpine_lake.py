"""Alpine sunrise, lit and rendered in Cycles with a genuine NB-FLIP water mesh.

blender -b --python tools/blender/alpine_lake.py -- --water-mesh water.obj
    --output artifacts/alpine-lake-sunrise.png --samples 192 --width 1800 --height 1200
The OBJ is already in Blender metres/Z-up (app x,y,z -> x,-z,y).
No proxy plane or shader-generated wave displacement replaces the solver surface.
"""
import argparse
import math
import random
import sys
from pathlib import Path
import bpy
import bmesh
from mathutils import Vector, noise

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--water-mesh', required=True)
parser.add_argument('--output', default='artifacts/alpine-lake-sunrise.png')
parser.add_argument('--samples', type=int, default=192)
parser.add_argument('--width', type=int, default=1800)
parser.add_argument('--height', type=int, default=1200)
parser.add_argument('--device', choices=['AUTO','CPU','METAL'], default='AUTO')
parser.add_argument('--no-render', action='store_true')
parser.add_argument('--dry-preview', action='store_true', help='Environment lookdev only; explicitly omit all liquid')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
water_path = Path(args.water_mesh).resolve()
if not args.dry_preview and not water_path.is_file():
    raise FileNotFoundError(f'Actual exported NB-FLIP surface is required: {water_path}')
out = Path(args.output).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
random.seed(20261028)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene['title'] = 'First Light — an alpine tarn'
scene['water_source'] = str(water_path)
scene['water_method'] = 'DRY ENVIRONMENT PREVIEW; no water' if args.dry_preview else 'uniform-narrow-band-flip; imported solver surface'
bpy.context.preferences.filepaths.save_version = 0


def mesh(name, verts, faces, mat=None, smooth=True):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    if mat:
        data.materials.append(mat)
    for face in data.polygons:
        face.use_smooth = smooth
    return obj


def material(name, color, rough=.8):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bs = mat.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*color, 1)
    bs.inputs['Roughness'].default_value = rough
    return mat


def stone_material(name, low, high, scale=3):
    mat = material(name, low)
    n, l = mat.node_tree.nodes, mat.node_tree.links
    bs = n.get('Principled BSDF')
    coordinates = n.new('ShaderNodeNewGeometry')
    tex = n.new('ShaderNodeTexNoise')
    l.new(coordinates.outputs['Position'], tex.inputs['Vector'])
    tex.inputs['Scale'].default_value = scale
    tex.inputs['Detail'].default_value = 6
    ramp = n.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = .18
    ramp.color_ramp.elements[0].color = (*low, 1)
    ramp.color_ramp.elements[1].position = .83
    ramp.color_ramp.elements[1].color = (*high, 1)
    l.new(tex.outputs['Fac'], ramp.inputs[0])
    l.new(ramp.outputs[0], bs.inputs['Base Color'])
    fine = n.new('ShaderNodeTexNoise')
    l.new(coordinates.outputs['Position'], fine.inputs['Vector'])
    fine.inputs['Scale'].default_value = 65
    fine.inputs['Detail'].default_value = 4
    bump = n.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = .28
    bump.inputs['Distance'].default_value = .035
    l.new(fine.outputs['Fac'], bump.inputs['Height'])
    l.new(bump.outputs['Normal'], bs.inputs['Normal'])
    return mat


def bed_height(x, y):
    r = math.hypot(x / 30, y / 38)
    return .3 + 2.8 * min(1.5, r ** 6) - .045 * math.sin(x * .7) * math.sin(y * .6)


def surface_grid(name, xmin, xmax, ymin, ymax, nx, ny, height, mat):
    vertices = []
    for j in range(ny + 1):
        y = ymin + (ymax - ymin) * j / ny
        for i in range(nx + 1):
            x = xmin + (xmax - xmin) * i / nx
            vertices.append((x, y, height(x, y)))
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    obj=mesh(name, vertices, faces, mat)
    uv=obj.data.uv_layers.new(name='World scale rock UV')
    for face in obj.data.polygons:
        for loop_index in face.loop_indices:
            co=obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            uv.data[loop_index].uv=(co.x/50,co.y/50)
    return obj


bed_mat = stone_material('Wet mineral gravel • granite, quartz and fine silt', (.045,.055,.038), (.18,.17,.12), 1.7)
surface_grid('Continuous solver-matched tarn bed', -100, 100, -55, 100, 400, 310, bed_height, bed_mat)

# Snow follows altitude and upward-facing rock, with noise eroding its boundary.
rock = stone_material('Alpine gneiss and snowfields', (.012,.021,.029), (.095,.11,.12), .055)
n,l = rock.node_tree.nodes, rock.node_tree.links
bs = n.get('Principled BSDF')
base_color = bs.inputs['Base Color'].links[0].from_socket
geometry = n.new('ShaderNodeNewGeometry')
position = n.new('ShaderNodeSeparateXYZ'); l.new(geometry.outputs['Position'],position.inputs[0])
normal = n.new('ShaderNodeSeparateXYZ'); l.new(geometry.outputs['Normal'],normal.inputs[0])
alt = n.new('ShaderNodeMapRange'); alt.inputs['From Min'].default_value=150; alt.inputs['From Max'].default_value=310
l.new(position.outputs['Z'],alt.inputs['Value'])
slope = n.new('ShaderNodeMapRange'); slope.inputs['From Min'].default_value=.25; slope.inputs['From Max'].default_value=.80
l.new(normal.outputs['Z'],slope.inputs['Value'])
mul = n.new('ShaderNodeMath'); mul.operation='MULTIPLY'; l.new(alt.outputs[0],mul.inputs[0]);l.new(slope.outputs[0],mul.inputs[1])
snow_noise=n.new('ShaderNodeTexNoise');snow_noise.inputs['Scale'].default_value=.025;snow_noise.inputs['Detail'].default_value=5
# Geometry Position provides world-scale snow patches instead of object-bounds noise.
l.new(geometry.outputs['Position'],snow_noise.inputs['Vector'])
sub=n.new('ShaderNodeMath');sub.operation='SUBTRACT';l.new(mul.outputs[0],sub.inputs[0]);l.new(snow_noise.outputs['Fac'],sub.inputs[1])
ramp=n.new('ShaderNodeMapRange');ramp.inputs['From Min'].default_value=-.15;ramp.inputs['From Max'].default_value=.15;l.new(sub.outputs[0],ramp.inputs[0])
mix=n.new('ShaderNodeMixRGB');l.new(ramp.outputs[0],mix.inputs[0]);l.new(base_color,mix.inputs[1]);mix.inputs[2].default_value=(.83,.89,.95,1);l.new(mix.outputs[0],bs.inputs['Base Color'])


# Metre-scale fractured relief, independent of the broad mountain silhouette.
crags=n.new('ShaderNodeTexNoise');crags.inputs['Scale'].default_value=.22;crags.inputs['Detail'].default_value=7;crags.inputs['Roughness'].default_value=.8
l.new(geometry.outputs['Position'],crags.inputs['Vector'])
crag_bump=n.new('ShaderNodeBump');crag_bump.inputs['Strength'].default_value=.65;crag_bump.inputs['Distance'].default_value=3.5
l.new(crags.outputs['Fac'],crag_bump.inputs['Height']);l.new(crag_bump.outputs['Normal'],bs.inputs['Normal'])
# CC0 photogrammetry by Rob Tuytel / Poly Haven, physically 50 m per tile.
texture_dir=Path(__file__).resolve().parents[2]/'assets/alpine-lake/textures'
for channel,suffix,color_space in [('color','diff','sRGB'),('normal','nor_gl','Non-Color'),('roughness','rough','Non-Color'),('height','disp','Non-Color')]:
    path=(texture_dir.parent/'alpine-strata-neutral.png') if channel=='color' else texture_dir/('aerial_rocks_02_'+suffix+'_2k.jpg')
    if not path.exists():raise FileNotFoundError('Reproducible rock texture missing: '+str(path))
    tex=n.new('ShaderNodeTexImage');tex.name='Photogrammetry '+channel;tex.image=bpy.data.images.load(str(path),check_existing=True);tex.image.colorspace_settings.name=color_space
    if channel=='color':l.new(tex.outputs['Color'],mix.inputs[1])
    elif channel=='roughness':l.new(tex.outputs['Color'],bs.inputs['Roughness'])
    elif channel=='normal':
        nm=n.new('ShaderNodeNormalMap');nm.inputs['Strength'].default_value=.85;l.new(tex.outputs['Color'],nm.inputs['Color']);l.new(nm.outputs[0],bs.inputs['Normal'])


def fbm(x,y,z=0):
    return noise.fractal(Vector((x,y,z)), .9, 2.1, 5)


def mountain_height(x,y):
    # Unequal, overlapping massifs leave a glacial valley down the centre.
    peaks=[(-390,700,370,290,310),(-80,1000,430,235,380),(260,760,340,245,290),(540,1200,480,320,470),(-760,1400,600,430,470),(20,1700,650,430,480)]
    h=0
    for px,py,ph,sx,sy in peaks:
        d=math.sqrt(((x-px)/sx)**2+((y-py)/sy)**2)
        theta=math.atan2((y-py)/sy,(x-px)/sx)
        gully=(.5+.5*math.sin(theta*7+d*2+px*.017))**3
        profile=ph*max(0,1-d*.65)**1.2
        h=max(h,profile-60*gully*min(1,d*2.5)*max(0,1-d*.65))
    # Fractal crags and a creased drainage texture break the smooth massifs.
    f=fbm(x*.008,y*.008,1.2)
    ridge=abs(noise.noise(Vector((x*.026,y*.026,2.7))))
    h *= .83 + .24*f + .17*ridge
    foothill=(max(0,abs(x)-35)*.18)*math.exp(-((y-150)/280)**2)
    return 3 + max(h,foothill)


surface_grid('Glaciated mountains • ridges and snow couloirs', -1800,1800,95,2200,600,360,mountain_height,rock)
# Smaller foothills catch morning sidelight and frame the far bank.
forest_floor=stone_material('Fir woodland floor',(.035,.05,.025),(.12,.15,.065),.4)
def foothill(x,y):
    side=max(0,abs(x)-25)
    hill=3+side*.19*math.exp(-((y-95)/190)**2)+max(0,y-80)*.025+2*abs(fbm(x*.03,y*.03))
    t=min(1,max(0,(y-42)/30));t=t*t*(3-2*t)
    return bed_height(x,y)*(1-t)+hill*t
surface_grid('Forested lateral moraine',-270,270,42,340,220,130,foothill,forest_floor)

# Stones are individual softened, deformed mineral bodies, including shallow
# submerged stones seen through the actual refractive liquid surface.
stone_mats=[stone_material('Tarn pebbles '+str(i),a,b,5) for i,(a,b) in enumerate([
    ((.12,.13,.13),(.44,.44,.40)),((.09,.065,.045),(.35,.26,.17)),((.17,.19,.18),(.51,.54,.51)),((.055,.07,.08),(.21,.26,.29))])]
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1)
proto=bpy.context.object
proto.name='Stone prototype'
for v in proto.data.vertices:
    v.co *= .91 + .17*noise.noise(v.co*3)
for p in proto.data.polygons:p.use_smooth=True
protos=[]
for mat in stone_mats:
    dat=proto.data.copy();dat.materials.append(mat);protos.append(dat)
bpy.data.objects.remove(proto,do_unlink=True)
for i in range(4200):
    x=random.uniform(-33,33);y=random.uniform(-38,38) if i<2000 else random.uniform(-35,-20)
    r=math.hypot(x/30,y/38)
    if r < .55 or r>1.17:continue
    size=random.uniform(.055,.24) if y>-25 else random.uniform(.10,.38)
    obj=bpy.data.objects.new('Water-polished shoreline stone',random.choice(protos));scene.collection.objects.link(obj)
    obj.location=(x,y,bed_height(x,y)+size*.24)
    obj.scale=(size*random.uniform(1,1.5),size*random.uniform(.7,1.2),size*.65)
    obj.rotation_euler=(random.uniform(-.3,.3),random.uniform(-.3,.3),random.uniform(0,6.28))

bark=material('Fir bark',(.065,.045,.028))
needles=[material('Fir foliage '+str(i),(.019+i*.006,.043+i*.009,.028+i*.004)) for i in range(4)]
# Individual branch fans with uneven tips avoid stacked cone silhouettes.
def tree_geometry(seed):
    rng=random.Random(seed);verts=[];faces=[]
    for layer in range(18):
        z=.8+layer*.32;rad=(1-z/7)*1.3
        for branch in range(8):
            theta=branch*math.tau/8+layer*.83+rng.uniform(-.3,.3)
            radius=rad*rng.uniform(.7,1.2)
            direction=Vector((math.cos(theta),math.sin(theta),-.22))
            side=Vector((-math.sin(theta),math.cos(theta),0))
            for twig in range(5):
                t=.25+twig*.15
                center=Vector((0,0,z+.2))+direction*radius*t
                length=radius*(1-t)*.22+.055
                # Crossed, irregular needle sprays have volume from every view.
                for tilt in [-.6,.4,1.4]:
                    cross=side*math.cos(tilt)+Vector((0,0,math.sin(tilt)))
                    tip=center+direction*length*1.8+Vector((0,0,.05))
                    a=len(verts);verts.extend([center-cross*length,tip,center+cross*length,center+Vector((0,0,length*.4))])
                    faces.extend([(a,a+1,a+3),(a+1,a+2,a+3)])
    trunk_start=len(verts)
    for z,radius in [(0,.095),(6.7,.018)]:
        verts.extend((math.cos(a*math.tau/8)*radius,math.sin(a*math.tau/8)*radius,z) for a in range(8))
    for a in range(8):faces.append((trunk_start+a,trunk_start+(a+1)%8,trunk_start+8+(a+1)%8,trunk_start+8+a))
    return verts,faces

# Reuse the detailed Blender-authored cedar geometry also used by the app.
# This replaces the old layered triangle prototypes with woody architecture
# and individual closed foliage envelopes, while retaining true instancing.
trees=[]
for asset in ['alpine-fir-0','alpine-fir-1']:
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=str(Path(__file__).resolve().parents[2]/'public/assets/alpine-lake'/(asset+'.glb')))
    imported=list(bpy.context.selected_objects)
    meshes=[o for o in imported if o.type=='MESH']
    empties=[o for o in imported if o.type!='MESH']
    for obj in imported:obj.select_set(obj.type=='MESH')
    bpy.context.view_layer.objects.active=meshes[0]
    bpy.ops.object.join();obj=bpy.context.object
    matrix=obj.matrix_world.copy()
    for v in obj.data.vertices:v.co=matrix@v.co
    zmin=min(v.co.z for v in obj.data.vertices);zmax=max(v.co.z for v in obj.data.vertices)
    scale=6.7/(zmax-zmin)
    for v in obj.data.vertices:v.co=Vector((v.co.x*scale,v.co.y*scale,(v.co.z-zmin)*scale))
    trees.append(obj.data)
    bpy.data.objects.remove(obj,do_unlink=True)
    for old in empties:
        if old.name in bpy.data.objects:bpy.data.objects.remove(old,do_unlink=True)
for i in range(7200):
    x=random.uniform(-220,220);y=random.uniform(43,150) if i<4600 else random.uniform(150,340)
    if abs(x)<26 and y<90:continue
    obj=bpy.data.objects.new('Subalpine fir',random.choice(trees));scene.collection.objects.link(obj)
    obj.location=(x,y,foothill(x,y)-.1)
    s=random.uniform(.55,1.8);obj.scale=(s,s,s*random.uniform(.85,1.25));obj.rotation_euler[2]=random.uniform(0,math.tau)
# Four near-shore trees frame the composition without blocking the water.
for x,y,s in [(-32,-3,2.6),(-36,12,2.4),(34,5,2.7),(31,28,2.0),(-27,31,2)]:
    obj=bpy.data.objects.new('Near-bank fir',trees[int(s*3)%len(trees)]);scene.collection.objects.link(obj);obj.location=(x,y,bed_height(x,y));obj.scale=(s,s,s)

# Import the simulation output exactly once; keep its world placement.
if not args.dry_preview:
    bpy.ops.wm.obj_import(filepath=str(water_path),forward_axis='Y',up_axis='Z')
    water=material('Liquid water • IOR 1.333',(.96,.99,1),.009)
    n,l=water.node_tree.nodes,water.node_tree.links
    bs=n.get('Principled BSDF');bs.inputs['Transmission Weight'].default_value=1;bs.inputs['IOR'].default_value=1.333
    absorb=n.new('ShaderNodeVolumeAbsorption');absorb.inputs['Color'].default_value=(.25,.67,.78,1);absorb.inputs['Density'].default_value=.025
    l.new(absorb.outputs[0],n.get('Material Output').inputs['Volume'])
    for obj in list(bpy.context.selected_objects):
        obj.name='NB-FLIP lake • measured free surface';obj.data.materials.clear();obj.data.materials.append(water)
        bm=bmesh.new();bm.from_mesh(obj.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5);bm.normal_update();bm.to_mesh(obj.data);bm.free()
        bounds=[obj.matrix_world@Vector(corner) for corner in obj.bound_box]
        print('NBFLIP_IMPORTED_BOUNDS',tuple(min(v[k] for v in bounds) for k in range(3)),tuple(max(v[k] for v in bounds) for k in range(3)))
        for p in obj.data.polygons:p.use_smooth=True
        obj['source']='Actual narrow-band FLIP export; no wave geometry substitution'

# Wisps are confined to the far water; they preserve foreground clarity.
mist=bpy.data.materials.new('Cold-air radiation mist');mist.use_nodes=True
n,l=mist.node_tree.nodes,mist.node_tree.links;n.clear()
outnode=n.new('ShaderNodeOutputMaterial');vol=n.new('ShaderNodeVolumePrincipled');vol.inputs['Color'].default_value=(.7,.79,.85,1);vol.inputs['Anisotropy'].default_value=.15
tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=3;tex.inputs['Detail'].default_value=3
r=n.new('ShaderNodeValToRGB');r.color_ramp.elements[0].position=.4;r.color_ramp.elements[0].color=(0,0,0,1);r.color_ramp.elements[1].position=.72;r.color_ramp.elements[1].color=(.035,.035,.035,1)
l.new(tex.outputs['Fac'],r.inputs[0]);l.new(r.outputs[0],vol.inputs['Density']);l.new(vol.outputs[0],outnode.inputs['Volume'])
for loc,scale in [((-7,21,3.2),(19,5,.7)),((10,29,3.8),(17,5,1.2)),((-23,42,5),(16,8,2))]:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,location=loc);obj=bpy.context.object;obj.name='Low drifting mist';obj.scale=scale;obj.data.materials.append(mist)

world=bpy.data.worlds.new('Clear high-altitude dawn');scene.world=world;world.use_nodes=True
n,l=world.node_tree.nodes,world.node_tree.links
sky=n.new('ShaderNodeTexSky');sky.sky_type='MULTIPLE_SCATTERING' if bpy.app.version >= (5,0,0) else 'NISHITA';sky.sun_elevation=math.radians(8);sky.sun_rotation=math.radians(-135);sky.altitude=1.5;sky.air_density=1;setattr(sky,'aerosol_density' if hasattr(sky,'aerosol_density') else 'dust_density',.25);sky.ozone_density=1.2
l.new(sky.outputs[0],n.get('Background').inputs[0]);n.get('Background').inputs['Strength'].default_value=.35
bpy.ops.object.light_add(type='SUN',location=(-100,-80,80));sun=bpy.context.object;sun.name='First warm sunrise';sun.data.energy=2.4;sun.data.color=(1,.71,.45);sun.data.angle=math.radians(.65)
sun_direction=Vector((math.cos(sky.sun_rotation)*math.cos(sky.sun_elevation),math.sin(sky.sun_rotation)*math.cos(sky.sun_elevation),math.sin(sky.sun_elevation)))
sun.rotation_euler=(-sun_direction).to_track_quat('-Z','Y').to_euler()

bpy.ops.object.camera_add(location=(0,-36.8,4.0));camera=bpy.context.object;scene.camera=camera
camera.rotation_euler=(Vector((0,260,27))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.lens=27;camera.data.sensor_width=36;camera.data.clip_end=10000
scene.render.engine='CYCLES';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
scene.cycles.max_bounces=12;scene.cycles.transmission_bounces=8;scene.cycles.volume_bounces=2
scene.render.resolution_x=args.width;scene.render.resolution_y=args.height;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.filepath=str(out)
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.35
scene.render.film_transparent=False
# Use GPU when installed; the file remains portable to CPU-only Blender.
if args.device=='CPU':
    scene.cycles.device='CPU'
else:
    try:
        prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
        for device in prefs.devices:device.use=True
        scene.cycles.device='GPU'
    except Exception:
        if args.device=='METAL':raise
        scene.cycles.device='CPU'
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')))
if not args.no_render:bpy.ops.render.render(write_still=True)
