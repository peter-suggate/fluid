"""Author metre-scale diving equipment, honed travertine and reusable living turf.

Run with Blender --background --python. All authoring coordinates below are
application Y-up metres; glTF export performs Blender's axis conversion.
Existing furniture assets and their preview are preserved.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'public/assets/pool-party'
SOURCE = ROOT/'assets/pool-party'
parser = argparse.ArgumentParser()
parser.add_argument('--no-render', action='store_true')
parser.add_argument('--samples', type=int, default=48)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
rng = np.random.default_rng(20261019)
assets = {}
active = None

def xyz(p): return (p[0], -p[2], p[1])
def collection(name):
    global active
    active = bpy.data.collections.new(name)
    scene.collection.children.link(active)
    assets[name] = active

def material(name, color, rough=.5, metal=0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metal
    return m

def image(name, pixels, noncolor=False):
    h,w,_ = pixels.shape
    im = bpy.data.images.new(name,width=w,height=h)
    if noncolor: im.colorspace_settings.name = 'Non-Color'
    rgba = np.ones((h,w,4),dtype=np.float32)
    rgba[:,:,:3] = pixels
    im.pixels.foreach_set(rgba.ravel())
    im.filepath_raw = str(OUT/(name+'.png'))
    im.file_format = 'PNG'
    im.save(); im.pack()
    return im

def textured(m, name, rgb, roughness, height, relief):
    nodes,links = m.node_tree.nodes,m.node_tree.links
    p = nodes.get('Principled BSDF')
    for label,pixels,socket in [('albedo',rgb,'Base Color'),('roughness',np.repeat(roughness[:,:,None],3,axis=2),'Roughness')]:
        t = nodes.new('ShaderNodeTexImage'); t.image=image(name+'-'+label,pixels,label!='albedo')
        links.new(t.outputs['Color'],p.inputs[socket])
    # Periodic central derivatives in metres. Export a real tangent normal map,
    # and bake it into the separate source shading-normal channel of the FVX.
    dx=(np.roll(height,-1,axis=1)-np.roll(height,1,axis=1))*.5*height.shape[1]*relief
    dy=(np.roll(height,-1,axis=0)-np.roll(height,1,axis=0))*.5*height.shape[0]*relief
    normal=np.stack((-dx,-dy,np.ones_like(dx)),axis=2)
    normal/=np.linalg.norm(normal,axis=2,keepdims=True)
    t=nodes.new('ShaderNodeTexImage');t.image=image(name+'-normal',normal*.5+.5,True)
    n=nodes.new('ShaderNodeNormalMap');links.new(t.outputs['Color'],n.inputs['Color']);links.new(n.outputs['Normal'],p.inputs['Normal'])

def finish(o,name,mat,bevel=0,smooth=True):
    o.name=name
    for c in list(o.users_collection): c.objects.unlink(o)
    active.objects.link(o);o.data.materials.append(mat)
    if bevel:
        b=o.modifiers.new('Soft manufactured edges','BEVEL');b.width=bevel;b.segments=4
    if o.type=='MESH' and smooth:
        for p in o.data.polygons:p.use_smooth=True
        b=o.modifiers.new('Weighted face normals','WEIGHTED_NORMAL');b.keep_sharp=True
    return o

def box(name,p,size,mat,bevel=.006):
    bpy.ops.mesh.primitive_cube_add(size=1,location=xyz(p));o=bpy.context.object
    o.scale=(size[0],size[2],size[1]);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return finish(o,name,mat,bevel)

def rod(name,a,b,r,mat):
    a,b=Vector(xyz(a)),Vector(xyz(b))
    bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=r,depth=(b-a).length,location=(a+b)/2)
    o=bpy.context.object;o.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler()
    return finish(o,name,mat,min(.003,r*.15))

def tube(name,points,r,mat):
    curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.bevel_depth=r;curve.bevel_resolution=4;curve.use_fill_caps=True
    s=curve.splines.new('POLY');s.points.add(len(points)-1)
    for p,co in zip(s.points,points):p.co=(*xyz(co),1)
    o=bpy.data.objects.new(name,curve);active.objects.link(o);curve.materials.append(mat)
    return o

steel=material('polished-316-stainless',(.60,.64,.66),.21,.98)
paint=material('ivory-powdercoat',(.74,.76,.72),.37)
rubber=material('charcoal-anti-slip',(.023,.027,.025),.86)
fiberglass=material('ivory-fibreglass',(.80,.82,.76),.30)
navy=material('lagoon-edge-inlay',(.025,.20,.24),.38)
grip=material('mineral-grit-deck',(.70,.72,.66),.78)
n=512;y,x=np.mgrid[:n,:n]/n
noise=rng.random((n,n))
textured(grip,'board-grit',np.clip(np.array([.70,.72,.66])[None,None,:]*(.93+.10*noise[:,:,None]),0,1),.71+.16*noise,noise,.00045)

collection('diving-board-ladder')
# Board anchor is the original simulation board's centre in XZ, at patio Y.
# The visual envelope encloses the unchanged 4 x .16 x .8 m collision box.
rotation=Matrix.Rotation(math.pi/40,4,'Y')
for name,p,size,mat,bevel in [
    ('fibreglass-springboard',(0,0,0),(4.012,.174,.812),fiberglass,.003),
    ('bonded-grit-walking-surface',(0,.091,0),(3.94,.012,.74),grip,.005),
    ('nose-wear-strip',(1.93,.099,0),(.055,.008,.73),navy,.002),
    *[(f'edge-inlay-{s}',(0,.015,s*.406),(3.92,.035,.012),navy,.003) for s in [-1,1]],
    *[(f'underside-reinforcement-{s}',(-.15,-.106,s*.23),(3.55,.055,.06),fiberglass,.01) for s in [-1,1]],
]:
    o=box(name,p,size,mat,bevel)
    o.matrix_world=Matrix.Translation(Vector((0,0,3.15)))@rotation@o.matrix_world
for s in [-1,1]:
    z=s*.46
    box('anchored-base-plate',(-1.65,.065,z),(.65,.07,.25),steel,.018)
    box('upright-box-section',(-1.65,1.61,z),(.13,3.05,.12),paint,.02)
    rod('diagonal-support',(-2.35,.08,z),(-1.65,2.70,z),.047,steel)
    box('brace-anchor-plate',(-2.35,.05,z),(.30,.045,.18),steel,.012)
    for xx in [-1.88,-1.42,-2.42,-2.28]:
        rod('anchor-stud',(xx,.08,z),(xx,.107,z),.021,steel)
    # Continuous curved return replaces sharp capsule elbows at the top.
    points=[(-2.95,.10,s*.36),(-1.99,3.08,s*.36),(-1.90,3.36,s*.40),(-1.90,3.90,s*.46)]
    points += [(-1.67-.23*math.cos(t),3.90+.23*math.sin(t),z) for t in np.linspace(0,math.pi/2,13)[1:]]
    points += [(-1.05,4.13,z)]
    points += [(-1.05+.18*math.sin(t),3.95+.18*math.cos(t),z) for t in np.linspace(0,math.pi/2,13)[1:]]
    points += [(-.87,3.25,z)]
    tube('continuous-stainless-handrail',points,.036,steel)
    box('ladder-shoe',(-2.95,.055,s*.36),(.24,.07,.16),rubber,.018)
    rod('rail-saddle',(-.87,3.24,z),(-.87,3.34,z),.052,steel)
for i in range(11):
    t=(i+1)/11;xx=-2.95+t*1.05;yy=.10+t*3.25
    box('folded-stainless-tread',(xx,yy,0),(.19,.045,.77),steel,.012)
    box('inset-rubber-tread',(xx,yy+.026,0),(.155,.014,.64),rubber,.006)
    for s in [-1,1]:rod('tread-fixing',(xx,yy+.02,s*.35),(xx,yy+.04,s*.35),.015,steel)
box('board-support-crossmember',(-1.65,3.19,0),(.32,.12,1.14),paint,.02)
box('elastomer-fulcrum',(-1.55,3.23,0),(.22,.075,.72),rubber,.018)

collection('pool-stainless-ladder')
# Anchor: pool wall x=4, patio y=2. Upper arcs terminate on the dry deck.
for s in [-1,1]:
    z=s*.40
    pts=[(-.30,-1.35,z),(-.30,.38,z)]
    pts += [(.05-.35*math.cos(t),.38+.35*math.sin(t),z) for t in np.linspace(0,math.pi,41)[1:]]
    pts += [(.40,.07,z)]
    tube('bent-stainless-rail',pts,.035,steel)
    rod('deck-escutcheon',(.40,.035,z),(.40,.065,z),.09,steel)
    rod('wall-spacer',(-.30,-1.25,z),(-.02,-1.25,z),.025,steel)
    box('wall-contact-pad',(-.025,-1.25,z),(.04,.12,.12),rubber,.018)
for i in range(4):
    yy=-1.20+i*.35
    box('pressed-stainless-step',(-.30,yy,0),(.24,.05,.79),steel,.015)
    box('anti-slip-step',(-.30,yy+.031,0),(.19,.018,.64),rubber,.008)
    for s in [-1,1]:rod('step-fastener',(-.30,yy+.02,s*.35),(-.30,yy+.045,s*.35),.016,steel)

collection('honed-travertine-paver')
stone=material('honed-warm-travertine',(.58,.51,.40),.65)
n=1024;y,x=np.mgrid[:n,:n]/n
# Broad sediment layers and small open pores have distinct spatial scales.
vein=np.sin(2*math.pi*(y*7+.12*np.sin(x*2*math.pi)+.025*np.sin(x*18*math.pi)))
cloud=np.sin(2*math.pi*x)*np.cos(4*math.pi*y)+.5*np.sin(8*math.pi*(x+y))
height=.12*vein+.05*cloud
pores=np.zeros((n,n))
for _ in range(520):
    cx,cy=rng.random(2);rx=rng.uniform(.001,.007);ry=rx*rng.uniform(.4,1.3)
    dx=np.minimum(abs(x-cx),1-abs(x-cx));dy=np.minimum(abs(y-cy),1-abs(y-cy))
    pores=np.maximum(pores,np.exp(-2*((dx/rx)**2+(dy/ry)**2)))
height-=pores*.8
variation=.92+.035*vein+.035*cloud-.22*pores+.012*rng.normal(size=(n,n))
rgb=np.clip(np.array([.59,.515,.405])[None,None,:]*variation[:,:,None],0,1)
textured(stone,'honed-travertine',rgb,np.clip(.57+.18*pores+.035*vein,.45,.86),height,.0018)
o=box('eased-edge-natural-stone',(0,.015,0),(.988,.030,.988),stone,.005)
# Metric planar UVs on top preserve the intended pore and sediment scale.
for loop in o.data.loops:
    co=o.data.vertices[loop.vertex_index].co
    o.data.uv_layers.active.data[loop.index].uv=(co.x+.5,co.y+.5)

collection('living-turf')
# Closed tapered blades, stratified roots, and a thin thatch bed. Four sectors
# on each axis keep the baker's BVHs local; every instance shares this image.
soil=material('turf-root-thatch',(.055,.072,.019),.98)
box('root-mat',(0,-.012,0),(1,.024,1),soil,.001)
greens=[material(f'grass-tone-{i}',c,.69+i*.015) for i,c in enumerate([
    (.065,.16,.018),(.085,.205,.025),(.11,.25,.035),(.15,.29,.048),(.18,.27,.055),(.24,.28,.085)])]
for gz in range(4):
    for gx in range(4):
        verts,faces,mats=[],[],[]
        for j in range(7):
            for i in range(7):
                # 784 clumps/m², two bowed leaves each; no billboard alpha.
                rx=-.5+(gx+(i+.15+.7*rng.random())/7)/4
                rz=-.5+(gz+(j+.15+.7*rng.random())/7)/4
                for leaf in range(2):
                    a=rng.uniform(0,2*math.pi);h=rng.uniform(.035,.082)
                    lean=rng.uniform(.009,.025);width=rng.uniform(.008,.014)
                    base=len(verts);tone=int(rng.choice(6,p=[.12,.22,.28,.23,.12,.03]))
                    for k in range(5):
                        t=k/4;w=width*(1-.91*t)/2;th=.0023*(1-.8*t)
                        xx=rx+math.cos(a)*lean*t*t;zz=rz+math.sin(a)*lean*t*t
                        for u,v in [(-w,-th),(w,-th),(w,th),(-w,th)]:
                            verts.append(xyz((xx+math.cos(a)*u-math.sin(a)*v,h*t-.003,zz+math.sin(a)*u+math.cos(a)*v)))
                    faces.append(tuple(base+k for k in [3,2,1,0]));mats.append(tone)
                    for k in range(4):
                        for side in range(4):
                            a0=base+k*4+side;a1=base+k*4+(side+1)%4
                            faces.append((a0,a1,a1+4,a0+4));mats.append(tone)
                    faces.append(tuple(base+16+k for k in range(4)));mats.append(min(5,tone+1))
        mesh=bpy.data.meshes.new('bowed-blades');mesh.from_pydata(verts,[],faces);mesh.update()
        o=bpy.data.objects.new(f'blade-cluster-{gx}-{gz}',mesh);active.objects.link(o)
        for mat in greens:mesh.materials.append(mat)
        for p,mat in zip(mesh.polygons,mats):p.material_index=mat;p.use_smooth=True
        # Ensure the same outward winding after coordinate conversion.
        bpy.context.view_layer.objects.active=o;o.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.mesh.normals_make_consistent(inside=False);bpy.ops.object.mode_set(mode='OBJECT');o.select_set(False)

bpy.ops.object.select_all(action='DESELECT')
manifest=json.loads((OUT/'manifest.json').read_text())
manifest['assets']=[a for a in manifest['assets'] if a['id'] not in assets]
for name,c in assets.items():
    for o in c.objects:o.select_set(True)
    bpy.context.view_layer.objects.active=next(iter(c.objects))
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_normals=True,export_texcoords=True,export_materials='EXPORT')
    manifest['assets'].append({'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','sourceCollection':name})
    bpy.ops.object.select_all(action='DESELECT')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'pool-party-landscape.blend'))

# A separate preview uses the real exported materials and metre-scale assembly.
for o in assets['pool-stainless-ladder'].objects:o.location+=Vector(xyz((3.2,0,0)))
for name in ['honed-travertine-paver','living-turf']:
    c=assets[name]
    copies=list(c.objects)
    positions=([(x+.5,z+.5) for x in range(-4,4) for z in range(-2,2)] if name.startswith('honed') else [(x+.5,z+.5) for x in range(-4,4) for z in [-3,2]])
    for xx,zz in positions:
        for o in copies:
            cp=o.copy();cp.data=o.data;c.objects.link(cp);cp.location+=Vector(xyz((xx,-.03,zz)))
    for o in copies:o.hide_render=True
world=bpy.data.worlds.new('afternoon-sky');world.use_nodes=True;scene.world=world
sky=world.node_tree.nodes.new('ShaderNodeTexSky');sky.sky_type='MULTIPLE_SCATTERING';sky.sun_elevation=math.radians(38);sky.sun_rotation=math.radians(125)
world.node_tree.links.new(sky.outputs['Color'],world.node_tree.nodes.get('Background').inputs['Color']);world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.35
bpy.ops.object.camera_add(location=xyz((7.8,6.5,10)))
cam=bpy.context.object;cam.rotation_euler=(Vector(xyz((-.3,1.65,0)))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=48;scene.camera=cam
scene.render.engine='CYCLES';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
scene.render.resolution_x=1400;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.8
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(OUT/'pool-party-landscape-preview.png')
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'pool-party-landscape-preview.blend'))
if not args.no_render:bpy.ops.render.render(write_still=True)
