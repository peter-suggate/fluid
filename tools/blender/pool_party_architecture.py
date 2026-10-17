"""Reproducible pool-side architecture and furniture; Blender, metre scale, Y up.
Opaque tinted glazing is deliberate: FVX currently retains opaque PBR only.
Existing assets and their sources are preserved. Run with -- --no-render to skip
only the separate Cycles presentation, never the exports.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'public/assets/pool-party';SOURCE=ROOT/'assets/pool-party'
parser=argparse.ArgumentParser();parser.add_argument('--no-render',action='store_true');parser.add_argument('--samples',type=int,default=48)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version=0
scene=bpy.context.scene;scene.unit_settings.system='METRIC'
rng=np.random.default_rng(20261020)

def xyz(p):return (p[0],-p[2],p[1])

class Builder:
    def __init__(self):self.assets={};self.active=None
    def collection(self,name):
        self.active=bpy.data.collections.new(name);scene.collection.children.link(self.active);self.assets[name]=self.active
    def finish(self,o,name,mat,bevel=0):
        o.name=name
        for c in list(o.users_collection):c.objects.unlink(o)
        self.active.objects.link(o);o.data.materials.append(mat)
        if bevel:
            m=o.modifiers.new('Eased manufactured edge','BEVEL');m.width=bevel;m.segments=4
        if o.type=='MESH':
            for p in o.data.polygons:p.use_smooth=True
            m=o.modifiers.new('Weighted corner normals','WEIGHTED_NORMAL');m.keep_sharp=True
        return o
    def box(self,name,p,size,mat,bevel=.006,uv=None):
        bpy.ops.mesh.primitive_cube_add(size=1,location=xyz(p));o=bpy.context.object;o.scale=(size[0],size[2],size[1])
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        if uv:
            # Axis choices are in Blender coordinates; spatial scale remains
            # explicit so wood grain follows a board rather than each cube UV.
            a,b,sa,sb,offset=uv
            for loop in o.data.loops:
                co=o.data.vertices[loop.vertex_index].co
                o.data.uv_layers.active.data[loop.index].uv=(co[a]*sa+offset,co[b]*sb+.5)
        return self.finish(o,name,mat,bevel)
    def rod(self,name,a,b,r,mat):
        a,b=Vector(xyz(a)),Vector(xyz(b));bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=r,depth=(b-a).length,location=(a+b)/2)
        o=bpy.context.object;o.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler()
        return self.finish(o,name,mat,min(.002,r*.18))

B=Builder()
def material(name,color,rough=.5,metal=0):
    m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal
    return m

def texture(m,name,rgb,rough,height,relief):
    nodes,links=m.node_tree.nodes,m.node_tree.links;p=nodes.get('Principled BSDF')
    dx=(np.roll(height,-1,1)-np.roll(height,1,1))*.5*height.shape[1]*relief
    dy=(np.roll(height,-1,0)-np.roll(height,1,0))*.5*height.shape[0]*relief
    normals=np.stack((-dx,-dy,np.ones_like(dx)),axis=2);normals/=np.linalg.norm(normals,axis=2,keepdims=True)
    for label,pixels,socket in [('albedo',rgb,'Base Color'),('roughness',np.repeat(rough[:,:,None],3,axis=2),'Roughness'),('normal',normals*.5+.5,'Normal')]:
        h,w,_=pixels.shape;im=bpy.data.images.new(name+'-'+label,width=w,height=h)
        if label!='albedo':im.colorspace_settings.name='Non-Color'
        rgba=np.ones((h,w,4),np.float32);rgba[:,:,:3]=np.clip(pixels,0,1);im.pixels.foreach_set(rgba.ravel())
        im.filepath_raw=str(OUT/(name+'-'+label+'.png'));im.file_format='PNG';im.save();im.pack()
        t=nodes.new('ShaderNodeTexImage');t.image=im
        if label=='normal':
            normal=nodes.new('ShaderNodeNormalMap');links.new(t.outputs['Color'],normal.inputs['Color']);links.new(normal.outputs['Normal'],p.inputs[socket])
        else:links.new(t.outputs['Color'],p.inputs[socket])

n=512;y,x=np.mgrid[:n,:n]/n
noise=rng.normal(size=(n,n))
grain=np.sin(2*math.pi*(x*51+.3*np.sin(y*2*math.pi)+.1*np.sin(y*12*math.pi)))
weather=np.sin(2*math.pi*x)*.5+.5*np.cos(6*math.pi*y+2*np.sin(2*math.pi*x))
cedar=material('silvered-cedar',(.31,.205,.125),.72)
variation=.88+.07*grain+.06*weather+.018*noise
texture(cedar,'silvered-cedar',np.array([.34,.235,.15])[None,None,:]*variation[:,:,None],np.clip(.67+.06*grain+.025*noise,.5,.9),.15*grain+.07*noise,.0007)
teak=material('table-oiled-teak',(.36,.185,.07),.40)
texture(teak,'table-oiled-teak',np.array([.40,.225,.095])[None,None,:]*(.91+.075*grain+.025*noise)[:,:,None],np.clip(.43+.045*grain,.3,.6),.12*grain+.04*noise,.0004)
stucco=material('warm-mineral-render',(.69,.67,.61),.88)
cloud=np.sin(2*math.pi*x)*np.sin(4*math.pi*y)
texture(stucco,'warm-mineral-render',np.array([.69,.67,.61])[None,None,:]*(.98+.012*cloud+.009*noise)[:,:,None],np.clip(.83+.035*noise,.7,.96),.08*noise,.00065)
limestone=material('pool-limestone',(.64,.59,.48),.61)
height=.08*np.sin(2*math.pi*(4*x+.1*np.sin(2*math.pi*y)))+.08*noise
texture(limestone,'pool-limestone',np.array([.65,.60,.49])[None,None,:]*(.96+.025*cloud+.012*noise)[:,:,None],np.clip(.60+.035*noise,.45,.8),height,.0007)
black=material('anodised-charcoal',(.025,.030,.032),.32,.65)
steel=material('brushed-hardware',(.55,.58,.60),.29,.95)
glass=material('smoked-blue-glazing',(.025,.055,.065),.13,.30)
grout=material('warm-pool-grout',(.36,.40,.37),.91)
rubber=material('furniture-foot-rubber',(.018,.023,.022),.87)

B.collection('cedar-fence-panel')
for i in range(12):
    xx=(i-5.5)*.146
    B.box('vertical-cedar-board',(xx,.97,0),(.135,1.82,.032),cedar,.004,uv=(0,2,4,.5,i*.173))
    for yy in [.45,1.38]:B.rod('stainless-ring-nail',(xx,yy,.015),(xx,yy,.022),.0065,steel)
for yy in [.45,1.38]:B.box('rear-horizontal-rail',(0,yy,-.045),(1.8,.085,.055),cedar,.005,uv=(2,0,6,.55,.2))
B.box('square-section-post',(-.90,.94,-.025),(.095,1.88,.095),cedar,.007,uv=(0,2,4,.5,.3))
B.box('beveled-post-cap',(-.90,1.90,-.025),(.115,.045,.115),cedar,.01)
B.box('weathering-cap-rail',(0,1.895,0),(1.8,.038,.065),cedar,.007,uv=(2,0,6,.55,.7))

B.collection('stucco-window-bay')
# Width 3.5 m, height 3.3 m, shallow cladding instead of a dense solid wall.
for xx in [-1.49,1.49]:B.box('stucco-jamb',(xx,1.65,0),(.52,3.3,.085),stucco,.008,uv=(0,2,1/3.5,1/3.3,.5))
B.box('stucco-sill-wall',(0,.30,0),(2.46,.60,.085),stucco,.006,uv=(0,2,1/3.5,1/3.3,.5))
B.box('stucco-lintel',(0,2.96,0),(2.46,.68,.085),stucco,.006,uv=(0,2,1/3.5,1/3.3,.5))
B.box('limestone-plinth',(0,.075,.025),(3.5,.15,.12),limestone,.007)
B.box('projecting-window-sill',(0,.585,.085),(2.59,.065,.24),limestone,.012)
B.box('recessed-tinted-glazing',(0,1.61,-.008),(2.40,2.00,.022),glass,.003)
for xx in [-1.23,0,1.23]:B.box('aluminium-window-mullion',(xx,1.61,.045),(.045,2.05,.085),black,.005)
for yy in [.60,2.62]:B.box('aluminium-window-transom',(0,yy,.045),(2.50,.045,.085),black,.005)
for xx in [-.10,.10]:
    B.rod('handle-spacer',(xx,1.47,.075),(xx,1.47,.11),.013,steel)
    B.rod('window-handle',(xx,1.42,.11),(xx,1.58,.11),.012,steel)
B.box('shadow-gap-at-eaves',(0,3.27,.04),(3.50,.035,.12),black,.002)
B.box('cedar-soffit',(0,3.33,.22),(3.50,.055,.55),cedar,.008,uv=(2,0,5,.5,.1))
B.box('metal-fascia',(0,3.40,.48),(3.50,.18,.045),black,.006)
B.box('roof-edge-flashing',(0,3.50,.23),(3.50,.035,.58),black,.004)

B.collection('limestone-coping')
B.box('rounded-coping-stone',(0,.070,0),(.497,.14,.298),limestone,.018,uv=(0,1,2,3,.5))
B.collection('limestone-coping-corner')
B.box('rounded-corner-stone',(0,.070,0),(.298,.14,.298),limestone,.018,uv=(0,1,3,3,.5))

B.collection('ceramic-waterline')
B.box('grouted-tile-backing',(0,.14,-.005),(1,.28,.012),grout,.001)
blues=[material(f'glazed-sea-tile-{i}',c,.18+i*.018) for i,c in enumerate([
    (.024,.20,.225),(.035,.26,.27),(.055,.31,.31),(.075,.29,.30),(.095,.36,.34),(.12,.38,.36)])]
for j in range(6):
    for i in range(24):
        B.box('glazed-mosaic-tessera',((i+.5)/24-.5,(j+.5)*.28/6,.004),(.0385,.043,.018),blues[int(rng.integers(6))],.002)

B.collection('teak-serving-table')
for i in range(12):B.box('slatted-teak-top',((i-5.5)*.108,.800,0),(.101,.050,2.50),teak,.008,uv=(0,1,5,.4,i*.179))
for zz in [-1.0,1.0]:B.box('table-cross-rail',(0,.738,zz),(1.15,.075,.065),teak,.006)
for xx in [-.50,.50]:
    B.box('long-apron',(xx,.70,0),(.055,.10,2.05),teak,.006,uv=(2,1,5,.5,.1))
    B.box('lower-stretcher',(xx,.20,0),(.04,.06,2.0),teak,.005)
    for zz in [-1.,1.]:
        B.box('table-leg',(xx,.385,zz),(.065,.77,.065),teak,.007,uv=(0,2,5,1.2,.3))
        B.box('table-foot',(xx,.015,zz),(.073,.03,.073),rubber,.006)
        B.rod('apron-bolt',(xx-.038,.71,zz),(xx+.038,.71,zz),.015,steel)

bpy.ops.object.select_all(action='DESELECT')
manifest=json.loads((OUT/'manifest.json').read_text());manifest['assets']=[a for a in manifest['assets'] if a['id'] not in B.assets]
for name,c in B.assets.items():
    for o in c.objects:o.select_set(True)
    bpy.context.view_layer.objects.active=next(iter(c.objects))
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_normals=True,export_texcoords=True,export_materials='EXPORT')
    manifest['assets'].append({'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','sourceCollection':name})
    bpy.ops.object.select_all(action='DESELECT')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'pool-party-architecture.blend'))

# Editable presentation, kept separate from the origin-space exports.
positions={'cedar-fence-panel':(3.1,0,-.5),'stucco-window-bay':(-.5,0,-1.),'limestone-coping':(-1.,0,2.1),'limestone-coping-corner':(-1.4,0,2.1),'ceramic-waterline':(-1.,0,1.8),'teak-serving-table':(1.0,0,1.0)}
for name,c in B.assets.items():
    for o in c.objects:o.location+=Vector(xyz(positions[name]))
B.collection('preview-only-stage');B.box('stone-patio',(0,-.09,.7),(10,.18,6),limestone,.01)
world=bpy.data.worlds.new('afternoon-sky');world.use_nodes=True;scene.world=world
sky=world.node_tree.nodes.new('ShaderNodeTexSky');sky.sky_type='MULTIPLE_SCATTERING';sky.sun_elevation=math.radians(38);sky.sun_rotation=math.radians(125)
world.node_tree.links.new(sky.outputs['Color'],world.node_tree.nodes.get('Background').inputs['Color']);world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.35
bpy.ops.object.camera_add(location=xyz((6.5,4.8,9)))
camera=bpy.context.object;camera.rotation_euler=(Vector(xyz((.7,1.2,.4)))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.lens=48;scene.camera=camera
scene.render.engine='CYCLES';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
scene.render.resolution_x=1400;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.8;scene.render.image_settings.file_format='PNG';scene.render.filepath=str(OUT/'pool-party-architecture-preview.png')
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'pool-party-architecture-preview.blend'))
if not args.no_render:bpy.ops.render.render(write_still=True)
