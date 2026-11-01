"""Author Beach in Blender before exporting the live scenery and collision grid.

blender -b --python-exit-code 1 --python tools/blender/beach.py
Optional -- --water-mesh path.obj --render path.png renders actual NB-FLIP output.
No ocean modifier, displaced water plane, or substitute liquid is authored.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Quaternion, Matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_color_image import linear_base_color_image
from lighthouse_rock_material import photographed_rock_material, assign_rock_uv

# Beach colour is independent of the lighthouse material tuning.
ROCK_FACTOR=(.58,.55,.50,1)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--water-mesh', type=Path)
parser.add_argument('--render', type=Path)
parser.add_argument('--samples', type=int, default=64)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
OUT = ROOT/'public/assets/beach'
SOURCE = ROOT/'assets/beach'
for path in [OUT, SOURCE]: path.mkdir(parents=True, exist_ok=True)
random.seed(20261101)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'


ROCKS = [(6.4,-5.6,1.1,.8,.8),(2.5,5.4,.85,.65,.75),(-1,7.2,1.25,.9,1.9),(8.2,7.1,.6,.5,.5)]


def sand_height(x, z):
    """Metres, app Y up. Shelf, gently curved foreshore, berm and dune toes."""
    shore = x + .3*math.sin(z*.35) + .12*math.sin(z*.83)
    slope = max(0, shore+9)/7
    dune = .65*math.exp(-((x-13)/2.5)**2)*(1+.22*math.sin(z*.67))
    ripple = .018*math.sin(x*8+math.sin(z*.7))*min(1, max(0, (x+6)/5))
    return .35+slope+dune+ripple


def rock_height(x,z,rock):
    rx,rz,a,b,c=rock;u,v=(x-rx)/a,(z-rz)/b
    q=max(abs(u)*.91,abs(v)*.9,abs(u+.62*v)*.72,abs(u-.85*v)*.64)
    top=max(0,1-q**3)**.65
    bedding=.07*math.sin(x*5.3+z*2.7)+.045*math.sin(z*9.2-x*3.1)
    joint=.055*max(0,1-abs(math.sin(x*10.4+z*3.8))/.20)
    return sand_height(rx,rz)-.5+(c+.28)*top+(bedding-joint)*top


def height(x,z):
    return max([sand_height(x,z)]+[rock_height(x,z,rock) for rock in ROCKS
        if abs(x-rock[0])<rock[2]*1.5 and abs(z-rock[1])<rock[3]*1.5])


def mesh(name, vertices, faces):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    bm = bmesh.new(); bm.from_mesh(data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    assert all(e.is_manifold for e in bm.edges), name
    bm.to_mesh(data); bm.free()
    ob = bpy.data.objects.new(name, data); scene.collection.objects.link(ob)
    for p in data.polygons: p.use_smooth = True
    return ob


# Closed sand bed. The exported simulation grid samples this bed together
# with the independently detailed rock meshes using the same height functions.
NX, NZ, SPACING = 257, 193, .125
heights = [height(-16+i%NX*SPACING, -12+i//NX*SPACING) for i in range(NX*NZ)]
verts = [(-16+i%NX*SPACING, 12-i//NX*SPACING, sand_height(-16+i%NX*SPACING,-12+i//NX*SPACING)) for i in range(NX*NZ)]
faces = [(i, i+1, i+1+NX, i+NX) for j in range(NZ-1) for i in range(j*NX, j*NX+NX-1)]
edge = list(range(NX))+[j*NX+NX-1 for j in range(1,NZ)]+list(range(NX*NZ-2,NX*(NZ-1)-1,-1))+[j*NX for j in range(NZ-2,0,-1)]
bottom = []
for i in edge:
    x,y,_ = verts[i]; bottom.append(len(verts)); verts.append((x,y,-.25))
for j in range(len(edge)):
    k=(j+1)%len(edge); faces.append((edge[j],bottom[j],bottom[k],edge[k]))
faces.append(tuple(reversed(bottom)))
bed = mesh('beach-sand', verts, faces)
uv = bed.data.uv_layers.new(name='UVMap')
for face in bed.data.polygons:
    for loop in face.loop_indices:
        p=bed.data.vertices[bed.data.loops[loop].vertex_index].co
        uv.data[loop].uv=((p.x+16)/32, (12-p.y)/24)

# Portable photographic-scale sand variation, shared by Cycles and glTF/FVX.
rng=np.random.default_rng(20261101)
z,x=np.mgrid[-12:12:1024j,-16:16:1024j]
wet=1/(1+np.exp(-(x-5.2)*2))
grain=rng.normal(0,.024,x.shape)
streak=.018*np.sin(x*2.1+np.sin(z*.8))+.009*np.sin(z*17+x*7)
rgba=np.ones((*x.shape,4),dtype=np.float32)
for c,(low,high) in enumerate(zip((.17,.125,.075),(.57,.43,.26))):
    rgba[:,:,c]=np.clip(low+(high-low)*wet+grain+streak,0,1)
sand=bpy.data.materials.new('Quartz sand | dark wet foreshore to dry dune'); sand.use_nodes=True
n,l=sand.node_tree.nodes,sand.node_tree.links
p=n.get('Principled BSDF'); p.inputs['Roughness'].default_value=.78
tex=n.new('ShaderNodeTexImage'); tex.image=linear_base_color_image('Beach sand albedo',rgba)
l.new(tex.outputs['Color'],p.inputs['Base Color'])
fine=n.new('ShaderNodeTexNoise'); fine.inputs['Scale'].default_value=750
bump=n.new('ShaderNodeBump'); bump.inputs['Distance'].default_value=.0015; bump.inputs['Strength'].default_value=.35
l.new(fine.outputs['Fac'],bump.inputs['Height']); l.new(bump.outputs[0],p.inputs['Normal'])
bed.data.materials.append(sand)


def material(name, color, rough=.8):
    m=bpy.data.materials.new(name); m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF'); p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough
    return m


wood=material('Salt bleached driftwood',(.26,.20,.135))
stone=photographed_rock_material()
# Keep Base Color directly image-linked through the GLB round trip. The FVX
# baker samples image inputs, but cannot evaluate glTF's factor/multiply node.
# Bake the coastal tint into the sRGB image instead of losing it to grey .8.
stone_nodes=stone.node_tree.nodes
stone_bsdf=stone_nodes.get('Principled BSDF')
stone_tint=stone_bsdf.inputs['Base Color'].links[0].from_node
stone_diffuse=stone_tint.inputs[1].links[0].from_node
stone_image=stone_diffuse.image
stone_pixels=np.empty(len(stone_image.pixels),dtype=np.float32)
stone_image.pixels.foreach_get(stone_pixels)
stone_pixels=stone_pixels.reshape((stone_image.size[1],stone_image.size[0],4))
stone_rgb=stone_pixels[:,:,:3]
stone_pixels[:,:,:3]=np.where(stone_rgb<=.04045,stone_rgb/12.92,((stone_rgb+.055)/1.055)**2.4)*np.asarray(ROCK_FACTOR[:3])
stone_diffuse.image=linear_base_color_image('Beach photographed mineral colour',stone_pixels)
stone.node_tree.links.new(stone_diffuse.outputs['Color'],stone_bsdf.inputs['Base Color'])
stone_nodes.remove(stone_tint)
rocks=[]
for index,rock in enumerate(ROCKS):
    rx,rz,a,b,c=rock;nx,nz=81,65
    rv=[(rx-a*1.5+ix*a*3/(nx-1),-(rz-b*1.5+iz*b*3/(nz-1)),
        rock_height(rx-a*1.5+ix*a*3/(nx-1),rz-b*1.5+iz*b*3/(nz-1),rock)) for iz in range(nz) for ix in range(nx)]
    rf=[(i,i+1,i+1+nx,i+nx) for j in range(nz-1) for i in range(j*nx,j*nx+nx-1)]
    edge=list(range(nx))+[j*nx+nx-1 for j in range(1,nz)]+list(range(nx*nz-2,nx*(nz-1)-1,-1))+[j*nx for j in range(nz-2,0,-1)]
    bottom=[]
    for i in edge:
        px,py,_=rv[i];bottom.append(len(rv));rv.append((px,py,sand_height(rx,rz)-.6))
    for j in range(len(edge)):
        k=(j+1)%len(edge);rf.append((edge[j],bottom[j],bottom[k],edge[k]))
    rf.append(tuple(reversed(bottom)))
    ob=mesh('Coastal rock | weathered bedding %d'%index,rv,rf);ob.data.materials.append(stone)
    assign_rock_uv(ob.data)
    # Centimetre mineral details use photographed colour, roughness and normals.
    rocks.append(ob)
grass=[material('Marram grass '+str(i),c) for i,c in enumerate([(.12,.14,.045),(.26,.25,.10),(.38,.32,.14)])]
props=[]
# Wind-combed dune grass; each blade is a closed taper, not a camera-facing card.
for tuft in range(62):
    x=random.uniform(10.5,15.5); z=random.uniform(-11.4,11.4)
    for blade in range(random.randint(7,12)):
        angle=random.uniform(0,math.tau); length=random.uniform(.3,.85); radius=random.uniform(.035,.065)
        bx=x+random.uniform(-.22,.22); bz=z+random.uniform(-.22,.22); h=height(bx,bz)-.04
        dx=math.cos(angle)*.18+.18; dz=math.sin(angle)*.18
        v=[]
        for t,r in [(0,radius),(.55,radius*.55),(1,.003)]:
            v.extend([(bx+dx*t*t+r*math.cos(a),-bz+dz*t*t+r*math.sin(a),h+length*t) for a in (0,math.tau/3,2*math.tau/3)])
        f=[(2,1,0),(6,7,8)]+[(j*3+i,j*3+(i+1)%3,(j+1)*3+(i+1)%3,(j+1)*3+i) for j in range(2) for i in range(3)]
        ob=mesh('Marram blade',v,f); ob.data.materials.append(random.choice(grass)); props.append(ob)
for i,(x,z,length) in enumerate([(8.2,-4.7,2.4),(9.4,6.5,1.1)]):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,radius=1,location=(x,-z,height(x,z)+.12))
    ob=bpy.context.object; ob.name='Tidal driftwood '+str(i); ob.scale=(.14,length/2,.12); ob.rotation_euler[2]=.65
    for v in ob.data.vertices: v.co*=random.uniform(.88,1.12)
    ob.data.materials.append(wood); props.append(ob)

# Dense, curved pinnate crowns authored as closed Blender meshes.
from beach_palms import build_palm
palm=build_palm(mesh,material)

# Floating Sargassum-style clump: tapered wet blades, branching stems and air
# bladders. Export analytic ellipsoids from the Blender source for rigid skins.
seaweed=[]; sea_parts=[]
weed_proxy={'dimensions_m':{'x':.9,'y':.375,'z':.6},'density_kg_m3':950}
weed_proxy['surfaceOffset_m']=weed_proxy['dimensions_m']['y']*(weed_proxy['density_kg_m3']/1025-.5)
kelp=material('Wet olive brown seaweed',(.09,.105,.018),.25)
bladder=material('Golden gas bladders',(.21,.145,.025),.32)
def weed_part(name,center,half,angle,mat):
    # Thin visible blades sit at the waterline above the resolved wet proxy's
    # centre of mass; they must not vanish beneath its hydrostatic draft.
    center=(center[0],center[1],center[2]+weed_proxy['surfaceOffset_m'])
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=6,radius=1,location=center)
    ob=bpy.context.object; ob.name=name; ob.scale=half; ob.rotation_euler=(random.uniform(-.65,.65),random.uniform(-.35,.35),angle); ob.data.materials.append(mat)
    for p in ob.data.polygons:p.use_smooth=True
    seaweed.append(ob)
    # Application axes x,y,z = Blender x,z,-y; rotation about Blender Z maps to Y.
    color=list(mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value[:3])
    q=ob.rotation_euler.to_quaternion()
    sea_parts.append({'p':[center[0],center[2],-center[1]],'d':[half[0],half[2],half[1]],
        'q':[q.w,q.x,q.z,-q.y],'color':color})
for i in range(27):
    a=random.uniform(0,math.tau);length=random.uniform(.10,.19)
    cx,cy=random.uniform(-.26,.26),random.uniform(-.21,.21);h=random.uniform(-.045,.035)
    weed_part('Seaweed overlapping blade',(cx,cy,h),(length,.022,.008),a,kelp)
    weed_part('Seaweed tangled stipe',(cx*.7,cy*.7,h-.015),(.11,.007,.007),a+.5,kelp)
    if i%2==0:weed_part('Buoyant gas bladder',(cx*.8,cy*.8,h+.01),(.025,.019,.023),a,bladder)
extent=[0,0,0]
for ob in seaweed:
    bpy.context.view_layer.update()
    for corner in ob.bound_box:
        p=ob.matrix_world@Vector(corner)
        for axis,value in enumerate((p.x,p.z,-p.y)):extent[axis]=max(extent[axis],abs(value)+.005)
(ROOT/'lib/core/imported-assets/beach-seaweed.generated.json').write_text(json.dumps({'extent':extent,'buoyancyProxy':weed_proxy,'roughness':kelp.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value,'parts':sea_parts},separators=(',',':'))+'\n')


def export(name, objects, cell):
    bpy.ops.object.select_all(action='DESELECT')
    for ob in objects: ob.select_set(True)
    bpy.context.view_layer.objects.active=objects[0]
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')),export_format='GLB',use_selection=True,
        export_apply=True,export_yup=True,export_normals=True,export_materials='EXPORT')
    return {'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','cellSize':cell,'bandCells':1}


manifest={'collection':{'id':'beach','label':'Beach'},'source':'beach.blend','assets':[
    export('beach-sand',[bed],.125),export('beach-dunes',props,.025),export('beach-palm',palm,.0125),export('beach-rocks',rocks,.04)]}
# Seaweed uses its exported analytic skin, not a static scenery volume.
bpy.ops.object.select_all(action='DESELECT')
for ob in seaweed:ob.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(OUT/'beach-seaweed.glb'),export_format='GLB',use_selection=True,export_yup=True)
# Place source instances after exporting their reusable local geometry.
bpy.context.view_layer.update()
palm_local=[ob.matrix_world.copy() for ob in palm]
for index,(x,z,yaw,scale) in enumerate([(11.3,-5.8,0,1),(12.9,1.8,2.2,1),(12.1,7.8,-.8,1)]):
    root=Matrix.Translation((x,-z,sand_height(x,z)))@Matrix.Rotation(yaw,4,'Z')@Matrix.Scale(scale,4)
    for source,local in zip(palm,palm_local):
        ob=source if index==0 else source.copy()
        if index:scene.collection.objects.link(ob)
        ob.matrix_world=root@local
for ob in seaweed:ob.hide_render=True;ob.hide_set(True)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
terrain={'kind':'grid','origin_m':{'x':-16,'z':-12},'spacing_m':SPACING,'size':{'nx':NX,'nz':NZ},
    'heights_m':[round(h,6) for h in heights]}
(ROOT/'lib/core/imported-assets/beach-terrain.json').write_text(json.dumps(terrain,separators=(',',':'))+'\n')

world=bpy.data.worlds.new('Coastal afternoon'); scene.world=world; world.use_nodes=True
n,l=world.node_tree.nodes,world.node_tree.links
coords=n.new('ShaderNodeTexCoord')
cloud=n.new('ShaderNodeTexNoise');cloud.inputs['Scale'].default_value=4.6;cloud.inputs['Detail'].default_value=5;cloud.inputs['Roughness'].default_value=.65
l.new(coords.outputs['Normal'],cloud.inputs['Vector'])
ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.53;ramp.color_ramp.elements[0].color=(.07,.27,.65,1)
ramp.color_ramp.elements[1].position=.68;ramp.color_ramp.elements[1].color=(1.2,1.25,1.3,1)
mid=ramp.color_ramp.elements.new(.59);mid.color=(.55,.68,.85,1)
l.new(cloud.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs[0],n.get('Background').inputs['Color']);n.get('Background').inputs['Strength'].default_value=.65
bpy.ops.object.light_add(type='SUN',location=(-10,-8,14));sun=bpy.context.object;sun.name='Warm coastal sunlight';sun.data.energy=3;sun.data.angle=.025
sun.rotation_euler=Vector((.62,-.67,-.41)).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(-10,-7,5.5))
camera=bpy.context.object; camera.name='Beach shoreline camera'; camera.rotation_euler=(Vector((10,0,4))-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.lens=27.4; scene.camera=camera
scene.render.engine='CYCLES'; scene.cycles.samples=args.samples; scene.cycles.use_denoising=True
scene.cycles.max_bounces=10; scene.cycles.transmission_bounces=8
scene.render.resolution_x=1400; scene.render.resolution_y=900; scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX'
scene['water_method']='uniform-narrow-band-flip; static source deliberately contains no water'
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'beach.blend'))
if args.water_mesh:
    provenance=Path(str(args.water_mesh)+'.json')
    if not provenance.exists():provenance=args.water_mesh.with_suffix('.json')
    report=json.loads(provenance.read_text())
    assert report['methodId']=='uniform-narrow-band-flip' and report['completedTime_s']>0
    assert report['objSha256']==hashlib.sha256(args.water_mesh.read_bytes()).hexdigest()
    bpy.ops.wm.obj_import(filepath=str(args.water_mesh.resolve()),forward_axis='Y',up_axis='Z')
    water=material('Simulated seawater dielectric',(.98,.99,1),.055)
    p=water.node_tree.nodes.get('Principled BSDF'); p.inputs['IOR'].default_value=1.333; p.inputs['Transmission Weight'].default_value=1
    absorption=water.node_tree.nodes.new('ShaderNodeVolumeAbsorption'); absorption.inputs['Color'].default_value=(.52,.82,.88,1); absorption.inputs['Density'].default_value=.16
    water.node_tree.links.new(absorption.outputs[0],water.node_tree.nodes.get('Material Output').inputs['Volume'])
    for ob in bpy.context.selected_objects:
        ob.name='NB-FLIP simulated beach water'; ob.data.materials.clear(); ob.data.materials.append(water)
        for p in ob.data.polygons:p.use_smooth=True
    scene['water_provenance']=json.dumps(report)
    scene['water_method']='uniform-narrow-band-flip; exported production free surface and GPU rigid poses'
    for body in report.get('rigidBodies',[]):
        if body.get('appearance')!='seaweed':continue
        p,q=body['position_m'],body['orientation']
        rotation=Quaternion((q['w'],q['x'],-q['z'],q['y']))
        position=Vector((p['x'],-p['z'],p['y']))
        for source in seaweed:
            ob=source.copy();ob.data=source.data;scene.collection.objects.link(ob)
            ob.hide_render=False;ob.hide_set(False)
            ob.location=position+rotation@source.location
            ob.rotation_mode='QUATERNION';ob.rotation_quaternion=rotation@source.rotation_euler.to_quaternion()
if args.render:
    args.render.parent.mkdir(parents=True,exist_ok=True)
    scene.render.filepath=str(args.render.resolve())
    bpy.ops.wm.save_as_mainfile(filepath=str(args.render.with_suffix('.blend').resolve()))
    bpy.ops.render.render(write_still=True)
