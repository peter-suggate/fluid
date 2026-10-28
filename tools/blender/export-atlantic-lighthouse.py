"""Export closed Blender geometry and face materials for the live SVO renderer.

No water is exported: NB-FLIP remains the live water authority.
The source .blend is authored by atlantic_lighthouse.py (--dry-preview works).
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector, noise
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lighthouse_rock_material import photographed_rock_material, assign_rock_uv, write_glb_rock_factor

ROOT = Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--blend',type=Path,default=ROOT/'assets/atlantic-lighthouse/storm-lighthouse.blend')
parser.add_argument('--asset',choices=['atlantic-lighthouse-tower','atlantic-lighthouse-rock'])
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
OUT=ROOT/'public/assets/atlantic-lighthouse'; OUT.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
original=list(bpy.context.scene.objects)
random.seed(20261028)
materials={}
assets=[]


def flat(name,color,rough=.7,metal=0):
    if name in materials:return materials[name]
    m=bpy.data.materials.get('Runtime | '+name) or bpy.data.materials.new('Runtime | '+name);m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal
    materials[name]=m
    return m


stone=[flat('Weathered granite '+str(i),(.07+i*.024,.071+i*.0225,.065+i*.02)) for i in range(15)]
rock=photographed_rock_material()
black=flat('Salted iron',(.021,.026,.03),.4,.65)
glass=flat('Dark reflective glazing',(.055,.085,.095),.18,.45)
mortar=flat('Recessed salt-aged lime mortar',(.235,.23,.205),.9)


def export(name,objects,cell,origin):
    bpy.ops.object.select_all(action='DESELECT')
    baked=[]
    deps=bpy.context.evaluated_depsgraph_get()
    for source in objects:
        data=bpy.data.meshes.new_from_object(source.evaluated_get(deps))
        for v in data.vertices:v.co=source.matrix_world@v.co-Vector(origin)
        ob=bpy.data.objects.new(name+'/'+source.name,data);bpy.context.scene.collection.objects.link(ob)
        data.materials.clear()
        if 'mortar' in source.name.lower():data.materials.append(mortar)
        elif 'Masonry' in source.name or 'Stone' in source.name or 'Foundation' in source.name:
            for m in stone:data.materials.append(m)
            data.update()
            for face in data.polygons:
                q=face.center
                stain=noise.noise_vector(Vector((q.x*2.2,q.y*2.2,q.z*.09)))[0]*.5+.5
                index=int(3+stain*13)
                if q.z<1.4:index-=3
                if q.y<-.7 and abs(q.x+.2)<.22 and (2.3<q.z<3.6 or 5<q.z<6.3):index-=5
                face.material_index=max(0,min(14,index))
        elif 'glazing' in source.name.lower():data.materials.append(glass)
        elif 'rock' in source.name.lower():
            data.materials.append(rock)
            assign_rock_uv(data,origin)
        else:data.materials.append(black)
        bm=bmesh.new();bm.from_mesh(data)
        bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.000001)
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
        bad=sum(not edge.is_manifold for edge in bm.edges)
        if bad:raise ValueError(source.name+': '+str(bad)+' nonmanifold edges')
        bm.to_mesh(data);bm.free();baked.append(ob)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in baked:ob.select_set(True)
    bpy.context.view_layer.objects.active=baked[0]
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')),export_format='GLB',use_selection=True,
        export_apply=True,export_yup=True,export_normals=True,export_materials='EXPORT')
    if name=='atlantic-lighthouse-rock':write_glb_rock_factor(OUT/(name+'.glb'))
    assets.append({'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','cellSize':cell,'bandCells':1,
        'sourceNames':[ob.name for ob in objects],'sourceCollection':'Atlantic lighthouse',
        'description':'Blender-authored closed static geometry; live fluid excluded'})
    for ob in baked:bpy.data.objects.remove(ob,do_unlink=True)
    print('EXPORTED',name,flush=True)


tower=[o for o in original if o.type=='MESH' and not any(s in o.name for s in ['rock shelf','water','haze','Fresnel lens','storm bank'])
       and not any(m and m.name.startswith('Lantern glass') for m in o.data.materials)
       # Native FVX is opaque: preserve the glazing/lens aperture, including rings.
       and min((o.matrix_world@Vector(c)).z for c in o.bound_box)>5.4]
if args.asset != 'atlantic-lighthouse-rock':export('atlantic-lighthouse-tower',tower,.04,(7,0,5.8))


rock_layout=json.loads((ROOT/'lib/core/imported-assets/atlantic-lighthouse-rock-layout.json').read_text())
def height(x,y):
    z=-y
    h=.15
    for r in rock_layout:
        dx,dz=x-r['x'],z-r['z'];c,s=math.cos(r['yaw']),math.sin(r['yaw'])
        u,v=(c*dx+s*dz)/r['rx'],(-s*dx+c*dz)/r['rz']
        q=max(abs(u),abs(v),abs(u+v)*.65,abs(u-v)*.65)
        top=r['height']+dx*r['tiltX']+dz*r['tiltZ']
        h=max(h,top-max(0,q-r['core'])*r['height']/(1-r['core']))
    h+=.07*math.sin(x*3.1+z*1.7)*math.sin(z*2.3-x*.4)*min(1,max(0,h-.15))
    return 5.8 if math.hypot(x-7,z)<1.55 else h


nx,ny=73,81
verts=[(-2+x*.25,-10+y*.25,height(-2+x*.25,-10+y*.25)) for y in range(ny) for x in range(nx)]
faces=[(x+nx*y,x+1+nx*y,x+1+nx*(y+1),x+nx*(y+1)) for y in range(ny-1) for x in range(nx-1)]
edge=list(range(nx))+[nx-1+nx*y for y in range(1,ny)]+[x+nx*(ny-1) for x in range(nx-2,-1,-1)]+[nx*y for y in range(ny-2,0,-1)]
bottom=[]
for i in edge:
    x,y,z=verts[i];bottom.append(len(verts));verts.append((x,y,-.2))
for j in range(len(edge)):
    k=(j+1)%len(edge);faces.append((edge[j],bottom[j],bottom[k],edge[k]))
faces.append(tuple(reversed(bottom)))
data=bpy.data.meshes.new('Closed authoritative rock mesh');data.from_pydata(verts,[],faces);data.update()
ob=bpy.data.objects.new('Fractured rock island',data);bpy.context.scene.collection.objects.link(ob)
for p in data.polygons:p.use_smooth=True
if args.asset != 'atlantic-lighthouse-tower':export('atlantic-lighthouse-rock',[ob],.125,(7,0,0))
bpy.data.objects.remove(ob,do_unlink=True)
if args.asset and (OUT/'manifest.json').exists():
    existing=json.loads((OUT/'manifest.json').read_text())['assets']
    replacement={a['id']:a for a in assets}
    assets=[replacement.get(a['id'],a) for a in existing]
(OUT/'manifest.json').write_text(json.dumps({'title':'Atlantic Lighthouse','assets':assets},indent=2)+'\n')
source_dir=ROOT/'assets/atlantic-lighthouse';source_dir.mkdir(parents=True,exist_ok=True)
for source in list(bpy.context.scene.objects):
    if 'NB-FLIP simulated' in source.name:bpy.data.objects.remove(source,do_unlink=True)
    elif source.name == 'Authoritative simulated rock shelf':
        for v in source.data.vertices:v.co.z=height(v.co.x,v.co.y)
        source.data.materials.clear();source.data.materials.append(rock);assign_rock_uv(source.data)
bpy.context.scene['water_method']='Live app NB-FLIP water; static authoring source contains no water'
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(source_dir/'storm-lighthouse.blend'))
print('MANIFEST',OUT/'manifest.json',flush=True)
