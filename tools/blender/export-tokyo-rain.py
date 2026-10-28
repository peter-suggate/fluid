"""Publish authored Tokyo geometry as compact reusable native-volume assets.
Never exports water, photographic rain, fog, lights, or camera geometry.
"""
import bpy,bmesh,json,math,sys,argparse,fnmatch
from pathlib import Path
from mathutils import Vector
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from tokyo_rain import build_scene
from base_color_image import linear_base_color_image
OUT=ROOT/'public/assets/tokyo-rain';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'assets/tokyo-rain';SOURCE.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--asset',default='*',help='Asset id or glob to rebuild; default all')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
build_scene()
original=list(bpy.context.scene.objects);deps=bpy.context.evaluated_depsgraph_get()
materials={};assets=[];instances=[]


def runtime_material(old):
    if old.name in materials:return materials[old.name]
    m=bpy.data.materials.new('Native '+old.name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF')
    src=old.node_tree.nodes.get('Principled BSDF') if old.use_nodes else None
    color=tuple(old.diffuse_color[:3]);p.inputs['Base Color'].default_value=(*color,1)
    if src:
        p.inputs['Roughness'].default_value=src.inputs['Roughness'].default_value
        p.inputs['Metallic'].default_value=src.inputs['Metallic'].default_value
    # Procedural surface colour becomes a real UV texture; the FVX importer
    # retains its spatial palette, unlike glTF's unbaked procedural nodes.
    if any(n.type=='TEX_NOISE' for n in old.node_tree.nodes):
        size=128;yy,xx=np.mgrid[:size,:size];rng=np.random.default_rng(47)
        coarse=(np.sin(xx*.12+np.sin(yy*.07)*3)+np.sin(yy*.18+xx*.06))*.14
        fine=rng.random((size,size))*.18-.09
        factor=np.clip(.83+coarse+fine,.45,1.3)
        if any(word in old.name.lower() for word in ['plaster','render']):
            streak=(.5+.5*np.cos(xx*math.tau/size*5+.22*np.sin(yy*math.tau/size)))**6
            factor*=1-.35*streak
        if 'cedar' in old.name.lower():factor=.64+.38*(.5+.5*np.sin(xx*.7+np.sin(yy*.04)*3))+fine
        px=np.ones((size,size,4),dtype=np.float32);px[:,:,:3]=np.array(color)*factor[:,:,None]
        im=linear_base_color_image(old.name+' baked grain',px)
        t=m.node_tree.nodes.new('ShaderNodeTexImage');t.image=im;m.node_tree.links.new(t.outputs['Color'],p.inputs['Base Color'])
    materials[old.name]=m;return m


def bounds(o):
    pts=[o.matrix_world@Vector(p) for p in o.bound_box]
    return [(min(p[i] for p in pts)+max(p[i] for p in pts))/2 for i in range(3)]


def export(name,sources,origin,cell):
    if not fnmatch.fnmatchcase(name,args.asset):return
    bpy.ops.object.select_all(action='DESELECT');objects=[]
    for source in sources:
        # Native static materials have no transmission. The transparent film
        # therefore stays optically empty, with resolved wet edges and ribs.
        if source.name.startswith('Clear eight-panel umbrella'):continue
        if name.startswith('tokyo-blade-'):
            # A neon plate is mounted clear of its coarser sampled backing.
            # Its thin source font otherwise disappears between FVX samples.
            source.location.y -= .08
            if source.type=='FONT':
                source.data.extrude=max(source.data.extrude,cell*.8)
                source.data.bevel_depth=max(source.data.bevel_depth,cell*.4)
            deps.update()
        if source.type=='CURVE' and source.name.startswith(('Umbrella steel rib','Umbrella wet perimeter')):
            source.data.bevel_depth=max(source.data.bevel_depth,cell*.55)
            deps.update()
        data=bpy.data.meshes.new_from_object(source.evaluated_get(deps))
        if not data.vertices:continue
        transform=source.matrix_world
        for v in data.vertices:v.co=transform@v.co-Vector(origin)
        # Curves are closed with finite envelopes; degenerate umbrella polar
        # vertices are welded before its thin shell is evaluated.
        bm=bmesh.new();bm.from_mesh(data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6);bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-6)
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
        boundaries=[e for e in bm.edges if e.is_boundary]
        if boundaries:bmesh.ops.holes_fill(bm,edges=boundaries,sides=0)
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bad=sum(not e.is_manifold for e in bm.edges)
        if bad:
            print('OMIT nonmanifold visual microdetail',source.name,bad,flush=True);bm.free();bpy.data.meshes.remove(data);continue
        bm.to_mesh(data);bm.free();data.update()
        oldm=list(data.materials);indices=[p.material_index for p in data.polygons];data.materials.clear()
        for m in oldm:data.materials.append(runtime_material(m))
        for p,idx in zip(data.polygons,indices):p.material_index=idx
        uv=data.uv_layers.new(name='World planar material UV')
        for p in data.polygons:
            normal=p.normal;axis=max(range(3),key=lambda a:abs(normal[a]));axes=[a for a in range(3) if a!=axis]
            for li in p.loop_indices:
                v=data.vertices[data.loops[li].vertex_index].co
                uv.data[li].uv=(v[axes[0]]*.65,v[axes[1]]*.65)
        o=bpy.data.objects.new(name+'/'+source.name,data);bpy.context.collection.objects.link(o);o.select_set(True);objects.append(o)
    bpy.context.view_layer.objects.active=objects[0]
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_normals=True,export_materials='EXPORT')
    assets.append({'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','cellSize':cell,'bandCells':1,'sourceNames':[o.name for o in sources]})
    for o in objects:bpy.data.objects.remove(o,do_unlink=True)
    print('EXPORTED',name,len(objects),flush=True)


skip=('Narrow old shop building','Flooded street','footpath','curb joints','Night mist','Airborne','Far street','Distant','End street','Soft sky','Rain in','Street photography','Overhead','Sagging facade')
# Emissive letters and rails are independently instanced. Keep their opaque
# enamel backing in the ordinary storefront so authored lamp intensity cannot
# turn an unlit black signboard into a luminous panel.
sign=('Neon sign edge','Japanese illuminated lettering')
for side,label,y in [(-1,'left',-3),(1,'right',-1.75)]:
    picked=[];signs=[];lights=[]
    for o in original:
        if o.type not in ('MESH','CURVE','FONT') or o.name.startswith(skip):continue
        x,by,z=bounds(o)
        if x*side<(1.8 if o.name.startswith(sign) else 2.25) or x*side>5.3 or not y-1.79<by<y+1.79:continue
        if o.name.startswith(('Vending','Drink can','Japanese drink','Coin slot')):continue
        if o.name.startswith(('Rain-streaked glazing','Ribbed red paper lantern')):continue
        if o.name.startswith(('Warm interior behind glass','Ribbed red paper lantern','Interior hanging strip lamp')):lights.append(o)
        else:(signs if o.name.startswith(sign) else picked).append(o)
    origin=(side*3,y,0)
    export('tokyo-shopfront-'+label,picked,origin,.06)
    export('tokyo-blade-'+label,signs,origin,.00625)
    export('tokyo-lights-'+label,lights,origin,.05)
    for i in range(7):
        for kind in ['shopfront','blade','lights']:
            if label=='right' and i%2==1 and kind=='lights':continue
            asset_id='tokyo-shopfront-closed-right' if label=='right' and i%2==1 and kind=='shopfront' else f'tokyo-{kind}-{label}'
            if kind=='blade' and i%3:asset_id+=f'-{i%3}'
            instances.append({'id':f'tokyo/{kind}/{label}/{i}','asset':asset_id,'position':[side*3,0,-y-i*3.6],'orientation':[1,0,0,0],'emission':4 if kind=='blade' else 3.2 if kind=='lights' else 0})
# Each outdoor lantern is one physical fixture. Grouping it with rear shop
# lamps placed the analytic emitter centre behind the facade and shadowed it.
lantern=next(o for o in original if o.name.startswith('Ribbed red paper lantern'))
# The emitter belongs to the paper, independently of glTF material ordering.
# Adding opaque ribs must not change the light to their dark wood color.
lantern_emission_color=list(lantern.data.materials[0].node_tree.nodes.get('Principled BSDF').inputs['Emission Color'].default_value[:3])
# Keep the source's actual bamboo hoops with the paper. Their 5 mm diameter
# vanished when they belonged only to the 6 cm storefront volume; sampling
# this small fixture separately preserves its authored silhouette/material.
lantern_ribs=[o for o in original if o.name.startswith('Lantern bamboo ribs')
              and abs(bounds(o)[0]-lantern.location.x)<.01
              and abs(bounds(o)[1]-lantern.location.y)<.01
              and abs(bounds(o)[2]-lantern.location.z)<.35]
assert len(lantern_ribs)==12, f'Expected 12 authored lantern hoops, found {len(lantern_ribs)}'
export('tokyo-lantern',[lantern,*lantern_ribs],tuple(lantern.location),.00625)
for i in range(7):
    for k in [-1,1]:
        instances.append({'id':f'tokyo/lantern/{i}/{k}','asset':'tokyo-lantern','position':[-2.64,2.57,3-i*3.6-k*.85],'orientation':[1,0,0,0],'emission':6.4,'emissionColorLinear':lantern_emission_color})
# Two additional authored Japanese sign designs per side interrupt repetition.
for side,label,y in [(-1,'left',-3),(1,'right',-1.75)]:
    for variant in [1,2]:
        sy=y+variant*3.6
        selected=[o for o in original if o.type in ('MESH','CURVE','FONT') and o.name.startswith(sign) and bounds(o)[0]*side>1.8 and sy-1.79<bounds(o)[1]<sy+1.79]
        export(f'tokyo-blade-{label}-{variant}',selected,(side*3,sy,0),.00625)
# The alternating closed store keeps the authored shutter relief and avoids
# repeating the open ramen interior along both sides of the street.
closed=[];cy=1.85
for o in original:
    if o.type not in ('MESH','CURVE','FONT') or o.name.startswith(skip+sign):continue
    x,by,z=bounds(o)
    if not (2.25<x<5.3 and cy-1.79<by<cy+1.79):continue
    if o.name.startswith(('Rain-streaked glazing','Interior hanging strip lamp','Warm interior','Ribbed red','Vending','Drink can','Japanese drink','Coin slot')):continue
    closed.append(o)
export('tokyo-shopfront-closed-right',closed,(3,cy,0),.06)
# The small modeled cyclist is its own reusable asset, independent of water.
figure=('Bicycle pedal','Pneumatic bicycle','Polished wheel','Stainless bicycle','Wheel axle','Bicycle tubular','Seat post','Leather saddle','Handlebar','Bicycle rear','Cyclist','Coat','Trouser','Rider','Umbrella','Clear eight')
export('tokyo-cyclist',[o for o in original if o.type in ('MESH','CURVE') and o.name.startswith(figure)],(.35,3.5,.43),.035)
instances.append({'id':'tokyo/cyclist','asset':'tokyo-cyclist','position':[.35,.43,-3.5],'orientation':[1,0,0,0]})
vending=('Japanese drink vending','Vending machine','Drink can','Coin slot')
export('tokyo-vending',[o for o in original if o.type in ('MESH','CURVE') and o.name.startswith(vending)],(2.58,-.4,.45),.035)
instances.append({'id':'tokyo/vending','asset':'tokyo-vending','position':[2.58,.45,.4],'orientation':[1,0,0,0]})
manifest={'version':1,'collection':{'id':'tokyo-rain','label':'Tokyo rain · Blender authored street'},'source':'tokyo-rain.blend','assets':assets}
if not assets:raise ValueError(f'No assets match {args.asset!r}')
if args.asset!='*' and (OUT/'manifest.json').exists():
    previous=json.loads((OUT/'manifest.json').read_text())['assets'];updated={a['id']:a for a in assets}
    manifest['assets']=[updated.pop(a['id'],a) for a in previous]+list(updated.values())
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'lib/core/imported-assets/tokyo-rain-instances.json').write_text(json.dumps(instances,indent=2)+'\n')
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'tokyo-rain.blend'))
