"""Export shared, closed static runtime assets from the authored forest.

Run with Blender -b --python-exit-code 1 --python this_file.py.
Then bake_mesh_volume.py -- --collection forest-waterfall. Source water,
foam, droplets, lighting and atmosphere are intentionally never exported.
Open leaves acquire finite visual envelopes for the sampled runtime LOD;
instance scales quantize to the authored asset sizes because field programs
are authored in metres and cannot scale their sampled distances.
"""
import bpy, bmesh, json, math, sys, random, hashlib
from pathlib import Path
from mathutils import Vector, Quaternion, noise

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'assets/forest-waterfall'
OUT=ROOT/'public/assets/forest-waterfall'
OUT.mkdir(parents=True,exist_ok=True)
TREES_ONLY='--trees-only' in sys.argv
previous_manifest=json.loads((OUT/'manifest.json').read_text()) if TREES_ONLY else None
sys.path.insert(0,str(Path(__file__).parent))
from waterfall_foliage import tree_species, _material
bpy.ops.wm.open_mainfile(filepath=str(SOURCE/'stillwater-grove.blend'))
original=list(bpy.context.scene.objects)
original_meshes={obj.data.name:obj.data for obj in original if obj.type=='MESH'}
prototypes=[]; assets=[]; instances=[]; material_cache={}


def flat_material(mat):
    if mat.name in material_cache:return material_cache[mat.name]
    p=next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None) if mat.use_nodes else None
    color=tuple(p.inputs['Base Color'].default_value[:3]) if p and not p.inputs['Base Color'].is_linked else tuple(mat.diffuse_color[:3])
    if 'bark' in mat.name:color=(.075,.044,.021)
    new=bpy.data.materials.new('Runtime | '+mat.name);new.use_nodes=True
    bs=new.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(*color,1)
    bs.inputs['Roughness'].default_value=p.inputs['Roughness'].default_value if p else .75
    material_cache[mat.name]=new
    return new


def stone_palette():
    mats=[]
    colors=[(.035,.042,.035),(.065,.075,.060),(.10,.115,.092),(.15,.165,.13),(.20,.215,.17),
            (.018,.045,.008),(.032,.075,.012),(.065,.12,.019),(.10,.17,.027),(.15,.22,.039)]
    for i,c in enumerate(colors):
        m=bpy.data.materials.new('Runtime mineral grain and moss '+str(i));m.use_nodes=True
        p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*c,1);p.inputs['Roughness'].default_value=.8
        mats.append(m)
    return mats


stone_materials=stone_palette()


def clean(obj,thickness=0):
    bpy.context.view_layer.objects.active=obj;obj.select_set(True)
    bm=bmesh.new();bm.from_mesh(obj.data)
    # Connected leaves intentionally share attachment positions without sharing
    # topology. Sub-millimetre offsets avoid accidental welds between touching
    # independent envelopes during glTF re-import validation.
    pending=set(bm.verts);r=random.Random(441)
    while pending:
        seed=pending.pop();group=[seed];stack=[seed]
        while stack:
            v=stack.pop()
            for edge in v.link_edges:
                other=edge.other_vert(v)
                if other in pending:pending.remove(other);group.append(other);stack.append(other)
        delta=Vector((r.uniform(-.0002,.0002),r.uniform(-.0002,.0002),r.uniform(-.0002,.0002)))
        for v in group:v.co+=delta
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-7)
    bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-6)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free()
    # Only open botanical surfaces need a shell; closed woody branches retain
    # their exact source volume. Solidify closes all disconnected leaf margins.
    bm=bmesh.new();bm.from_mesh(obj.data)
    boundary=any(not e.is_manifold for e in bm.edges);bm.free()
    if boundary:
        mod=obj.modifiers.new('Finite closed botanical sampling envelope','SOLIDIFY')
        mod.thickness=thickness;mod.offset=0;mod.use_even_offset=False
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bm=bmesh.new();bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-6)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bad=sum(not e.is_manifold for e in bm.edges)
    if bad:raise ValueError(f'{obj.name}: {bad} nonmanifold edges after closure')
    bm.to_mesh(obj.data);bm.free();obj.data.update()
    obj.select_set(False)


def export_asset(name,sources,scale,cell,stone=False,origin=None):
    bpy.ops.object.select_all(action='DESELECT')
    objects=[]
    col=bpy.data.collections.new(name);bpy.context.scene.collection.children.link(col)
    for i,source in enumerate(sources):
        if isinstance(source,bpy.types.Object):
            deps=bpy.context.evaluated_depsgraph_get()
            mesh=bpy.data.meshes.new_from_object(source.evaluated_get(deps))
            source_name=source.name
        else:mesh=source.copy();source_name=source.name
        obj=bpy.data.objects.new(name+'/'+str(i),mesh);col.objects.link(obj)
        for v in mesh.vertices:
            if origin is not None:v.co-=Vector(origin)
            v.co*=scale
        indices=[face.material_index for face in mesh.polygons]
        materials=[flat_material(m) for m in mesh.materials]
        mesh.materials.clear()
        for m in materials:mesh.materials.append(m)
        for face,index in zip(mesh.polygons,indices):face.material_index=index
        if stone:
            mesh.materials.clear()
            for m in stone_materials:mesh.materials.append(m)
            mesh.update()
            for face in mesh.polygons:
                q=face.center
                n=noise.noise_vector(q*3.7)[0]*.5+.5
                up=face.normal.z
                face.material_index=min(4,int(n*5))+(5 if up+n>.65 else 0)
        clean(obj,cell*1.25)
        objects.append(obj);prototypes.append(obj)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:obj.select_set(True)
    bpy.context.view_layer.objects.active=objects[0]
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')),export_format='GLB',use_selection=True,
        export_apply=True,export_yup=True,export_normals=True,export_materials='EXPORT')
    assets.append({'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','cellSize':cell,'bandCells':1,
                   'sourceCollection':name,'sourceNames':[s.name for s in sources],
                   'runtimeScale':scale,'botanicalEnvelopeMetres':cell*1.25 if not stone else 0})
    print('EXPORTED',name,len(objects),flush=True)
    return name


def add_instance(source,asset,position=None,orientation=None):
    p=source.location if position is None else Vector(position)
    if abs(p.x)>40 or p.y < -24 or p.y >45:return
    q=source.rotation_euler.to_quaternion() if orientation is None else orientation
    instances.append({'id':'forest/'+str(len(instances)),'asset':asset,
        'position':[p.x,p.z+3,-p.y],'yaw':source.rotation_euler.z,
        'orientation':[q.w,q.x,q.z,-q.y],'sourceName':source.name})


# Mature spreading hardwoods, pale multiple-stem alders, and tiered evergreen
# cedars use distinct authored trunks and crown architectures at discrete sizes.
seeds=[340,411,482]
tree_sizes=[.95,.65,.23]
tree_cells=[.08,.06,.025]
tree_assets=[]
source_leaves=list(original_meshes['Individual folded canopy leaves 340'].materials)
source_bark=original_meshes['Tree branching architecture 340'].materials[0]


def varied_tree(seed,species):
    mats=source_leaves
    if species!='broadleaf':
        base=(.065,.16,.028) if species=='alder' else (.025,.075,.045)
        mats=[_material('Runtime '+species+' foliage '+str(i),tuple(c*(.65+.16*i) for c in base),.62,True) for i in range(6)]
    wood,leaves,count=tree_species(seed,species,mats,source_bark)
    colors=([(c*.9,c*.88,c*.75) for c in [.19,.25,.31,.38,.46,.52]]+[(.035,.04,.03),(.07,.075,.06)]) if species=='alder' else [
        (.026,.016,.009),(.042,.026,.014),(.065,.04,.023),(.09,.058,.033),
        (.12,.075,.042),(.15,.095,.055),(.045,.06,.024),(.075,.09,.035)]
    wood.materials.clear()
    for i,color in enumerate(colors):wood.materials.append(_material('Trunk '+species+' grain '+str(i),color,.91))
    for face in wood.polygons:
        p=face.center
        grain=noise.noise_vector(Vector((p.x*7,p.y*7,p.z*.8)))[0]*.5+.5
        index=min(5,int(grain*6))
        if species=='alder':
            # Irregular horizontal lenticels, carried by real exported face
            # materials rather than a Blender-only procedural shader.
            marks=math.sin(p.z*18+math.sin(p.x*9+p.y*7)*1.7)
            if marks>.82 and grain<.55:index=6 if grain<.35 else 7
        elif face.normal.z>.45 and p.z<2.5 and grain>.45:index=6+(grain>.7)
        face.material_index=index
    return wood,leaves,count


for i,(seed,size,h) in enumerate(zip(seeds,tree_sizes,tree_cells)):
    wood,leaf,count=varied_tree(seed,'broadleaf')
    tree_assets.append(export_asset('forest-tree-'+str(i),[wood,leaf],size,h))
    assets[-1].update(species='Spreading buttressed broadleaf',leafCount=count)
species_assets={}
for species,seed in [('alder',1361),('cedar',1742)]:
    wood,leaf,count=varied_tree(seed,species)
    for i,(size,h) in enumerate(zip(tree_sizes[:2],tree_cells[:2])):
        name=export_asset('forest-'+species+'-'+str(i),[wood,leaf],size,h)
        species_assets[(species,i)]=name
        assets[-1].update(species='Pale multiple-stem alder' if species=='alder' else 'Tiered evergreen cedar',leafCount=count)
for obj in original:
    if obj.type=='MESH' and obj.data.name.startswith('Tree branching architecture'):
        scale=sum(obj.scale)/3
        index=min(range(3),key=lambda i:abs(math.log(scale/tree_sizes[i])))
        choice=int(hashlib.sha256(obj.name.encode()).hexdigest()[:8],16)%10
        species='alder' if choice<3 else 'cedar' if choice<5 else 'broadleaf'
        asset=species_assets[(species,index)] if index<2 and species!='broadleaf' else tree_assets[index]
        add_instance(obj,asset)

if TREES_ONLY:
    # Keep every non-tree asset and instance byte-for-byte in the manifest;
    # only the tree references change, with their stable IDs and placements.
    replacement={entry['sourceName']:entry['asset'] for entry in instances}
    instances=previous_manifest['instances']
    for entry in instances:
        if entry['sourceName'] in replacement:entry['asset']=replacement[entry['sourceName']]
    previous_manifest['assets']=[entry for entry in previous_manifest['assets'] if not entry['id'].startswith(('forest-tree-','forest-alder-','forest-cedar-'))]+assets
    previous_manifest['treeSpecies']=['buttressed broadleaf','multiple-stem alder','tiered evergreen cedar']
    previous_manifest['instances']=instances
    for obj in original:bpy.data.objects.remove(obj,do_unlink=True)
    # Load only existing non-tree runtime collections into the curated source.
    with bpy.data.libraries.load(str(SOURCE/'forest-waterfall-runtime.blend'),link=False) as (old,new):
        new.collections=[name for name in old.collections if name.startswith('forest-') and not name.startswith(('forest-tree-','forest-alder-','forest-cedar-'))]
    for col in new.collections:
        if col:bpy.context.scene.collection.children.link(col)
    bpy.context.scene.world=None
    bpy.context.scene['runtime_scope']='Static forest prototypes only; water supplied by app narrow-band FLIP.'
    bpy.context.scene['tree_species']='Buttressed broadleaf, multiple-stem alder, tiered evergreen cedar; real leaf and branch geometry.'
    updated_source=SOURCE/'forest-waterfall-runtime-updated.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(updated_source))
    updated_source.replace(SOURCE/'forest-waterfall-runtime.blend')
    (OUT/'manifest.json').write_text(json.dumps(previous_manifest,indent=2)+'\n')
    (ROOT/'lib/core/imported-assets/forest-waterfall-instances.json').write_text(json.dumps(instances,indent=2)+'\n')
    print('FINISHED TREE UPDATE',len(assets),'tree assets',len(instances),'stable instances',flush=True)
    raise SystemExit(0)

# The source fern has fine secondary lobes; the finite pitch preserves the
# larger pinnae while progressively omitting sub-cell tips, without blob proxies.
fern_assets=[]
for i,(size,h) in enumerate([(.65,.025),(1,.035),(1.3,.045)]):
    fern_assets.append(export_asset('forest-fern-'+str(i),[original_meshes['Pinnate fern with finely divided fronds '+str(700+i)]],size,h))
for obj in original:
    if obj.type=='MESH' and obj.data.name.startswith('Pinnate fern with finely divided fronds'):
        idx=min(range(3),key=lambda i:abs(obj.scale.x-[.65,1,1.3][i]))
        add_instance(obj,fern_assets[idx])

broad_assets=[]
for i,size in enumerate([.85,1.6]):
    broad_assets.append(export_asset('forest-broadleaf-'+str(i),[original_meshes['Glossy woodland broadleaf rosette '+str(810+i)]],size,.025 if i==0 else .04))
for obj in original:
    if obj.type=='MESH' and obj.data.name.startswith('Glossy woodland broadleaf rosette'):
        add_instance(obj,broad_assets[0 if obj.scale.x<1.2 else 1])

sedge_asset=export_asset('forest-sedge',[original_meshes['Riparian sedge arching blades 530']],1,.025)
ivy_asset=export_asset('forest-ivy',[original_meshes['Trailing cliff ivy 970']],1,.035)
for obj in original:
    if obj.type=='MESH' and obj.data.name.startswith('Riparian sedge arching blades'):add_instance(obj,sedge_asset)
    if obj.type=='MESH' and obj.data.name.startswith('Trailing cliff ivy'):add_instance(obj,ivy_asset)

# Reuse real irregular rock meshes at four authored sizes, preserving placement
# and full rotations. Large ledges keep exact source dimensions for the falls.
rock_objects=[o for o in original if o.type=='MESH' and any(o.name.startswith(s) for s in ['Shore boulder','Hero waterside moss stone','Cliff buttress','Rounded submerged river stone'])]
rock_sizes=[.45,1.25,2.8,4.8]
rock_assets=[]
for i,size in enumerate(rock_sizes):
    src=min(rock_objects,key=lambda o:abs(max(o.dimensions)-size))
    norm=size/max(src.dimensions)
    rock_assets.append(export_asset('forest-rock-'+str(i),[src],norm,.04 if i<2 else .08,True))
for obj in rock_objects:
    idx=min(range(4),key=lambda i:abs(math.log(max(obj.dimensions)/rock_sizes[i])))
    add_instance(obj,rock_assets[idx])

ledge_objects=[o for o in original if o.name.startswith('Cascade ledge')]
for i,obj in enumerate(ledge_objects):
    asset=export_asset('forest-cascade-ledge-'+str(i),[obj],1,.08,True)
    add_instance(obj,asset)

# Main fallen trunk is one reusable curved solid. Fine litter/root decoration
# remains source-only at this runtime LOD, instead of thousands of unique fields.
log=next(o for o in original if o.name.startswith('Old fallen tree'))
log_origin=(-12,-2,1.3)
log_asset=export_asset('forest-fallen-log',[log],1,.06,False,log_origin)
add_instance(log,log_asset,log_origin,Quaternion())

# Delete every original object, including ALL authored water/mist/foam, from the
# separate runtime library source. Only static, origin-space prototypes remain.
for obj in original:bpy.data.objects.remove(obj,do_unlink=True)
for col in list(bpy.data.collections):
    if not col.objects and not col.children:bpy.data.collections.remove(col)
bpy.context.scene.world=None
bpy.context.scene['runtime_scope']='Static authored forest prototypes; all water is supplied by the app narrow-band FLIP solver.'
bpy.context.scene['finite_lod']='Discrete tree/plant sizes; closed botanical envelopes at native 25–80mm sampling pitch.'
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'forest-waterfall-runtime.blend'))
manifest={'version':1,'collection':{'id':'forest-waterfall','label':'Forest waterfall'},
          'source':'forest-waterfall-runtime.blend','assets':assets,'instances':instances,
          'runtimeScope':'Static environment only; no waterfall/pond/foam geometry. Native app FLIP water.',
          'finiteLod':'Actual mesh-derived botanical envelopes, discrete size variants; source render geometry retained separately.',
          'excludedSourceCategories':['water','foam','droplets','mist','lights','cameras','continuous terrain','fine exposed roots'],
          'terrainPolicy':'Native scene collision heightfield and terrain-shell; imported rock/plant surfaces complement it.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'lib/core/imported-assets/forest-waterfall-instances.json').write_text(json.dumps(instances,indent=2)+'\n')
print('FINISHED',len(assets),'assets',len(instances),'instances',flush=True)
