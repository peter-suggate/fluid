"""Matched source-geometry Cycles diagnostic of a registered live voxel scene.
See the JSON sidecar for lighting and reconstruction differences.
"""
import argparse,hashlib,json,math,sys,struct
from pathlib import Path
import bpy,bmesh
from mathutils import Vector,Quaternion,Matrix
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--layout',required=True)
p.add_argument('--water-mesh',required=True)
p.add_argument('--output',required=True)
p.add_argument('--sun-angle',type=float,default=.065,help='Cycles sun angular diameter in radians; diagnostic live default .065, physical .53 degree sun .009250245')
p.add_argument('--width',type=int,default=800);p.add_argument('--height',type=int,default=560);p.add_argument('--samples',type=int,default=64)
p.add_argument('--no-render',action='store_true')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if not math.isfinite(a.sun_angle) or not 0<a.sun_angle<math.pi:p.error('--sun-angle must be a finite angular diameter between zero and pi radians')
layout=json.loads(Path(a.layout).read_text());out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version=0
scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene['source_scene_hash']=layout['sceneHash'];scene['water_method']='Recorded production narrow-band FLIP contour; see render provenance sidecar'
water_path=Path(a.water_mesh).resolve();water_report=Path(str(water_path)+'.json')
if not water_report.exists():water_report=water_path.with_suffix('.json')
provenance={'renderer':'Blender Cycles','rendererSourceSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'renderSettings':{'width':a.width,'height':a.height,'samples':a.samples,'device':'CPU'},'layoutSource':str(Path(a.layout).resolve()),'layoutSceneHash':layout['sceneHash'],
    'waterMeshSource':str(water_path),'waterMeshSha256':hashlib.sha256(water_path.read_bytes()).hexdigest(),
    'waterExportProvenance':json.loads(water_report.read_text()) if water_report.exists() else None,
    'assetMeshes':{i['asset']:{'path':i['file'],'sha256':hashlib.sha256(Path(i['file']).read_bytes()).hexdigest()} for i in {v['asset']:v for v in layout['instances']}.values()},
    'sunAngularDiameter_rad':a.sun_angle,
    'notes':['Static source GLB layout and camera match the exported live scene; interactive rendering uses FVX isosurfaces.',
             'Water is the supplied recorded NB-FLIP run, not freshly simulated here; it may predate current solver edits.',
             'Terrain uses published base color and roughness, without procedural modulation; sky uses the documented diagnostic approximation.',
             f'Cycles sun angular diameter is {a.sun_angle} radians; the live default shadow aperture is .065 radians. Matching the numeric angle does not imply identical shadow filtering.']}
water_provenance=provenance['waterExportProvenance'] or {}
if water_provenance.get('objSha256') and water_provenance['objSha256']!=provenance['waterMeshSha256']:raise ValueError('Water OBJ differs from its recorded export hash')
capture_path=water_provenance.get('captureReport')
if capture_path and Path(capture_path).exists():
    capture=json.loads(Path(capture_path).read_text());sentinels=capture.get('gpuExecutionSentinels',[])
    export_checks=[entry for entry in sentinels if entry.get('label')=='export-rendered-water-mesh']
    if not export_checks or any(entry.get('actual')!=entry.get('expected') or not entry.get('actual') for entry in export_checks):raise ValueError('Captured water export lacks a valid execution sentinel')
    provenance['captureValidation']={'snapshotExportValidated':True,'wholeCaptureSuccessful':capture.get('successful'),
        'executionSentinels':sentinels,'laterFailure':capture.get('failure')}
    if capture.get('successful') is False:provenance['notes'].append('The exported snapshot passed its execution sentinel before a later capture/stress failure. This reference does not validate the whole capture run or its performance measurements.')
out.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
scene['render_provenance_file']=str(out.with_suffix('.json'));scene['water_mesh_sha256']=provenance['waterMeshSha256']
point=lambda q:Vector((q['x'],-q['z'],q['y']))
vec=lambda q:Vector((q[0],-q[2],q[1]))
objects_by_key={}
explicit_light_keys={record["sourceKey"] for record in layout["lights"]}
def rotation(q):return Quaternion((q['w'],q['x'],-q['z'],q['y']))
def emission_for_visible_paths(mat,strength,replace_diffuse=False):
    # Live direct light records already provide this emitter's illumination.
    # Retain surface emission for camera/glossy/transmission without adding a
    # second diffuse light source in the offline comparison.
    if strength<=0:return
    n,l=mat.node_tree.nodes,mat.node_tree.links;bs=n.get('Principled BSDF')
    if bs is None:return
    base=bs.inputs['Base Color'];em=bs.inputs['Emission Color']
    if base.is_linked:l.new(base.links[0].from_socket,em)
    else:em.default_value=base.default_value
    if not replace_diffuse:
        bs.inputs['Emission Strength'].default_value=strength
        return
    path=n.new('ShaderNodeLightPath');invert=n.new('ShaderNodeMath');invert.operation='SUBTRACT';invert.inputs[0].default_value=1;l.new(path.outputs['Is Diffuse Ray'],invert.inputs[1])
    scale=n.new('ShaderNodeMath');scale.operation='MULTIPLY';scale.inputs[1].default_value=strength;l.new(invert.outputs[0],scale.inputs[0]);l.new(scale.outputs[0],bs.inputs['Emission Strength'])


# Full source GLBs (live rendering uses FVX isosurfaces, not GLB triangles).
# Instance origin placement exactly inverts FVX centering.
prototypes={}
for inst in layout['instances']:
    asset=inst['asset']
    if hashlib.sha256(Path(inst['file']).read_bytes()).hexdigest()!=inst['sha256']:raise ValueError('Source asset changed since layout export: '+asset)
    if asset in prototypes:continue
    bpy.ops.object.select_all(action='DESELECT');bpy.ops.import_scene.gltf(filepath=inst['file'])
    imported=list(bpy.context.selected_objects);parts=[o for o in imported if o.type=='MESH'];empties=[o for o in imported if o.type!='MESH']
    for o in imported:o.select_set(o.type=='MESH')
    bpy.context.view_layer.objects.active=parts[0]
    if len(parts)>1:bpy.ops.object.join()
    obj=bpy.context.object
    obj.data.transform(obj.matrix_world);obj.data.update();prototypes[asset]=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
    for o in empties:
        if o.name in bpy.data.objects:bpy.data.objects.remove(o,do_unlink=True)
for inst in layout['instances']:
    obj=bpy.data.objects.new(inst['id'],prototypes[inst['asset']]);scene.collection.objects.link(obj)
    obj.matrix_world=Matrix.Translation(point(inst['origin']))@rotation(inst['orientation']).to_matrix().to_4x4()
    for key in inst['proxyKeys']:objects_by_key[key]=obj
    if inst['material']['emission']>0:
        obj.data=obj.data.copy()
        for slot in obj.material_slots:
            slot.material=slot.material.copy();emission_for_visible_paths(slot.material,inst['material']['emission'],any(key in explicit_light_keys for key in inst['proxyKeys']))

for primitive in layout['primitives']:
    kind=primitive['kind']
    if kind=='box':
        bpy.ops.mesh.primitive_cube_add(size=2);obj=bpy.context.object
        h=primitive['halfSize_m'];obj.scale=(h['x'],h['z'],h['y'])
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    elif kind=='cone':
        bpy.ops.mesh.primitive_cone_add(vertices=64,radius1=primitive['baseRadius_m'],radius2=primitive['topRadius_m'],depth=2*primitive['halfHeight_m']);obj=bpy.context.object
    else:raise ValueError('Unsupported primitive '+kind)
    obj.name=primitive['key'];obj.location=point(primitive['center_m']);obj.rotation_mode='QUATERNION';obj.rotation_quaternion=rotation(primitive.get('orientation',dict(w=1,x=0,y=0,z=0)))
    objects_by_key[primitive['key']]=obj
    mat=bpy.data.materials.new(obj.name);mat.use_nodes=True;bs=mat.node_tree.nodes.get('Principled BSDF');m=primitive['material'];bs.inputs['Base Color'].default_value=(*m['colorLinear'],1);bs.inputs['Roughness'].default_value=m['roughness'];emission_for_visible_paths(mat,m['emission'],primitive['key'] in explicit_light_keys);obj.data.materials.append(mat)


# Exact sampled terrain coordinates, with a neutral mineral ground shader.
g=layout['terrain']['grid'];nx=g['size']['nx'];nz=g['size']['nz'];h=g['spacing_m'];origin=g['origin_m']
verts=[(origin['x']+(i%nx)*h,-(origin['z']+(i//nx)*h),height) for i,height in enumerate(g['heights_m'])]
faces=[]
for z in range(nz-1):
    for x in range(nx-1):
        i=z*nx+x;faces.append((i,i+nx,i+nx+1,i+1))
mesh=bpy.data.meshes.new('Exact live terrain');mesh.from_pydata(verts,[],faces);mesh.update();obj=bpy.data.objects.new('Exact live terrain',mesh);scene.collection.objects.link(obj)
mat=bpy.data.materials.new('Neutral terrain reference');mat.use_nodes=True;bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(*layout['terrainMaterial']['baseColorLinear'],1);bs.inputs['Roughness'].default_value=layout['terrainMaterial']['roughness'];mesh.materials.append(mat)
for face in mesh.polygons:face.use_smooth=True

# Keep OBJ optical normals when supplied. Welding/recomputing would silently
# replace production phi gradients with geometric triangle averages.
water_positions=[];water_normals=[];water_corners=[]
for line in water_path.read_text().splitlines():
    fields=line.split()
    if not fields:continue
    if fields[0]=='v':water_positions.append(tuple(float(v) for v in fields[1:4]))
    elif fields[0]=='vn':water_normals.append(tuple(float(v) for v in fields[1:4]))
    elif fields[0]=='f':water_corners.extend(fields[1:])
expected_normals={}
if water_normals:
    for token in water_corners:
        fields=token.split('/')
        if len(fields)!=3 or not fields[2]:raise ValueError('Incomplete supplied water optical normals')
        position=water_positions[int(fields[0])-1];normal=water_normals[int(fields[2])-1]
        position=struct.unpack('fff',struct.pack('fff',*position))
        expected_normals.setdefault(position,[]).append(Vector(normal).normalized())
bpy.ops.wm.obj_import(filepath=str(water_path),forward_axis='Y',up_axis='Z')
water=bpy.data.materials.new('Live water optics');water.use_nodes=True;n,l=water.node_tree.nodes,water.node_tree.links;bs=n.get('Principled BSDF');optics=layout['optics']
bs.inputs['Base Color'].default_value=(1,1,1,1);bs.inputs['IOR'].default_value=optics['indexOfRefraction'];bs.inputs['Roughness'].default_value=optics['roughness'];bs.inputs['Transmission Weight'].default_value=1
absorb=n.new('ShaderNodeVolumeAbsorption');absorb.inputs['Color'].default_value=(*[max(0,1-v) for v in optics['absorption_mInv']],1);absorb.inputs['Density'].default_value=1;l.new(absorb.outputs[0],n.get('Material Output').inputs['Volume'])
normal_error=0.;attribute_error=0.;validated_corners=0
for obj in list(bpy.context.selected_objects):
    obj.name='Live NB-FLIP surface';obj.data.materials.clear();obj.data.materials.append(water)
    if water_normals:
        if not obj.data.has_custom_normals:raise ValueError('OBJ importer discarded production optical normals')
        # OBJ import has already published custom loop normals. Do not change
        # topology, recompute normals or toggle polygon smoothing afterward.
        for loop,corner_normal in zip(obj.data.loops,obj.data.corner_normals):
            coordinate=tuple(obj.data.vertices[loop.vertex_index].co)
            candidates=expected_normals.get(coordinate)
            if candidates is None:raise ValueError('OBJ import changed a production water position')
            error=min((corner_normal.vector-source).length for source in candidates)
            normal_error=max(normal_error,error);validated_corners+=1
        # Blender internally compresses custom split normals, with larger error
        # on degenerate/sliver faces. A float corner attribute preserves the
        # actual solver vectors for the optical BSDF, independently of that
        # mesh-normal encoding. The surface positions/winding remain untouched.
        values=[]
        for token,loop in zip(water_corners,obj.data.loops):
            fields=token.split('/');source_position=water_positions[int(fields[0])-1]
            source_position=struct.unpack('fff',struct.pack('fff',*source_position))
            if tuple(obj.data.vertices[loop.vertex_index].co)!=source_position:raise ValueError('OBJ importer reordered water face corners; cannot bind optical attribute')
            values.extend(Vector(water_normals[int(fields[2])-1]).normalized())
        attribute=obj.data.attributes.new(name='NBFLIP_optical_normal',type='FLOAT_VECTOR',domain='CORNER')
        attribute.data.foreach_set('vector',values)
        for i,entry in enumerate(attribute.data):attribute_error=max(attribute_error,(entry.vector-Vector(values[3*i:3*i+3])).length)
        if attribute_error>1e-6:raise ValueError('Optical normal attribute did not preserve float32 source')
        node=n.new('ShaderNodeAttribute');node.attribute_name=attribute.name
        normalize=n.new('ShaderNodeVectorMath');normalize.operation='NORMALIZE';l.new(node.outputs['Vector'],normalize.inputs[0]);l.new(normalize.outputs['Vector'],bs.inputs['Normal'])
    else:
        bm=bmesh.new();bm.from_mesh(obj.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5);bm.normal_update();bm.to_mesh(obj.data);bm.free()
        for face in obj.data.polygons:face.use_smooth=True
if water_normals and validated_corners!=len(water_corners):raise ValueError('OBJ import changed water corner count')
provenance['waterNormals']={'mode':'production-optical' if water_normals else 'legacy-averaged-triangle',
    'suppliedNormals':len(water_normals),'validatedCorners':validated_corners,'maximumImportedCustomNormalError':normal_error,'maximumOpticalAttributeError':attribute_error,
    'opticalStorage':'float32 corner attribute interpolated and normalized in BSDF' if water_normals else 'averaged geometric normals',
    'topologyWelded':not bool(water_normals)}

lighting=layout['lighting'];environment=lighting['environment'];world=bpy.data.worlds.new('Live authored sky palette');world.use_nodes=True;scene.world=world
n,l=world.node_tree.nodes,world.node_tree.links;coords=n.new('ShaderNodeTexCoord');sep=n.new('ShaderNodeSeparateXYZ');l.new(coords.outputs['Normal'],sep.inputs[0]);remap=n.new('ShaderNodeMapRange');remap.inputs['From Min'].default_value=-1;remap.inputs['From Max'].default_value=1;l.new(sep.outputs['Z'],remap.inputs[0]);mix=n.new('ShaderNodeMixRGB');mix.inputs[1].default_value=(*environment['lowerRadianceLinear'],1);mix.inputs[2].default_value=(*environment['upperRadianceLinear'],1);l.new(remap.outputs[0],mix.inputs[0]);l.new(mix.outputs[0],n.get('Background').inputs['Color']);n.get('Background').inputs['Strength'].default_value=environment['diffuseScale']
# Preserve published record count/order; approximate physical unit conversion.
light_reports=[]
for record in layout['lights']:
    kind=record['kind'];light_type={'directional':'SUN','point':'POINT','spot':'SPOT','rectangleArea':'AREA','sphereArea':'POINT'}[kind]
    bpy.ops.object.light_add(type=light_type,location=vec(record['position_m']));obj=bpy.context.object;obj.name='Published '+record['sourceKey'];data=obj.data
    data.color=record['colorLinear'];data.energy=record['intensity'] if kind=='directional' else 4*math.pi*record['intensity']
    if kind=='directional':data.angle=a.sun_angle;obj.rotation_euler=(-vec(record['direction'])).to_track_quat('-Z','Y').to_euler()
    else:
        if kind=='rectangleArea':
            data.shape='RECTANGLE';data.size=2*record['halfWidth_m'];data.size_y=2*record['halfHeight_m'];data.energy=math.pi*4*record['halfWidth_m']*record['halfHeight_m']*record['intensity']
        else:data.shadow_soft_size=record['radius_m']
        if kind=='sphereArea':data.energy*=4*math.pi*record['radius_m']**2
        if kind in ('rectangleArea','spot'):obj.rotation_euler=vec(record['direction']).to_track_quat('-Z','Y').to_euler()
        if kind=='spot':
            data.spot_size=2*math.acos(record['cone']['cosOuter']);data.spot_blend=1-math.acos(record['cone']['cosInner'])/math.acos(record['cone']['cosOuter'])
        if record['range_m']>0:raise ValueError('Finite authored range needs explicit reference falloff mapping')
    owner=objects_by_key.get(record['sourceKey'])
    if owner:
        collection=bpy.data.collections.new('Exclude emitter '+record['sourceKey']);collection.objects.link(owner)
        collection.collection_objects[0].light_linking.link_state='EXCLUDE'
        obj.light_linking.blocker_collection=collection
    light_reports.append(dict(sourceKey=record['sourceKey'],kind=kind,cyclesEnergy=data.energy,ownerShadowExcluded=bool(owner)))
provenance['lightConversion']=light_reports
provenance['notes'] += [
 'Published direct light records are explicit Cycles lights. For fixtures with explicit light records only, visible emitter radiance is suppressed on diffuse rays to avoid duplicate diffuse mesh illumination. Other emissive surfaces retain full emission; source emitter is excluded from its own light shadow linking as in the live owner contract.',
 'Point/spot conversion uses 4*pi*intensity to match far-field inverse-square irradiance. Live 1m attenuation cap and spot beam interpolation differ from Cycles; near-field illumination is not an exact reference.',
 'Cycles source GLB material images preserve image detail; live reconstructs voxel surfaces and samples mapped images. Live procedural primitive surface variations are not reproduced.',
 'No depth of field, volumetric haze, rain streaks, water bump or compositor bloom. AgX implementation and environment angular interpolation are not identical to the live renderer.'
]
out.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
camera=layout['camera'];bpy.ops.object.camera_add(location=point(camera['position_m']));cam=bpy.context.object;scene.camera=cam;cam.rotation_euler=(point(camera['target_m'])-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.sensor_fit='VERTICAL';cam.data.sensor_height=24;cam.data.lens=12/camera['tanHalfFov'];cam.data.clip_end=10000
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=a.samples;scene.cycles.use_denoising=True;scene.cycles.max_bounces=12;scene.cycles.transmission_bounces=8
scene.render.resolution_x=a.width;scene.render.resolution_y=a.height;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.filepath=str(out);scene.view_settings.view_transform='AgX';scene.view_settings.exposure=math.log2(lighting['grade']['exposure'])
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')));
if not a.no_render:bpy.ops.render.render(write_still=True)
