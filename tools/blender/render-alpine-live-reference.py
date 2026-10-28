"""Cycles diagnostic of the exact interactive lake asset layout and camera.
Uses the same GLBs, physical terrain and real NB-FLIP contour. The continuous
terrain material is a neutral proxy for the live procedural terrain shader.
No extra trees, photographic mountain backdrop, mist or authored water added.
"""
import argparse,hashlib,json,math,sys
from pathlib import Path
import bpy,bmesh
from mathutils import Vector,Quaternion,Matrix
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--layout',default='artifacts/water-scenes/alpine-lake-sunrise/live-layout.json')
p.add_argument('--water-mesh',required=True)
p.add_argument('--output',default='artifacts/water-scenes/alpine-lake-sunrise/cycles-live-reference.png')
p.add_argument('--sun-angle',type=float,default=.065,help='Cycles sun angular diameter in radians; diagnostic live default .065, physical .53 degree sun .009250245')
p.add_argument('--width',type=int,default=1200);p.add_argument('--height',type=int,default=800);p.add_argument('--samples',type=int,default=64)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if not math.isfinite(a.sun_angle) or not 0<a.sun_angle<math.pi:p.error('--sun-angle must be a finite angular diameter between zero and pi radians')
layout=json.loads(Path(a.layout).read_text());out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version=0
scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene['source_scene_hash']=layout['sceneHash'];scene['water_method']='Recorded production narrow-band FLIP contour; see render provenance sidecar'
water_path=Path(a.water_mesh).resolve();water_report=water_path.with_suffix('.json')
provenance={'renderer':'Blender Cycles','layoutSource':str(Path(a.layout).resolve()),'layoutSceneHash':layout['sceneHash'],
    'waterMeshSource':str(water_path),'waterMeshSha256':hashlib.sha256(water_path.read_bytes()).hexdigest(),
    'waterExportProvenance':json.loads(water_report.read_text()) if water_report.exists() else None,
    'assetMeshes':{i['asset']:{'path':i['file'],'sha256':hashlib.sha256(Path(i['file']).read_bytes()).hexdigest()} for i in {v['asset']:v for v in layout['instances']}.values()},
    'sunAngularDiameter_rad':a.sun_angle,
    'notes':['Static source GLB layout and camera match the exported live scene; interactive rendering uses FVX isosurfaces.',
             'Water is the supplied recorded NB-FLIP run, not freshly simulated here; it may predate current solver edits.',
             'Terrain material and sky use the documented Cycles diagnostic approximations.',
             f'Cycles sun angular diameter is {a.sun_angle} radians; the live default shadow aperture is .065 radians. Matching the numeric angle does not imply identical shadow filtering.']}
out.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
scene['render_provenance_file']=str(out.with_suffix('.json'));scene['water_mesh_sha256']=provenance['waterMeshSha256']
point=lambda q:Vector((q['x'],-q['z'],q['y']))

# Full source GLBs (live rendering uses FVX isosurfaces, not GLB triangles).
# Instance origin placement exactly inverts FVX centering.
prototypes={}
for inst in layout['instances']:
    asset=inst['asset']
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
    q=inst['orientation'];rotation=Quaternion((q['w'],q['x'],-q['z'],q['y']))
    obj.matrix_world=Matrix.Translation(point(inst['origin']))@rotation.to_matrix().to_4x4()

# Exact sampled terrain coordinates, with a neutral mineral ground shader.
g=layout['terrain']['grid'];nx=g['size']['nx'];nz=g['size']['nz'];h=g['spacing_m'];origin=g['origin_m']
verts=[(origin['x']+(i%nx)*h,-(origin['z']+(i//nx)*h),height) for i,height in enumerate(g['heights_m'])]
faces=[]
for z in range(nz-1):
    for x in range(nx-1):
        i=z*nx+x;faces.append((i,i+nx,i+nx+1,i+1))
mesh=bpy.data.meshes.new('Exact live terrain');mesh.from_pydata(verts,[],faces);mesh.update();obj=bpy.data.objects.new('Exact live terrain',mesh);scene.collection.objects.link(obj)
mat=bpy.data.materials.new('Neutral terrain reference');mat.use_nodes=True;bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.14,.12,.09,1);bs.inputs['Roughness'].default_value=.85;mesh.materials.append(mat)
for face in mesh.polygons:face.use_smooth=True

# No ocean modifier, re-meshing or visual replacement of the fluid.
bpy.ops.wm.obj_import(filepath=str(Path(a.water_mesh).resolve()),forward_axis='Y',up_axis='Z')
water=bpy.data.materials.new('Live water optics');water.use_nodes=True;n,l=water.node_tree.nodes,water.node_tree.links;bs=n.get('Principled BSDF');optics=layout['optics']
bs.inputs['Base Color'].default_value=(1,1,1,1);bs.inputs['IOR'].default_value=optics['indexOfRefraction'];bs.inputs['Roughness'].default_value=optics['roughness'];bs.inputs['Transmission Weight'].default_value=1
absorb=n.new('ShaderNodeVolumeAbsorption');absorb.inputs['Color'].default_value=(*[max(0,1-v) for v in optics['absorption_mInv']],1);absorb.inputs['Density'].default_value=1;l.new(absorb.outputs[0],n.get('Material Output').inputs['Volume'])
for obj in list(bpy.context.selected_objects):
    obj.name='Live NB-FLIP surface';obj.data.materials.clear();obj.data.materials.append(water)
    bm=bmesh.new();bm.from_mesh(obj.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5);bm.normal_update();bm.to_mesh(obj.data);bm.free()
    for face in obj.data.polygons:face.use_smooth=True

lighting=layout['lighting'];environment=lighting['environment'];world=bpy.data.worlds.new('Live authored sky palette');world.use_nodes=True;scene.world=world
n,l=world.node_tree.nodes,world.node_tree.links;coords=n.new('ShaderNodeTexCoord');sep=n.new('ShaderNodeSeparateXYZ');l.new(coords.outputs['Normal'],sep.inputs[0]);remap=n.new('ShaderNodeMapRange');remap.inputs['From Min'].default_value=-1;remap.inputs['From Max'].default_value=1;l.new(sep.outputs['Z'],remap.inputs[0]);mix=n.new('ShaderNodeMixRGB');mix.inputs[1].default_value=(*environment['lowerRadianceLinear'],1);mix.inputs[2].default_value=(*environment['upperRadianceLinear'],1);l.new(remap.outputs[0],mix.inputs[0]);l.new(mix.outputs[0],n.get('Background').inputs['Color']);n.get('Background').inputs['Strength'].default_value=environment['diffuseScale']
key=lighting['directional'];d=key['direction'];sun_direction=Vector((d[0],-d[2],d[1])).normalized();bpy.ops.object.light_add(type='SUN');sun=bpy.context.object;sun.data.color=key['colorLinear'];sun.data.energy=key['intensity'];sun.data.angle=a.sun_angle;sun.rotation_euler=(-sun_direction).to_track_quat('-Z','Y').to_euler()
camera=layout['camera'];bpy.ops.object.camera_add(location=point(camera['position_m']));cam=bpy.context.object;scene.camera=cam;cam.rotation_euler=(point(camera['target_m'])-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.sensor_fit='VERTICAL';cam.data.sensor_height=24;cam.data.lens=12/camera['tanHalfFov'];cam.data.clip_end=10000
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=a.samples;scene.cycles.use_denoising=True;scene.cycles.max_bounces=12;scene.cycles.transmission_bounces=8
scene.render.resolution_x=a.width;scene.render.resolution_y=a.height;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.filepath=str(out);scene.view_settings.view_transform='AgX';scene.view_settings.exposure=math.log2(lighting['grade']['exposure'])
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')));bpy.ops.render.render(write_still=True)
