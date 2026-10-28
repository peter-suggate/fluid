"""Author a bounded field of closed glacial cobbles along the existing dry bank.
The same GLB is the Cycles source and input to the common FVX baker. No liquid
or collision-bed geometry is authored here. Origins use (0,0,-36) in app metres.
"""
import argparse,json,math,random,sys
from pathlib import Path
import bpy,bmesh
from mathutils import Vector,Quaternion,noise
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,default=ROOT/'public/assets/alpine-lake')
p.add_argument('--source-dir',type=Path,default=ROOT/'assets/alpine-lake')
p.add_argument('--count',type=int,default=1200)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
a.output_dir.mkdir(parents=True,exist_ok=True);a.source_dir.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version=0
rng=random.Random(712813)
# Read the actual exported height samples rather than approximating their
# bilinear interpolation independently. This script follows the current bed.
layout=json.loads((ROOT/'artifacts/water-scenes/alpine-lake-sunrise/live-layout.json').read_text())
g=layout['terrain']['grid'];nx=g['size']['nx'];nz=g['size']['nz'];h=g['spacing_m'];origin=g['origin_m'];heights=g['heights_m']
def bed(x,z):
    gx=max(0,min(nx-1,(x-origin['x'])/h));gz=max(0,min(nz-1,(z-origin['z'])/h))
    ix=min(nx-2,math.floor(gx));iz=min(nz-2,math.floor(gz));fx=gx-ix;fz=gz-iz
    return sum(heights[(iz+dz)*nx+ix+dx]*(fx if dx else 1-fx)*(fz if dz else 1-fz) for dz in range(2) for dx in range(2))
mat=bpy.data.materials.new('Photographed neutral mineral cobbles');mat.use_nodes=True
nodes,links=mat.node_tree.nodes,mat.node_tree.links;bs=nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.83
tex=nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(ROOT/'assets/alpine-lake/alpine-strata-neutral.png'));tex.image.colorspace_settings.name='sRGB';links.new(tex.outputs['Color'],bs.inputs['Base Color'])
# Blue-noise packing: no rows, no intersecting component shells. Broad, uneven
# deposit density leaves soil visible between stones and around their margins.
positions=[];bins={};cell=1.1
for attempt in range(90000):
    if len(positions)>=a.count:break
    x=rng.uniform(-39,39);z=rng.uniform(-41,-33)
    density=.28+.72*(.5+.5*math.sin(x*.24+math.sin(z*.9)))
    if rng.random()>density or bed(x,z)<2.42:continue
    r=rng.uniform(.21,.48);key=(math.floor(x/cell),math.floor(z/cell))
    nearby=[s for dx in (-1,0,1) for dz in (-1,0,1) for s in bins.get((key[0]+dx,key[1]+dz),[])]
    if any(math.hypot(x-px,z-pz)<r+pr+.025 for px,pz,pr in nearby):continue
    positions.append((x,z,r));bins.setdefault(key,[]).append((x,z,r))
winding_corrections=0
for i,(x,z,r) in enumerate(positions):
    bm=bmesh.new();bmesh.ops.create_icosphere(bm,subdivisions=2,radius=1)
    rz=r*rng.uniform(.36,.65);rx=r*rng.uniform(.76,.94);ry=r*rng.uniform(.68,.94)
    angle=rng.uniform(0,math.tau);cs,ss=math.cos(angle),math.sin(angle)
    for v in bm.verts:
        p=v.co.copy();variation=1+.1*noise.noise(Vector((p.x*2+i*.17,p.y*2,p.z*2)))
        px,py=p.x*rx*variation,p.y*ry*variation
        v.co=(x+cs*px-ss*py,-z-36+ss*px+cs*py,bed(x,z)+rz*.27+p.z*rz*variation)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    # Recalculation alone can choose inward orientation for translated closed
    # components. Compute volume relative to the component centroid, avoiding
    # cancellation against the world origin, and explicitly require outward.
    center=sum((v.co for v in bm.verts),Vector())/len(bm.verts)
    volume6=sum((f.verts[0].co-center).dot((f.verts[1].co-center).cross(f.verts[2].co-center)) for f in bm.faces)
    if volume6<0:
        bmesh.ops.reverse_faces(bm,faces=list(bm.faces));winding_corrections+=1
    assert abs(volume6)>1e-7
    bm.normal_update()
    mesh=bpy.data.meshes.new(f'Closed mineral cobble {i}');bm.to_mesh(mesh);bm.free();mesh.materials.append(mat)
    uv=mesh.uv_layers.new(name='Photographic mineral sample');u0,v0=rng.random(),rng.random()
    for face in mesh.polygons:
        face.use_smooth=True
        for li in face.loop_indices:
            co=mesh.vertices[mesh.loops[li].vertex_index].co;uv.data[li].uv=(u0+co.x*.2,v0+co.y*.2)
    obj=bpy.data.objects.new(f'Cobbles/{i:04d}',mesh);bpy.context.scene.collection.objects.link(obj)
    obj['ground_height_m']=bed(x,z);obj['source_radius_m']=r
bpy.ops.object.select_all(action='SELECT')
name='alpine-shore-scree'
bpy.ops.export_scene.gltf(filepath=str(a.output_dir/(name+'.glb')),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_normals=True,export_materials='EXPORT')
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(a.source_dir/'alpine-shore-scree.blend'))
manifest=a.output_dir/'manifest.json';data=json.loads(manifest.read_text()) if manifest.exists() else {'collection':{'id':'alpine-lake','label':'Alpine Lake'},'assets':[]}
entry={'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','cellSize':.1,'bandCells':1}
data['assets']=[v for v in data['assets'] if v['id']!=name]+[entry];manifest.write_text(json.dumps(data,indent=2)+'\n')
(a.source_dir/'alpine-shore-provenance.json').write_text(json.dumps({'asset':name,'sourceSceneHash':layout['sceneHash'],'origin_m':[0,0,-36],'cellSize_m':.1,'stoneCount':len(positions),'orientation':'positive centroid-relative signed volume','windingCorrections':winding_corrections,'minimumGroundHeight_m':min(bed(x,z) for x,z,_ in positions),'waterline_m':2.3,'positions_x_z_radius_m':positions},indent=2)+'\n')
print('ALPINE_SHORE_EXPORTED',len(positions),'closed cobbles; corrected outward winding',winding_corrections,flush=True)
