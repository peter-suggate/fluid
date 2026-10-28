"""Bake optional FVM1 textures/mapping from an existing GLB + FVX1 pair.

Run in Blender. This does not modify the FVX or source geometry. Only nodes
belonging to sign-changing FVX cells need mappings; other records remain zero.
"""
import argparse,gzip,hashlib,json,math,struct,sys,tempfile
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mesh_sampling import inside,barycentric,oct16,atomic_write
from normal_orientation import outward_shading_normal,signed_outward_gradient
from surface_mapping import affine_uv,rgba_mips,srgb_decode
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--asset',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--texture-size',type=int,default=2048);p.add_argument('--publish',action='store_true',help='Atomically publish companion and matching FVX header/metadata after baking')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if any(not v or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in v) for v in (a.collection,a.asset)):p.error('Simple collection/asset names required')
if a.texture_size<1 or a.texture_size>4096 or a.texture_size&(a.texture_size-1):p.error('Texture size must be a power of two <=4096')
out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
source=ROOT/'public/assets'/a.collection/(a.asset+'.glb');fvx=source.with_suffix('.fvx.gz')
blob=source.read_bytes();assert blob[:4]==b'glTF'
chunks={};at=12
while at<len(blob):
 n,kind=struct.unpack_from('<II',blob,at);chunks[kind]=blob[at+8:at+8+n];at+=8+n
model=json.loads(chunks[0x4e4f534a]);binary=chunks[0x004e4942]
raw=gzip.decompress(fvx.read_bytes());assert raw[:4]==b'FVX1';hn=struct.unpack_from('<I',raw,4)[0];h=json.loads(raw[8:8+hn]);payload=memoryview(raw)[8+hn:]
if h.get('solidRule','even-odd') != 'even-odd':
 p.error('FVM mapping currently requires even-odd FVX; component-union mapping orientation is not implemented')
record_dtype=np.dtype([('distance','<f4'),('gradient','<u4'),('shade','<u4'),('material','<u4')]);brick_bytes=12+729*16
needed=set();origins=[]
for bi in range(h['brickCount']):
 off=bi*brick_bytes;origin=struct.unpack_from('<3i',payload,off);origins.append(origin)
 distance=np.frombuffer(payload,record_dtype,729,off+12)['distance'].reshape(9,9,9)
 negative=distance<0;lo=np.ones((8,8,8),bool);hi=np.zeros((8,8,8),bool)
 for z in range(2):
  for y in range(2):
   for x in range(2):
    cell=negative[z:z+8,y:y+8,x:x+8];lo&=cell;hi|=cell
 for z,y,x in np.argwhere(hi&~lo):
  for dz in range(2):
   for dy in range(2):
    for dx in range(2):needed.add((int(origin[0]+x+dx),int(origin[1]+y+dy),int(origin[2]+z+dz)))
print(f'{a.asset}: {len(needed)} unique nodes bordering genuine FVX surfaces',flush=True)
# Read exact embedded images at bake time. The runtime only receives raw RGBA.
image_cache={}
with tempfile.TemporaryDirectory(prefix='fvm-images-') as temporary:
 def image(index):
  if index not in image_cache:
   info=model['images'][index];view=model['bufferViews'][info['bufferView']];offset=view.get('byteOffset',0)
   path=Path(temporary)/(str(index)+('.png' if info['mimeType']=='image/png' else '.jpg'));path.write_bytes(binary[offset:offset+view['byteLength']])
   im=bpy.data.images.load(str(path),check_existing=False);im.colorspace_settings.name='Non-Color'
   pix=np.empty(len(im.pixels),np.float32);im.pixels.foreach_get(pix);image_cache[index]=pix.reshape(im.size[1],im.size[0],4).copy();bpy.data.images.remove(im)
  return image_cache[index]
 def texture(info):
  if info.get('texCoord',0)!=0 or info.get('extensions'):raise ValueError('Unsupported UV set/texture transform; export transformed TEXCOORD_0 first')
  t=model['textures'][info['index']];sampler=model.get('samplers',[{}])[t.get('sampler',0)] if model.get('samplers') else {}
  if sampler.get('wrapS',10497)!=10497 or sampler.get('wrapT',10497)!=10497:raise ValueError('FVM1 initial texture ABI requires repeat wrapping')
  return image(t['source'])
 descriptors=[];layers=[]
 for mat in model.get('materials',[]):
  pm=mat.get('pbrMetallicRoughness',{});color=texture(pm['baseColorTexture']) if 'baseColorTexture'in pm else np.ones((1,1,4),np.float32)
  mr=texture(pm['metallicRoughnessTexture']) if 'metallicRoughnessTexture'in pm else np.ones((1,1,4),np.float32)
  normal=texture(mat['normalTexture']) if 'normalTexture'in mat else None
  color_layer=0xffffffff
  if 'baseColorTexture' in pm or 'metallicRoughnessTexture' in pm:
   color_layer=len(layers);layers.append(rgba_mips(srgb_decode(color[:,:,:3]),mr[:,:,1],a.texture_size))
  normal_layer=0xffffffff
  if normal is not None:
   normal_layer=len(layers);layers.append(rgba_mips(normal[:,:,:3],np.ones(normal.shape[:2],np.float32),a.texture_size,normal=True))
  descriptors.append({'name':mat.get('name',''),'baseColorFactor':pm.get('baseColorFactor',[1,1,1,1])[:3],'roughnessFactor':pm.get('roughnessFactor',1),'metallicFactor':pm.get('metallicFactor',1),'colorRoughnessLayer':color_layer,'normalLayer':normal_layer,'normalStrength':mat.get('normalTexture',{}).get('scale',1)})
# Reimport the unchanged source only in this offline baker.
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
for mat in list(bpy.data.materials):bpy.data.materials.remove(mat)
bpy.ops.import_scene.gltf(filepath=str(source));deps=bpy.context.evaluated_depsgraph_get()
material_names={m.get('name',''):i for i,m in enumerate(model['materials'])}
parts=[];mapping_records=[bytes(64)]; mapping_keys={}
for obj in bpy.context.scene.objects:
 if obj.type!='MESH':continue
 evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh();mesh.calc_loop_triangles();matrix=obj.matrix_world;nm=matrix.to_3x3().inverted().transposed()
 axis=lambda v:Vector((v.x,v.z,-v.y))
 vertices=[axis(matrix@v.co) for v in mesh.vertices];triangles=[tuple(t.vertices) for t in mesh.loop_triangles]
 normals=[axis(nm@n.vector).normalized() for n in mesh.corner_normals]
 bounds=np.asarray(vertices);bvh=BVHTree.FromPolygons(vertices,triangles,all_triangles=True);lo=bounds.min(axis=0);hi=bounds.max(axis=0)
 ids=[]
 for tri in mesh.loop_triangles:
  mapping=None
  if mesh.uv_layers.active:
   uvs=[mesh.uv_layers.active.data[i].uv for i in tri.loops];mapping=affine_uv([vertices[i] for i in tri.vertices],uvs)
  if mapping is None:ids.append(0);continue
  u,v,tangent,sign,geon=mapping
  # Source winding does not define outwardness. Use the same closed-solid parity.
  midpoint=sum((vertices[i] for i in tri.vertices),Vector())/3;probe=max(h['cellSize']*1e-3,1e-6)
  if inside(bvh,midpoint+Vector(geon)*probe,h['cellSize']*1e-5,(lo,hi)):sign=-sign
  material=material_names[mesh.materials[tri.material_index].name]
  encoded=struct.pack('<12f4I',*u,*v,*tangent,sign,material,0,0,0)
  # Identical affine charts share a descriptor, without merging discontinuous UVs.
  if encoded not in mapping_keys:mapping_keys[encoded]=len(mapping_records);mapping_records.append(encoded)
  ids.append(mapping_keys[encoded])
 parts.append({'bvh':bvh,'vertices':vertices,'triangles':triangles,'normals':[tuple(normals[i] for i in t.loops) for t in mesh.loop_triangles],'ids':ids,'lo':lo,'hi':hi})
 evaluated.to_mesh_clear()
cache={};base=np.asarray(h['origin']);pitch=h['cellSize']
for ni,key in enumerate(sorted(needed,key=lambda v:(v[2],v[1],v[0]))):
 point=Vector(base+np.asarray(key)*pitch);best=float('inf');chosen=0;shade=Vector((0,0,1))
 for part in parts:
  if any(point[i]<part['lo'][i]-2*pitch or point[i]>part['hi'][i]+2*pitch for i in range(3)):continue
  nearest,normal,tri,d=part['bvh'].find_nearest(point)
  if nearest is None:continue
  sign=-1 if inside(part['bvh'],point,pitch*1e-5,(part['lo'],part['hi'])) else 1
  if sign*d>=best:continue
  best=sign*d;chosen=part['ids'][tri];weights=barycentric(nearest,*[part['vertices'][i] for i in part['triangles'][tri]])
  shade=sum((n*w for n,w in zip(part['normals'][tri],weights)),Vector()).normalized()
  grad=signed_outward_gradient(point,nearest,d,sign,normal,lambda q:inside(part['bvh'],q,pitch*1e-5,(part['lo'],part['hi'])),max(pitch*1e-3,1e-7))
  shade=outward_shading_normal(shade,grad)
 cache[key]=struct.pack('<II',chosen,oct16(shade))
 if ni%10000==0:print(f'{a.asset}: mapping {ni}/{len(needed)} nodes',flush=True)
nodes=bytearray();zero=bytes(8)
for x,y,z in origins:
 for dz in range(9):
  for dy in range(9):
   for dx in range(9):nodes.extend(cache.get((x+dx,y+dy,z+dz),zero))
maps=b''.join(mapping_records);textures=b''.join(layers)
header={'version':1,'asset':a.asset,'sourceSha256':hashlib.sha256(blob).hexdigest(),'fvxPayloadSha256':hashlib.sha256(payload).hexdigest(),'origin':h['origin'],'dimensions':h['dimensions'],'cellSize':pitch,'brickCount':h['brickCount'],'brickCells':8,'nodeStrideBytes':8,'mappingStrideBytes':64,'mappingCount':len(mapping_records),'mappingByteOffset':0,'nodeByteOffset':len(maps),'textureByteOffset':len(maps)+len(nodes),'textureWidth':a.texture_size,'textureHeight':a.texture_size,'textureLayerCount':len(layers),'mipLevelCount':int(math.log2(a.texture_size))+1,'textureFormat':'rgba8unorm-srgb','textureLayout':'layer-major-mip-major','textureOrigin':'blender-uv-bottom-left','materials':descriptors,'mappedUniqueNodes':len(cache),'nodeOrder':'FVX1 brick order; x-fastest 9x9x9 within brick','normalEncoding':'oct16x2 unperturbed outward base normal; normal texture mapped mean unit vectors explicitly sRGB-encoded'}
encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%4);data=b'FVM1'+struct.pack('<I',len(encoded))+encoded+maps+nodes+textures
result=out/(a.asset+'.fvm.gz')
atomic_write(result,data,compressed=True)
atomic_write(out/(a.asset+'-surface-mapping.json'),(json.dumps(header,indent=2)+'\n').encode())
descriptor={'url':'/assets/'+a.collection+'/'+a.asset+'.fvm.gz','sha256':hashlib.sha256(data).hexdigest()}
atomic_write(out/(a.asset+'-surface-mapping-descriptor.json'),(json.dumps(descriptor,indent=2)+'\n').encode())
if a.publish:
 current=gzip.decompress(fvx.read_bytes());length=struct.unpack_from('<I',current,4)[0];current_header=json.loads(current[8:8+length]);current_payload=current[8+length:]
 if hashlib.sha256(current_payload).hexdigest()!=header['fvxPayloadSha256']:raise RuntimeError('FVX changed during companion bake; refusing publication')
 current_header['surfaceMapping']=descriptor
 encoded_header=json.dumps(current_header,separators=(',',':')).encode();encoded_header+=b' '*((-len(encoded_header))%4)
 atomic_write(source.with_suffix('.fvm.gz'),data,compressed=True)
 atomic_write(fvx,b'FVX1'+struct.pack('<I',len(encoded_header))+encoded_header+current_payload,compressed=True)
 atomic_write(ROOT/'lib/core/imported-assets/metadata'/(a.asset+'-volume.json'),(json.dumps(current_header,indent=2)+'\n').encode())
print(json.dumps({'path':str(result),'rawBytes':len(data),'compressedBytes':result.stat().st_size,'mappingCount':len(mapping_records),'mappedUniqueNodes':len(cache),'layers':len(layers)},indent=2),flush=True)
