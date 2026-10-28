"""Import GLB and bake sparse, signed mesh volumes with two normal channels.

FVX1: magic, uint32 JSON byte length, padded UTF-8 header, then fixed bricks.
Each brick has int32 xyz (cell origin), followed by 9^3 node records:
float32 distance in metres, oct16x2 geometric gradient, oct16x2 shading normal,
uint32 material index (zero = no surface). All words are little endian.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import struct
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from texture_filter import TextureMipChain, uv_cell_metric
from normal_orientation import outward_shading_normal, signed_outward_gradient
from mesh_components import split_part
parser = argparse.ArgumentParser()
parser.add_argument('--asset', default='all')
parser.add_argument('--collection', default='pool-party', help='Collection folder under public/assets containing manifest.json')
parser.add_argument('--cell', type=float)
parser.add_argument('--band-cells', type=float)
parser.add_argument('--solid-rule', choices=('even-odd','component-union'), help='Default even-odd per object; component-union unions separate closed connected shells')
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if not args.collection or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.collection):
    parser.error('--collection must be a simple asset collection folder name')
if args.band_cells is not None and (not math.isfinite(args.band_cells) or args.band_cells < 1):
    parser.error('--band-cells must be at least one source cell')
OUT = ROOT/'public/assets'/args.collection
METADATA = ROOT/'lib/core/imported-assets/metadata'
B = 8
RAY = Vector((1, .371390676, .529113257)).normalized()


from mesh_sampling import oct16, inside, barycentric, atomic_write


def image_input(socket):
    """Resolve glTF's image/normal-map/separate-channel shader wiring."""
    for link in socket.links:
        node = link.from_node
        if node.type == 'TEX_IMAGE' and node.image:
            im = node.image
            pixels = np.empty(len(im.pixels), dtype=np.float32)
            im.pixels.foreach_get(pixels)
            pixels = pixels.reshape((im.size[1], im.size[0], 4))[:,:,:3]
            if im.colorspace_settings.name == 'sRGB':
                pixels = np.where(pixels <= .04045, pixels/12.92, ((pixels+.055)/1.055)**2.4)
            return pixels
        if node.type == 'NORMAL_MAP': return image_input(node.inputs['Color'])
        if node.type in ('SEPARATE_COLOR', 'SEPRGB'):
            pixels = image_input(node.inputs[0])
            if pixels is not None:
                channel = list(node.outputs).index(link.from_socket)
                return np.repeat(pixels[:,:,channel:channel+1], 3, axis=2)
    return None



def bake(name, h, band_cells=3, solid_rule='even-odd'):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(OUT/(name+'.glb')))
    deps = bpy.context.evaluated_depsgraph_get()
    materials, material_ids, parts, textures, variants = [], {}, [], {}, {}
    roughness_maps, normal_maps = {}, {}
    # Import glTF back through Blender so this tests the actual exported asset.
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH': continue
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        check = bmesh.new()
        check.from_mesh(mesh)
        bmesh.ops.remove_doubles(check, verts=list(check.verts), dist=1e-6)
        if any(not e.is_manifold for e in check.edges):
            raise ValueError(f'{name}/{obj.name} is not watertight; cannot assign a solid sign')
        check.free()
        transform = obj.matrix_world
        normal_matrix = transform.to_3x3().inverted().transposed()
        def axis(p): return Vector((p.x, p.z, -p.y))
        vertices = [axis(transform@v.co) for v in mesh.vertices]
        triangles = [tuple(t.vertices) for t in mesh.loop_triangles]
        corner_normals = [axis(normal_matrix@n.vector).normalized() for n in mesh.corner_normals]
        normals = [tuple(corner_normals[i] for i in t.loops) for t in mesh.loop_triangles]
        ids=[]
        for t in mesh.loop_triangles:
            mat=mesh.materials[t.material_index]
            if mat.name not in material_ids:
                p=next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
                color=list(p.inputs['Base Color'].default_value[:3])
                texture = None
                for link in p.inputs['Base Color'].links:
                    if link.from_node.type == 'TEX_IMAGE' and link.from_node.image:
                        image=link.from_node.image
                        pixels=np.empty(len(image.pixels),dtype=np.float32)
                        image.pixels.foreach_get(pixels)
                        pixels=pixels.reshape((image.size[1], image.size[0], 4))[:,:,:3]
                        if image.colorspace_settings.name == 'sRGB':
                            pixels=np.where(pixels <= .04045, pixels/12.92, ((pixels+.055)/1.055)**2.4)
                        texture = pixels
                        color=pixels.mean(axis=(0,1)).tolist()
                materials.append({'name':mat.name,'baseColorLinear':color,
                                  'roughness':p.inputs['Roughness'].default_value,
                                  'metallic':p.inputs['Metallic'].default_value})
                material_ids[mat.name]=len(materials)
                if texture is not None: textures[len(materials)]=TextureMipChain(texture)
                rough=image_input(p.inputs['Roughness'])
                normal=image_input(p.inputs['Normal'])
                if rough is not None: roughness_maps[len(materials)]=TextureMipChain(rough)
                if normal is not None: normal_maps[len(materials)]=TextureMipChain(normal,normal=True)
            ids.append(material_ids[mat.name])
        uvs=[tuple(mesh.uv_layers.active.data[i].uv.copy() for i in t.loops) for t in mesh.loop_triangles] if mesh.uv_layers.active else None
        texture_lods=[]
        if uvs:
            for indices,uv,material in zip(triangles,uvs,ids):
                if not any(material in maps for maps in (textures,roughness_maps,normal_maps)):
                    texture_lods.append((0.,0.,0.));continue
                metric=uv_cell_metric([vertices[i] for i in indices],uv,h)
                texture_lods.append(tuple(maps[material].lod(metric) if material in maps else 0.
                                          for maps in (textures,roughness_maps,normal_maps)))
        bounds=np.array([tuple(v) for v in vertices])
        parts.append({'bvh':BVHTree.FromPolygons(vertices,triangles,all_triangles=True),
                      'vertices':vertices,'triangles':triangles,'normals':normals,'ids':ids,
                      'uvs':uvs,'texture_lods':texture_lods,
                      'lo':bounds.min(axis=0),'hi':bounds.max(axis=0)})
        evaluated.to_mesh_clear()
    if solid_rule == 'component-union':
        parts=[component for part in parts for component in split_part(part)]
        print(f'{name}: component-union uses {len(parts)} closed connected parts',flush=True)
    band=band_cells*h
    lo=np.min([p['lo'] for p in parts],axis=0)
    hi=np.max([p['hi'] for p in parts],axis=0)
    origin=np.floor((lo-band)/h)*h
    dims=(np.ceil((hi+band-origin)/(B*h)).astype(int)*B)
    candidates={}
    # Conservative expanded bounds index the same parts for a lattice node,
    # regardless of which overlapping brick emits it. This avoids scanning
    # every blade cluster / ladder fitting for every sample.
    part_bins={}
    for pi,part in enumerate(parts):
        begin=np.maximum(0,np.floor((part['lo']-band-origin)/(B*h)).astype(int))
        end=np.minimum(dims//B,np.ceil((part['hi']+band-origin)/(B*h)).astype(int))
        bin_end=np.minimum(dims//B,np.floor((part['hi']+band-origin)/(B*h)).astype(int))
        for z in range(begin[2],bin_end[2]+1):
            for y in range(begin[1],bin_end[1]+1):
                for x in range(begin[0],bin_end[0]+1):
                    part_bins.setdefault((x,y,z),[]).append(pi)
        for z in range(begin[2],end[2]):
            for y in range(begin[1],end[1]):
                for x in range(begin[0],end[0]):
                    c=Vector(origin+(np.array([x,y,z])*B+B/2)*h)
                    nearest,_,_,d=part['bvh'].find_nearest(c)
                    if nearest is None: continue
                    if d > math.sqrt(3)*B*h/2+band and not inside(part['bvh'],c,h*1e-5,(part['lo'],part['hi'])): continue
                    candidates.setdefault((x*B,y*B,z*B),[]).append(pi)
    data=bytearray()
    # Cache integer lattice coordinates, including shared brick faces. The exact
    # same node is emitted bit-for-bit in every neighbouring brick.
    cache={}
    negative=0
    for bi,(base,indices) in enumerate(sorted(candidates.items(), key=lambda kv:(kv[0][2],kv[0][1],kv[0][0]))):
        data.extend(struct.pack('<3i',*base))
        for z in range(9):
            for y in range(9):
                for x in range(9):
                    key=(base[0]+x,base[1]+y,base[2]+z)
                    if key not in cache:
                        p=Vector(origin+np.array(key)*h)
                        best=band; gradient=shade=Vector((0,0,0)); material=0
                        # Node ownership must be independent of the brick's part
                        # candidate list; nearby neighbouring parts may overlap.
                        for pi in part_bins.get(tuple(k//B for k in key),[]):
                            part=parts[pi]
                            if any(p[a]<part['lo'][a]-band or p[a]>part['hi'][a]+band for a in range(3)): continue
                            nearest,normal,tri,d=part['bvh'].find_nearest(p)
                            if nearest is None: continue
                            sign=-1 if inside(part['bvh'],p,h*1e-5,(part['lo'],part['hi'])) else 1
                            signed=sign*d
                            if signed >= best: continue
                            best=signed
                            gradient=signed_outward_gradient(p,nearest,d,sign,normal,
                                lambda q: inside(part['bvh'],q,h*1e-5,(part['lo'],part['hi'])),
                                max(h*1e-3,1e-7))
                            verts=[part['vertices'][i] for i in part['triangles'][tri]]
                            weights=barycentric(nearest,*verts)
                            shade=sum((n*w for n,w in zip(part['normals'][tri],weights)),Vector()).normalized()
                            if shade.dot(normal)<0: shade=-shade
                            material=part['ids'][tri]
                            if part['uvs'] and (material in textures or material in roughness_maps or material in normal_maps):
                                uv=sum((uv*w for uv,w in zip(part['uvs'][tri],weights)),Vector((0,0)))
                                source=materials[material-1]
                                color_lod,rough_lod,normal_lod=part['texture_lods'][tri]
                                rgb=textures[material].sample(uv,color_lod) if material in textures else np.array(source['baseColorLinear'])
                                rough=float(roughness_maps[material].sample(uv,rough_lod)[0]) if material in roughness_maps else source['roughness']
                                if material in normal_maps:
                                    uv0,uv1,uv2=part['uvs'][tri];du,dv=uv1-uv0,uv2-uv0
                                    det=du.x*dv.y-du.y*dv.x
                                    if abs(det)>1e-12:
                                        e1,e2=verts[1]-verts[0],verts[2]-verts[0]
                                        tangent=(e1*dv.y-e2*du.y)/det
                                        bitangent=(e2*du.x-e1*dv.x)/det
                                        tangent=(tangent-shade*tangent.dot(shade)).normalized()
                                        cross=shade.cross(tangent)
                                        cross*=1 if cross.dot(bitangent)>=0 else -1
                                        n=normal_maps[material].sample(uv,normal_lod)
                                        shade=(tangent*float(n[0])+cross*float(n[1])+shade*float(n[2])).normalized()
                                # Quantise perceptually; keep the palette bounded while
                                # retaining spatial wood grain, weave and stone variation.
                                srgb=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
                                q=tuple(np.clip(np.rint(srgb*31),0,31).astype(int))
                                # Eight roughness buckets are sufficient for honed
                                # stone/grit without multiplying the palette by 256.
                                rq=round(rough*7)/7 if material in roughness_maps else rough
                                key_material=(material,*q,rq)
                                if key_material not in variants:
                                    c=np.array(q)/31
                                    c=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
                                    materials.append({**source,'name':source['name']+'/'+','.join(map(str,q))+f'/r{rq:.3f}',
                                                      'baseColorLinear':c.tolist(),'roughness':rq})
                                    variants[key_material]=len(materials)
                                material=variants[key_material]
                        # Parity, not source winding, defines a closed solid's
                        # outward side. Apply after tangent-space normal detail
                        # so valid detail survives and inward source shells do
                        # not publish inward lighting normals into the FVX.
                        shade=outward_shading_normal(shade,gradient)
                        if best < 0: negative+=1
                        cache[key]=struct.pack('<fIII',max(-band,best),oct16(gradient),oct16(shade),material)
                    data.extend(cache[key])
        if bi%100==0: print(f'{name}: {bi}/{len(candidates)} bricks',flush=True)
    header={'version':1,'units':'metres','upAxis':'Y','asset':name,'source':name+'.glb',
            'origin':origin.tolist(),'dimensions':dims.tolist(),'cellSize':h,'bandWidth':band,
            'brickCount':len(candidates),'brickCells':8,'nodeStrideBytes':16,'materials':materials,
            'sign':'negative-inside','solidRule':solid_rule,'texturePolicy':'Closest-triangle UV footprint-aware linear-color and roughness area mips; tangent-space unit-vector normal mips renormalized after filtering; isotropic largest-Jacobian-axis LOD; independent geometric gradient; metallic retained',
            'uniqueNodes':len(cache),'insideNodes':negative}
    if name in ('honed-travertine-paver', 'hero-ground-tile'):
        # A proof over the sampled geometry, not an assumption from a material
        # name. Inset past the bevel and its interpolation apron. The source
        # shading normals may vary freely; the distance must remain a slab.
        half_x, half_z, bottom, top = ((.45, .45, 0, .03) if name == 'honed-travertine-paver'
                                     else (.8, .5, -.05, 0))
        for key, record in cache.items():
            p=origin+np.array(key)*h
            if abs(p[0]) <= half_x and abs(p[2]) <= half_z:
                expected=max(bottom-p[1],p[1]-top)
                expected=max(-band,min(band,expected))
                if abs(struct.unpack_from('<f',record)[0]-expected)>2e-6:
                    raise ValueError('Stone slab coarsening certificate does not match its geometry')
        header['horizontalSlab']={'minimum':[-half_x,bottom,-half_z],'maximum':[half_x,top,half_z]}
    encoded=json.dumps(header,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%4)
    raw=b'FVX1'+struct.pack('<I',len(encoded))+encoded+data
    atomic_write(OUT/(name+'.fvx.gz'),raw,compressed=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    atomic_write(METADATA/(name+'-volume.json'),(json.dumps(header,indent=2)+'\n').encode())
    print(f'{name}: {len(raw)/1e6:.1f} MB, {len(cache)} unique nodes, {negative} inside',flush=True)


presets = [('teak-lounger',.01),('garden-parasol',.01),('insulated-cooler',.01),('refreshment-set',.005),
               ('diving-board-ladder',.00625),('pool-stainless-ladder',.00625),
               ('honed-travertine-paver',.00625),('living-turf',.003125),
               ('cedar-fence-panel',.00625),('stucco-window-bay',.00625),
               ('limestone-coping',.00625),('limestone-coping-corner',.00625),
               ('ceramic-waterline',.00625),('teak-serving-table',.00625)]
bands = {}
solid_rules = {}
if args.collection != 'pool-party':
    assets = json.loads((OUT/'manifest.json').read_text())['assets']
    presets = [(a['id'], a['cellSize']) for a in assets]
    bands = {a['id']: a.get('bandCells', 3) for a in assets}
    solid_rules = {a['id']: a.get('solidRule', 'even-odd') for a in assets}
    if any(rule not in ('even-odd','component-union') for rule in solid_rules.values()):
        parser.error('Manifest solidRule must be even-odd or component-union')
if args.asset != 'all' and args.asset not in dict(presets):
    parser.error(f'Unknown asset {args.asset} in {args.collection}')
for name,h in presets:
    if args.asset in ('all',name): bake(name,args.cell or h,
        args.band_cells if args.band_cells is not None else bands.get(name,3),
        args.solid_rule or solid_rules.get(name,'even-odd'))
