"""Author a closed, textured alpine ridge and irregular conifers for the live renderer.
The editable Blender mesh retains its full triangles; an independent 1.25 m
distant-ridge FVX sampling provides the renderer's native sparse scene representation.
"""
import argparse,sys
import bpy,bmesh,json,math,random
import numpy as np
from pathlib import Path
from mathutils import Vector,noise
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'public/assets/alpine-lake')
parser.add_argument('--source-dir',type=Path,default=ROOT/'assets/alpine-lake')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
OUT=args.output_dir.resolve();SOURCE=args.source_dir.resolve()
OUT.mkdir(parents=True,exist_ok=True);SOURCE.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version=0
random.seed(20261028)

def height(x,y):
    peaks=[(-49,30,42,22,25),(-19,45,50,19,27),(16,28,38,19,26),(48,51,49,26,30),(-74,60,56,28,34)]
    h=0
    for px,py,ph,sx,sy in peaks:
        d=math.sqrt(((x-px)/sx)**2+((y-py)/sy)**2)
        theta=math.atan2((y-py)/sy,(x-px)/sx)
        profile=ph*max(0,1-d*.65)**1.1
        # Radial erosion cuts continuous gullies down from each summit.
        # Their amplitude vanishes at the summit, so no needle spikes form.
        gully=(.5+.5*math.sin(theta*7+d*2+px*.17))**3
        eroded=profile-8*gully*min(1,d*2.5)*max(0,1-d*.65)
        h=max(h,eroded)
    # Broad erosion shapes carry the massif; small rock detail is additive
    # metres, not a fraction of the full mountain height at every vertex.
    crag=noise.fractal(Vector((x*.055,y*.055,2)),.9,2.1,4)
    crease=abs(noise.noise(Vector((x*.12,y*.12,1.2))))
    detail=noise.fractal(Vector((x*.4,y*.4,5)),.9,2,3)*.8
    h=h*(.9+.18*crag+.10*crease)+detail
    return max(.2,h*min(1,y/8))

nx,ny=320,160
verts=[(-80+160*i/nx,75*j/ny,height(-80+160*i/nx,75*j/ny)) for j in range(ny+1) for i in range(nx+1)]
faces=[]
for j in range(ny):
    for i in range(nx):
        a=j*(nx+1)+i
        faces.extend([(a,a+1,a+nx+2),(a,a+nx+2,a+nx+1)])
top_count=len(faces)
# Explicit closed skirt and base; this is a genuine watertight sampled volume.
boundary=list(range(nx+1))+[j*(nx+1)+nx for j in range(1,ny+1)]+[ny*(nx+1)+i for i in range(nx-1,-1,-1)]+[j*(nx+1) for j in range(ny-1,0,-1)]
base=[]
for i in boundary:base.append(len(verts));verts.append((verts[i][0],verts[i][1],-.5))
for k,a in enumerate(boundary):
    nxt=(k+1)%len(boundary);faces.append((a,base[k],base[nxt],boundary[nxt]))
faces.append(tuple(reversed(base)))
data=bpy.data.meshes.new('Fractured glacial ridge with closed strata');data.from_pydata(verts,[],faces);data.update()
obj=bpy.data.objects.new('alpine-mountain-ridge',data);bpy.context.scene.collection.objects.link(obj)
# Derive a neutral mineral color map from the licensed aerial strata. Keep
# its photographed luminance/detail while reducing vegetation-like ochre.
# This is a texture build step; the unmodified source and checksums remain.
source_color=bpy.data.images.load(str(ROOT/'assets/alpine-lake/textures/aerial_rocks_02_diff_2k.jpg'),check_existing=True)
pixels=np.asarray(source_color.pixels[:],dtype=np.float32).reshape((-1,4))
luminance=pixels[:,:3]@np.array([.2126,.7152,.0722],dtype=np.float32)
pixels[:,:3]=(.15*pixels[:,:3]+.85*luminance[:,None])*.8
neutral=bpy.data.images.new('Neutral alpine photographic strata',width=source_color.size[0],height=source_color.size[1])
neutral.colorspace_settings.name='sRGB'
neutral.pixels.foreach_set(pixels.ravel());neutral.file_format='PNG';neutral.filepath_raw=str(SOURCE/'alpine-strata-neutral.png');neutral.save()
# The common baker already preserves image color, roughness and tangent
# normals. Keep the same photogrammetry source in offline and native assets.
mat=bpy.data.materials.new('Alpine photographic gneiss');mat.use_nodes=True
n,l=mat.node_tree.nodes,mat.node_tree.links;bs=n.get('Principled BSDF')
bs.inputs['Roughness'].default_value=.85
for channel,suffix,color_space in [('color','diff','sRGB'),('normal','nor_gl','Non-Color'),('roughness','rough','Non-Color')]:
    path=(SOURCE/'alpine-strata-neutral.png') if channel=='color' else ROOT/'assets/alpine-lake/textures'/('aerial_rocks_02_'+suffix+'_2k.jpg')
    tex=n.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(path),check_existing=True);tex.image.colorspace_settings.name=color_space
    if channel=='color':l.new(tex.outputs['Color'],bs.inputs['Base Color'])
    elif channel=='roughness':l.new(tex.outputs['Color'],bs.inputs['Roughness'])
    else:
        nm=n.new('ShaderNodeNormalMap');nm.inputs['Strength'].default_value=1;l.new(tex.outputs['Color'],nm.inputs['Color']);l.new(nm.outputs[0],bs.inputs['Normal'])
data.materials.append(mat)
uv=data.uv_layers.new(name='Distant macro rock strata')
for f in data.polygons:
    for li in f.loop_indices:
        co=data.vertices[data.loops[li].vertex_index].co;uv.data[li].uv=(co.x/5,co.y/5)
for i in range(8):
    t=i/7;color=(.48+.35*t,.56+.34*t,.65+.29*t)
    mat=bpy.data.materials.new('Old glacier snow '+str(i));mat.use_nodes=True;mat.diffuse_color=(*color,1)
    p=mat.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=.7;data.materials.append(mat)
for f in data.polygons:
    x,y,z=f.center;n=noise.fractal(Vector((x*1.2,y*1.2,z*.8)),.9,2,4)
    # Snow persists on gentler high slopes and in sheltered gullies; exposed
    # steep crags remain rock instead of carrying an even horizontal cap.
    snow_altitude=z+noise.noise(Vector((x*.15,y*.15,z*.12)))*3
    snow_slope=.84-min(50,max(0,z))*.008
    covered=snow_altitude>23 and f.normal.z>snow_slope
    f.material_index=1+max(0,min(7,int((n*.5+.5)*7))) if covered else 0
    f.use_smooth=True
# Distant peaks rise roughly 150 metres above the shore. The low front skirt joins
# the real shore; stretching it outward avoids a detached backdrop or a blue
# gap beneath the distant massif. Scale the FVX spacing with the source, so
# sparse memory remains bounded while native triangles retain the silhouette.
for vertex in data.vertices:
    x,y,z=vertex.co
    vertex.co=(x*2.5,15*y if y<8 else 100+2.5*y,2.5*(z-.2))
data.update()
bm=bmesh.new();bm.from_mesh(data);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));assert all(e.is_manifold for e in bm.edges);bm.to_mesh(data);bm.free()
bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
bpy.ops.export_scene.gltf(filepath=str(OUT/'alpine-mountain-ridge.glb'),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_normals=True,export_materials='EXPORT')
assets=[{'id':'alpine-mountain-ridge','mesh':'alpine-mountain-ridge.glb','volume':'alpine-mountain-ridge.fvx.gz','cellSize':1.25,'bandCells':1}]
# Warp woody branches and their attached foliage together. The source cedars'
# identical horizontal tiers become uneven branching crowns without breaking
# their closed mesh topology or inventing disconnected floating leaves.
for variant in range(2):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=str(ROOT/'public/assets/forest-waterfall'/('forest-cedar-'+str(variant)+'.glb')))
    selected=list(bpy.context.selected_objects);parts=[o for o in selected if o.type=='MESH']
    phase=variant*1.7+.37
    for part in parts:
        part.data.transform(part.matrix_world);part.matrix_world.identity()
        for vert in part.data.vertices:
            x,y,z=vert.co;radius=math.hypot(x,y);theta=math.atan2(y,x)
            radial=min(1,radius/1.3);t=min(1,max(0,(z-1)/2))
            wave=(.75*math.sin(2*theta+z*.7+phase)+.38*math.cos(5*theta-z*.42+phase))*radial*t
            vert.co=(x+.025*z*math.sin(z*.24+phase),y+.035*z*math.sin(z*.2+phase),z+wave)
        part.data.update()
    name='alpine-fir-'+str(variant)
    bpy.ops.object.select_all(action='DESELECT')
    for part in parts:part.select_set(True)
    bpy.context.view_layer.objects.active=parts[0]
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_normals=True,export_materials='EXPORT')
    assets.append({'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','cellSize':(.04 if variant==0 else .03),'bandCells':1,'solidRule':'component-union'})
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'alpine-runtime.blend'))
existing_ids={asset['id'] for asset in assets}
if (OUT/'manifest.json').exists():
    assets.extend(asset for asset in json.loads((OUT/'manifest.json').read_text())['assets'] if asset['id'] not in existing_ids)
(OUT/'manifest.json').write_text(json.dumps({'collection':{'id':'alpine-lake','label':'Alpine Lake'},'source':'alpine-runtime.blend','assets':assets},indent=2)+'\n')
print('ALPINE_RUNTIME_EXPORTED',len(verts),len(faces),flush=True)
