"""Atlantic storm, rendered in Cycles from an actual narrow-band FLIP mesh.

blender -b --python tools/blender/atlantic_lighthouse.py -- \
  --water-mesh artifacts/photoreal/atlantic-lighthouse.obj --output artifacts/atlantic-lighthouse/storm.png
OBJ coordinates are metres, Z-up, already mapped from application (x,y,z)
to Blender (x,-z,y). No analytic/procedural liquid geometry is substituted.
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
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lighthouse_rock_material import photographed_rock_material, assign_rock_uv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--water-mesh', type=Path)
parser.add_argument('--dry-preview', action='store_true', help='Environment-only framing preview; no water, never a final result')
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--samples', type=int, default=128)
parser.add_argument('--width', type=int, default=1600)
parser.add_argument('--height', type=int, default=1100)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
if not args.dry_preview and (args.water_mesh is None or not args.water_mesh.is_file()):
    raise FileNotFoundError('A real NB-FLIP mesh is required: ' + str(args.water_mesh))
args.output.parent.mkdir(parents=True, exist_ok=True)
random.seed(20261028)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.render.engine = 'CYCLES'
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 10
scene.cycles.transmission_bounces = 8
scene.render.resolution_x = args.width
scene.render.resolution_y = args.height
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = .3
scene['water_provenance'] = str(args.water_mesh.resolve()) if args.water_mesh else 'DRY PREVIEW — NO WATER'
scene['water_method'] = 'DRY PREVIEW — NO WATER' if args.dry_preview else 'uniform-narrow-band-flip; imported simulated free surface'


def material(name, lo, hi, rough=.7, scale=2.5, bump=.05):
    m = bpy.data.materials.new(name); m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    p = n.get('Principled BSDF'); p.inputs['Roughness'].default_value = rough
    tc = n.new('ShaderNodeTexCoord')
    no = n.new('ShaderNodeTexNoise'); no.inputs['Scale'].default_value = scale
    no.inputs['Detail'].default_value = 6; no.inputs['Roughness'].default_value = .78
    l.new(tc.outputs['Object'], no.inputs['Vector'])
    ramp = n.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = .18; ramp.color_ramp.elements[0].color = (*lo, 1)
    ramp.color_ramp.elements[1].position = .8; ramp.color_ramp.elements[1].color = (*hi, 1)
    l.new(no.outputs['Fac'], ramp.inputs[0]); l.new(ramp.outputs[0], p.inputs['Base Color'])
    fine = n.new('ShaderNodeTexNoise'); fine.inputs['Scale'].default_value = 65
    fine.inputs['Detail'].default_value = 3
    l.new(tc.outputs['Object'], fine.inputs['Vector'])
    b = n.new('ShaderNodeBump'); b.inputs['Strength'].default_value = .6; b.inputs['Distance'].default_value = bump
    l.new(fine.outputs['Fac'], b.inputs['Height']); l.new(b.outputs[0], p.inputs['Normal'])
    return m


rock = photographed_rock_material()
stone = [material('Salt-worn granite course %02d' % i, (.16 + i*.009, .16 + i*.008, .14 + i*.007),
                  (.43 + i*.018, .42 + i*.017, .36 + i*.016), .76, 4, .026) for i in range(7)]
mortar = material('Salted lime mortar', (.18, .18, .16), (.29, .28, .25), .9, 13, .003)
for mat in stone:
    ns,ls=mat.node_tree.nodes,mat.node_tree.links
    p=ns.get('Principled BSDF');base=p.inputs['Base Color'].links[0].from_socket
    tc=ns.new('ShaderNodeTexCoord');stretch=ns.new('ShaderNodeVectorMath');stretch.operation='MULTIPLY';stretch.inputs[1].default_value=(2.2,2.2,.09)
    ls.new(tc.outputs['Object'],stretch.inputs[0]);stain=ns.new('ShaderNodeTexNoise');stain.inputs['Scale'].default_value=1.2;stain.inputs['Detail'].default_value=3
    ls.new(stretch.outputs[0],stain.inputs['Vector']);ramp=ns.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position=.28;ramp.color_ramp.elements[0].color=(.18,.2,.21,1)
    ramp.color_ramp.elements[1].position=.73;ramp.color_ramp.elements[1].color=(1,1,1,1)
    ls.new(stain.outputs['Fac'],ramp.inputs[0]);mix=ns.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.85
    ls.new(base,mix.inputs[1]);ls.new(ramp.outputs[0],mix.inputs[2]);ls.new(mix.outputs[0],p.inputs['Base Color'])
iron = material('Weathered black iron', (.008, .011, .013), (.045, .05, .053), .36, 9, .009)
iron.node_tree.nodes.get('Principled BSDF').inputs['Metallic'].default_value = .72
rust = material('Oxidised handrail edges', (.04, .015, .006), (.22, .07, .022), .62, 6, .008)
glass = bpy.data.materials.new('Lantern glass'); glass.use_nodes = True
gp = glass.node_tree.nodes.get('Principled BSDF'); gp.inputs['Base Color'].default_value = (.8, .9, .88, 1)
gp.inputs['Transmission Weight'].default_value = 1; gp.inputs['Roughness'].default_value = .07; gp.inputs['IOR'].default_value = 1.45


def mesh(name, verts, faces, mat, smooth=False):
    data = bpy.data.meshes.new(name); data.from_pydata(verts, [], faces); data.update()
    ob = bpy.data.objects.new(name, data); scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    for p in data.polygons: p.use_smooth = smooth
    return ob


def cylinder(name, radius, depth, location, mat, vertices=64, radius2=None):
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=radius, radius2=radius if radius2 is None else radius2,
                                    depth=depth, location=location)
    ob = bpy.context.object; ob.name = name; ob.data.materials.append(mat)
    bevel = ob.modifiers.new('Worn edge', 'BEVEL'); bevel.width = .018; bevel.segments = 2
    for p in ob.data.polygons: p.use_smooth = True
    return ob


def rod(name, a, b, radius, mat):
    middle = (Vector(a) + Vector(b)) / 2
    ob = cylinder(name, radius, (Vector(b)-Vector(a)).length, middle, mat, 12)
    ob.rotation_euler = (Vector(b)-Vector(a)).to_track_quat('Z', 'Y').to_euler()
    return ob


def terrain_height(x, y):
    z = -y
    height = .15
    for r in rock_layout:
        dx,dz=x-r['x'],z-r['z'];c,s=math.cos(r['yaw']),math.sin(r['yaw'])
        u,v=(c*dx+s*dz)/r['rx'],(-s*dx+c*dz)/r['rz']
        q=max(abs(u),abs(v),abs(u+v)*.65,abs(u-v)*.65)
        top=r['height']+dx*r['tiltX']+dz*r['tiltZ']
        height=max(height,top-max(0,q-r['core'])*r['height']/(1-r['core']))
    height+=.07*math.sin(x*3.1+z*1.7)*math.sin(z*2.3-x*.4)*min(1,max(0,height-.15))
    return 5.8 if math.hypot(x-7,z)<1.55 else height


# Identical 25 cm lattice to the physical collision heightfield.
rock_layout=json.loads((Path(__file__).resolve().parents[2]/'lib/core/imported-assets/atlantic-lighthouse-rock-layout.json').read_text())
verts = [(x*.25-20, y*.25-16, terrain_height(x*.25-20, y*.25-16)) for y in range(129) for x in range(161)]
faces = [(i, i+1, i+162, i+161) for y in range(128) for x in range(160) for i in [x+161*y]]
rock_shelf=mesh('Authoritative simulated rock shelf', verts, faces, rock, True)
assign_rock_uv(rock_shelf.data)

# Individual curved masonry blocks retain real recessed joints and bevels.
base = 5.8
cylinder('Foundation footing embedded in bedrock', 1.52, .5, (7, 0, 5.85), stone[2])
# Real pointing closes the joints without filling the hollow tower. The former
# empty cracks became deep black gaps when sampled on the native 4 cm lattice.
segments=128
mortar_vertices=[]
for z in [base,base+43*.208]:
    outer=1.35-(z-base)*.049-.012
    for radius in [outer,outer-.19]:
        for i in range(segments):
            a=i*math.tau/segments
            mortar_vertices.append((7+radius*math.cos(a),radius*math.sin(a),z))
mortar_faces=[]
for i in range(segments):
    j=(i+1)%segments;n=segments
    mortar_faces.extend([(i,j,2*n+j,2*n+i),(n+j,n+i,3*n+i,3*n+j),
                         (i,n+i,n+j,j),(2*n+j,3*n+j,3*n+i,2*n+i)])
mesh('Recessed lime mortar masonry shell',mortar_vertices,mortar_faces,mortar,True)
for course in range(43):
    z0 = base + course*.208; z1 = z0+.202
    radius0 = 1.35-(z0-base)*.049; radius1 = 1.35-(z1-base)*.049
    count = 25
    for block in range(count):
        a0 = (block+(course%2)*.5)*math.tau/count+.004
        a1 = a0+math.tau/count-.008
        pts = []
        for z, radius in [(z0, radius0), (z1, radius1)]:
            for rad in [radius-.2, radius]:
                for j in range(4):
                    a = a0+(a1-a0)*j/3
                    pts.append((7+rad*math.cos(a), rad*math.sin(a), z))
        fs = []
        for j in range(3):
            fs.extend([(4+j, 5+j, 13+j, 12+j), (j, 8+j, 9+j, 1+j), (j, 1+j, 5+j, 4+j), (8+j, 12+j, 13+j, 9+j)])
        fs.extend([(0, 4, 12, 8), (3, 11, 15, 7)])
        ob = mesh('Masonry %02d-%02d' % (course, block), pts, fs, random.choice(stone))
        bevel = ob.modifiers.new('Salt softened arris', 'BEVEL'); bevel.width = .009; bevel.segments = 1

# Belt courses, door, small recessed window, lantern and gallery.
for z, radius in [(6.05, 1.44), (9.1, 1.24), (14.58, 1.04), (14.85, 1.38)]:
    cylinder('Stone stringcourse', radius, .19, (7, 0, z), stone[4])
bpy.ops.mesh.primitive_cube_add(size=1, location=(6.74, -1.25, 6.68))
door = bpy.context.object; door.name = 'Salt-blackened oak entrance'; door.scale = (.64, .1, 1.53); door.data.materials.append(iron)
for z in [9.4, 12.1]:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(6.8, -(1.35-(z-base)*.049), z))
    ob = bpy.context.object; ob.name = 'Deep narrow window'; ob.scale = (.31, .05, .68); ob.data.materials.append(iron)
    rod('Window mullion', (6.8, ob.location.y-.04, z-.31), (6.8, ob.location.y-.04, z+.31), .014, rust)
    for x,zz,scale,name in [(6.8-.18,z,(.055,.13,.8),'jamb'),(6.8+.18,z,(.055,.13,.8),'jamb'),
                             (6.8,z+.4,(.46,.15,.13),'lintel'),(6.8,z-.4,(.48,.24,.13),'drip sill')]:
        bpy.ops.mesh.primitive_cube_add(size=1,location=(x,ob.location.y-.04,zz))
        frame=bpy.context.object;frame.name='Stone window '+name;frame.scale=scale;frame.data.materials.append(stone[2])
        bevel=frame.modifiers.new('Salt worn sill edge','BEVEL');bevel.width=.015;bevel.segments=2
cylinder('Lantern plinth', .98, .28, (7, 0, 15.13), iron)
cylinder('Lantern glazing', .8, 1.3, (7, 0, 15.88), glass, 16)
cylinder('Roof eave', 1.04, .12, (7, 0, 16.58), iron)
cylinder('Conical copper lantern roof', 1.04, .75, (7, 0, 17), iron, 64, .06)
rod('Lightning conductor', (7, 0, 17.35), (7, 0, 18), .025, iron)
for i in range(16):
    a = i*math.tau/16; x, y = 7+.81*math.cos(a), .81*math.sin(a)
    rod('Lantern glazing bar', (x,y,15.24), (x,y,16.55), .025, iron)
    x,y = 7+1.27*math.cos(a), 1.27*math.sin(a)
    rod('Gallery stanchion', (x,y,14.92), (x,y,15.64), .022, iron)
for z in [15.2, 15.63]:
    bpy.ops.mesh.primitive_torus_add(major_radius=1.27, minor_radius=.022, major_segments=96, minor_segments=8, location=(7,0,z))
    bpy.context.object.name='Continuous iron gallery rail'; bpy.context.object.data.materials.append(iron)
bulb = bpy.data.materials.new('Warm Fresnel lamp'); bulb.use_nodes=True
bp=bulb.node_tree.nodes.get('Principled BSDF'); bp.inputs['Emission Color'].default_value=(1,.4,.09,1); bp.inputs['Emission Strength'].default_value=2.2
cylinder('Fresnel lens', .12, .4, (7,0,15.87), bulb)
for z in [15.58+i*.06 for i in range(11)]:
    bpy.ops.mesh.primitive_torus_add(major_radius=.265, minor_radius=.022, major_segments=48, minor_segments=8, location=(7,0,z))
    bpy.context.object.data.materials.append(glass)

# Only imported fluid geometry is allowed to carry the water material.
water_objects = []
if not args.dry_preview:
    bpy.ops.wm.obj_import(filepath=str(args.water_mesh.resolve()), forward_axis='Y', up_axis='Z')
    water_objects = list(bpy.context.selected_objects)
water = bpy.data.materials.new('Atlantic seawater — physically transmitted NB-FLIP surface'); water.use_nodes=True
n,l=water.node_tree.nodes, water.node_tree.links
p=n.get('Principled BSDF'); p.inputs['Base Color'].default_value=(.76,.88,.89,1)
p.inputs['Transmission Weight'].default_value=1; p.inputs['IOR'].default_value=1.333; p.inputs['Roughness'].default_value=.075
absorption=n.new('ShaderNodeVolumeAbsorption'); absorption.inputs['Color'].default_value=(.24,.62,.70,1); absorption.inputs['Density'].default_value=.11
l.new(absorption.outputs[0],n.get('Material Output').inputs['Volume'])
tex=n.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value=8; tex.inputs['Detail'].default_value=4
bump=n.new('ShaderNodeBump'); bump.inputs['Strength'].default_value=.13; bump.inputs['Distance'].default_value=.004
l.new(tex.outputs['Fac'],bump.inputs['Height']); l.new(bump.outputs[0],p.inputs['Normal'])
# Surface aeration is a shading approximation on the genuine simulated mesh,
# not a replacement wave or a separately fabricated splash geometry.
geo=n.new('ShaderNodeNewGeometry'); xyz=n.new('ShaderNodeSeparateXYZ'); l.new(geo.outputs['Position'],xyz.inputs[0])
height=n.new('ShaderNodeMapRange'); height.inputs['From Min'].default_value=3.5; height.inputs['From Max'].default_value=5.0
l.new(xyz.outputs['Z'],height.inputs['Value'])
foamnoise=n.new('ShaderNodeTexNoise'); foamnoise.inputs['Scale'].default_value=5; foamnoise.inputs['Detail'].default_value=5
l.new(geo.outputs['Position'],foamnoise.inputs['Vector'])
patch=n.new('ShaderNodeMath'); patch.operation='MULTIPLY'; l.new(height.outputs[0],patch.inputs[0]); l.new(foamnoise.outputs['Fac'],patch.inputs[1])
foamramp=n.new('ShaderNodeValToRGB'); foamramp.color_ramp.elements[0].position=.25; foamramp.color_ramp.elements[1].position=.58
l.new(patch.outputs[0],foamramp.inputs[0])
curvature=n.new('ShaderNodeMapRange'); curvature.inputs['From Min'].default_value=.54; curvature.inputs['From Max'].default_value=.62
l.new(geo.outputs['Pointiness'],curvature.inputs['Value'])
foamamount=n.new('ShaderNodeMath'); foamamount.operation='MULTIPLY'; l.new(foamramp.outputs[0],foamamount.inputs[0]); l.new(curvature.outputs[0],foamamount.inputs[1])
foam=n.new('ShaderNodeBsdfPrincipled'); foam.inputs['Base Color'].default_value=(.72,.79,.78,1); foam.inputs['Roughness'].default_value=.46
l.new(bump.outputs[0],foam.inputs['Normal'])
mix=n.new('ShaderNodeMixShader'); l.new(foamamount.outputs[0],mix.inputs[0]); l.new(p.outputs[0],mix.inputs[1]); l.new(foam.outputs[0],mix.inputs[2])
l.new(mix.outputs[0],n.get('Material Output').inputs['Surface'])
for ob in water_objects:
    ob.name='NB-FLIP simulated Atlantic water'; ob.data.materials.clear(); ob.data.materials.append(water)
    ob['source_mesh']=str(args.water_mesh.resolve())
    bm=bmesh.new(); bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=.00001)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces)); bm.to_mesh(ob.data); bm.free()
    for polygon in ob.data.polygons: polygon.use_smooth=True

# A physical sky behind volumetric storm banks, with a low warm opening.
world=bpy.data.worlds.new('Atlantic storm cloud cover'); scene.world=world; world.use_nodes=True
n,l=world.node_tree.nodes,world.node_tree.links
bg=n.get('Background'); bg.inputs['Strength'].default_value=.018
sky=n.new('ShaderNodeTexSky'); sky.sky_type='MULTIPLE_SCATTERING'; sky.sun_elevation=.16; sky.sun_rotation=-2.35; sky.aerosol_density=3
l.new(sky.outputs['Color'],bg.inputs['Color'])
for i,(location,scale) in enumerate([((0,65,31),(185,60,27)), ((-65,110,26),(160,70,22))]):
    cloud=bpy.data.materials.new('Atlantic volumetric storm bank '+str(i)); cloud.use_nodes=True
    cn,cl=cloud.node_tree.nodes,cloud.node_tree.links;cn.clear()
    out=cn.new('ShaderNodeOutputMaterial'); vol=cn.new('ShaderNodeVolumePrincipled')
    vol.inputs['Color'].default_value=(.57,.63,.69,1);vol.inputs['Anisotropy'].default_value=.45
    tc=cn.new('ShaderNodeTexCoord'); noise=cn.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=4
    noise.inputs['Detail'].default_value=5;noise.inputs['Roughness'].default_value=.72;cl.new(tc.outputs['Generated'],noise.inputs['Vector'])
    density=cn.new('ShaderNodeMapRange');density.inputs['From Min'].default_value=.39;density.inputs['From Max'].default_value=.67
    density.inputs['To Max'].default_value=.18;cl.new(noise.outputs['Fac'],density.inputs['Value'])
    sep=cn.new('ShaderNodeSeparateXYZ');cl.new(tc.outputs['Generated'],sep.inputs[0])
    falloff=cn.new('ShaderNodeValToRGB');falloff.color_ramp.elements[0].position=.02;falloff.color_ramp.elements[1].position=.25
    top=falloff.color_ramp.elements.new(.8);top.color=(1,1,1,1);top=falloff.color_ramp.elements.new(.99);top.color=(0,0,0,1)
    cl.new(sep.outputs['Z'],falloff.inputs[0]);mult=cn.new('ShaderNodeMath');mult.operation='MULTIPLY'
    cl.new(density.outputs[0],mult.inputs[0]);cl.new(falloff.outputs[0],mult.inputs[1]);cl.new(mult.outputs[0],vol.inputs['Density']);cl.new(vol.outputs[0],out.inputs['Volume'])
    bpy.ops.mesh.primitive_cube_add(size=1,location=location);ob=bpy.context.object;ob.name='Volumetric storm bank '+str(i);ob.scale=scale;ob.data.materials.append(cloud)
ld=bpy.data.lights.new('Late afternoon break in storm', 'SUN'); ld.energy=2.6; ld.angle=.04; ld.color=(1,.94,.85)
lo=bpy.data.objects.new(ld.name,ld); scene.collection.objects.link(lo); lo.rotation_euler=Vector((.55,.55,-.63)).to_track_quat('-Z','Y').to_euler()
ld=bpy.data.lights.new('Soft sky fill', 'AREA'); ld.energy=2400; ld.shape='DISK'; ld.size=22; ld.color=(.55,.7,1)
lo=bpy.data.objects.new(ld.name,ld); scene.collection.objects.link(lo); lo.location=(-4,-3,20)

# Thin maritime haze gives distance a measurable optical depth.
mist=bpy.data.materials.new('Salt haze'); mist.use_nodes=True
mn=mist.node_tree.nodes; mn.clear(); mo=mn.new('ShaderNodeOutputMaterial'); mv=mn.new('ShaderNodeVolumeScatter')
mv.inputs['Color'].default_value=(.62,.71,.8,1); mv.inputs['Density'].default_value=.0035; mv.inputs['Anisotropy'].default_value=.28
mist.node_tree.links.new(mv.outputs[0],mo.inputs['Volume'])
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,20))
ob=bpy.context.object; ob.name='Maritime atmospheric haze'; ob.scale=(160,160,60); ob.data.materials.append(mist)

bpy.ops.object.camera_add(location=(-17.3,-25.4,5.5))
cam=bpy.context.object; cam.name='Storm photograph — 36 mm'; target=Vector((2,0,8.4))
cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler(); cam.data.lens=36
cam.data.dof.use_dof=False
scene.camera=cam
scene.render.filepath=str(args.output.resolve())
bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix('.blend').resolve()))
bpy.ops.render.render(write_still=True)
args.output.with_suffix('.json').write_text(json.dumps({'scene':'atlantic-lighthouse','waterMesh':str(args.water_mesh.resolve()) if args.water_mesh else None,
    'waterMeshSha256':hashlib.sha256(args.water_mesh.read_bytes()).hexdigest() if args.water_mesh else None,
    'waterMethod':'DRY PREVIEW — NO WATER' if args.dry_preview else 'uniform-narrow-band-flip','foam':'procedural surface aeration shading; no secondary particle simulation',
    'samples':args.samples,'width':args.width,'height':args.height},indent=2)+'\n')
