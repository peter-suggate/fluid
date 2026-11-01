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
from mathutils import Vector, noise
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lighthouse_rock_material import photographed_rock_material, assign_rock_uv
from lighthouse_terrain import terrain_height
from lighthouse_weathering import whitewash_material, tower_uv

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
scene.view_settings.exposure = .55
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


# Dry geometry is sampled at 6.25 cm, independently of the 25 cm fluid lattice.
# The same sculpted height function is sampled more coarsely for fluid collision.
step = .0625
nx, ny = 385, 321
verts = [(-5+x*step, -10+y*step, terrain_height(-5+x*step, -10+y*step))
         for y in range(ny) for x in range(nx)]
faces = [(i, i+1, i+nx+1, i+nx) for y in range(ny-1) for x in range(nx-1) for i in [x+nx*y]]
rock_shelf = mesh('Authoritative simulated rock shelf', verts, faces, rock, True)
assign_rock_uv(rock_shelf.data)

# A closed tapered masonry shell with actual chipped plaster relief, not rows
# of identical decorative blocks. Its cavity remains open for fluid collision.
base = 5.8
body_height = 8.6
whitewash = whitewash_material()
def tower_radius(z):
    t = max(0, min(body_height, z-base))
    return 1.60 - .058*t - .035*math.sin(math.pi*t/body_height)

cylinder('Foundation footing embedded in bedrock', 1.72, .42, (7, 0, 5.86), stone[2])
segments, rings = 192, 145
vertices = []
for side in range(2):
    for row in range(rings+1):
        z = base + row/rings*body_height
        radius = tower_radius(z)
        for col in range(segments):
            a = col*math.tau/segments
            chip = noise.noise_vector(Vector((math.cos(a)*13, math.sin(a)*13, z*7)))[0]
            weather = noise.noise_vector(Vector((math.cos(a)*4, math.sin(a)*4, z*.8)))[0]
            relief = -.026*max(0, chip+.24)*max(0, weather+.4) + .004*chip
            r = radius + relief if side == 0 else radius-.24
            vertices.append((7+r*math.cos(a), r*math.sin(a), z))
faces = []
layer = segments*(rings+1)
for row in range(rings):
    for col in range(segments):
        i = row*segments+col; j = row*segments+(col+1)%segments
        faces.extend([(i,j,j+segments,i+segments), (layer+i+segments,layer+j+segments,layer+j,layer+i)])
for col in range(segments):
    j = (col+1)%segments; top = rings*segments
    faces.extend([(col,layer+col,layer+j,j), (top+j,layer+top+j,layer+top+col,top+col)])
shell = mesh('Whitewashed chipped masonry shell', vertices, faces, whitewash, True)
tower_uv(shell.data)

# Staggered deep windows follow the stair inside the tower; the entry is part
# of the tower masonry, not a detached keeper house.
openings=[(6.7,-1.15,.62,1.62),(9,-1.05,.34,.70),(10.7,-2.15,.34,.74),
          (12.55,-2.4,.32,.68),(12.55,-1.12,.32,.68)]
for z,a,width,height in openings:
    r=tower_radius(z); c,s=math.cos(a),math.sin(a); rotation=a-math.pi/2
    bpy.ops.mesh.primitive_cube_add(size=1,location=(7+r*c,r*s,z))
    cutter=bpy.context.object;cutter.scale=(width,.85,height);cutter.rotation_euler.z=rotation
    bpy.context.view_layer.objects.active=cutter;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    cut=shell.modifiers.new('Deep masonry opening','BOOLEAN');cut.operation='DIFFERENCE';cut.object=cutter
    bpy.context.view_layer.objects.active=shell;bpy.ops.object.modifier_apply(modifier=cut.name)
    bpy.data.objects.remove(cutter,do_unlink=True)
    bpy.ops.mesh.primitive_cube_add(size=1,location=(7+(r-.2)*c,(r-.2)*s,z))
    pane=bpy.context.object;pane.name='Salt-blackened entry' if z<7 else 'Deep narrow window'
    pane.scale=(width-.035,.07,height-.03);pane.rotation_euler.z=rotation;pane.data.materials.append(iron)
    for tangent,dz,w,h,label in [(-width/2-.035,0,.065,height+.16,'jamb'),
                                 (width/2+.035,0,.065,height+.16,'jamb'),
                                 (0,height/2+.045,width+.2,.13,'lintel'),
                                 (0,-height/2-.055,width+.23,.12,'drip sill')]:
        bpy.ops.mesh.primitive_cube_add(size=1,location=(7+(r+.035)*c-tangent*s,(r+.035)*s+tangent*c,z+dz))
        frame=bpy.context.object;frame.name='Stone opening '+label;frame.scale=(w,.18,h);frame.rotation_euler.z=rotation
        frame.data.materials.append(whitewash);bevel=frame.modifiers.new('Worn stone edge','BEVEL');bevel.width=.016;bevel.segments=2

# Projecting gabled doorway surround, anchored to the tower itself.
a=-1.15;c,s=math.cos(a),math.sin(a)
def entrance_point(t,r,z):return (7+r*c-t*s,r*s+t*c,z)
for t in [-.43,.43]:
    bpy.ops.mesh.primitive_cube_add(size=1,location=entrance_point(t,1.65,6.66))
    ob=bpy.context.object;ob.name='Stone entry pier';ob.scale=(.18,.56,1.68);ob.rotation_euler.z=a-math.pi/2;ob.data.materials.append(whitewash)
v=[entrance_point(t,r,z) for r in [1.36,1.96] for t,z in [(-.57,7.47),(.57,7.47),(0,7.87)]]
mesh('Stone entry gabled canopy',v,[(0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)],whitewash)
for i in range(3):
    bpy.ops.mesh.primitive_cube_add(size=1,location=entrance_point(0,1.8+i*.24,5.92-i*.07))
    ob=bpy.context.object;ob.name='Stone threshold step';ob.scale=(1.05,.34,.18);ob.rotation_euler.z=a-math.pi/2;ob.data.materials.append(stone[2])

# A substantial dark watch room between two distinct gallery levels.
for z,r,depth in [(6.04,1.65,.16),(14.24,1.18,.16),(14.43,1.48,.18)]:
    cylinder('Stone cornice and stringcourse',r,depth,(7,0,z),whitewash)
cylinder('Black iron watch room',1.10,.9,(7,0,14.99),iron,96)
cylinder('Lower gallery iron lip',1.49,.10,(7,0,14.55),iron)
cylinder('Upper lantern gallery deck',1.36,.14,(7,0,15.51),iron)
cylinder('Lantern plinth',.95,.20,(7,0,15.65),iron)
cylinder('Lantern glazing',.86,1.25,(7,0,16.32),glass,16)
cylinder('Roof eave',1.06,.12,(7,0,17.0),iron)
cylinder('Conical copper lantern roof',1.06,.68,(7,0,17.39),iron,64,.08)
cylinder('Roof finial ball',.11,.19,(7,0,17.81),iron,32,.06)
rod('Lightning conductor',(7,0,17.87),(7,0,18.4),.022,iron)
for i in range(16):
    a=i*math.tau/16;c,s=math.cos(a),math.sin(a)
    rod('Lantern glazing bar',(7+.87*c,.87*s,15.72),(7+.87*c,.87*s,16.96),.025,iron)
    for r,low,high in [(1.40,14.54,15.18),(1.28,15.57,16.09)]:
        rod('Gallery stanchion',(7+r*c,r*s,low),(7+r*c,r*s,high),.025,iron)
        cylinder('Gallery post base collar',.052,.07,(7+r*c,r*s,low+.03),iron,16)
    rod('Gallery radial corbel',(7+1.12*c,1.12*s,13.97),(7+1.43*c,1.43*s,14.38),.06,iron)
    rod('Watch room panel seam',(7+1.105*c,1.105*s,14.61),(7+1.105*c,1.105*s,15.41),.014,rust)
    rod('Standing seam roof rib',(7+1.03*c,1.03*s,17.07),(7+.10*c,.10*s,17.74),.022,iron)
for r,z in [(1.40,14.84),(1.40,15.18),(1.28,15.83),(1.28,16.09)]:
    bpy.ops.mesh.primitive_torus_add(major_radius=r,minor_radius=.022,major_segments=96,minor_segments=8,location=(7,0,z))
    bpy.context.object.name='Continuous iron gallery rail';bpy.context.object.data.materials.append(iron)
bulb=bpy.data.materials.new('Warm Fresnel lamp');bulb.use_nodes=True
bp=bulb.node_tree.nodes.get('Principled BSDF');bp.inputs['Emission Color'].default_value=(1,.4,.09,1);bp.inputs['Emission Strength'].default_value=3
cylinder('Fresnel lens',.16,.4,(7,0,16.3),bulb)
for z in [16.03+i*.055 for i in range(11)]:
    bpy.ops.mesh.primitive_torus_add(major_radius=.265,minor_radius=.022,major_segments=48,minor_segments=8,location=(7,0,z))
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
sky=n.new('ShaderNodeTexSky'); sky.sky_type='MULTIPLE_SCATTERING'; sky.sun_elevation=.27; sky.sun_rotation=-1.03; sky.aerosol_density=3
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
ld=bpy.data.lights.new('Late afternoon break in storm', 'SUN'); ld.energy=6; ld.angle=.055; ld.color=(1,.43,.13)
lo=bpy.data.objects.new(ld.name,ld); scene.collection.objects.link(lo); lo.rotation_euler=Vector((-.7,.67,-.25)).to_track_quat('-Z','Y').to_euler()
ld=bpy.data.lights.new('Soft sky fill', 'AREA'); ld.energy=1100; ld.shape='DISK'; ld.size=22; ld.color=(.55,.7,1)
lo=bpy.data.objects.new(ld.name,ld); scene.collection.objects.link(lo); lo.location=(-4,-3,20)

# Thin maritime haze gives distance a measurable optical depth.
mist=bpy.data.materials.new('Salt haze'); mist.use_nodes=True
mn=mist.node_tree.nodes; mn.clear(); mo=mn.new('ShaderNodeOutputMaterial'); mv=mn.new('ShaderNodeVolumeScatter')
mv.inputs['Color'].default_value=(.62,.71,.8,1); mv.inputs['Density'].default_value=.0012; mv.inputs['Anisotropy'].default_value=.28
mist.node_tree.links.new(mv.outputs[0],mo.inputs['Volume'])
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,20))
ob=bpy.context.object; ob.name='Maritime atmospheric haze'; ob.scale=(160,160,60); ob.data.materials.append(mist)

bpy.ops.object.camera_add(location=(-12.9,-28.5,5.16))
cam=bpy.context.object; cam.name='Storm photograph — 36 mm'; target=Vector((8,0,8.4))
cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler(); cam.data.lens=36
cam.data.dof.use_dof=False
scene.camera=cam
scene.render.filepath=str(args.output.resolve())
scene['dry_geometry_spacing_m']=.0625
scene['fluid_geometry_spacing_m']=.25
scene['reference']='Weathered white lighthouse on fractured rocks, low warm sunlight; no keeper house'
scene.cycles.device='GPU'
preferences=bpy.context.preferences.addons['cycles'].preferences
preferences.compute_device_type='METAL'
preferences.get_devices()
for device in preferences.devices: device.use=(device.type=='METAL')
bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix('.blend').resolve()))
bpy.ops.render.render(write_still=True)
args.output.with_suffix('.json').write_text(json.dumps({'scene':'atlantic-lighthouse','waterMesh':str(args.water_mesh.resolve()) if args.water_mesh else None,
    'waterMeshSha256':hashlib.sha256(args.water_mesh.read_bytes()).hexdigest() if args.water_mesh else None,
    'waterMethod':'DRY PREVIEW — NO WATER' if args.dry_preview else 'uniform-narrow-band-flip','foam':'procedural surface aeration shading; no secondary particle simulation',
    'samples':args.samples,'width':args.width,'height':args.height},indent=2)+'\n')
