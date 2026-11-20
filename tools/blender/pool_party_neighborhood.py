"""Author the editable residential setting for Poolparty. Blender coordinates
are converted from the scene's Y-up metres. Each module exports at its origin.
"""
from pathlib import Path
from math import atan2, cos, hypot, sin, exp
from mathutils import noise, Vector
import bpy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'public/assets/pool-neighborhood'
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
bpy.context.preferences.filepaths.save_version = 0


def material(name, rgb, roughness=.65, metallic=0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    shader = m.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*rgb, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metallic
    return m


lime = material('Limewashed warm white', (.67,.60,.48))
cedar = material('Honey cedar', (.29,.136,.060))
shadow = material('Oiled cedar end grain', (.125,.066,.035))
zinc = material('Graphite folded zinc', (.047,.058,.063), .51, .34)
steel = material('Powder coated frames', (.032,.043,.044), .34, .4)
glass = material('Blue grey smoked pane', (.048,.11,.13), .12, .45)
stone = material('Sawn limestone', (.47,.40,.31), .73)
linen = material('Natural linen', (.68,.65,.53), .9)
tile = material('Red clay roof', (.31,.098,.048), .8)
ridge = material('Weathered clay ridges', (.35,.126,.065), .85)
plaster = material('Sage painted render', (.37,.41,.31))
grass = material('Mown fescue', (.083,.145,.042), .97)
soil = material('Earth', (.09,.057,.028), 1)


def collection(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c


def box(c, name, center, size, mat, pitch=0):
    x, y, z = center
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x,-z,y))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = (size[0], size[2], size[1])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.rotation_euler.x = pitch
    obj.data.materials.append(mat)
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    c.objects.link(obj)
    return obj


def roof_half(c, name, x, y1, y2, z1, z2, width, mat, thickness=.18):
    pitch = -atan2(y2-y1,z2-z1)
    return box(c,name,(x,(y1+y2)/2,(z1+z2)/2),
               (width,thickness,hypot(z2-z1,y2-y1)),mat,pitch)


def export(c):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in c.objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = c.objects[0]
    bpy.ops.export_scene.gltf(filepath=str(OUT/(c.name+'.glb')), use_selection=True, export_yup=True)


p = collection('garden-pavilion')
# A real room behind the open sliders. The eave projects past the threshold.
box(p,'Limestone floor',(0,.015,-2.65),(10.9,.23,6.65),stone)
box(p,'West return',(-5.3,1.51,-2.9),(.23,3.0,5.85),lime)
box(p,'East return',(5.3,1.51,-2.9),(.23,3.0,5.85),lime)
interior = material('Warm clay kitchen plaster', (.29,.24,.17))
box(p,'Rear wall',(0,1.91,-5.72),(10.7,3.84,.22),interior)
for x in [-5.3,5.3]:
    # Close the high end of the cheek wall under the sloping roof.
    roof_half(p,'Raked side fascia',x,3.57,2.94,-5.84,.10,.23,lime,.58)
box(p,'Walnut display shelf',(-1.8,2.27,-5.35),(5.48,.075,.52),shadow)
ceramic = material('Green glazed stoneware',(.12,.22,.17),.28)
for i in range(5):
    x=-3.7+i*.23
    box(p,'Upright cookbook',(x,2.52,-5.40),(.14,.42+.055*sin(i*1.8),.28),cedar if i%2 else linen)
for x,r,h in [(-1.2,.18,.33),(-.70,.11,.27),(.05,.15,.37)]:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=10,location=(x,5.4,2.30+h/2))
    o=bpy.context.object;o.name='Stoneware on kitchen shelf';o.scale=(r,r,h/2);o.data.materials.append(ceramic)
    for coll in list(o.users_collection):coll.objects.unlink(o)
    p.objects.link(o)
for x, width in [(-4.62,1.6),(.9,1.0),(4.86,.9)]:
    box(p,'Street-facing pier',(x,1.45,0),(width,2.9,.24),lime)
box(p,'Lintel above sliders',(0,2.94,.01),(10.8,.44,.31),lime)
for x in [-3.76,-1.75,.34,1.48,4.30]:
    box(p,'Recessed slider jamb',(x,1.42,-.06),(.115,2.68,.17),steel)
for i in range(10):
    box(p,'Cedar privacy screen',(-3.69+i*.192,1.41,-.10),(.109,2.60,.14),cedar)
for y in [.17,2.64]:
    box(p,'Privacy screen rail',(-2.82,y,-.12),(1.95,.12,.15),shadow)
for x in [-3.78,-1.72,1.53,4.29]:
    box(p,'Flush threshold',(x+.04,.14,.08),(.12,.12,.24),steel)
box(p,'Kitchen island',(.1,.52,-3.14),(3.5,.98,.84),cedar)
box(p,'Honed kitchen counter',(.1,1.055,-3.14),(3.72,.115,.98),stone)
box(p,'Deep cabinet bank',(-1.8,.63,-5.24),(5.35,1.2,.67),shadow)
box(p,'Kitchen splashback',(-1.8,1.58,-5.55),(5.35,.64,.12),linen)
for x in [-1.7,-.65]:
    box(p,'Oak stool seat',(x,.62,-2.25),(.53,.11,.45),cedar)
    for leg in [-.17,.17]:
        box(p,'Stool side',(x+leg,.32,-2.25),(.09,.56,.35),shadow)
for i in range(11):
    box(p,'Folded linen curtain',(3.12+i*.106,1.4,-.18+sin(i*1.8)*.08),(.125,2.53,.12),linen)
roof_half(p,'Deep zinc shed roof',0,3.96,3.21,-6.22,1.35,11.6,zinc)
roof_half(p,'Cedar soffit',0,3.80,3.05,-6.20,1.31,11.53,cedar)
for i in range(20):
    roof_half(p,'Standing seam',-5.49+i*.58,4.075,3.325,-6.22,1.35,.065,zinc,.066)
for x in [-5.2,.94,5.2]:
    box(p,'Veranda column',(x,1.53,1.08),(.15,3.05,.15),shadow)
box(p,'Front gutter',(0,3.19,1.4),(11.68,.16,.16),zinc)
for x in [-5.28,5.28]:
    box(p,'Rainwater leader',(x,1.48,.23),(.12,2.86,.13),zinc)
export(p)

c = collection('neighbor-cottage')
box(c,'Rendered house',(0,1.82,0),(7.6,3.65,6.25),plaster)
for side in [-1,1]:
    roof_half(c,'Clay pitched roof',0,4.98,3.53,0,side*3.62,8.4,tile)
    for i in range(1,10):
        t = i/10
        box(c,'Overlapping tile courses',(0,5.08-1.45*t,side*3.62*t),(8.4,.12,.15),ridge)
for x in [-2.45,0,2.45]:
    for side in [-1,1]:
        box(c,'Timber window casing',(x,1.94,side*3.18),(1.57,1.51,.16),linen)
        box(c,'Recessed glazing',(x,1.94,side*3.28),(1.32,1.27,.14),glass)
        box(c,'Central sash',(x,1.94,side*3.37),(.11,1.32,.12),linen)
# The roof encloses a loft. Both gable ends face a neighboring garden.
for side in [-1,1]:
    end = side*3.78
    section=[(-3.12,3.55),(3.12,3.55),(0,4.91)]
    vertices=[(end+side*d,-z,h) for d in [-.12,.10] for z,h in section]
    faces=[(0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)]
    if side==1: faces=[tuple(reversed(f)) for f in faces]
    mesh=bpy.data.meshes.new('Plastered gable end')
    mesh.from_pydata(vertices,[],faces);mesh.materials.append(plaster);mesh.update()
    c.objects.link(bpy.data.objects.new('Plastered gable end',mesh))
    for z in [-1.62,1.40]:
        box(c,'Garden-facing window casing',(side*3.84,1.85,z),(.22,1.60,1.49),linen)
        box(c,'Garden-facing sash glazing',(side*3.95,1.85,z),(.17,1.35,1.25),glass)
        box(c,'Garden-facing mullion',(side*4.04,1.85,z),(.12,1.41,.10),linen)
        box(c,'Garden-facing stone sill',(side*3.96,1.03,z),(.39,.13,1.65),stone)
    for eave in [-1,1]:
        roof_half(c,'Painted bargeboard',side*4.12,4.96,3.51,0,eave*3.60,.14,linen,.23)
    box(c,'Masonry footing',(side*3.83,.16,0),(.18,.32,6.29),stone)
box(c,'Garden door frame',(.98,1.04,3.18),(1.03,2.07,.21),linen)
box(c,'Timber garden door',(.98,1.04,3.30),(.83,1.90,.13),cedar)
box(c,'Glazed door light',(.98,1.60,3.39),(.55,.59,.08),glass)
box(c,'Clay chimney stack',(2.48,4.7,-.68),(.79,2.1,.85),tile)
box(c,'Chimney coping',(2.48,5.8,-.68),(.98,.20,1.01),stone)
export(c)


# Two neighbors built for their individual plots. The lane house has a wide
# hipped roof and a loggia. The uphill saltbox has a much steeper silhouette.
def roof_solid(c, name, ring, summit, sides, mat):
    vertices=[(x,-z,y) for x,y,z in ring+summit]
    polygons=[tuple(reversed(range(4))), *sides]
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(vertices,[],polygons);mesh.materials.append(mat);mesh.update()
    c.objects.link(bpy.data.objects.new(name,mesh))

def garden_window(c, x, y, z, width, height, facing, trim):
    box(c,'Window casing',(x,y,z),(width+.22,height+.21,.16),trim)
    box(c,'Recessed glass',(x,y,z+facing*.093),(width,height,.14),glass)
    box(c,'Mid stile',(x,y,z+facing*.18),(.11,height+.02,.10),trim)
    box(c,'Drip sill',(x,y-height/2-.10,z+facing*.08),(width+.32,.13,.33),stone)

bungalow=collection('lane-bungalow')
brick=material('Oxide toned brick',(.28,.162,.091),.9)
slate=material('Patinated slate blue metal',(.068,.098,.111),.61,.25)
fascia=material('Oyster painted trim',(.56,.59,.53))
box(bungalow,'Brick plinth',(0,.06,0),(9.4,.78,5.7),brick)
box(bungalow,'Low rendered dwelling',(0,1.56,0),(9.28,2.78,5.62),lime)
# CCW in Blender XY for the top of the roof.
roof_solid(bungalow,'Hipped slate roof',
    [(-5.12,2.99,3.23),(5.12,2.99,3.23),(5.12,2.99,-3.23),(-5.12,2.99,-3.23)],
    [(-2.18,4.52,0),(2.18,4.52,0)],
    [(0,1,5,4),(1,2,5),(2,3,4,5),(3,0,4)],slate)
box(bungalow,'Ridge flashing',(0,4.54,0),(4.48,.14,.19),zinc)
for side in [-1,1]:
    box(bungalow,'Eave fascia',(0,2.93,side*3.22),(10.27,.19,.15),fascia)
    box(bungalow,'End fascia',(side*5.08,2.93,0),(.16,.19,6.46),fascia)
for x in [-2.86,.57,3.0]:
    garden_window(bungalow,x,1.72,-2.85,2.08 if x<0 else 1.48,1.23,-1,fascia)
garden_window(bungalow,-2.68,1.63,2.85,2.59,1.63,1,fascia)
garden_window(bungalow,1.43,1.72,2.85,1.79,1.23,1,fascia)
box(bungalow,'Entry surround',(-.27,1.35,2.89),(1.18,2.21,.21),cedar)
box(bungalow,'Porch door',(-.27,1.33,3.03),(.92,2.01,.115),shadow)
box(bungalow,'Loggia slab',(-1.76,.18,3.67),(5.85,.34,1.96),stone)
roof_half(bungalow,'Veranda awning',-1.76,2.98,2.66,2.73,4.78,5.95,slate,.18)
for x in [-4.51,.94]:
    box(bungalow,'Veranda post',(x,1.49,4.52),(.18,2.42,.18),cedar)
box(bungalow,'Brick chimney',(-2.84,4.04,-.72),(.92,2.12,1.10),brick)
box(bungalow,'Stone chimney cap',(-2.84,5.14,-.72),(1.08,.19,1.23),stone)
export(bungalow)

saltbox=collection('hill-saltbox')
boards=material('Weathered blue board and batten',(.137,.211,.223),.83)
battens=material('Sunward edge of painted boards',(.17,.246,.25),.78)
tin=material('Weathered pale galvanised steel',(.38,.402,.389),.57,.38)
box(saltbox,'Basalt footing',(0,.12,0),(7.66,.96,6.44),stone)
box(saltbox,'Boarded ground floor',(0,1.72,0),(7.54,2.91,6.32),boards)
# Unequal pitches make the small front loft quite different from the far slope.
roof_solid(saltbox,'Enclosed loft',
    [(-3.77,3.1,3.16),(3.77,3.1,3.16),(3.77,3.1,-3.16),(-3.77,3.1,-3.16)],
    [(-3.77,5.42,1.04),(3.77,5.42,1.04)],
    [(0,1,5,4),(1,2,5),(2,3,4,5),(3,0,4)],boards)
for z,y in [(3.59,2.85),(-3.60,2.94)]:
    roof_half(saltbox,'Long-run metal roof',0,5.60,y,1.04,z,8.30,tin)
    for k in range(13):
        roof_half(saltbox,'Raised folded seam',-3.89+k*.65,5.71,y+.11,1.04,z,.072,tin,.065)
for z,side in [(3.18,1),(-3.18,-1)]:
    for k in range(18):
        box(saltbox,'Vertical cladding batten',(-3.56+.42*k,1.80,z),(.079,2.59,.058),battens)
    for x in [-2.4,2.23]:
        garden_window(saltbox,x,1.88,z+side*.08,1.39,1.55,side,linen)
box(saltbox,'Entry door surround',(-.33,1.44,3.29),(1.25,2.30,.22),linen)
box(saltbox,'Mustard painted front door',(-.33,1.40,3.43),(.97,2.10,.13),cedar)
box(saltbox,'Four-light transom',(-.33,2.12,3.53),(.70,.37,.10),glass)
box(saltbox,'Front porch landing',(-.38,.30,3.85),(2.68,.45,1.48),stone)
box(saltbox,'Garden step',(-.38,.14,4.74),(2.52,.26,.61),stone)
roof_half(saltbox,'Door canopy',-.38,3.06,2.75,3.08,4.65,2.92,zinc)
for x in [-1.65,.88]:
    box(saltbox,'Porch column',(x,1.63,4.45),(.145,2.18,.145),linen)
# Broad gable windows read from the pool-side approach as well as the lane.
for side in [-1,1]:
    for z,y,w,h in [(-1.31,1.95,1.56,1.36),(1.16,1.89,1.45,1.45),(.86,4.00,.86,.82)]:
        box(saltbox,'Gable window surround',(side*3.82,y,z),(.18,h+.19,w+.20),linen)
        box(saltbox,'Gable window pane',(side*3.94,y,z),(.15,h,w),glass)
        box(saltbox,'Gable window sill',(side*4.00,y-h/2-.08,z),(.33,.14,w+.33),stone)
box(saltbox,'Chimney',(-2.25,5.09,-.51),(.77,2.1,.82),brick)
box(saltbox,'Chimney cowl',(-2.25,6.20,-.51),(.91,.19,.97),stone)
export(saltbox)

# Map materials across the whole site in metres. The colour field belongs to
# the terrain UVs, so moving the camera never slides a track across a slope.
def landscape_material(name, extent, hills):
    mat = material(name, (.26,.32,.17), .97)
    n = 384
    im = bpy.data.images.new(name, width=n, height=n, alpha=False)
    im.colorspace_settings.name = 'sRGB'
    pixels = []
    for j in range(n):
        z = (.5 - (j + .5) / n) * extent
        for i in range(n):
            x = ((i + .5) / n - .5) * extent
            wide = noise.fractal(Vector((x*.10,z*.10,.84)), 1.2, 2.03, 4)
            fine = noise.noise_vector(Vector((x*.71,z*.71,.45))).x
            if hills:
                # Mixed pasture, copses and open ground follow the slope.
                t = min(1,max(0,.50 + .85*wide + .18*fine))
                col = (.25+.29*t, .32+.24*t, .24+.115*t)
            else:
                t = min(1,max(0,.53+.78*wide+.11*fine))
                col = (.30+.16*t, .38+.14*t, .23+.085*t)
                # Gravel lane beyond the fences, with two garden drives.
                d = min(abs(z+20.8-1.5*sin(x*.10))-.88,
                        abs(x-22.2-.8*sin(z*.13))-.82,
                        max(abs(x-3.2)-.77,abs(z+17.4)-3.7),
                        max(abs(z-1.8)-.76,abs(x-19.6)-2.7))
                mix = min(1,max(0,.42-(d+fine*.09)*4))
                col = tuple(v*(1-mix)+(c+.027*fine)*mix for v,c in zip(col,(.63,.59,.50)))
            pixels.extend((*col,1))
    im.pixels.foreach_set(pixels);im.pack()
    tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=im
    mat.node_tree.links.new(tex.outputs['Color'],mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    return mat

def ground_uv(mesh, extent):
    uv=mesh.uv_layers.new(name='Site metres')
    for loop in mesh.loops:
        co=mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv=(.5+co.x/extent,.5+co.y/extent)

g = collection('neighborhood-ground')
# Concentric rectangular contours leave the exact pool domain open. The first
# two rings overlap turf from below, then undulate into the neighboring plots.
rings = [(7.9,5.9),(10,8),(13,11),(18,16),(24,22),(32,30)]
verts, faces = [], []
for rx, rz in rings:
    for side in range(4):
        for step in range(16):
            t = step/8-1
            x,z = [(rx*t,-rz),(rx,rz*t),(-rx*t,rz),(-rx,-rz*t)][side]
            amount = max(0,min(1,(max(abs(x)-10,abs(z)-8))/10))
            y = -.09 + amount*(.40*sin(x*.16)+.36*cos(z*.27)+.08*z)
            verts.append((x,-z,y))
count = len(verts)
verts += [(x,y,z-.83) for x,y,z in verts]
for ring in range(len(rings)-1):
    for i in range(64):
        a=ring*64+i; b=ring*64+(i+1)%64
        faces.extend([(a,b,b+64,a+64),(count+a+64,count+b+64,count+b,count+a)])
for i in range(64):
    j=(i+1)%64
    for start in [0,64*(len(rings)-1)]:
        a,b=start+i,start+j
        faces.append((a,count+a,count+b,b) if start==0 else (b,count+b,count+a,a))
mesh=bpy.data.meshes.new('Sloping residential lawns')
mesh.from_pydata(verts,[],faces)
mesh.materials.append(landscape_material('Residential lawns and gravel lane', 68, False))
mesh.materials.append(soil)
ground_uv(mesh,68)
mesh.update()
for i,poly in enumerate(mesh.polygons):
    poly.material_index=0 if i < 2*(len(rings)-1)*64 and i%2==0 else 1
obj=bpy.data.objects.new('Continuous ground around the pool',mesh)
g.objects.link(obj)
export(g)
# Low wooded ridges close the long views beyond the surrounding lots. Their
# inner ring follows the local topography, with all relief in the geometry.
r = collection('wooded-horizon')
hill_rings = [(28,26),(32,30),(35,33),(38,36),(41,39),(44,42),(47,45)]
vertices, polygons = [], []
for k,(rx,rz) in enumerate(hill_rings):
    for side in range(4):
        for step in range(32):
            t=step/16-1
            x,z=[(rx*t,-rz),(rx,rz*t),(-rx*t,rz),(-rx,-rz*t)][side]
            theta=atan2(z,x)
            near=-1.62+.4*sin(x*.16)+.36*cos(z*.27)+.08*z
            shift=(.56*sin(x*.09+z*.03)+.62*cos(z*.08-x*.05))
            fade=min(1,k*.26)
            # Unequal summits and the low gap beyond the north-east garden.
            height=near+fade*(1.5+2.0*sin(theta*2-.4)+1.2*sin(theta*5+.6)+
                    4.1*exp(-((x+24)**2+(z-29)**2)/330)+
                    2.5*exp(-((x-18)**2+(z+36)**2)/240)+shift)
            height+=fade*.78*noise.fractal(Vector((x*.12,z*.12,1.42)),1.2,2.02,3)
            vertices.append((x,-z,height))
n=len(vertices)
vertices += [(x,y,z-3.2) for x,y,z in vertices]
for k in range(len(hill_rings)-1):
    for i in range(128):
        a=k*128+i;b=k*128+(i+1)%128
        polygons.extend([(a,b,b+128,a+128),(n+a+128,n+b+128,n+b,n+a)])
for i in range(128):
    for start in [0,128*(len(hill_rings)-1)]:
        a=start+i;b=start+(i+1)%128
        polygons.append((a,n+a,n+b,b) if start==0 else (b,n+b,n+a,a))
mesh=bpy.data.meshes.new('Wooded foothills');mesh.from_pydata(vertices,[],polygons)
mesh.materials.append(landscape_material('Pasture and wooded slopes',100,True))
ground_uv(mesh,100)
for face in mesh.polygons: face.use_smooth=True
mesh.update();r.objects.link(bpy.data.objects.new('Rolling farmland',mesh));export(r)

# One closed terrain carries the lawns, approaches, foothills and the far
# skyline. A continuous top and soil underside prevent an overlap from
# casting an artificial dark line at the old property and field boundaries.
far=collection('valley-farmland')
rings=[(7.9,5.9),(10,8),(13,11),(18,16),(24,22),(28,26),(32,30),
       (35,33),(38,36),(41,39),(44,42),(47,45),(51,49),(55,53),
       (60,58),(65,63),(72,69),(78,75)]
def smooth(lo,hi,v):
    t=max(0,min(1,(v-lo)/(hi-lo)))
    return t*t*(3-2*t)
vertices,faces=[],[]
for rx,rz in rings:
    for side in range(4):
        for step in range(40):
            t=step/20-1
            x,z=[(rx*t,-rz),(rx,rz*t),(-rx*t,rz),(-rx,-rz*t)][side]
            theta=atan2(z,x)
            relief=.4*sin(x*.16)+.36*cos(z*.27)+.08*z
            amount=max(0,min(1,max(abs(x)-10,abs(z)-8)/10))
            lawn=-.09+amount*relief
            hill=(-1.62+relief+1.5+2*sin(theta*2-.4)+1.2*sin(theta*5+.6)
                  +4.1*exp(-((x+24)**2+(z-29)**2)/330)+2.5*exp(-((x-18)**2+(z+36)**2)/240)
                  +.56*sin(x*.09+z*.03)+.62*cos(z*.08-x*.05)
                  +.78*noise.fractal(Vector((x*.12,z*.12,1.42)),1.2,2.02,3))
            crest=(5.8+3.4*sin(theta*3-.5)+1.8*cos(theta*7+.4)+.11*(rx-47)
                   +1.8*sin(x*.073+z*.046)+1.3*cos(z*.087-x*.031)
                   +.8*noise.fractal(Vector((x*.05,z*.05,4.14)),1,2,3))
            y=lawn+(hill-lawn)*smooth(28,41,rx)
            y+=(crest-y)*smooth(47,60,rx)
            if rx==78:y-=4.7
            vertices.append((x,-z,y))
n=len(vertices);vertices += [(x,y,z-4.2) for x,y,z in vertices]
for k in range(len(rings)-1):
    for i in range(160):
        a=160*k+i;b=160*k+(i+1)%160
        faces.extend([(a,b,b+160,a+160),(n+a+160,n+b+160,n+b,n+a)])
for i in range(160):
    for start in [0,160*(len(rings)-1)]:
        a=start+i;b=start+(i+1)%160
        faces.append((a,n+a,n+b,b) if start==0 else (b,n+b,n+a,a))
mesh=bpy.data.meshes.new('Residential lawns to open farmland')
mesh.from_pydata(vertices,[],faces)
pasture=material('Garden grass, gravel and pasture',(.20,.28,.14),1)
size=640; extent=160
im=bpy.data.images.new('Valley land use',width=size,height=size,alpha=False)
im.colorspace_settings.name='sRGB';pixels=[]
for j in range(size):
    for i in range(size):
        x=((i+.5)/size-.5)*extent;z=(.5-(j+.5)/size)*extent
        u=(x+.26*z+2.7*sin(z*.065))*.057
        v=(z-.2*x+2.1*cos(x*.086))*.072
        fine=noise.noise_vector(Vector((x*.71,z*.71,.45))).x
        broad=noise.fractal(Vector((x*.094,z*.094,2.84)),1.1,2,3)
        field=.5+.5*sin(12.23*(u//1)+25.19*(v//1))
        hedge=max(0,1-10*min(abs(u-round(u)),abs(v-round(v)))-.18*fine)
        distance=max(abs(x),abs(z)+2)
        t=smooth(24,38,distance)
        grass_tone=.5+.65*broad+.13*fine
        lawn=(.27+.13*grass_tone,.35+.13*grass_tone,.19+.08*grass_tone)
        crop=(.32+.13*field+.080*broad,.41+.085*field+.07*broad,.25+.045*field+.045*broad)
        col=[a*(1-t)+b*(1-.28*hedge)*t for a,b in zip(lawn,crop)]
        lane=min(abs(z+20.8-1.5*sin(x*.10))-.88,abs(x-22.2-.8*sin(z*.13))-.82,
                 max(abs(x-3.2)-.77,abs(z+17.4)-3.7),max(abs(z-1.8)-.76,abs(x-19.6)-2.7))
        gravel=(1-smooth(26,39,distance))*(1-smooth(-.18,.16,lane+.07*fine))
        pixels.extend((*[a*(1-gravel)+(b+.027*fine)*gravel for a,b in zip(col,(.63,.59,.50))],1))
im.pixels.foreach_set(pixels);im.pack()
tex=pasture.node_tree.nodes.new('ShaderNodeTexImage');tex.image=im
pasture.node_tree.links.new(tex.outputs['Color'],pasture.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
mesh.materials.append(pasture);ground_uv(mesh,extent);mesh.update()
for face in mesh.polygons:face.use_smooth=True
far.objects.link(bpy.data.objects.new('Continuous valley landscape',mesh));export(far)


# Lance-shaped grey-green leaves and flower whorls around a woody crown.
import random
from mathutils import Vector
rand=random.Random(8147)
lav=collection('lavender-border')
leaf_mats=[material('Sage leaf',(.19,.26,.14)),material('Leaf shade',(.086,.145,.076)),
           material('Leaf light',(.24,.31,.19)),material('Woody stem',(.17,.145,.08)),
           material('Lavender spikes',(.33,.20,.41)), material('Faded lavender',(.43,.32,.46))]
vtx,idx,mi=[],[],[]
def shoot(a,b,r,mat):
    axis=(b-a).normalized();side=axis.cross(Vector((0,0,1)))
    if side.length<.001:side=Vector((1,0,0))
    side.normalize();other=axis.cross(side);start=len(vtx)
    for t,rad in [(0,.24*r),(.32,r),(.78,.78*r),(1,.13*r)]:
        for k in range(5):
            angle=k*1.256637
            vtx.append(a.lerp(b,t)+rad*(cos(angle)*side+sin(angle)*other))
    for ring in range(3):
        for k in range(5):
            m=start+ring*5;idx.append((m+k,m+(k+1)%5,m+5+(k+1)%5,m+5+k));mi.append(mat)
    idx.extend([tuple(range(start+4,start-1,-1)),tuple(range(start+15,start+20))]);mi.extend([mat,mat])
for i in range(152):
    ang=rand.random()*6.283;radius=.42*rand.random()**.5;u=Vector((cos(ang),sin(ang),0))
    a=u*radius*.20;b=u*radius+Vector((0,0,.30+.27*(1-radius/.44)+rand.random()*.17))
    shoot(a,b,.012,3)
    for k in range(2,8):
        joint=a.lerp(b,k*.096)
        for sign in (-1,1):
            y=ang+sign*(.5+rand.random()*.9);v=Vector((cos(y),sin(y),.48))
            shoot(joint,joint+v*(.10+rand.random()*.086),.018+rand.random()*.012,rand.randrange(3))
    if i%3==0:
        tip=b+u*.035+Vector((0,0,.09));shoot(b,tip,.022,4+i%2)
mesh=bpy.data.meshes.new('Leaves and blooms');mesh.from_pydata(vtx,[],idx)
for m in leaf_mats:mesh.materials.append(m)
for f,k in zip(mesh.polygons,mi):f.material_index=k;f.use_smooth=True
mesh.update();lav.objects.link(bpy.data.objects.new('Lavender shrub',mesh));export(lav)

# A high-resolution native cable and lamp assembly keeps the wire at its
# measured diameter even when the player chooses a wider fluid cell.
strand = collection('terrace-festoon')
rubber=material('Braided outdoor cable',(.032,.029,.024),.82)
socket=material('Dark brass lampholder',(.145,.098,.058),.48,.32)
bulb=material('Daylight opal glass',(.81,.78,.68),.26,.03)
def link_piece(obj,coll,mat):
    for group in list(obj.users_collection):group.objects.unlink(obj)
    coll.objects.link(obj);obj.data.materials.append(mat)
for x in [-7.5,7.5]:
    box(strand,'Cedar festoon support',(x,1.59,0),(.093,3.18,.093),cedar)
wire=bpy.data.curves.new('Cable catenary','CURVE');wire.dimensions='3D';wire.resolution_u=2;wire.bevel_depth=.011;wire.bevel_resolution=2;wire.use_fill_caps=True
spline=wire.splines.new('POLY');spline.points.add(72)
for i,pnt in enumerate(spline.points):
    t=i/72;pnt.co=(-7.5+15*t,0,3.12-.58*4*t*(1-t),1)
o=bpy.data.objects.new('Outdoor lighting flex',wire);strand.objects.link(o);wire.materials.append(rubber)
bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o;bpy.ops.object.convert(target='MESH')
for i in range(1,15):
    t=i/15;x=-7.5+15*t;y=3.12-.58*4*t*(1-t)
    bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=.023,depth=.064,location=(x,0,y-.032))
    link_piece(bpy.context.object,strand,socket)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=14,ring_count=9,radius=1,location=(x,0,y-.093))
    o=bpy.context.object;o.scale=(.039,.039,.056);link_piece(o,strand,bulb)
export(strand)

# Store an editable source with the modules laid out separately.
for obj in p.objects: obj.location.x -= 14
for obj in c.objects: obj.location.x += 15
for obj in bungalow.objects: obj.location.y -= 15
for obj in saltbox.objects: obj.location.y += 15
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'assets/pool-neighborhood/neighborhood.blend'),compress=True)
