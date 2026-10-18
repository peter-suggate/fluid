"""Build the complete X10 porcelain garden in Blender. Run export-hero-layout.ts first.
Every visible solid is exported as a watertight GLB; instances share FVX volumes.
The saved presentation includes a water surface only to preview the live fluid.
"""
import argparse, json, math, random, sys
from pathlib import Path
import bpy, bmesh
import numpy as np
from mathutils import Vector, Quaternion
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'assets/hero-garden-blender'; OUT = ROOT/'public/assets/hero-garden-blender'
L = json.loads((SOURCE/'layout.json').read_text())
p = argparse.ArgumentParser(); p.add_argument('--no-render', action='store_true'); p.add_argument('--samples', type=int, default=48)
args = p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene; scene.unit_settings.system = 'METRIC'
rng = random.Random(20261018)
assets = {}; instances = []; active = None

def xyz(p): return (p[0], -p[2], p[1])
def mat(name, rgb, rough=.5, metal=0):
    m=bpy.data.materials.new(name); m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value=(*rgb,1); bs.inputs['Roughness'].default_value=rough; bs.inputs['Metallic'].default_value=metal
    return m
clay=mat('Sand glazed porcelain basin',(.58,.33,.14),.27)
# Linear RGB palettes: varied living foliage, warm mineral ground and brass.
leafm=[mat('Garden leaf '+str(i),rgb,.38+i*.045) for i,rgb in enumerate([
    (.045,.16,.025),(.085,.26,.035),(.16,.34,.055),(.065,.22,.09),(.25,.39,.085)])]
bark=mat('Warm brown textured bark',(.19,.082,.032),.78)
stone=[mat('Honed limestone '+str(i),rgb,.56+i*.035) for i,rgb in enumerate([
    (.38,.31,.22),(.52,.42,.29),(.64,.55,.40),(.40,.43,.38),(.57,.48,.36)])]
groundmat=mat('Warm earth garden soil',(.28,.17,.085),.88)
copingmat=mat('Honey sandstone coping',(.58,.33,.14),.62)
rubber=mat('Forest green braided garden hose',(.012,.075,.035),.67)
brass=mat('Warm satin brass nozzle',(.64,.38,.115),.28,.85)
sagem=[mat('Silver sage leaf '+str(i),rgb,.48) for i,rgb in enumerate([
    (.16,.31,.24),(.24,.40,.31),(.29,.43,.33),(.18,.34,.28),(.36,.49,.35)])]

def collection(name,cell,band=3):
    global active
    name='hero-'+name
    active=bpy.data.collections.new(name); scene.collection.children.link(active)
    assets[name]={'collection':active,'cell':cell,'band':band}
    return name

def finish(o,name,m,smooth=True):
    o.name=name
    for c in list(o.users_collection): c.objects.unlink(o)
    active.objects.link(o); o.data.materials.append(m)
    if o.type=='MESH':
        for f in o.data.polygons: f.use_smooth=smooth
    return o

def mesh(name,verts,faces,m,closed=True):
    d=bpy.data.meshes.new(name); d.from_pydata([xyz(v) for v in verts],[],faces); d.update()
    if closed:
        bm=bmesh.new();bm.from_mesh(d);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(d);bm.free()
    o=bpy.data.objects.new(name,d); active.objects.link(o); d.materials.append(m)
    for f in d.polygons:f.use_smooth=True
    return o

def box(name,pos,size,m,bevel=.002):
    bpy.ops.mesh.primitive_cube_add(size=1,location=xyz(pos));o=bpy.context.object;o.scale=(size[0],size[2],size[1]);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    finish(o,name,m,False)
    if bevel:
        mod=o.modifiers.new('Hand eased edge','BEVEL');mod.width=bevel;mod.segments=3
        mod=o.modifiers.new('Corner normals','WEIGHTED_NORMAL');mod.keep_sharp=True
    return o

def ellipsoid(name,pos,scale,m,seed=0):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,location=xyz(pos));o=bpy.context.object
    for v in o.data.vertices:
        q=v.co; f=1+.035*math.sin(q.x*7+seed)*math.sin(q.y*9+q.z*5)
        q*=f
    o.scale=(scale[0],scale[2],scale[1]);return finish(o,name,m)

def tube(name,points,radii,m,sides=10):
    # Parallel frames and capped rings keep each branch closed, including roots.
    verts=[];faces=[]
    for i,point in enumerate(points):
        tangent=Vector(points[min(i+1,len(points)-1)])-Vector(points[max(i-1,0)]);tangent.normalize()
        side=tangent.cross(Vector((0,1,0)))
        if side.length<.01:side=tangent.cross(Vector((1,0,0)))
        side.normalize();up=tangent.cross(side).normalized()
        for j in range(sides):
            angle=2*math.pi*j/sides; r=radii[i]*(1+.045*math.sin(j*5+i*.7))
            verts.append(tuple(Vector(point)+r*(math.cos(angle)*side+math.sin(angle)*up)))
    for i in range(len(points)-1):
        for j in range(sides):a=i*sides+j;b=i*sides+(j+1)%sides;faces.append((a,b,b+sides,a+sides))
    faces.extend([tuple(reversed(range(sides))),tuple((len(points)-1)*sides+j for j in range(sides))])
    return mesh(name,verts,faces,m)

def leaf(name,start,tip,width,m):
    # Solid, folded lanceolate leaf: thin enough to read botanically, thick enough
    # to survive a 1.56 mm source field rather than disappearing as an alpha card.
    a,b=Vector(start),Vector(tip);axis=b-a;side=axis.cross(Vector((0,1,0)))
    if side.length<1e-5:side=Vector((1,0,0))
    side.normalize();normal=side.cross(axis).normalized();verts=[]
    for sign in [-1,1]:
        for t,w in [(0,.08),(.3,.82),(.62,1),(.88,.5),(1,.035)]:
            center=a+axis*t+Vector((0,axis.length*.18*math.sin(math.pi*t),0))
            for across in [-1,0,1]:
                verts.append(tuple(center+side*width*w*across+normal*(sign*(.0035 if width>=.008 else .0018 if width<.003 else .0025)+abs(across)*width*.12)))
    faces=[]
    for layer in [0,15]:
        for i in range(4):
            for j in range(2):
                q=layer+i*3+j;f=(q,q+1,q+4,q+3);faces.append(f if layer else tuple(reversed(f)))
    border=[0,1,2,5,8,11,14,13,12,9,6,3]
    for a,b in zip(border,border[1:]+border[:1]):faces.append((a,b,b+15,a+15))
    return mesh(name,verts,faces,m)

def instance(asset,pos=(0,0,0),yaw=0,id=None):
    instances.append({'id':id or asset+'/'+str(len(instances)),'asset':asset,'position':list(pos),'yaw':yaw})

def height(x,z):
    t=L['terrain'];i=min(t['nx']-1,max(0,round((x+.9)/t['spacing'])));j=min(t['nz']-1,max(0,round((z+.6)/t['spacing'])))
    return t['heights'][j][i]

def outside_coping(x,z):
    inside=False;distance=1e10;curve=L['curve']
    for a,b in zip(curve,curve[1:]+curve[:1]):
        dx,dz=b[0]-a[0],b[1]-a[1];den=dx*dx+dz*dz
        t=max(0,min(1,((x-a[0])*dx+(z-a[1])*dz)/den)) if den else 0
        distance=min(distance,math.hypot(x-a[0]-t*dx,z-a[1]-t*dz))
        if (a[1]>z)!=(b[1]>z) and x<(b[0]-a[0])*(z-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return not inside and distance>.045

# Exactly the solver's pond profile, rebuilt as a smooth closed Blender mesh.
# The 2 mm surface allowance prevents coincident render/physics surfaces.
asset=collection('pond-and-ground',.00625)
t=L['terrain'];nx,nz=t['nx'],t['nz'];verts=[];faces=[]
for j in range(nz):
    for i in range(nx):verts.append((-.9+i*t['spacing'],t['heights'][j][i]+.002,-.6+j*t['spacing']))
for j in range(nz-1):
    for i in range(nx-1):
        a=j*nx+i;faces.append((a,a+nx,a+nx+1,a+1))
border=list(range(nx))+[j*nx+nx-1 for j in range(1,nz)]+list(range(nx*nz-2,nx*(nz-1)-1,-1))+[j*nx for j in range(nz-2,0,-1)]
base=len(verts)
for i in border: x,y,z=verts[i];verts.append((x,.10,z))
for k,i in enumerate(border):q=(k+1)%len(border);faces.append((i,border[q],base+q,base+k))
faces.append(tuple(base+i for i in reversed(range(len(border)))))
o=mesh('Continuous porcelain pond casting',verts,faces,clay)
# One closed collision-matched mesh, with separate materials for its surfaces.
# Classify against the actual pond rail so soil never inherits the basin glaze.
o.data.materials.append(groundmat);o.data.materials.append(copingmat)
centres=np.array([tuple(sum(verts[v][axis] for v in face)/len(face) for axis in (0,2)) for face in faces])
xs,zs=centres[:,0],centres[:,1];inside=np.zeros(len(faces),dtype=bool);distance2=np.full(len(faces),np.inf)
for a,b in zip(L['curve'],L['curve'][1:]+L['curve'][:1]):
    dx,dz=b[0]-a[0],b[1]-a[1];den=dx*dx+dz*dz
    t=np.clip(((xs-a[0])*dx+(zs-a[1])*dz)/den,0,1) if den else np.zeros(len(faces))
    distance2=np.minimum(distance2,(xs-a[0]-t*dx)**2+(zs-a[1]-t*dz)**2)
    if abs(dz)>1e-12:inside^=((a[1]>zs)!=(b[1]>zs)) & (xs<(b[0]-a[0])*(zs-a[1])/dz+a[0])
signed=np.sqrt(distance2)*np.where(inside,-1,1)
for face,d in zip(o.data.polygons,signed):face.material_index=1 if d>.045 else 2 if d>=-.040 else 0
# Decimate coplanar triangles only: keep the shared curved basin profile intact.
instance(asset)

# Retain the old small tile asset for previously saved scenes; the new surround
# needs its own ID so cached volumes and saved tile instances remain valid.
collection('ground-tile',.0125,1)
active.use_fake_user=True
box('Legacy garden ground tile',(0,-.025,0),(1.85,.05,1.25),groundmat,0)
# One watertight surround replaces the repeated tile grid. Its inner opening
# is inset under the pond casting, and its top sits 2 mm below the finish so
# two coplanar owners never compete.
asset=collection('ground-surround',.025,1)
outer=[(-4.5,-3),(4.5,-3),(4.5,3),(-4.5,3)]
inner=[(-.85,-.55),(.85,-.55),(.85,.55),(-.85,.55)]
verts=[(x,y,z) for y in (0,-.05) for x,z in outer+inner];faces=[]
for i in range(4):
    j=(i+1)%4
    faces.extend([(i,j,4+j,4+i),(8+i,12+i,12+j,8+j),
                  (i,8+i,8+j,j),(4+i,4+j,12+j,12+i)])
o=mesh('Continuous garden ground surround',verts,faces,groundmat)
for face in o.data.polygons:face.use_smooth=False
instance(asset,(0,.300,0),id='ground/surround')

# Individually sculpted cap and stem shapes preserve the graded boulder family.
for idx,s in enumerate(L['boulders']):
    asset=collection('boulder-'+str(idx),.003125);w=s['width_m'];h=s['height_m'];r=w/2
    tube('Eroded limestone pedestal',[(0,0,0),(.03*w,.3*h,0),(-.035*w,.61*h,.015*w)],[r*.48,r*.36,r*.53],stone[idx%5],24)
    rings=[(h*.47,.50),(h*.57,.85),(h*.67,1),(h*.79,.97),(h*.92,.73),(h,.08)];vs=[];fs=[]
    for k,(y,rr) in enumerate(rings):
        for j in range(48):
            a=j*2*math.pi/48;noise=1+.035*math.sin(a*5+idx)+.017*math.cos(a*11)
            vs.append((r*rr*noise*math.cos(a),y+.005*math.sin(a*3)*rr,r*rr*noise*.88*math.sin(a)))
    for k in range(len(rings)-1):
        for j in range(48):a=k*48+j;b=k*48+(j+1)%48;fs.append((a,b,b+48,a+48))
    fs.extend([tuple(reversed(range(48))),tuple((len(rings)-1)*48+j for j in range(48))]);mesh('Undercut weathered stone cap',vs,fs,stone[(idx+1)%5])
    x,z=s['at_m'];instance(asset,(x,height(x,z)+s['base_m'],z),id='stone/boulder-'+str(idx))

for idx,s in enumerate(L['stepping']):
    asset=collection('stepping-'+str(idx),.003125);r=s['width_m']/2;h=s['height_m']
    bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=r,depth=h,location=xyz((0,h/2,0)))
    o=finish(bpy.context.object,'Honed stone stepping disc',stone[2],False)
    mod=o.modifiers.new('Soft worn disc edge','BEVEL');mod.width=.003;mod.segments=4
    mod=o.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
    x,z=s['at_m'];instance(asset,(x,s['base_m']+.001,z),id='stone/path-'+str(idx))

# Shared river pebbles, distributed along the existing three graded beds.
pebbles=[]
for i in range(4):
    asset=collection('pebble-'+str(i),.0015625 if i<2 else .003125);r=[.006,.010,.018,.03][i]
    ellipsoid('Water worn river pebble',(0,r*.53,0),(r,r*.6,r*.76),stone[i],i+1);pebbles.append(asset)
for row in L['layout']:
    if row['species']!='pebble-bed':continue
    path=row['run']['path_m'];count={'stone/bed-shore':150,'stone/bed-near':100}.get(row['key'].replace('layout/',''),80)
    for i in range(count):
        t=rng.random()*(len(path)-1);k=int(t);f=t-k;x=path[k][0]*(1-f)+path[k+1][0]*f;z=path[k][1]*(1-f)+path[k+1][1]*f
        x+=rng.uniform(-1,1)*row['run']['halfWidth_m'];z+=rng.uniform(-1,1)*row['run']['halfWidth_m']
        size=rng.choices(range(4),[4,5,3,1])[0];yaw=rng.random()*math.tau
        if outside_coping(x,z):instance(pebbles[size],(x,height(x,z)+.001,z),yaw)

# Botanical assets: closed leaf meshes and tapered, irregular branching.
def tree(species,h):
    asset=collection(species,.003125 if species=='oak' else .0025)
    tall=species in ['oak','birch','pine','willow'];trunk=h*(.56 if tall else .38)
    tube('Tapered fluted trunk',[(0,0,0),(-h*.018,h*.16,0),(h*.025,trunk*.7,h*.015),(0,trunk,0)],[h*.047,h*.033,h*.022,h*.012],bark,16)
    for j in range(6):
        a=j*math.tau/6; tube('Surface root',[(0,h*.08,0),(math.cos(a)*h*.08,.008,math.sin(a)*h*.08),(math.cos(a)*h*.14,0,math.sin(a)*h*.14)],[h*.026,h*.014,.002],bark,8)
    branches=18 if species=='oak' else 12
    for j in range(branches):
        a=j*2.39996;level=j/branches
        if species=='pine':y=h*(.28+.66*level);reach=h*.35*(1-.8*level)
        else:y=trunk*(.45+.5*level);reach=h*(.36 if tall else .45)*rng.uniform(.72,1.05)
        start=(0,y,0);end=(math.cos(a)*reach,y+h*(.22 if tall else .30),math.sin(a)*reach)
        mid=(end[0]*.5,y+h*.09,end[2]*.5)
        tube('Primary branch',[start,mid,end],[h*.017,h*.011,.002],bark,8)
        for k in range(4):
            aa=a+(k-1.5)*.65;rr=h*(.12 if tall else .16);tip=(end[0]+math.cos(aa)*rr,end[1]+h*rng.uniform(.02,.13),end[2]+math.sin(aa)*rr)
            if species=='willow':tip=(tip[0],tip[1]-h*.23,tip[2])
            tube('Fine twig',[end,tip],[.0025,.0012],bark,6)
            for n in range(5 if species=='oak' else 4):
                t=(n+.5)/5;at=tuple(end[d]*(1-t)+tip[d]*t for d in range(3));ang=aa+(-1 if n%2 else 1)*1.1
                length=h*(.10 if species=='oak' else .15);width=length*(.29 if species not in ['pine','willow','juniper'] else .12)
                target=(at[0]+math.cos(ang)*length,at[1]+length*.3,at[2]+math.sin(ang)*length)
                leaf('Folded '+species+' leaf',at,target,width,(sagem if species in ['juniper','hebe'] else leafm)[(j+k+n)%5])
    return asset
sizes={'oak':.73,'birch':.23,'pine':.22,'willow':.19,'azalea':.20,'juniper':.19,'boxwood':.22,'hebe':.20}
trees={s:tree(s,h) for s,h in sizes.items()}
# Replace the generic oak with the actual X10 specimen. Running the generic
# builder above preserves its random-stream consumption, so every other asset
# and planting position stays unchanged when the oak is rebuilt.
active=assets[trees['oak']]['collection']
for obj in list(active.objects):bpy.data.objects.remove(obj,do_unlink=True)
assets[trees['oak']]['cell']=.0025
reference=L['oak']
wood=next(g['children'] for g in reference['children'] if g['id'].endswith('/structure'))
sprays=next(g['children'] for g in reference['children'] if g['id'].endswith('/foliage'))
for branch in wood:
    center=Vector(tuple(branch['place']['position'][axis] for axis in ('x','y','z')))
    points=[tuple(center+Vector(tuple(p['position'][axis] for axis in ('x','y','z')))) for p in branch['points']]
    radii=[p['radius'] for p in branch['points']]
    tube(branch['id'],points,radii,bark,12 if max(radii)>.012 else 8)
    # Round the major fork junctions; the original uses shared endpoint spheres.
    if radii[-1]>.006:ellipsoid(branch['id']+'/fork',points[-1],(radii[-1],)*3,bark)

for spray in sprays:
    # Small, rounded oak leaves occupy the original oriented spray envelopes.
    # Their seeded scatter retains the original canopy gaps and tiered boughs.
    r=random.Random(spray['seed']);q=spray['place']['orientation'];rotation=Quaternion((q['w'],q['x'],q['y'],q['z']))
    center=Vector(tuple(spray['place']['position'][axis] for axis in ('x','y','z')))
    rx,ry,rz=spray['form']['radii_m'];ry*=spray['form']['flatten']
    vs=[];fs=[]
    for n in range(12):
        angle=n*2.399963+r.uniform(-.25,.25);radius=math.sqrt((n+.25)/12)*.72
        at=Vector((math.cos(angle)*rx*radius,r.uniform(-.6,.6)*ry,math.sin(angle)*rz*radius))
        heading=angle+r.uniform(-.65,.65);length=r.uniform(.018,.028);width=length*r.uniform(.25,.32)
        axis=Vector((math.cos(heading),r.uniform(-.18,.5),math.sin(heading))).normalized()
        side=axis.cross(Vector((0,1,0))).normalized();normal=side.cross(axis).normalized()
        profile=[(0,.12),(.12,.48),(.23,.88),(.33,.61),(.44,1),(.55,.70),(.66,.91),(.77,.56),(.88,.55),(1,.04)]
        start=len(vs);rows=len(profile);layer=rows*3
        for sign in (-1,1):
            for t,w in profile:
                mid=at+axis*length*(t-.5)+normal*(length*.10*math.sin(math.pi*t))
                for across in (-1,0,1):
                    local=mid+side*width*w*across+normal*(sign*.0017+abs(across)*width*.13)
                    vs.append(tuple(center+rotation@local))
        for off in (0,layer):
            for row in range(rows-1):
                for across in range(2):
                    a=start+off+row*3+across;face=(a,a+1,a+4,a+3);fs.append(face if off else tuple(reversed(face)))
        border=[0,1,2]+[row*3+2 for row in range(1,rows)]+[layer-2,layer-3]+[row*3 for row in range(rows-2,0,-1)]
        for a,b in zip(border,border[1:]+border[:1]):fs.append((start+a,start+b,start+b+layer,start+a+layer))
    mesh(spray['id'],vs,fs,leafm[r.randrange(len(leafm))])
# Preserve X10's original planted base; no invented radial roots intrude on the pond.
x,z=reference['place']['ground'];instance(trees['oak'],(x,height(x,z),z),id='tree/oak')

def rosette(grass):
    asset=collection('grass' if grass else 'rosette',.0015625)
    for j in range(22 if grass else 15):
        a=j*2.39996;reach=rng.uniform(.018,.04);h=rng.uniform(.025,.065) if grass else rng.uniform(.018,.045)
        leaf('Arching grass blade' if grass else 'Succulent keeled leaf',(math.cos(a)*.003,0,math.sin(a)*.003),(math.cos(a)*reach,h,math.sin(a)*reach),.0025 if grass else .006,(leafm if grass else sagem)[j%5])
    return asset
rosettes={'grass-tuft':rosette(True),'air-plant':rosette(False)}
for s in L['stands']:
    x,z=s['at_m'];instance(trees.get(s['species']) or rosettes[s['species']],(x,height(x,z)+.001,z),math.atan2(x,z),id='stress/'+s['species']+'-'+str(s['index']))
for row in L['layout']:
    if row['species']=='rosette':
        x,z=row['at_m'];instance(rosettes['air-plant'],(x,height(x,z),z),row['yaw_rad'],id=row['key'])
for i,(x,z) in enumerate([(-.72,-.18),(.44,.42),(.72,-.32)]):instance(rosettes['grass-tuft' if i==2 else 'air-plant'],(x,height(x,z),z))
# The background is also Blender geometry, with sparse planted islands.
for i in range(32):
    a=rng.random()*math.tau;r=rng.uniform(1.2,3.2);x,z=r*math.cos(a),r*.7*math.sin(a)
    instance(rosettes['grass-tuft'] if i%3 else trees['hebe'],(x,.302,z),a)

asset=collection('hose-and-nozzle',.0025)
mouth=Vector(tuple(L['mouth'][k] for k in ['x','y','z']));impact=Vector(tuple(L['impact'][k] for k in ['x','y','z']));aim=(impact-mouth).normalized()
points=[tuple(mouth-aim*.032),tuple(mouth-aim*.075),(.365,.372,.255),(.455,.398,.305),(.575,.33,.365),(.720,.315,.440),(.890,.315,.520),(1.06,.315,.600)]
curve=bpy.data.curves.new('Continuous draped rubber tube','CURVE');curve.dimensions='3D';curve.resolution_u=12;curve.bevel_depth=.012;curve.bevel_resolution=4;curve.use_fill_caps=True
spline=curve.splines.new('BEZIER');spline.bezier_points.add(len(points)-1)
for b,co in zip(spline.bezier_points,points):b.co=xyz(co);b.handle_left_type='AUTO';b.handle_right_type='AUTO'
o=bpy.data.objects.new('Draped forest green hose',curve);active.objects.link(o);curve.materials.append(rubber)
bpy.context.view_layer.objects.active=o;o.select_set(True);bpy.ops.object.convert(target='MESH');o.select_set(False)
# Annular ferrule with a real open bore, aligned to the simulation's jet.
side=aim.cross(Vector((0,1,0))).normalized();up=aim.cross(side).normalized();vs=[];fs=[]
for d,r in [(-.04,.016),(0,.016),(0,.0105),(-.04,.0105)]:
    for j in range(48):a=j*math.tau/48;vs.append(tuple(mouth+aim*d+r*(math.cos(a)*side+math.sin(a)*up)))
for k in range(4):
    for j in range(48):fs.append((k*48+j,k*48+(j+1)%48,((k+1)%4)*48+(j+1)%48,((k+1)%4)*48+j))
mesh('Hollow machined nozzle',vs,fs,brass)
for k in range(6):
    center=mouth-aim*(.008+k*.0045)
    bpy.ops.mesh.primitive_torus_add(major_segments=40,minor_segments=8,major_radius=.016,minor_radius=.0008,location=xyz(center))
    o=bpy.context.object;o.rotation_euler=Vector(xyz(aim)).to_track_quat('Z','Y').to_euler();finish(o,'Nozzle grip ring',brass)
instance(asset,id='hose')

# Export local-space reusable collections, then assemble an editable full scene.
bpy.ops.object.select_all(action='DESELECT')
manifest={'version':1,'source':'hero-garden.blend','assets':[],'instances':instances}
for name,entry in assets.items():
    c=entry['collection']
    for o in c.objects:o.select_set(True)
    bpy.context.view_layer.objects.active=next(iter(c.objects))
    bpy.ops.export_scene.gltf(filepath=str(OUT/(name+'.glb')),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_normals=True,export_materials='EXPORT')
    manifest['assets'].append({'id':name,'mesh':name+'.glb','volume':name+'.fvx.gz','cellSize':entry['cell'],'bandCells':entry['band'],'sourceCollection':name})
    bpy.ops.object.select_all(action='DESELECT')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'lib/core/imported-assets/hero-garden-instances.json').write_text(json.dumps(instances,indent=2)+'\n')
# Unlink the original collections from the scene; collection instances retain them.
for entry in assets.values():scene.collection.children.unlink(entry['collection'])
for i in instances:
    o=bpy.data.objects.new(i['id'],None);o.instance_type='COLLECTION';o.instance_collection=assets[i['asset']]['collection'];scene.collection.objects.link(o)
    o.location=xyz(i['position']);o.rotation_euler[2]=i['yaw']
# Preview water is deliberately absent from every exported asset.
water=mat('Preview only pond water',(.035,.19,.16),.10)
bs=water.node_tree.nodes.get('Principled BSDF');bs.inputs['Transmission Weight'].default_value=.78;bs.inputs['IOR'].default_value=1.333
active=bpy.data.collections.new('Preview water only - runtime uses fluid');scene.collection.children.link(active)
verts=[];faces=[]
t=L['terrain']
for j in range(nz-1):
    for i in range(nx-1):
        if max(t['heights'][j][i],t['heights'][j+1][i],t['heights'][j][i+1],t['heights'][j+1][i+1])>=L['waterline']-.002:continue
        a=len(verts)
        for xx,zz in [(i,j),(i,j+1),(i+1,j+1),(i+1,j)]:verts.append((-.9+xx*t['spacing'],L['waterline'],-.6+zz*t['spacing']))
        faces.append((a,a+1,a+2,a+3))
mesh('Preview fluid surface',verts,faces,water,closed=False)
world=bpy.data.worlds.new('Soft daylight');world.use_nodes=True;scene.world=world
world.node_tree.nodes['Background'].inputs[0].default_value=(.72,.81,1,1);world.node_tree.nodes['Background'].inputs[1].default_value=.45
bpy.ops.object.light_add(type='AREA',location=xyz((-2.5,4,1.4)));o=bpy.context.object;o.data.energy=260;o.data.shape='DISK';o.data.size=2.2;o.rotation_euler=(Vector(xyz((0,.3,0)))-o.location).to_track_quat('-Z','Y').to_euler()
c=L['camera'];az=c['azimuth_rad'];el=c['elevation_rad'];dist=c['distance_m'];target=Vector(tuple(c['target_m'][k] for k in ['x','y','z']))
pos=target+Vector((math.sin(az)*math.cos(el),math.sin(el),math.cos(az)*math.cos(el)))*dist
bpy.ops.object.camera_add(location=xyz(pos));cam=bpy.context.object;cam.rotation_euler=(Vector(xyz(target))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='PERSP';cam.data.sensor_fit='VERTICAL';cam.data.angle=2*math.atan(c['tanHalfFov']);scene.camera=cam
scene.render.engine='CYCLES';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
scene.render.resolution_x=1600;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=.3;scene.render.image_settings.file_format='PNG';scene.render.filepath=str(OUT/'hero-garden-preview.png')
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'hero-garden.blend'))
if not args.no_render:bpy.ops.render.render(write_still=True)
