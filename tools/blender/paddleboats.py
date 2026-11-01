"""Reproducible Blender source, GLB, and analytic-part BVH for live moving skins.
Run: Blender --background --python-exit-code 1 --python tools/blender/paddleboats.py
The runtime intersects these same box/ellipsoid/cylinder transforms; fluid
colliders are deliberately separate, resolved at the FLIP grid spacing.
"""
import bpy, math, json, os
from mathutils import Vector, Matrix, Quaternion
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
C=Matrix(((1,0,0),(0,0,-1),(0,1,0)))
parts=[]; materials={}; objects=[]
colors={'paint':(.038,.16,.17),'cream':(.88,.79,.59),'wood':(.30,.12,.038),'woodlight':(.48,.24,.085),'brass':(.65,.39,.09),'dark':(.025,.031,.033),'glass':(.035,.12,.16),'red':(.48,.055,.036),'rope':(.55,.39,.20)}
def mat(key):
 if key not in materials:
  m=bpy.data.materials.new(key); m.diffuse_color=(*colors[key],1); m.use_nodes=True
  bs=m.node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=(*colors[key],1); bs.inputs['Roughness'].default_value=.27 if key in ('paint','glass','brass') else .48; bs.inputs['Metallic'].default_value=.75 if key=='brass' else 0
  materials[key]=m
 return materials[key]
def part(name,kind,p,d,color,angle=0,axis=(0,0,1)):
 q=Quaternion(Vector(axis),angle); bp=C@Vector(p); bq=(C@q.to_matrix()@C.transposed()).to_quaternion()
 if kind==0: bpy.ops.mesh.primitive_cube_add(size=2)
 elif kind==1: bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=1)
 else:
  bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=1,depth=2)
  # Blender cylinders are Z-aligned, runtime cylinders Y-aligned.
  for v in bpy.context.object.data.vertices: v.co=Vector((v.co.x,v.co.z,-v.co.y))
 o=bpy.context.object;o.name=name;o.location=bp;o.rotation_mode='QUATERNION';o.rotation_quaternion=bq
 # Mesh coordinates transformed so app local Y is up.
 for v in o.data.vertices: v.co=C@Vector((v.co.x*d[0],v.co.y*d[1],v.co.z*d[2]))
 o.data.materials.append(mat(color));objects.append(o)
 if kind!=0:
  for poly in o.data.polygons: poly.use_smooth=True
 parts.append({'name':name,'kind':kind,'p':p,'d':d,'q':[q.w,q.x,q.y,q.z],'color':colors[color],'paint':color=='paint'})
 return o
def box(n,p,d,c,**kw):return part(n,0,p,d,c,**kw)
def ell(n,p,d,c):return part(n,1,p,d,c)
def rod(n,a,b,r,c):
 a,b=Vector(a),Vector(b);v=b-a;q=Vector((0,1,0)).rotation_difference(v.normalized());axis,angle=q.to_axis_angle()
 return part(n,2,(a+b)/2,(r,v.length/2,r),c,angle,axis)
def ring(n,center,r,t,c,plane='xy',start=0,end=2*math.pi,count=28):
 for i in range(count):
  def p(a):
   if plane=='xy':return (center[0]+r*math.cos(a),center[1]+r*math.sin(a),center[2])
   return (center[0],center[1]+r*math.sin(a),center[2]+r*math.cos(a))
  rod(n,p(start+(end-start)*i/count),p(start+(end-start)*(i+1)/count),t,c)
# Long launch hull; all upper detail fits above the actual box displacement.
box('Hull / painted central planking',(0,0,0),(.68,.14,.27),'paint')
ell('Rounded bow',(.64,0,0),(.18,.14,.27),'paint')
box('Teak rubbing strake',(0,.13,0),(.72,.025,.295),'wood')
ell('Bow rubbing strake',(.64,.13,0),(.19,.025,.295),'wood')
for k in range(15):
 z=(k-7)*.037
 box('Individual deck plank',(0,.162,z),(.71,.009,.0175),'woodlight' if k%3 else 'wood')
box('Cabin',(0,.31,0),(.31,.14,.205),'cream')
for side in (-1,1):
 for x in (-.21,0,.21):
  box('Window brass frame',(x,.335,side*.209),(.075,.082,.008),'brass')
  box('Blue glazed window',(x,.335,side*.219),(.061,.068,.004),'glass')
  box('Window centre mullion',(x,.335,side*.225),(.004,.069,.004),'cream')
 for x in (-.57,-.44,.44,.57):
  ring('Brass porthole',(x,.025,side*.273),.037,.006,'brass',count=12)
  part('Porthole glass',2,(x,.025,side*.275),(.029,.004,.029),'glass',math.pi/2,(1,0,0))
 for x in (-.65,-.48,.46,.64):
  rod('Rail stanchion',(x,.17,side*.27),(x,.34,side*.27),.007,'brass')
 for y in (.25,.34):
  rod('Fore rail',(.4,y,side*.27),(.70,y,side*.27),.006,'brass')
  rod('Aft rail',(-.7,y,side*.27),(-.4,y,side*.27),.006,'brass')
 for x in (-.59,.54):
  part('Mooring bollard',2,(x,.20,side*.17),(.015,.03,.015),'brass')
  rod('Bollard crossbar',(x-.035,.22,side*.17),(x+.035,.22,side*.17),.008,'brass')
box('Cabin roof',(0,.465,0),(.36,.025,.25),'paint')
box('Roof brass edge',(0,.445,0),(.37,.007,.26),'brass')
part('Smokestack',2,(-.12,.615,0),(.048,.13,.048),'dark')
for y in (.51,.69,.735):part('Stack copper bands',2,(-.12,y,0),(.054,.012,.054),'brass')
part('Stack mouth',2,(-.12,.749,0),(.041,.003,.041),'dark')
part('Steam whistle',2,(.03,.57,.065),(.012,.07,.012),'brass')
box('Pilot house',(.25,.53,0),(.09,.06,.15),'cream')
box('Pilot forward glass',(.344,.54,0),(.003,.043,.125),'glass')
box('Pilot roof',(.25,.60,0),(.12,.012,.18),'wood')
for z in (-.23,.23):rod('Sternwheel bearing support',(-.59,.15,z),(-1.0,.075,z),.025,'paint')
rod('Wheel axle',(-1.0,.075,-.31),(-1.0,.075,.31),.025,'brass')
# Name boards and engraved-looking brass insets.
for side in (-1,1):
 box('Nameplate',(.02,.19,side*.299),(.18,.024,.006),'dark')
 for k in range(9):box('Nameplate lettering',(-.13+k*.033,.191,side*.306),(.009,.012,.0015),'brass')
hull=list(parts);hullobjects=list(objects);parts=[];objects=[]
# One rotating quarter. Its origin is the FLIP paddle centre at radial +X.
# Four identical quarters form the wheel, with four broad teak paddle flights.
r=.235
box('Paddle / teak flight',(0,0,0),(.065,.025,.25),'woodlight')
for z in (-.19,0,.19):box('Paddle iron strap',(0,-.028,z),(.067,.004,.012),'dark')
for side in (-1,1):
 rod('Wheel spoke',(-r,0,side*.22),(0,0,side*.22),.012,'red')
 ring('Wheel iron rim',(-r,0,side*.22),r,.012,'red',start=-math.pi/4,end=math.pi/4,count=10)
 for z in (side*.22,):part('Paddle bolt',2,(0,-.032,z),(.009,.005,.009),'brass')
wheel=list(parts);wheelobjects=list(objects)
# Export runtime data: pre-order BVH with escape indices, same authored solids.
def export_skin(ps):
 def bounds(p):
  q=Quaternion(p['q']).to_matrix();d=Vector(p['d']);e=Vector([sum(abs(q[a][b])*d[b] for b in range(3)) for a in range(3)]);c=Vector(p['p']);return c-e,c+e
 nodes=[];ordered=[]
 def build(items):
  lo=Vector([min(bounds(p)[0][a] for p in items) for a in range(3)]);hi=Vector([max(bounds(p)[1][a] for p in items) for a in range(3)])
  idx=len(nodes);nodes.append({'lo':list(lo),'hi':list(hi),'part':-1,'end':0})
  if len(items)==1:nodes[idx]['part']=len(ordered);ordered.append(items[0])
  else:
   axis=max(range(3),key=lambda a:hi[a]-lo[a]);items.sort(key=lambda p:p['p'][axis]);mid=len(items)//2;build(items[:mid]);build(items[mid:])
  nodes[idx]['end']=len(nodes)
 build(ps[:]);return {'parts':ordered,'nodes':nodes,'extent':[max(abs(nodes[0]['lo'][a]),abs(nodes[0]['hi'][a])) for a in range(3)]}
def serial(x):
 if isinstance(x,(Vector,Quaternion)):return list(x)
 raise TypeError(type(x))
parts=[];objects=[]
box('Basin plinth',(0,-.18,0),(6.62,.03,1.42),'dark')
box('Basin floor',(0,-.08,0),(6.55,.08,1.35),'paint')
for z in (-1.25,1.25):
 box('Long basin wall',(0,.4,z),(6.5,.4,.05),'cream')
 box('Long coping',(0,.82,z),(6.55,.02,.10),'woodlight')
for x in (-6.45,6.45):
 box('End basin wall',(x,.4,0),(.05,.4,1.2),'cream')
 box('End coping',(x,.82,0),(.10,.02,1.15),'woodlight')
for k in range(63):
 box('Measuring tick',(-6.2+k*.2,.843,-1.25),(.006,.003,.065 if k%5==0 else .035),'dark')
basin=list(parts);basinobjects=list(objects)
out={'hull':export_skin(hull),'wheel':export_skin(wheel),'basin':basin}
with open(ROOT+'/lib/core/imported-assets/paddleboats.generated.json','w') as f:json.dump(out,f,default=serial,separators=(',',':'))
# Save origin-space editable source and individual GLBs (wheel = one quarter).
for group,name in ((hullobjects,'paddleboat'),(wheelobjects,'paddlewheel-quarter'),(basinobjects,'test-basin')):
 bpy.ops.object.select_all(action='DESELECT')
 for o in group:o.select_set(True)
 bpy.ops.export_scene.gltf(filepath=ROOT+'/public/assets/paddleboats/'+name+'.glb',use_selection=True,export_format='GLB')
for group,name in ((hullobjects,'Launch'),(wheelobjects,'Wheel quarter'),(basinobjects,'Test basin')):
 collection=bpy.data.collections.new(name);bpy.context.scene.collection.children.link(collection)
 for o in group:
  for old in list(o.users_collection):old.objects.unlink(o)
  collection.objects.link(o)
bpy.ops.wm.save_as_mainfile(filepath=ROOT+'/assets/paddleboats/paddleboats.blend')
# Presentation: move quarter assemblies onto two opposed boats, never exported as physics.
for o in wheelobjects:bpy.data.objects.remove(o,do_unlink=True)
for o in hullobjects:o.hide_render=True
for boat,(x,yaw) in enumerate(((-4.95,0),(4.95,math.pi))):
 root=bpy.data.objects.new('Boat assembly',None);bpy.context.collection.objects.link(root);root.location=(x,0,.61);root.rotation_euler.z=yaw
 for o in hullobjects:
  c=o.copy();c.data=o.data.copy();bpy.context.collection.objects.link(c);c.hide_render=False;c.parent=root
  if boat==0 and c.data.materials and c.data.materials[0].name=='paint':c.data.materials[0]=mat('red')
 for k in range(4):
  parts=[];objects=[]
  for p in wheel:
   # Build quarter from original app transforms, rotate around local Z.
   q=Quaternion((0,0,1),k*math.pi/2);v=q@(Vector(p['p'])+Vector((r,0,0)))+Vector((-1.0,.075,0));qq=q@Quaternion(p['q']);axis,ang=qq.to_axis_angle()
   key=next((key for key,c in colors.items() if tuple(c)==tuple(p['color'])),'wood')
   o=part(p['name'],p['kind'],v,p['d'],key,ang,axis);o.parent=root
# Tiled basin / still water is for the Blender preview ONLY.
parts=[];objects=[]
water=bpy.data.materials.new('Preview water only');water.use_nodes=True;bs=water.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.10,.31,.29,1);bs.inputs['Roughness'].default_value=.1;bs.inputs['Transmission Weight'].default_value=.6;bs.inputs['IOR'].default_value=1.333
w=box('Preview water',(0,.6,0),(6.399,.003,1.199),'glass');w.data.materials[0]=water
bpy.ops.object.camera_add(location=(6,-7,5.8));cam=bpy.context.object;cam.rotation_euler=(Vector((0,0,.6))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=13.8;bpy.context.scene.camera=cam
bpy.ops.object.light_add(type='AREA',location=(1,-3,7));bpy.context.object.data.energy=1900;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=5
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=32;scene.world.color=(.25,.25,.25);scene.render.resolution_x=1400;scene.render.resolution_y=950;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.filepath=ROOT+'/public/assets/paddleboats/preview.png';bpy.ops.render.render(write_still=True)
