"""Rain in Shinjuku — editable Cycles set lit by its practical signs.

Real water is imported from the application's NB-FLIP OBJ export in metres,
Z-up (x,-z,y). With no --water-mesh the street is deliberately left dry: this
script never silently replaces simulated water with a plane.
"""
import argparse
import math
from pathlib import Path
import random
import sys
import bpy
import bmesh
from mathutils import Vector, Matrix, noise

ROOT = Path(__file__).resolve().parents[2]
R = random.Random(20261028)


def mat(name, color, rough=.5, metal=0, emission=0):
    m = bpy.data.materials.new(name); m.diffuse_color = (*color, 1); m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metal
    if emission:
        p.inputs['Emission Color'].default_value = (*color, 1)
        p.inputs['Emission Strength'].default_value = emission
    return m


def textured(name, color, scale=9, rough=.5, bump=.025):
    m=mat(name,color,rough); n=m.node_tree.nodes; l=m.node_tree.links; p=n.get('Principled BSDF')
    noise=n.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value=scale; noise.inputs['Detail'].default_value=5
    ramp=n.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color=(*(v*.33 for v in color),1); ramp.color_ramp.elements[1].color=(*(min(1,v*1.8) for v in color),1)
    l.new(noise.outputs['Fac'],ramp.inputs[0]); l.new(ramp.outputs[0],p.inputs['Base Color'])
    fine=n.new('ShaderNodeTexNoise'); fine.inputs['Scale'].default_value=scale*20
    b=n.new('ShaderNodeBump'); b.inputs['Strength'].default_value=.45; b.inputs['Distance'].default_value=bump
    l.new(fine.outputs['Fac'],b.inputs['Height']); l.new(b.outputs[0],p.inputs['Normal'])
    return m


def cube(name, loc, size, material, bevel=.015):
    x,y,z=(v/2 for v in size)
    verts=[(-x,-y,-z),(-x,-y,z),(-x,y,-z),(-x,y,z),(x,-y,-z),(x,-y,z),(x,y,-z),(x,y,z)]
    data=bpy.data.meshes.new(name);data.from_pydata(verts,[],[(0,4,6,2),(1,3,7,5),(0,1,5,4),(2,6,7,3),(0,2,3,1),(4,5,7,6)])
    o=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(o);o.location=loc
    if material: o.data.materials.append(material)
    if bevel:
        b=o.modifiers.new('Worn eased edges','BEVEL'); b.width=bevel; b.segments=2
        o.modifiers.new('Weighted corner normals','WEIGHTED_NORMAL')
    return o


def rod(name,a,b,r,material,vertices=12):
    d=Vector(b)-Vector(a); mid=(Vector(a)+Vector(b))/2
    verts=[(r*math.cos(i*math.tau/vertices),r*math.sin(i*math.tau/vertices),z) for z in [-d.length/2,d.length/2] for i in range(vertices)]
    faces=[tuple(range(vertices-1,-1,-1)),tuple(range(vertices,vertices*2))]
    faces += [(i,(i+1)%vertices,(i+1)%vertices+vertices,i+vertices) for i in range(vertices)]
    data=bpy.data.meshes.new(name);data.from_pydata(verts,[],faces);data.materials.append(material)
    o=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(o);o.location=mid;o.rotation_euler=d.to_track_quat('Z','Y').to_euler()
    for p in data.polygons:p.use_smooth=len(p.vertices)==4
    return o


def curve(name, pts, radius, material):
    c=bpy.data.curves.new(name,'CURVE'); c.dimensions='3D'; c.bevel_depth=radius; c.bevel_resolution=2
    s=c.splines.new('POLY'); s.points.add(len(pts)-1)
    for p,co in zip(s.points,pts):p.co=(*co,1)
    o=bpy.data.objects.new(name,c); bpy.context.collection.objects.link(o); o.data.materials.append(material); return o


def sphere(name,loc,scale,material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,location=loc);o=bpy.context.object;o.name=name;o.scale=scale;o.data.materials.append(material)
    for p in o.data.polygons:p.use_smooth=True
    return o


def area(name,loc,target,color,power,size=2):
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.color=color;data.shape='DISK';data.size=size
    o=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler();return o


def text(name,body,loc,size,material,font=None,rotation=(math.pi/2,0,0)):
    c=bpy.data.curves.new(name,'FONT');c.body=body;c.align_x='CENTER';c.align_y='CENTER';c.size=size;c.extrude=.002;c.bevel_depth=.001
    if font:c.font=font
    o=bpy.data.objects.new(name,c);bpy.context.collection.objects.link(o);o.location=loc;o.rotation_euler=rotation;o.data.materials.append(material);return o


def build_scene(water_mesh_path=None):
    R.seed(20261028)
    noise.seed_set(20261028)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    sc=bpy.context.scene;sc.unit_settings.system='METRIC';sc['title']='Rain in Shinjuku';sc['water_mode']='Imported narrow-band FLIP surface' if water_mesh_path else 'No fluid imported; dry environment preview'
    concrete=textured('Damp soot-stained render',(.19,.18,.16),3,.58,.025)
    brick=textured('Weathered ochre plaster',(.25,.13,.075),7,.64,.025)
    tile=textured('Wet black ceramic',(.027,.034,.04),8,.22,.009)
    asphalt=textured('Wet fine asphalt aggregate',(.034,.04,.046),18,.19,.014)
    curb=textured('Granite curb',(.19,.2,.2),14,.34,.015)
    steel=textured('Oxidised zinc',(.14,.17,.18),9,.38,.007);steel.node_tree.nodes.get('Principled BSDF').inputs['Metallic'].default_value=.7
    # Rain streaks sit in the plaster's colour, separate from fine mineral bump.
    for plaster in [concrete,brick]:
        ns=plaster.node_tree.nodes;ls=plaster.node_tree.links;bs=ns.get('Principled BSDF')
        coord=ns.new('ShaderNodeTexCoord');stretch=ns.new('ShaderNodeVectorMath');stretch.operation='MULTIPLY';stretch.inputs[1].default_value=(2.8,2.8,.16);ls.new(coord.outputs['Object'],stretch.inputs[0])
        stains=ns.new('ShaderNodeTexNoise');stains.inputs['Scale'].default_value=3;stains.inputs['Detail'].default_value=3;ls.new(stretch.outputs[0],stains.inputs['Vector'])
        ramp=ns.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.38;ramp.color_ramp.elements[0].color=(.35,.38,.4,1);ramp.color_ramp.elements[1].position=.65;ramp.color_ramp.elements[1].color=(1,1,1,1);ls.new(stains.outputs['Fac'],ramp.inputs[0])
        old=bs.inputs['Base Color'].links[0].from_socket;mix=ns.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.55;ls.new(old,mix.inputs[1]);ls.new(ramp.outputs[0],mix.inputs[2]);ls.new(mix.outputs[0],bs.inputs['Base Color'])
    black=mat('Rubber and cables',(.008,.012,.014),.45);wood=textured('Oiled cedar',(.13,.045,.014),12,.3,.009)
    warm=mat('Warm paper lantern', (1,.27,.055),.5,emission=2.7)
    red=mat('Coral neon', (1,.012,.065),.25,emission=5)
    cyan=mat('Cyan neon',(.015,.55,1),.25,emission=5)
    cream=mat('Warm cream lettering',(1,.73,.4),.3,emission=2)
    white=mat('Fluorescent light',(.65,.82,1),.2,emission=5)
    glass=mat('Shop window glass',(.92,.97,1),.035);p=glass.node_tree.nodes.get('Principled BSDF');p.inputs['Transmission Weight'].default_value=.98;p.inputs['IOR'].default_value=1.46
    upper_glass=mat('Dark upper window glass',(.025,.035,.055),.12,.18)
    bottle_mats=[mat('Sake amber glass '+str(i),(.07+.02*i,.035,.008),.18) for i in range(2)]
    fontfile=Path('/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc');font=bpy.data.fonts.load(str(fontfile)) if fontfile.exists() else None
    cube('Flooded street bed',(0,2,.09),(7,30,.18),asphalt)
    for side in [-1,1]:
        cube('Continuous raised footpath',(side*3.05,3,.31),(.48,34,.26),curb)
        for j in range(84):cube('Individual curb joints',(side*2.81,-10+j*.42,.32),(.12,.412,.27),curb,.007)
        for j in range(7):
            y=-3+j*3.6+(1.25 if side>0 else 0)
            wall=brick if (j+side)%3==0 else concrete
            cube('Narrow old shop building',(side*4.12,y,5.9),(2.1,3.57,5.8+R.uniform(-.5,.5)),wall,.035)
            cube('Weather-streaked upper facade',(side*3.04,y,5.86),(.055,3.53,5.7),wall,.012)
            cube('Recessed shop rear wall',(side*5.1,y,1.73),(.14,3.57,2.6),wood,.015)
            for end in [-1,1]:cube('Shop party wall',(side*4.1,y+end*1.74,1.73),(2.0,.10,2.6),wall,.01)
            cube('Shop interior floor',(side*4.1,y,.49),(2,3.5,.1),tile,.01)
            cube('Black ceramic shopfront plinth',(side*3.055,y,.57),(.08,3.5,.22),tile,.005)
            cube('Black ceramic shopfront lintel',(side*3.055,y,2.79),(.08,3.5,.24),tile,.005)
            for pier in [-1.65,-.56,.56,1.65]:cube('Ceramic facade pier',(side*3.055,y+pier,1.7),(.11,.1,2.3),tile,.006)
            # Doors, glazing, frames and muntins along side facades.
            for k in range(3):
                yy=y-1.12+k*1.08
                # Real recessed interiors replace flat luminous window cards.
                cube('Interior cedar counter',(side*3.85,yy,1.1),(.45,.98,.12),wood,.01)
                for leg in [-1,1]:rod('Counter timber leg',(side*3.85,yy+leg*.35,.5),(side*3.85,yy+leg*.35,1.05),.035,wood)
                cube('Interior wall shelf',(side*4.8,yy,1.9),(.3,1.0,.045),wood,.008)
                for bottle in range(5):
                    bottle_y=yy-.36+bottle*.18
                    bottle_mat=bottle_mats[bottle%2]
                    rod('Sake bottle body',(side*4.72,bottle_y,1.925),(side*4.72,bottle_y,2.13),.04,bottle_mat)
                    rod('Sake bottle neck',(side*4.72,bottle_y,2.13),(side*4.72,bottle_y,2.2),.019,bottle_mat)
                cube('Interior hanging strip lamp',(side*4.04,yy,2.55),(.09,.72,.035),warm,.005)
                if not (side>0 and j in [1,3,5]):area('Shop practical warm light',(side*3.65,yy,2.46),(side*2.1,yy,1.0),(1,.46,.18),22,.65)
                cube('Rain-streaked glazing',(side*2.995,yy,1.63),(.025,.93,1.88),glass,.003)
                for edge in [-1,1]:cube('Cedar window jamb',(side*2.96,yy+edge*.49,1.64),(.1,.065,2.05),wood,.008)
                for z in [.67,1.62,2.61]:cube('Cedar transom',(side*2.96,yy,z),(.1,1.04,.065),wood,.006)
            cube('Shop awning fascia',(side*2.85,y,2.94),(.48,3.5,.22),wood)
            for k in range(22):cube('Awning weathered slat',(side*2.81,y-1.7+k*.16,3.03),(.55,.145,.08),tile,.005)
            for floor in range(2):
                for k in [-1,1]:
                    yy=y+k*.85;zz=4.2+floor*2.1
                    cube('Upper aluminium window',(side*3.045,yy,zz),(.09,.97,1.14),steel)
                    cube('Upper dark reflective glazing',(side*2.985,yy,zz),(.025,.82,.98),upper_glass,.003)
                    cube('Window sill',(side*2.9,yy,zz-.59),(.29,1.15,.08),concrete)
                    cube('Window mullion',(side*2.96,yy,zz),(.06,.035,1),steel,.003)
            # AC condenser and its grille.
            acy=y+.9;cube('Air conditioning condenser',(side*2.75,acy,3.5),(.5,.8,.49),steel)
            for g in range(9):cube('Condenser grille',(side*2.485,acy-.34+g*.085,3.5),(.025,.027,.38),black,.002)
            rod('Rainwater downpipe',(side*2.88,y-1.66,.5),(side*2.88,y-1.66,7.8),.045,steel)
            for z in [1,3.6,6.8]:cube('Downpipe collar',(side*2.88,y-1.66,z),(.13,.13,.06),steel)
            # Blade signs face camera and illuminate actual water.
            sx=side*(2.53 if j%2 else 2.4);sy=y-.9;sz=3.1+(j%3)*.48
            cube('Projecting enamel sign',(sx,sy,sz),(.66,.16,1.72),black,.025)
            emit=red if side<0 else cyan
            for edge in [-1,1]:rod('Neon sign edge',(sx+edge*.29,sy-.095,sz-.78),(sx+edge*.29,sy-.095,sz+.78),.012,emit)
            word=['居酒屋','らーめん','酒場','珈琲','食堂','東京','焼鳥'][j]
            for k,ch in enumerate(word):text('Japanese illuminated lettering',ch,(sx,sy-.103,sz+.48-k*.4),.34,emit,font)
            area('Neon spill',(sx,sy-.25,sz),(0,sy-1,.3), (1,.025,.09) if side<0 else (.025,.42,1),60,1.2)
            if side>0 and j in [1,3,5]:
                # Shops closed for the evening break the repeated bright frontage.
                for syy in [y-1.08,y,y+1.08]:
                    cube('Closed galvanised roller shutter',(side*2.94,syy,1.6),(.06,.94,1.93),steel,.004)
                    for ri in range(28):cube('Corrugated shutter seam',(side*2.90,syy,.68+ri*.069),(.025,.94,.012),black,.001)
            if side<0:
                for k in [-1,1]:
                    ly=y+k*.85
                    sphere('Ribbed red paper lantern',(side*2.64,ly,2.57),(.18,.18,.31),warm)
                    for r in range(12):
                        z=2.3+r*.045;rad=.18*math.sqrt(max(.08,1-((z-2.57)/.31)**2))
                        curve('Lantern bamboo ribs',[(side*2.64+rad*math.cos(a*math.tau/32),ly+rad*math.sin(a*math.tau/32),z) for a in range(33)],.0025,wood)
        # Utility conduits run across the facades.
        for z in [3.35,6.9]:curve('Sagging facade power line',[(side*2.9,-6+i*.4,z-.10*math.sin(i*.3)) for i in range(80)],.012,black)
    # A vending machine, stacked crates and service entrance interrupt the
    # otherwise repeated shopfront rhythm with everyday street-scale objects.
    machine=mat('Vending machine faded red enamel',(.35,.018,.024),.28,.3)
    cube('Japanese drink vending machine',(2.58,-.4,1.38),(.53,.83,1.86),machine,.025)
    cube('Vending machine illuminated display',(2.27,-.4,1.64),(.035,.72,.94),cream,.006)
    for row in range(3):
        for col in range(5):
            yy=-.7+col*.14;zz=1.28+row*.26
            canmat=[red,cyan,warm,cream][(row+col)%4]
            rod('Drink can behind vending window',(2.24,yy,zz),(2.24,yy,zz+.15),.038,canmat)
    cube('Coin slot',(2.25,-.12,1.09),(.03,.09,.024),black,.002)
    cube('Vending machine delivery slot',(2.25,-.42,.72),(.03,.55,.15),black,.006)
    crate=mat('Battered beer crate green plastic',(.022,.11,.052),.5)
    for i in range(3):
        z=.46+i*.25
        for side in [-1,1]:
            cube('Stacked beer crate edge',(-2.48+side*.22,2.5,z+.12),(.045,.35,.24),crate,.008)
            cube('Stacked beer crate end',(-2.48,2.5+side*.16,z+.12),(.45,.045,.24),crate,.008)
        for j in range(5):cube('Crate ventilation slots',(-2.71,2.37+j*.065,z+.13),(.006,.035,.11),black,.001)
    for y in [-1.5,1.0]:
        bagmat=textured('Creased wet refuse sack',(.009,.01,.012),35,.63,.008)
        bag=sphere('Rain-wet bin bag',(-2.5,y,.64),(.21,.17,.22),bagmat)
        for v in bag.data.vertices:
            p=v.co;angle=math.atan2(p.y,p.x);f=1+.23*noise.noise_vector(p*7)[0]+.11*math.sin(angle*13)
            taper=1-.6*max(0,(p.z-.3)/.7)
            p.x*=f*taper;p.y*=f*taper;p.z=max(-.8,p.z)
        bag.data.update()
        rod('Tied bin bag neck',(-2.5,y,.82),(-2.49,y,.9),.025,black)
    # Horizontal illuminated ramen fascia, distinct from the blade signs.
    cube('Wide ramen bar fascia',(-2.56,1.1,2.91),(.07,2.9,.4),wood,.015)
    text('Ramen shop horizontal name','麺屋  一番',(-2.605,1.1,2.91),.28,cream,font,rotation=(math.pi/2,0,-math.pi/2))
    # Far end gives the alley believable depth and a pool of pale light.
    cube('End street building',(0,24,5),(14,2,10),concrete,.03)
    warm_office=mat('Distant occupied office',(.36,.12,.025),.4,emission=.7)
    for x in [-4,-1.8,.5,2.8,5]:
        for z in [2,4.5,7]:
            cube('Distant office window',(x,22.98,z),(1.2,.03,1.45),warm_office if x==.5 and z==2 else upper_glass,.01)
            for dx in [-.63,0,.63]:cube('Distant window jamb',(x+dx,22.91,z),(.07,.09,1.54),steel,.006)
            for dz in [-.76,.05,.76]:cube('Distant window rail',(x,22.91,z+dz),(1.3,.09,.07),steel,.006)
    # Small mounted floodlight apertures replace an exposed central light disk.
    # The warm shop practical sits in front of the cyclist to reveal the coat.
    for name,loc,target,color,power,wall in [
        ('Far blue',(2.55,18,4),(0,3.5,1.4),(.22,.5,1),180,3.04),
        ('Ramen warm',(-2.4,0,2.3),(.35,3.5,1.4),(1,.34,.11),80,-2.96),
    ]:
        d=(Vector(target)-Vector(loc)).normalized()
        shell=cube(name+' practical housing',Vector(loc)-d*.09,(.28,.28,.14),steel,.012)
        shell.rotation_euler=d.to_track_quat('Z','Y').to_euler()
        lens=rod(name+' luminous aperture',Vector(loc)-d*.014,Vector(loc)+d*.014,.09,mat(name+' LED',color,.3,emission=power),32)
        rod(name+' wall bracket',(wall,loc[1],loc[2]+.12),(loc[0],loc[1],loc[2]+.12),.025,steel)
        area(name+' practical beam',Vector(loc)+d*.02,target,color,power,.18)
    rod('Distant streetlamp pole',(2.75,17,.45),(2.75,17,5.7),.055,steel)
    rod('Distant streetlamp arm',(2.75,17,5.7),(1.7,17,5.7),.035,steel)
    cube('Distant streetlamp hood',(1.7,17,5.68),(.48,.24,.09),steel,.025)
    cube('Distant streetlamp diffuser',(1.7,17,5.62),(.4,.19,.025),cream,.008)
    area('Practical distant streetlamp',(1.7,17,5.57),(1.7,16,0),(1,.57,.25),80,.4)
    for y in [2,8,14]:
        curve('Overhead drooping cables',[(-3.1+i*.062,y,.68*((i/100-.5)*2)**2+7) for i in range(101)],.014,black)
    # A detailed city bicycle with a rider, turned slightly across the alley.
    # Bike axis is local Y, its wheels are YZ disks.
    before_figure=set(bpy.context.scene.objects)
    bx=.35;by=3.5;ground=.43
    chrome=mat('Bicycle brushed chrome',(.28,.32,.34),.19,.85);paint=mat('Bicycle burgundy enamel',(.16,.012,.025),.2,.55)
    for wy in [by-.65,by+.65]:
        pts=[(bx,wy+.34*math.cos(a*math.tau/64),ground+.35+.34*math.sin(a*math.tau/64)) for a in range(65)]
        curve('Pneumatic bicycle tyre',pts,.027,black)
        curve('Polished wheel rim',[(x,y,ground+.35+(z-ground-.35)*.93) for x,y,z in pts],.011,chrome)
        for a in range(24):rod('Stainless bicycle spoke',(bx,wy,ground+.35),(bx,wy+.315*math.cos(a*math.tau/24),ground+.35+.315*math.sin(a*math.tau/24)),.0014,chrome,6)
        rod('Wheel axle',(bx-.055,wy,ground+.35),(bx+.055,wy,ground+.35),.025,chrome)
    A=(bx,by-.65,ground+.35);B=(bx,by,ground+.35);C=(bx,by-.2,ground+.91);D=(bx,by+.52,ground+.95);E=(bx,by+.65,ground+.35)
    for a,b in [(A,B),(A,C),(B,C),(C,D),(D,B),(D,E)]:rod('Bicycle tubular frame',a,b,.021,paint)
    rod('Seat post',C,(bx,by-.23,ground+1.04),.014,chrome);sphere('Leather saddle',(bx,by-.25,ground+1.05),(.11,.19,.035),black)
    rod('Handlebar stem',D,(bx,by+.50,ground+1.18),.018,chrome);rod('Handlebar',(bx-.24,by+.5,ground+1.18),(bx+.24,by+.5,ground+1.18),.017,chrome)
    cube('Bicycle rear luggage rack',(bx,by-.65,ground+.75),(.26,.45,.035),chrome,.005)
    coat=textured('Rider midnight waterproof coat',(.019,.025,.032),35,.67,.008);pants=mat('Rider dark denim',(.015,.022,.03),.67);skin=mat('Warm skin in shade',(.32,.16,.10),.5)
    # Tailored waterproof coat with shoulder/waist/hem contours.
    verts=[]
    profile=[(1.47,.21,.14),(1.58,.23,.16),(1.84,.18,.16),(2.11,.24,.16),(2.2,.15,.12)]
    rings=17
    for ring in range(rings):
        z=1.47+(2.2-1.47)*ring/(rings-1)
        low=next((k for k in range(4) if profile[k][0]<=z<=profile[k+1][0]),3)
        t=(z-profile[low][0])/(profile[low+1][0]-profile[low][0]);rx=profile[low][1]*(1-t)+profile[low+1][1]*t;ry=profile[low][2]*(1-t)+profile[low+1][2]*t
        for i in range(32):
            a=i*math.tau/32;fold=.009*math.sin(a*7+z*31)+.006*math.sin(z*53+a*3)
            verts.append((bx+(rx+fold)*math.cos(a),by-.08+(ry+fold)*math.sin(a),z+.009*math.sin(a*5)))
    faces=[tuple(range(31,-1,-1)),tuple(range((rings-1)*32,rings*32))]
    for j in range(rings-1):
        for i in range(32):faces.append((j*32+i,j*32+(i+1)%32,(j+1)*32+(i+1)%32,(j+1)*32+i))
    mesh=bpy.data.meshes.new('Tailored coat');mesh.from_pydata(verts,[],faces);mesh.materials.append(coat);ob=bpy.data.objects.new('Cyclist waterproof jacket',mesh);bpy.context.collection.objects.link(ob)
    for p in mesh.polygons:p.use_smooth=True
    sub=ob.modifiers.new('Soft cloth contours','SUBSURF');sub.levels=2
    curve('Coat back seam',[(bx,by-.25,z) for z in [1.55,1.8,2.08]],.002,black)
    sphere('Cyclist head',(bx,by+.01,2.36),(.105,.105,.14),skin)
    sphere('Cyclist hood',(bx,by-.035,2.39),(.126,.125,.15),coat)
    rod('Cyclist raised raincoat collar',(bx,by-.035,2.16),(bx,by-.035,2.29),.095,coat)
    curve('Hood stitched seam',[(bx,by-.035-.128*math.sin(a),2.39+.153*math.cos(a)) for a in [i*math.pi/28 for i in range(18)]],.0015,black)
    packmat=textured('Weathered waxed-canvas backpack',(.046,.034,.019),28,.78,.008)
    cube('Cyclist backpack', (bx,by-.285,1.88),(.36,.18,.46),packmat,.055)
    cube('Cyclist backpack pocket',(bx,by-.39,1.80),(.29,.055,.18),packmat,.026)
    for strap in [-1,1]:
        curve('Cyclist backpack shoulder strap',[(bx+strap*.13,by-.33,2.10),(bx+strap*.18,by-.11,2.15),(bx+strap*.17,by+.065,2.04),(bx+strap*.15,by+.07,1.75)],.022,packmat)
    for side in [-1,1]:
        rod('Coat upper sleeve',(bx+side*.18,by-.02,2.10),(bx+side*.24,by+.10,1.88),.071,coat)
        sphere('Coat elbow fold',(bx+side*.24,by+.10,1.88),(.072,.075,.08),coat)
        rod('Coat forearm sleeve',(bx+side*.24,by+.10,1.88),(bx+side*.21,by+.45,1.65),.059,coat)
        sphere('Cyclist hand',(bx+side*.21,by+.45,1.64),(.045,.047,.06),skin)
        hip=(bx+side*.11,by-.22,1.5);knee=(bx+side*.14,by+.28,1.10+side*.08);foot=(bx+side*.13,by+side*.14,.78+side*.08)
        rod('Bicycle pedal crank',(bx+side*.08,by,.78),foot,.011,chrome)
        cube('Bicycle pedal',(foot[0],foot[1],foot[2]-.035),(.14,.075,.025),black,.004)
        rod('Trouser upper leg',hip,knee,.078,pants);rod('Trouser lower leg',knee,foot,.057,pants);sphere('Rider shoe',(foot[0],foot[1]+.05,foot[2]),(.07,.16,.045),black)
    # Transparent umbrella: eight curved gores, real thin transmission.
    umbrella=bpy.data.materials.new('Clear thin vinyl | Fresnel sheet');umbrella.use_nodes=True
    ns=umbrella.node_tree.nodes;ls=umbrella.node_tree.links;ns.clear();out=ns.new('ShaderNodeOutputMaterial')
    transparent=ns.new('ShaderNodeBsdfTransparent');gloss=ns.new('ShaderNodeBsdfGlossy');gloss.inputs['Roughness'].default_value=.035
    fresnel=ns.new('ShaderNodeFresnel');fresnel.inputs['IOR'].default_value=1.46
    mix=ns.new('ShaderNodeMixShader');ls.new(fresnel.outputs[0],mix.inputs[0]);ls.new(transparent.outputs[0],mix.inputs[1]);ls.new(gloss.outputs[0],mix.inputs[2]);ls.new(mix.outputs[0],out.inputs['Surface'])
    verts=[];faces=[];ux=bx;uy=by-.03;uz=2.98
    verts.append((ux,uy,uz))
    for ring in range(1,13):
        rad=.78*ring/12
        for a in range(97):
            angle=a*math.tau/96;z=uz-.36*(rad/.78)**1.65-.025*math.sin(angle*4)**2*(rad/.78)
            verts.append((ux+rad*math.cos(angle),uy+rad*math.sin(angle),z))
    for a in range(96):faces.append((0,1+a,2+a))
    for ring in range(11):
        for a in range(96):q=1+ring*97+a;faces.append((q,q+97,q+98,q+1))
    data=bpy.data.meshes.new('Umbrella gores');data.from_pydata(verts,[],faces);data.materials.append(umbrella);ob=bpy.data.objects.new('Clear eight-panel umbrella',data);bpy.context.collection.objects.link(ob)
    bm=bmesh.new();bm.from_mesh(data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(data);bm.free()
    for p in data.polygons:p.use_smooth=True
    sol=ob.modifiers.new('Thin vinyl film','SOLIDIFY');sol.thickness=.0005
    for a in range(8):
        ang=a*math.tau/8
        curve('Umbrella steel rib',[(ux+.78*t/20*math.cos(ang),uy+.78*t/20*math.sin(ang),uz-.36*(t/20)**1.65-.008) for t in range(21)],.003,chrome)
    curve('Umbrella wet perimeter',[(ux+.78*math.cos(i*math.tau/96),uy+.78*math.sin(i*math.tau/96),uz-.36) for i in range(97)],.004,chrome)
    rod('Umbrella shaft',(ux,uy,1.75),(ux,uy,3.03),.006,chrome)
    # A three-quarter pose exposes the bicycle wheels and bent riding posture.
    bpy.context.view_layer.update()
    pivot=Vector((bx,by,ground));turn=Matrix.Translation((0,0,.18-(ground+.35-.34-.027)))@Matrix.Translation(pivot)@Matrix.Rotation(math.radians(-32),4,'Z')@Matrix.Translation(-pivot)
    for ob in set(bpy.context.scene.objects)-before_figure:ob.matrix_world=turn@ob.matrix_world
    # Street dressing: drain, wall menu, bollards and damp paper scraps.
    for y in [-3,4,10]:
        cube('Storm drain metal frame',(-2.5,y,.195),(.44,.66,.025),steel,.008)
        for k in range(10):cube('Storm drain slot',(-2.5,y-.28+k*.06,.211),(.35,.024,.004),black,0)
    for side in [-1,1]:
        for y in [0,7,13]:
            rod('Protective street bollard',(side*2.7,y,.44),(side*2.7,y,1.13),.075,steel)
            sphere('Bollard rounded cap',(side*2.7,y,1.13),(.075,.075,.04),steel)
    menu=cube('Ramen menu board',(-2.47,-.75,1.15),(.5,.06,.9),wood,.015)
    text('Menu title','らーめん',(-2.47,-.79,1.36),.11,cream,font)
    text('Menu price','¥ 850',(-2.47,-.79,1.12),.13,cream,font)
    for k in range(12):
        x=R.choice([-1,1])*R.uniform(2.7,2.82);y=R.uniform(-6,12)
        ob=cube('Damp fallen leaf',(x,y,.46),(.025,.06,.002),wood,0);ob.rotation_euler[2]=R.random()*6
    # Slight aerial perspective with low-density participating media.
    fog=bpy.data.materials.new('Humid night air');fog.use_nodes=True;n=fog.node_tree.nodes;n.clear();out=n.new('ShaderNodeOutputMaterial');v=n.new('ShaderNodeVolumePrincipled');v.inputs['Density'].default_value=.009;v.inputs['Color'].default_value=(.29,.37,.46,1);v.inputs['Anisotropy'].default_value=.2;fog.node_tree.links.new(v.outputs['Volume'],out.inputs['Volume']);cube('Night mist volume',(0,7,5),(10,40,14),fog,0)
    # Fine rain streaks are photographic atmospheric detail. Bulk water remains
    # solely the imported solver mesh, including all impact deformations.
    rain=mat('Backlit rain streak',(.4,.5,.6),.15,emission=.16)
    for i in range(300):
        x=R.uniform(-2.7,2.7);y=R.uniform(-6,18);z=R.uniform(.7,7)
        rod('Airborne rain streak',(x,y,z),(x-.012,y-.006,z-R.uniform(.06,.16)),.0006,rain,4)
    if water_mesh_path:
        bpy.ops.wm.obj_import(filepath=str(Path(water_mesh_path).resolve()),forward_axis='Y',up_axis='Z')
        water=mat('NB-FLIP water | 1.333 IOR',(.92,.97,1),.035)
        n=water.node_tree.nodes;l=water.node_tree.links;p=n.get('Principled BSDF');p.inputs['Transmission Weight'].default_value=1;p.inputs['IOR'].default_value=1.333
        # Subpixel capillary detail only; all visible silhouette and bulk motion
        # come from the exported fluid, not procedural geometry.
        tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=110;tex.inputs['Detail'].default_value=2
        bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.16;bump.inputs['Distance'].default_value=.00045;l.new(tex.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs[0],p.inputs['Normal'])
        absorption=n.new('ShaderNodeVolumeAbsorption');absorption.inputs['Color'].default_value=(.56,.68,.76,1);absorption.inputs['Density'].default_value=.13;l.new(absorption.outputs[0],n.get('Material Output').inputs['Volume'])
        for o in bpy.context.selected_objects:
            o.name='Live NB-FLIP water surface';o.data.materials.clear();o.data.materials.append(water)
            bm=bmesh.new();bm.from_mesh(o.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-5);bm.to_mesh(o.data);bm.free()
            for p in o.data.polygons:p.use_smooth=True
    world=bpy.data.worlds.new('Clouded midnight sky');world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.032,.051,.09,1);world.node_tree.nodes['Background'].inputs[1].default_value=.18;sc.world=world
    area('Soft sky over street',(0,2,10),(0,2,0),(.24,.4,.67),80,7)
    bpy.ops.object.camera_add(location=(.75,-10.4,1.35));cam=bpy.context.object;cam.name='Street photography · 35 mm';cam.rotation_euler=(Vector((-.05,5.8,2.6))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=37;cam.data.sensor_width=36
    cam.data.dof.use_dof=True;cam.data.dof.focus_distance=9;cam.data.dof.aperture_fstop=2.8;sc.camera=cam
    sc.render.engine='CYCLES';sc.cycles.samples=96;sc.cycles.use_denoising=True;sc.cycles.max_bounces=12;sc.cycles.transmission_bounces=8;sc.cycles.transparent_max_bounces=8
    sc.render.resolution_x=1600;sc.render.resolution_y=1000;sc.render.resolution_percentage=100
    sc.view_settings.view_transform='AgX';sc.view_settings.look='AgX - Medium High Contrast';sc.view_settings.exposure=.4
    if hasattr(sc,'compositing_node_group'):
        nt=bpy.data.node_groups.new('Photographic highlight bloom','CompositorNodeTree');sc.compositing_node_group=nt
        nt.interface.new_socket(name='Image',in_out='OUTPUT',socket_type='NodeSocketColor')
        comp=nt.nodes.new('NodeGroupOutput')
    else:
        sc.use_nodes=True;nt=sc.node_tree;nt.nodes.clear();comp=nt.nodes.new('CompositorNodeComposite')
    rl=nt.nodes.new('CompositorNodeRLayers');gl=nt.nodes.new('CompositorNodeGlare');
    if 'Type' in gl.inputs:
        gl.inputs['Type'].default_value='Fog Glow';gl.inputs['Quality'].default_value='High';gl.inputs['Threshold'].default_value=1.5;gl.inputs['Strength'].default_value=.25
    else:gl.glare_type='FOG_GLOW';gl.quality='HIGH'
    nt.links.new(rl.outputs['Image'],gl.inputs['Image']);nt.links.new(gl.outputs['Image'],comp.inputs['Image'])

    return sc


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--water-mesh');parser.add_argument('--output',default=str(ROOT/'artifacts/photoreal/tokyo-rain.png'));parser.add_argument('--samples',type=int,default=96);parser.add_argument('--width',type=int,default=1600);parser.add_argument('--height',type=int,default=1000);parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    sc=build_scene(args.water_mesh);sc.cycles.samples=args.samples;sc.render.resolution_x=args.width;sc.render.resolution_y=args.height
    try:
        prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
        for d in prefs.devices:d.use=d.type=='METAL'
        if any(d.type=='METAL' for d in prefs.devices):sc.cycles.device='GPU'
    except (TypeError,RuntimeError):pass
    out=Path(args.output).resolve();out.parent.mkdir(parents=True,exist_ok=True);sc.render.filepath=str(out);sc.render.image_settings.file_format='PNG'
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')))
    if not args.no_render:bpy.ops.render.render(write_still=True)

if __name__=='__main__':main()
