"""Closed coconut-palm geometry with curved, paired pinnate leaflets.

Three age groups give the crown upright spears, broad green fronds and lower
drooping fronds. Fine geometry survives the native 12.5 mm SVO asset bake.
"""
import math
import random
import bpy
import numpy as np
from mathutils import Vector
from base_color_image import linear_base_color_image


def build_palm(mesh, material):
    rng=random.Random(8913)
    result=[]
    bark=material('Coconut trunk | fibrous leaf scars',(.28,.22,.15),.88)
    yy,xx=np.mgrid[0:1:2048j,0:1:512j]
    nr=np.random.default_rng(8913)
    scars=np.maximum(0,1-np.abs(np.sin(yy*math.pi*52+.12*np.sin(xx*25)))/.18)
    fibres=.028*np.sin(xx*240+np.sin(yy*19))+.018*np.sin(xx*630+yy*2)
    pixels=np.ones((*xx.shape,4),dtype=np.float32)
    for c,base in enumerate((.31,.25,.18)):
        pixels[:,:,c]=np.clip(base+fibres-.085*scars+nr.normal(0,.015,xx.shape),0,1)
    n,l=bark.node_tree.nodes,bark.node_tree.links
    tex=n.new('ShaderNodeTexImage');tex.image=linear_base_color_image('Coconut trunk fibres and irregular leaf scars',pixels)
    l.new(tex.outputs['Color'],n.get('Principled BSDF').inputs['Base Color'])
    bump=n.new('ShaderNodeBump');bump.inputs['Distance'].default_value=.012;bump.inputs['Strength'].default_value=.4
    l.new(tex.outputs['Color'],bump.inputs['Height']);l.new(bump.outputs[0],n.get('Principled BSDF').inputs['Normal'])
    greens=[material('Coconut leaf '+str(i),c,.57) for i,c in enumerate([
        (.065,.14,.016),(.09,.19,.025),(.13,.22,.035),(.17,.25,.047),(.20,.23,.065),(.13,.17,.025)])]
    brown=material('Old coconut leaf sheath',(.22,.14,.065),.87)
    nut=material('Green coconuts',(.22,.25,.07),.6)
    rings,sides=181,24;v=[]
    for j in range(rings):
        t=j/(rings-1)
        radius=(.23-.095*t+.065*math.exp(-t*25))*(1+.022*math.sin(t*math.tau*52))
        for k in range(sides):
            a=k*math.tau/sides
            v.append((.64*t*t+radius*math.cos(a),.16*math.sin(t*2)+radius*math.sin(a),5.6*t))
    f=[tuple(reversed(range(sides))),tuple(range((rings-1)*sides,rings*sides))]
    f += [(j*sides+k,j*sides+(k+1)%sides,(j+1)*sides+(k+1)%sides,(j+1)*sides+k) for j in range(rings-1) for k in range(sides)]
    trunk=mesh('Coconut | tapered leaning trunk',v,f);trunk.data.materials.append(bark);result.append(trunk)
    uv=trunk.data.uv_layers.new(name='UVMap')
    for face in trunk.data.polygons:
        for loop in face.loop_indices:
            idx=trunk.data.loops[loop].vertex_index
            uv.data[loop].uv=(idx%sides/sides,idx//sides/(rings-1))
    crown=Vector((.64,.16*math.sin(2),5.6))
    for branch in range(27):
        a=branch*2.39996+rng.uniform(-.17,.17)
        if branch<7:length,lift,drop=rng.uniform(1.9,2.6),rng.uniform(2.2,3.1),rng.uniform(.2,.5)
        elif branch<20:length,lift,drop=rng.uniform(2.9,3.6),rng.uniform(.7,1.5),rng.uniform(.7,1.35)
        else:length,lift,drop=rng.uniform(2.4,3.1),rng.uniform(.1,.45),rng.uniform(1.8,2.5)
        forward=Vector((math.cos(a),math.sin(a),0));side=Vector((-math.sin(a),math.cos(a),0))
        def stem(t):return crown+forward*(length*t)+side*(.12*math.sin(t*3))+Vector((0,0,lift*math.sin(t*math.pi*.72)-drop*t*t))
        verts=[];faces=[]
        # Rachis is a continuous tapered closed tube, not disconnected cones.
        for j in range(25):
            t=j/24;c=stem(t);r=.033*(1-t)+.004
            for k in range(6):
                angle=k*math.tau/6;verts.append(tuple(c+side*(r*math.cos(angle))+Vector((0,0,r*math.sin(angle)))))
        faces.extend([tuple(reversed(range(6))),tuple(range(24*6,25*6))])
        faces.extend((j*6+k,j*6+(k+1)%6,(j+1)*6+(k+1)%6,(j+1)*6+k) for j in range(24) for k in range(6))
        # Alternating leaflets twist and droop; they are neither flat triangles
        # nor identical radial spikes. Closed folded ribbons have real thickness.
        for j in range(4,47):
            t=j/49
            for sign in (-1,1):
                # Stagger the pairs enough to survive glTF's float32 weld:
                # coincident closed leaf caps would become a four-face edge.
                root=stem(t+sign*.004+rng.uniform(-.001,.001))
                length_leaf=(.25+.8*math.sin(math.pi*t)**.8)*rng.uniform(.83,1.12)
                sweep=rng.uniform(.28,.55);droop=rng.uniform(.23,.48)
                direction=side*sign+forward*sweep
                transverse=forward-side*sign*sweep;transverse.normalize()
                width=rng.uniform(.027,.043);start=len(verts)
                for row in range(7):
                    u=row/6
                    c=root+direction*(length_leaf*u)+Vector((0,0,.08*math.sin(u*math.pi)-droop*u*u))
                    w=max(.002,width*math.sin(math.pi*(u*.94+.03))**.65)
                    fold=.012*math.sin(math.pi*u)
                    # Diamond section produces the central fold visible in sun.
                    for offset in [-transverse*w,Vector((0,0,.012+fold)),transverse*w,Vector((0,0,-.012))]:
                        verts.append(tuple(c+offset))
                faces.extend([tuple(start+i for i in (3,2,1,0)),tuple(start+24+i for i in range(4))])
                faces.extend((start+r*4+k,start+r*4+(k+1)%4,start+(r+1)*4+(k+1)%4,start+(r+1)*4+k) for r in range(6) for k in range(4))
        ob=mesh('Coconut | pinnate frond %02d'%branch,verts,faces)
        ob.data.materials.append(greens[branch%len(greens)]);result.append(ob)
    # Overlapping leaf bases and a small, irregular fruit cluster complete the
    # crown/trunk transition, which otherwise resembles a bundle on a pole.
    for i in range(11):
        a=i*2.39996
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1,location=crown+Vector((.24*math.cos(a),.24*math.sin(a),-.1-.09*(i%3))))
        ob=bpy.context.object;ob.name='Coconut fruit';ob.scale=(.12,.14,.18);ob.data.materials.append(nut)
        for p in ob.data.polygons:p.use_smooth=True
        result.append(ob)
    for i in range(8):
        a=i*math.tau/8
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1,location=crown+Vector((.11*math.cos(a),.11*math.sin(a),-.18)))
        ob=bpy.context.object;ob.name='Coconut fibrous leaf base';ob.scale=(.16,.16,.42);ob.data.materials.append(brown);result.append(ob)
    return result
