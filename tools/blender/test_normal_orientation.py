"""Blender fixture: closed reversed-winding solids retain outward FVX optics.

Run blender -b --python-exit-code 1 --python this-file.py.
The signed gradient is derived from analytic inside/outside, independently of
the BVH triangle normals, as the volume baker's parity classifier does.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from normal_orientation import outward_shading_normal, signed_outward_gradient

quads=[(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
triangles=[t for a,b,c,d in quads for t in ((a,b,c),(a,c,d))]
checks=0
for centre,scale in [(Vector((0,0,0)),Vector((1,1,1))),
                     (Vector((1000,-250,300)),Vector((1,1,1))),
                     (Vector((24,-13,19)),Vector((.2,.11,.4)))]:
    vertices=[Vector((x*scale.x,y*scale.y,z*scale.z))+centre for z in (-1,1) for y in (-1,1) for x in (-1,1)]
    for reversed_winding in (False,True):
        faces=[tuple(reversed(t)) if reversed_winding else t for t in triangles]
        bvh=BVHTree.FromPolygons(vertices,faces,all_triangles=True)
        for axis in range(3):
            for side in (-1,1):
                for distance in (.94,1.0,1.06):
                    p=Vector((.13*scale.x,.21*scale.y,.17*scale.z))+centre;p[axis]=centre[axis]+side*distance*scale[axis]
                    nearest,face,_,d=bvh.find_nearest(p)
                    gradient=signed_outward_gradient(p,nearest,d,-1 if distance<1 else 1,face,
                        lambda q: all(abs(q[a]-centre[a])<scale[a] for a in range(3)),1e-4)
                    expected=Vector((0,0,0));expected[axis]=side
                    assert gradient.dot(expected)>.99999, (reversed_winding,axis,side,distance,d,tuple(gradient),tuple(face))
                    assert (face.dot(expected)<0)==reversed_winding
                    # Model a nontrivial normal-map perturbation in the source's
                    # hemisphere, then verify its detail is retained after repair.
                    tangent=Vector((0,0,0));tangent[(axis+1)%3]=.3
                    mapped=(expected+tangent).normalized()
                    authored=-mapped if reversed_winding else mapped
                    fixed=outward_shading_normal(authored,gradient)
                    assert (fixed-mapped).length<1e-6
                    if not reversed_winding:assert fixed is authored
                    assert fixed.dot(expected)>.95
                    checks+=1
# Degenerate/unowned samples must not invent a normal.
zero=Vector((0,0,0));assert outward_shading_normal(zero,zero) is zero
print(f'OUTWARD_NORMAL_FIXTURE: {checks} closed-surface cases passed')
