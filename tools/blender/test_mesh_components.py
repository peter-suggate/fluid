"""Analytic overlap, reversed winding and connected cavity regression."""
import bpy,sys
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mesh_components import triangle_components,split_part
from mesh_sampling import inside

def part(vertices,triangles):
    return dict(vertices=vertices,triangles=triangles,normals=[(Vector((1,0,0)),)*3]*len(triangles),ids=list(range(len(triangles))),uvs=None,texture_lods=[],lo=tuple(min(v[a] for v in vertices) for a in range(3)),hi=tuple(max(v[a] for v in vertices) for a in range(3)),bvh=BVHTree.FromPolygons(vertices,triangles,all_triangles=True))
def box(center,reverse=False):
    bpy.ops.mesh.primitive_cube_add(size=2,location=center)
    obj=bpy.context.object;m=obj.data;m.calc_loop_triangles()
    verts=[obj.matrix_world@v.co for v in m.vertices]
    tris=[tuple(reversed(t.vertices)) if reverse else tuple(t.vertices) for t in m.loop_triangles]
    return verts,tris

def signed(p,q):
    d=q['bvh'].find_nearest(p)[3]
    return -d if inside(q['bvh'],p,1e-6,(q['lo'],q['hi'])) else d

checks=0
for reverse in (False,True):
    a,ta=box((-.5,0,0));b,tb=box((.5,0,0),reverse)
    combined=part(a+b,ta+[tuple(i+len(a) for i in t) for t in tb]);pieces=split_part(combined)
    assert len(pieces)==2
    for point,expected in [((0,0,0),True),((1.25,0,0),True),((-1.25,0,0),True),((2,0,0),False)]:
        p=Vector(point);assert (min(signed(p,q) for q in pieces)<0)==expected;checks+=1
    assert not inside(combined['bvh'],Vector((0,0,0)),1e-6,(combined['lo'],combined['hi']))
    # Attribute identity follows original triangles exactly, including winding.
    for q in pieces:
        for local,original in enumerate(q['ids']):
            assert [tuple(q['vertices'][v]) for v in q['triangles'][local]]==[tuple(combined['vertices'][v]) for v in combined['triangles'][original]]
    checks+=1
bpy.ops.mesh.primitive_torus_add(major_segments=48,minor_segments=16,major_radius=2,minor_radius=.5)
obj=bpy.context.object;m=obj.data;m.calc_loop_triangles();v=[x.co.copy() for x in m.vertices];t=[tuple(x.vertices) for x in m.loop_triangles]
q=part(v,t);pieces=split_part(q);assert len(pieces)==1
for point,expected in [((0,0,0),False),((2,0,0),True),((3,0,0),False)]:assert (min(signed(Vector(point),p) for p in pieces)<0)==expected;checks+=1
# Duplicated corners (GLB UV seams) must weld without modifying their positions.
v2=[];t2=[]
for tri in t:
 t2.append(tuple(range(len(v2),len(v2)+3)));v2.extend(v[i].copy() for i in tri)
assert len(triangle_components(v2,t2))==1;checks+=1
print(f'COMPONENT_UNION_FIXTURE: {checks} topology, overlap and cavity cases passed')
