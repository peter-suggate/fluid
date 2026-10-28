"""Connected, coincident-vertex welded triangle groups for explicit solid unions.

This partitions topology only; it never changes source positions, attributes or
winding. A connected concave shell keeps its own odd/even interior, while separate
closed overlapping shells can be unioned using min(component signed distance).
"""
def triangle_components(vertices, triangles, tolerance=1e-6):
    import math
    parents=list(range(len(vertices)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    bins={}
    for i,v in enumerate(vertices):
        key=tuple(math.floor(float(x)/tolerance) for x in v)
        for z in range(-1,2):
            for y in range(-1,2):
                for x in range(-1,2):
                    for j in bins.get((key[0]+x,key[1]+y,key[2]+z),()):
                        if sum((float(v[a])-float(vertices[j][a]))**2 for a in range(3))<=tolerance*tolerance:
                            union(i,j)
        bins.setdefault(key,[]).append(i)
    for a,b,c in triangles:union(a,b);union(b,c)
    groups={}
    for i,t in enumerate(triangles):groups.setdefault(find(t[0]),[]).append(i)
    return list(groups.values())


def split_part(part):
    """Split BVH input while preserving every triangle's corner attributes."""
    from mathutils.bvhtree import BVHTree
    import numpy as np
    groups=triangle_components(part['vertices'],part['triangles'])
    result=[]
    for indices in groups:
        used=sorted({v for i in indices for v in part['triangles'][i]})
        mapping={v:i for i,v in enumerate(used)}
        vertices=[part['vertices'][v] for v in used]
        triangles=[tuple(mapping[v] for v in part['triangles'][i]) for i in indices]
        bounds=np.array([tuple(v) for v in vertices])
        component={'vertices':vertices,'triangles':triangles,
            'bvh':BVHTree.FromPolygons(vertices,triangles,all_triangles=True),
            'lo':bounds.min(axis=0),'hi':bounds.max(axis=0)}
        for attribute in ('normals','ids','uvs','texture_lods'):
            values=part[attribute]
            component[attribute]=[values[i] for i in indices] if values else values
        result.append(component)
    return result
