"""Shared closed-mesh parity and sampled normal helpers for offline bakes."""
from mathutils import Vector
RAY = Vector((1, .371390676, .529113257)).normalized()

def oct16(n):
    if n.length_squared < 1e-20: return 0
    n = n/(abs(n.x)+abs(n.y)+abs(n.z))
    x, y = n.x, n.y
    if n.z < 0:
        x, y = (1-abs(y))*(1 if x >= 0 else -1), (1-abs(x))*(1 if y >= 0 else -1)
    return int(round((x*.5+.5)*65535)) | (int(round((y*.5+.5)*65535)) << 16)


def inside(bvh, point, epsilon, bounds=None):
    # Odd/even intersections classify concave closed components, including cup
    # interiors. A fixed oblique ray avoids axis-aligned edges in authored props.
    # BVH coordinates are float32. Sub-ULP advances can hit the same triangle
    # indefinitely, especially at the canopy's 3 m elevation.
    epsilon = max(epsilon, 1e-5)
    if bounds is not None:
        # Dense forest crowns can intersect many independent closed leaves.
        # Bound the walk by the ray's exit from the actual component AABB,
        # not an arbitrary intersection count or an unbounded retry loop.
        lo, hi = bounds
        exit_distance = min((float(hi[a])-point[a])/RAY[a] for a in range(3))
        if exit_distance < 0: return False
        epsilon = max(epsilon, max(abs(v) for v in point)*2e-6)
        travelled = 0.0
        count = 0
        seen = set()
        stalled = 0
        while travelled <= exit_distance + epsilon:
            p = point + RAY*travelled
            hit, _, tri, distance = bvh.ray_cast(p, RAY, exit_distance-travelled+epsilon)
            if hit is None: return count % 2 == 1
            if tri not in seen:
                count += 1; seen.add(tri); stalled = 0
            else:
                # A straight ray cannot cross one planar triangle twice.
                # Grazing float32 hits occasionally return that triangle again;
                # do not count it twice, and advance monotonically from the
                # original origin rather than accumulating rounded hit vectors.
                stalled += 1
            step = epsilon * (2 ** min(stalled, 8))
            hit_distance = max(0.0, (hit-point).dot(RAY))
            travelled = max(travelled+step, hit_distance+step)
            if len(seen) > 65536:
                raise ValueError('Bounded dense-mesh parity exceeded 65536 distinct triangles')
        return count % 2 == 1
    p = point.copy()
    count = 0
    for _ in range(256):
        hit, _, _, _ = bvh.ray_cast(p, RAY)
        if hit is None: return count % 2 == 1
        count += 1
        p = hit + RAY*epsilon
    raise ValueError('Ray parity failed to escape mesh')


def barycentric(p, a, b, c):
    u, v, w = b-a, c-a, p-a
    uu, uv, vv, wu, wv = u.dot(u), u.dot(v), v.dot(v), w.dot(u), w.dot(v)
    det = uu*vv-uv*uv
    if abs(det) < 1e-24: return (1, 0, 0)
    y, z = (vv*wu-uv*wv)/det, (uu*wv-uv*wu)/det
    return (1-y-z, y, z)



def atomic_write(path, data, compressed=False):
    """Publish a complete artifact without exposing a partial gzip to readers."""
    import gzip, os, tempfile
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as output:
            temporary=Path(output.name)
            if compressed:
                with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0) as gz:gz.write(data)
            else:output.write(data)
            output.flush();os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
