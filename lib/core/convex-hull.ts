/** A point as three numbers, the form generated geometry is authored in. */
export type HullPoint = readonly [number, number, number];

const sub = (a: HullPoint, b: HullPoint): HullPoint => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: HullPoint, b: HullPoint): HullPoint => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: HullPoint, b: HullPoint) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

interface Face { a: number; b: number; c: number; normal: HullPoint; offset: number }

/**
 * The vertices of the convex hull of `points`, in input order.
 *
 * A support map only needs the extreme points: the maximum of `dot(p, u)` over
 * a point cloud is attained at a hull vertex for every direction `u`, so the
 * hull's vertices give the cloud's support function exactly at a fraction of
 * its cost. Incremental construction: each point outside the current hull
 * (beyond a face by more than `tolerance`) replaces the faces it sees by a fan
 * from their horizon. Points within `tolerance` of the hull are dropped, so
 * the support of the result is within `tolerance` of the cloud's; callers
 * check that with `hullSupportDeficit`.
 *
 * Throws on a degenerate (flat or empty) cloud: a hull with no volume is not a
 * collision envelope, and silently returning one would leak.
 */
export function convexHullVertices(points: readonly HullPoint[], tolerance = 1e-9): HullPoint[] {
  const unique: HullPoint[] = [];
  const seen = new Set<string>();
  for (const p of points) { const key = p.join(","); if (!seen.has(key)) { seen.add(key); unique.push(p); } }
  if (unique.length < 4) throw new Error(`A convex hull needs four points, not ${unique.length}`);
  // Initial tetrahedron: the extreme pair along x, the farthest from their
  // line, and the farthest from their plane.
  let i0 = 0, i1 = 0;
  unique.forEach((p, i) => { if (p[0] < unique[i0]![0]) i0 = i; if (p[0] > unique[i1]![0]) i1 = i; });
  const line = sub(unique[i1]!, unique[i0]!);
  let i2 = -1, best = 0;
  unique.forEach((p, i) => { const d = Math.hypot(...cross(line, sub(p, unique[i0]!))); if (d > best) { best = d; i2 = i; } });
  if (i2 < 0 || best <= tolerance) throw new Error("A convex hull needs points that are not collinear");
  const planeNormal = cross(line, sub(unique[i2]!, unique[i0]!));
  let i3 = -1; best = 0;
  unique.forEach((p, i) => { const d = Math.abs(dot(planeNormal, sub(p, unique[i0]!))) / Math.hypot(...planeNormal); if (d > best) { best = d; i3 = i; } });
  if (i3 < 0 || best <= tolerance) throw new Error("A convex hull needs points that are not coplanar");
  const centroid: HullPoint = [0, 1, 2].map((a) => (unique[i0]![a]! + unique[i1]![a]! + unique[i2]![a]! + unique[i3]![a]!) / 4) as unknown as HullPoint;
  const face = (a: number, b: number, c: number): Face => {
    let normal = cross(sub(unique[b]!, unique[a]!), sub(unique[c]!, unique[a]!));
    const length = Math.hypot(...normal);
    normal = [normal[0] / length, normal[1] / length, normal[2] / length];
    // Orient outward: the interior centroid lies behind every face.
    if (dot(normal, sub(centroid, unique[a]!)) > 0) return face(a, c, b);
    return { a, b, c, normal, offset: dot(normal, unique[a]!) };
  };
  let faces: Face[] = [face(i0, i1, i2), face(i0, i1, i3), face(i0, i2, i3), face(i1, i2, i3)];
  // Farthest points first: the large hull forms early and most interior
  // points are then rejected by a single sweep of its faces.
  const order = unique.map((_, i) => i).filter((i) => i !== i0 && i !== i1 && i !== i2 && i !== i3)
    .sort((a, b) => Math.hypot(...sub(unique[b]!, centroid)) - Math.hypot(...sub(unique[a]!, centroid)));
  for (const index of order) {
    const p = unique[index]!;
    const visible = faces.filter((f) => dot(f.normal, p) - f.offset > tolerance);
    if (visible.length === 0) continue;
    const edges = new Set<string>();
    for (const f of visible) for (const [u, v] of [[f.a, f.b], [f.b, f.c], [f.c, f.a]]) edges.add(`${u},${v}`);
    const horizon: [number, number][] = [];
    for (const key of edges) { const [u, v] = key.split(",").map(Number) as [number, number]; if (!edges.has(`${v},${u}`)) horizon.push([u, v]); }
    faces = faces.filter((f) => !visible.includes(f));
    for (const [u, v] of horizon) faces.push(face(u, v, index));
  }
  const used = new Set<number>();
  for (const f of faces) { used.add(f.a); used.add(f.b); used.add(f.c); }
  return unique.filter((_, i) => used.has(i));
}

/** The support of `points` along `direction`: the largest `dot(p, direction)`. */
export function pointSupport(points: readonly HullPoint[], direction: HullPoint): number {
  let best = -Infinity;
  for (const p of points) best = Math.max(best, dot(p, direction));
  return best;
}

/**
 * How far the support of `vertices` falls short of the support of `cloud`,
 * worst case over `directions` — zero for an exact hull.
 */
export function hullSupportDeficit(cloud: readonly HullPoint[], vertices: readonly HullPoint[], directions: readonly HullPoint[]): number {
  let worst = 0;
  for (const d of directions) worst = Math.max(worst, pointSupport(cloud, d) - pointSupport(vertices, d));
  return worst;
}

/** `count` unit directions spread evenly over the sphere (Fibonacci lattice), plus the six axes. */
export function sphereDirections(count: number): HullPoint[] {
  const result: HullPoint[] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (2 * (i + 0.5)) / count, r = Math.sqrt(1 - y * y), t = golden * i;
    result.push([r * Math.cos(t), y, r * Math.sin(t)]);
  }
  return result;
}
