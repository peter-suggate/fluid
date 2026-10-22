import { fluidSolidWorldForScene, sampleSolidWorld, type SolidWorld, type SolidWorldPage } from '../../../core/solid-world';
import { proceduralSolidField, sampleSolidNodes } from '../../../core/procedural-solid';
import { sceneCellSizes_m } from '../../../core/scene-lattice-dimensions';
import type { SceneDescription, Vec3 } from '../../../core/model';
export type CollisionShape = { kind: 'box'; center: Vec3; half: Vec3 } | { kind: 'hull'; points: Vec3[] };
type HullShape = Extract<CollisionShape, { kind: 'hull' }>;
export interface CollisionChunk { key: string; signature: string; shapes: CollisionShape[] }
type SampledBrick = { values: readonly number[]; valid: Uint32Array };
type CachedChunk = { page?: SolidWorldPage; values?: readonly number[]; valid?: Uint32Array; chunk?: CollisionChunk };
/** Owned by one physics world. Reuse only immutable source pages and identical
 * sampled validity; an edited hole must never retain its former convex hull. */
export class SolidCollisionCache {
  private context = '';
  private ground?: SolidWorld['ground'];
  private entries = new Map<string, CachedChunk>();
  prepare(context: string, ground: SolidWorld['ground']) {
    if (context !== this.context || ground !== this.ground) this.entries.clear();
    this.context = context; this.ground = ground;
  }
  get(key: string, page: SolidWorldPage | undefined, sampled: SampledBrick | undefined): CachedChunk | undefined {
    const entry = this.entries.get(key);
    if (!entry || entry.page !== page || entry.values !== sampled?.values
      || Boolean(entry.valid) !== Boolean(sampled?.valid)) return;
    if (entry.valid && sampled && entry.valid.some((word, index) => word !== sampled.valid[index])) return;
    return entry;
  }
  set(key: string, page: SolidWorldPage | undefined, sampled: SampledBrick | undefined, chunk?: CollisionChunk) {
    this.entries.set(key, { page, values: sampled?.values, valid: sampled?.valid.slice(), chunk });
  }
  retain(keys: ReadonlySet<string>) { for (const key of this.entries.keys()) if (!keys.has(key)) this.entries.delete(key); }
}
const corners = Array.from({ length: 8 }, (_, i) => [i & 1, (i >> 1) & 1, (i >> 2) & 1]);
const tetrahedra = [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]];
type Plane = { point: Vec3; normal: Vec3 };
const subtract = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const dot = (a: Vec3, b: Vec3) => a.x * b.x + a.y * b.y + a.z * b.z;
/** Merge only a certified convex sampled volume. Every boundary triangle's
 * outward plane must contain every solid vertex on its inside. A cavity or
 * concavity fails this test and keeps the original clipped tetrahedra. */
function mergeConvexSlice(shapes: HullShape[], planes: Plane[]): HullShape[] {
  if (!planes.length || shapes.length < 2) return shapes;
  const points = [...new Map(shapes.flatMap(shape => shape.points)
    .map(p => [[p.x, p.y, p.z].map(v => v.toFixed(8)).join(','), p] as const)).values()];
  if (planes.some(plane => points.some(p => dot(subtract(p, plane.point), plane.normal) > 1e-6))) return shapes;
  return [{ kind: 'hull', points }];
}
/** Fixed-resolution physics geometry. Never samples camera-dependent SVO meshes. */
export function compileSolidChunks(scene: SceneDescription, positions: readonly Vec3[] = [], cache?: SolidCollisionCache): CollisionChunk[] {
  const world = fluidSolidWorldForScene(scene), field = proceduralSolidField(scene), h = sceneCellSizes_m(scene);
  const origin = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  cache?.prepare(JSON.stringify([origin, h, world.regions]), world.ground);
  const sourcePages = cache ? new Map(world.pages.map(page => [page.coordinate.join(','), page])) : undefined;
  const pages = new Set<string>(world.pages.map(page => page.coordinate.join(',')));
  for (const brick of scene.solidBoundary?.bricks ?? []) pages.add(brick.origin.map(v => Math.floor(v / 8)).join(','));
  const addBox = (lo: readonly number[], hi: readonly number[]) => {
    for (let z = Math.floor(lo[2] / 8); z <= Math.floor(hi[2] / 8); z++) for (let y = Math.floor(lo[1] / 8); y <= Math.floor(hi[1] / 8); y++) for (let x = Math.floor(lo[0] / 8); x <= Math.floor(hi[0] / 8); x++) {
      pages.add(`${x},${y},${z}`);
      if (pages.size > 32768) throw new Error('Interactive collision coverage exceeds 32768 chunks');
    }
  };
  for (const region of world.regions ?? []) addBox(region.minimum, region.maximumExclusive.map(v => v - 1));
  if (world.ground) for (const p of positions) {
    const c = [p.x, p.y, p.z].map((v, a) => (v - origin[a]) / h[a]);
    const r = h.map(v => Math.ceil(3 / v));
    addBox(c.map((v, a) => v - r[a]), c.map((v, a) => v + r[a]));
  }
  const point = (p: readonly number[]): Vec3 => ({ x: origin[0] + p[0] * h[0], y: origin[1] + p[1] * h[1], z: origin[2] + p[2] * h[2] });
  const result: CollisionChunk[] = [];
  for (const key of [...pages].sort()) {
    const page = sourcePages?.get(key), sampled = field?.bricks.get(key), retained = cache?.get(key, page, sampled);
    if (retained) { if (retained.chunk) result.push(retained.chunk); continue; }
    const base = key.split(',').map(v => Number(v) * 8), full = new Uint8Array(512), shapes: CollisionShape[] = [];
    // Keep edge strips separate: interpolated SDF corners need not be convex,
    // even when the broad face between them is planar. One-cell slices let us
    // merge that face without bridging concavities or edited cells.
    const slices = Array.from({ length: 8 }, () => ({ shapes: [] as HullShape[], planes: [] as Plane[], complete: true }));
    for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const slice = slices[z];
      const c = [base[0] + x, base[1] + y, base[2] + z] as const, at = field?.at(c);
      if (!at) { slice.complete = false; full[x + 8 * (y + 8 * z)] = sampleSolidWorld(world, c).solidFraction > 0 ? 1 : 0; continue; }
      // Exact surface nodes can acquire roundoff-sized signed distances.
      // Snap these before clipping to avoid effectively zero-volume hulls.
      const values = corners.map(d => sampleSolidNodes(at.values, [at.local[0] + d[0], at.local[1] + d[1], at.local[2] + d[2]])[3]).map(v => Math.abs(v) < 1e-10 ? 0 : v);
      if (values.every(v => v <= 0)) { full[x + 8 * (y + 8 * z)] = 1; continue; }
      if (values.every(v => v >= 0)) continue;
      // Clip the six cell tetrahedra against phi=0. This preserves cavities and
      // provides a bounded piecewise-linear approximation of the sampled field.
      for (const tet of tetrahedra) {
        const points: number[][] = [];
        const surface: number[][] = [];
        for (const i of tet) if (values[i] <= 0) points.push(corners[i].map((v, a) => v + c[a]));
        for (const i of tet) if (values[i] === 0) surface.push(corners[i].map((v, a) => v + c[a]));
        for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
          const a = tet[i], b = tet[j];
          if ((values[a] < 0) === (values[b] < 0)) continue;
          const t = values[a] / (values[a] - values[b]);
          const p = corners[a].map((v, axis) => c[axis] + v + t * (corners[b][axis] - v));
          points.push(p); surface.push(p);
        }
        if (surface.length >= 3 && tet.some(i => values[i] > 0)) {
          // Solve the tetrahedron's linear field directly. Crossing triangles
          // can be arbitrarily small near a lattice node, making their cross
          // products too ill-conditioned to certify a supporting plane.
          const edges = tet.slice(1).map(i => corners[i].map((v, a) => v - corners[tet[0]][a]));
          const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
          const reciprocal = [cross(edges[1], edges[2]), cross(edges[2], edges[0]), cross(edges[0], edges[1])];
          const determinant = edges[0].reduce((sum, v, a) => sum + v * reciprocal[0][a], 0);
          const gradient = h.map((spacing, a) => reciprocal.reduce((sum, e, i) => sum + e[a] * (values[tet[i + 1]] - values[tet[0]]), 0) / determinant / spacing);
          const length = Math.hypot(...gradient);
          if (length > 1e-12) slice.planes.push({ point: point(surface[0]), normal: { x: gradient[0] / length, y: gradient[1] / length, z: gradient[2] / length } });
        }
        const unique = [...new Map(points.map(p => [p.map(v => v.toFixed(9)).join(','), p])).values()];
        if (unique.length >= 4) slice.shapes.push({ kind: 'hull', points: unique.map(point) });
      }
    }
    // Greedy boxes span complete adjacent cells in all three axes.
    const index = (x: number, y: number, z: number) => x + 8 * (y + 8 * z);
    for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      if (!full[index(x, y, z)]) continue;
      let X = x + 1, Y = y + 1, Z = z + 1;
      while (X < 8 && full[index(X, y, z)]) X++;
      while (Y < 8 && Array.from({ length: X - x }, (_, i) => full[index(x + i, Y, z)]).every(Boolean)) Y++;
      while (Z < 8 && Array.from({ length: (X - x) * (Y - y) }, (_, i) => full[index(x + i % (X - x), y + Math.floor(i / (X - x)), Z)]).every(Boolean)) Z++;
      for (let zz = z; zz < Z; zz++) for (let yy = y; yy < Y; yy++) for (let xx = x; xx < X; xx++) full[index(xx, yy, zz)] = 0;
      shapes.push({ kind: 'box', center: point([base[0] + (x + X) / 2, base[1] + (y + Y) / 2, base[2] + (z + Z) / 2]), half: { x: (X - x) * h[0] / 2, y: (Y - y) * h[1] / 2, z: (Z - z) * h[2] / 2 } });
    }
    for (const slice of slices) shapes.push(...(slice.complete ? mergeConvexSlice(slice.shapes, slice.planes) : slice.shapes));
    const chunk = shapes.length ? { key, shapes, signature: JSON.stringify(shapes) } : undefined;
    if (chunk) result.push(chunk);
    cache?.set(key, page, sampled, chunk);
  }
  cache?.retain(pages);
  return result;
}
