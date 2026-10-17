import { fluidSolidWorldForScene, sampleSolidWorld } from '../../../core/solid-world';
import { proceduralSolidField, sampleSolidNodes } from '../../../core/procedural-solid';
import { sceneCellSizes_m } from '../../../core/scene-lattice-dimensions';
import type { SceneDescription, Vec3 } from '../../../core/model';
export type CollisionShape = { kind: 'box'; center: Vec3; half: Vec3 } | { kind: 'hull'; points: Vec3[] };
export interface CollisionChunk { key: string; signature: string; shapes: CollisionShape[] }
const corners = Array.from({ length: 8 }, (_, i) => [i & 1, (i >> 1) & 1, (i >> 2) & 1]);
const tetrahedra = [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]];
/** Fixed-resolution physics geometry. Never samples camera-dependent SVO meshes. */
export function compileSolidChunks(scene: SceneDescription, positions: readonly Vec3[] = []): CollisionChunk[] {
  const world = fluidSolidWorldForScene(scene), field = proceduralSolidField(scene), h = sceneCellSizes_m(scene);
  const origin = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
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
    const base = key.split(',').map(v => Number(v) * 8), full = new Uint8Array(512), shapes: CollisionShape[] = [];
    for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const c = [base[0] + x, base[1] + y, base[2] + z] as const, at = field?.at(c);
      if (!at) { full[x + 8 * (y + 8 * z)] = sampleSolidWorld(world, c).solidFraction > 0 ? 1 : 0; continue; }
      const values = corners.map(d => sampleSolidNodes(at.values, [at.local[0] + d[0], at.local[1] + d[1], at.local[2] + d[2]])[3]);
      if (values.every(v => v <= 0)) { full[x + 8 * (y + 8 * z)] = 1; continue; }
      if (values.every(v => v >= 0)) continue;
      // Clip the six cell tetrahedra against phi=0. This preserves cavities and
      // provides a bounded piecewise-linear approximation of the sampled field.
      for (const tet of tetrahedra) {
        const points: number[][] = [];
        for (const i of tet) if (values[i] <= 0) points.push(corners[i].map((v, a) => v + c[a]));
        for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
          const a = tet[i], b = tet[j];
          if ((values[a] < 0) === (values[b] < 0)) continue;
          const t = values[a] / (values[a] - values[b]);
          points.push(corners[a].map((v, axis) => c[axis] + v + t * (corners[b][axis] - v)));
        }
        const unique = [...new Map(points.map(p => [p.map(v => v.toFixed(9)).join(','), p])).values()];
        if (unique.length >= 4) shapes.push({ kind: 'hull', points: unique.map(point) });
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
    if (shapes.length) result.push({ key, shapes, signature: JSON.stringify(shapes) });
  }
  return result;
}
