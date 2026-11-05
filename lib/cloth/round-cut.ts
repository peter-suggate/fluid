import type { SceneDescription } from '../core/model';
import type { SolidWorldVoxelPatch } from '../core/solid-world';
import { sceneCellSizes_m } from '../core/scene-lattice-dimensions';
import { solidPatchKindId } from '../core/solid-patch/kind';
import { subcellBounds } from '../core/solid-patch/ellipsoid-bounds';
import type { ClothCutHalfSpace, ClothStrokeCut } from './edit';
import type { ClothVector } from './types';

/**
 * The sculpt shapes a cloth surface can be cut by, read from the same patch the
 * voxel solids rasterize.
 *
 * A box, or a cuboid at subcell bounds, is clipped exactly. The rounded kinds
 * (the pen's nib, the shape tool's sphere and cylinder) are the inscribed
 * quadric of `../core/solid-patch/inscribed-quadric`; a planar mesh can only
 * hold a polygonal boundary of it, so the region is the intersection of
 * half-spaces whose distance from the quadric is at most an eighth of a cell —
 * the resolution the pen centres its stamps at, and finer than the solid
 * lattice resolves the same shape. Kinds without a surface operation here keep
 * their own id, and the cloth edit refuses them explicitly.
 */
export function clothCutForSolidPatch(scene: Pick<SceneDescription, 'container' | 'voxelDomain'>, patch: SolidWorldVoxelPatch): ClothStrokeCut {
  const kind = solidPatchKindId(patch), cell = sceneCellSizes_m(scene);
  const origin = [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m];
  const toBox = (minimum: readonly number[], maximum: readonly number[]) => ({
    minimum_m: minimum.map((v, a) => origin[a]! + v * cell[a]!) as unknown as ClothVector,
    maximum_m: maximum.map((v, a) => origin[a]! + v * cell[a]!) as unknown as ClothVector,
  });
  if (kind === 'box') return {kind, box: toBox(patch.minimum, patch.maximumExclusive)};
  if (kind !== 'cuboid' && kind !== 'ellipsoid' && kind !== 'cylinder') return {kind, box: toBox(patch.minimum, patch.maximumExclusive)};
  const subcell = patch.payload !== undefined, bounds = subcell ? subcellBounds(patch) : {minimum: patch.minimum, maximumExclusive: patch.maximumExclusive};
  const box = toBox(bounds.minimum, bounds.maximumExclusive);
  if (kind === 'cuboid') return {kind: 'box', box};
  const flatAxis = kind === 'cylinder' ? (patch.axis ?? 1) : undefined;
  // Same flatness rule as the inscribed quadric: one whole cell is not a curve.
  const half = [0, 1, 2].map(a => 0.5 * (bounds.maximumExclusive[a]! - bounds.minimum[a]!));
  const radii = half.map((h, a) => a !== flatAxis && (h > 0.5 || subcell) ? h * cell[a]! : Infinity) as unknown as ClothVector;
  const centre = [0, 1, 2].map(a => 0.5 * (box.minimum_m[a]! + box.maximum_m[a]!)) as unknown as ClothVector;
  const halfSpaces = clothRoundCutHalfSpaces(centre, radii, Math.min(...cell) / 8);
  return halfSpaces.length ? {kind, box, halfSpaces} : {kind: 'box', box};
}

/** Half-spaces n·x <= offset whose intersection lies within `tolerance_m` of
 * the axis-aligned ellipsoid (three finite radii) or elliptic cylinder (two;
 * an infinite radius is a flat axis). Fewer than two round axes is the box. */
export function clothRoundCutHalfSpaces(centre_m: ClothVector, radii_m: ClothVector, tolerance_m: number): ClothCutHalfSpace[] {
  if (!(tolerance_m > 0) || !centre_m.every(Number.isFinite) || !radii_m.every(r => r > 0))
    throw new Error('Rounded cloth cut needs a finite centre, positive radii and a positive tolerance');
  const round = [0, 1, 2].filter(a => Number.isFinite(radii_m[a]!));
  if (round.length < 2) return [];
  const largest = Math.max(...round.map(a => radii_m[a]!));
  let directions: readonly ClothVector[] | undefined, cover = 0;
  if (round.length === 2) {
    for (let k = 8; ; k *= 2) {
      cover = Math.PI / k;
      if (largest * deviation(cover) <= tolerance_m || k >= 1024) {
        directions = Array.from({length: k}, (_, i) => {
          const u: [number, number, number] = [0, 0, 0], angle = 2 * Math.PI * i / k;
          u[round[0]!] = Math.cos(angle); u[round[1]!] = Math.sin(angle);
          return u;
        });
        break;
      }
    }
  } else {
    for (let level = 0; ; level++) {
      const sphere = geodesic(level);
      cover = sphere.cover;
      if (largest * deviation(cover) <= tolerance_m || level >= 4) { directions = sphere.directions; break; }
    }
  }
  // The polytope tangent to the unit sphere overshoots by up to 1/cos(cover);
  // scaling it to straddle the quadric halves the worst distance either way.
  const scale = 2 / (1 + 1 / Math.cos(cover));
  return directions!.map(u => {
    const n = [0, 1, 2].map(a => Number.isFinite(radii_m[a]!) ? u[a]! / radii_m[a]! : 0), length = Math.hypot(...n);
    const normal = n.map(v => v / length) as unknown as ClothVector;
    return {normal, offset_m: normal[0] * centre_m[0] + normal[1] * centre_m[1] + normal[2] * centre_m[2] + scale / length};
  });
}

/** Relative distance between the straddling polytope and the unit sphere. */
const deviation = (cover: number) => (1 / Math.cos(cover) - 1) / (1 + 1 / Math.cos(cover));

const spheres: {directions: ClothVector[]; cover: number}[] = [];
/** Icosphere vertex directions; `cover` is the largest angle from a face's
 * centre to its corners, where the tangent polytope stands farthest out. */
function geodesic(level: number): {directions: readonly ClothVector[]; cover: number} {
  const held = spheres[level];
  if (held) return held;
  const t = (1 + Math.sqrt(5)) / 2;
  let vertices: ClothVector[] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let faces: [number, number, number][] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const unit = (v: readonly number[]): ClothVector => { const l = Math.hypot(...v); return [v[0]! / l, v[1]! / l, v[2]! / l]; };
  vertices = vertices.map(unit);
  for (let l = 0; l < level; l++) {
    const middle = new Map<string, number>(), next: [number, number, number][] = [];
    const mid = (a: number, b: number) => {
      const key = a < b ? `${a},${b}` : `${b},${a}`, old = middle.get(key);
      if (old !== undefined) return old;
      vertices.push(unit([0, 1, 2].map(k => vertices[a]![k]! + vertices[b]![k]!)));
      middle.set(key, vertices.length - 1);
      return vertices.length - 1;
    };
    for (const [a, b, c] of faces) {
      const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
  }
  let cover = 0;
  for (const [a, b, c] of faces) {
    // The tangent planes at a face's three directions meet on its circumcentre.
    const p = vertices[a]!, u = [0, 1, 2].map(k => vertices[b]![k]! - p[k]!), v = [0, 1, 2].map(k => vertices[c]![k]! - p[k]!);
    const centre = unit([u[1]! * v[2]! - u[2]! * v[1]!, u[2]! * v[0]! - u[0]! * v[2]!, u[0]! * v[1]! - u[1]! * v[0]!]);
    cover = Math.max(cover, Math.acos(Math.min(1, Math.abs(centre[0] * p[0] + centre[1] * p[1] + centre[2] * p[2]))));
  }
  return spheres[level] = {directions: vertices, cover};
}
