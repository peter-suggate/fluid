import type { SceneDescription } from '../core/model';
import { sceneLatticeDimensions } from '../core/scene-lattice-dimensions';
import { bakeProceduralSolid } from '../core/procedural-solid';
import { sourceFieldDistance } from './source-fields';
import type { WetClothSourceField } from './source-types';
import type { ClothVector } from './types';

/** Sample the actual source SDF at the original fluid lattice nodes. The
 * existing procedural-solid representation is one authority for rendering,
 * hit testing, cell/face collision and local edits; there is no proxy AABB. */
export function withStaticSourceSolids(scene: SceneDescription): SceneDescription {
  const program = scene.wetClothSource;
  if (!program) return scene;
  const moving = new Set(program.scripts.map(s => s.group));
  const roots = program.fields.filter(f => f.root && f.usage === 'solid' && !moving.has(f.group));
  if (!roots.length) return scene;
  if (scene.solidBoundary || scene.solidVoxels.length) throw new Error('Source solid compilation needs an empty authored solid layer');
  const dimensions = sceneLatticeDimensions(scene), h = scene.voxelDomain.finestCellSize_m;
  if (dimensions.some(n => n % 8) || [scene.container.width_m,scene.container.height_m,scene.container.depth_m].some((v,i) => Math.abs(v / dimensions[i]! - h) > 1e-12)) throw new Error('Source solids require their unchanged isotropic lattice spacing');
  const origin: ClothVector = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  const exact = (p: readonly number[]) => {
    const world: ClothVector = [origin[0]+p[0]!*h, origin[1]+p[1]!*h, origin[2]+p[2]!*h];
    return Math.min(...roots.map(f => sourceFieldDistance(program.fields, f.index, world))) / h;
  };
  const baked = bakeProceduralSolid(dimensions, [h,h,h], staticSourceBrickCull(dimensions, exact, roots.some(f => sourceFieldUsesMesh(program.fields, f.index)) ? Infinity : 1));
  // Same output as sampling every node: a far brick is skipped or filled, as its
  // exact samples would have it, and every node of a near brick is exact.
  const hasMovingSolids = program.fields.some(f => f.root && f.usage === 'solid' && moving.has(f.group));
  const requiredCapabilities = [...program.requiredCapabilities.filter(c => c !== 'source-solid-collision' || hasMovingSolids), 'source-static-sdf'];
  return {...scene, ...baked, wetClothSource: {...program, requiredCapabilities, bakedSolidFieldIndices: roots.map(f => f.index)}};
}

/** Brick-level cull for bakeProceduralSolid's sampler, in cell units. A brick
 * whose centre lies further from the zero set than its half-diagonal times the
 * field's Lipschitz bound has nodes of one sign, so the bake skips (air) or
 * fills (solid) it whatever the values; nodes no near brick holds return that
 * sign instead of an exact sample. Primitive, intersection and union distances
 * are 1-Lipschitz; a mesh field (`lipschitz` Infinity) is sampled everywhere. */
export function staticSourceBrickCull(dimensions: readonly [number, number, number], exact: (p: readonly number[]) => number, lipschitz: number): (p: readonly [number, number, number]) => number {
  if (!(lipschitz >= 1)) throw new RangeError('A distance field has Lipschitz bound of at least one');
  if (lipschitz === Infinity) return exact;
  const n = dimensions.map(v => v / 8), reach = 4 * Math.sqrt(3) * lipschitz, sign = new Int8Array(n[0]! * n[1]! * n[2]!);
  for (let z = 0; z < n[2]!; z++) for (let y = 0; y < n[1]!; y++) for (let x = 0; x < n[0]!; x++) {
    const d = exact([8 * x + 4, 8 * y + 4, 8 * z + 4]);
    if (!Number.isFinite(d)) throw new RangeError('Static source field must be finite');
    // 1e-6 cells keeps every far node's f32 sample strictly signed.
    sign[x + n[0]! * (y + n[1]! * z)] = Math.abs(d) - reach > 1e-6 ? Math.sign(d) : 0;
  }
  return p => {
    const range = p.map((v, a) => [v % 8 === 0 && v > 0 ? v / 8 - 1 : v >> 3, Math.min(v >> 3, n[a]! - 1)]);
    let found = 0;
    for (let z = range[2]![0]!; z <= range[2]![1]!; z++) for (let y = range[1]![0]!; y <= range[1]![1]!; y++) for (let x = range[0]![0]!; x <= range[0]![1]!; x++) {
      const s = sign[x + n[0]! * (y + n[1]! * z)]!;
      if (!s) return exact(p);
      if (found && s !== found) throw new Error('Static source cull found a node in far bricks of both signs');
      found = s;
    }
    return found;
  };
}

/** True when a source field's distance reads a mesh (directly or through an intersection). */
export function sourceFieldUsesMesh(fields: readonly WetClothSourceField[], index: number, seen = new Set<number>()): boolean {
  const f = fields.find(v => v.index === index);
  if (!f || seen.has(index)) throw new Error(`Invalid or cyclic source field ${index}`);
  if (f.kind === 'file') return true;
  seen.add(index);
  return (f.children ?? []).some(c => sourceFieldUsesMesh(fields, c, new Set(seen)));
}
