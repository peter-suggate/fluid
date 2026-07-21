/** Paper-compatible 18-bit same-or-finer local topology descriptor. */

import { createOctreePowerSite, type OctreePowerSite, type PowerVec3 } from "./octree-power-geometry";
import { OCTREE_CUBE_TRANSFORMS, transformPowerVector, type CubeTransform } from "./octree-power-topology";

export const OCTREE_POWER_SAME_OR_FINER_BITS = 18;
export const OCTREE_POWER_SAME_OR_FINER_MASK = (1 << OCTREE_POWER_SAME_OR_FINER_BITS) - 1;
export const OCTREE_POWER_SAME_OR_COARSER_FLAG = 0x8000_0000;
export const OCTREE_POWER_SAME_OR_COARSER_MASK = 0x1ff;

/** Six faces followed by twelve edges, in lexicographic signed-direction order. */
export const OCTREE_POWER_NEIGHBOR_DIRECTIONS: readonly PowerVec3[] = Object.freeze((() => {
  const directions: PowerVec3[] = [];
  for (let x = -1; x <= 1; x += 1) for (let y = -1; y <= 1; y += 1) for (let z = -1; z <= 1; z += 1) {
    const nonzero = Number(x !== 0) + Number(y !== 0) + Number(z !== 0);
    if (nonzero === 1) directions.push([x, y, z]);
  }
  for (let x = -1; x <= 1; x += 1) for (let y = -1; y <= 1; y += 1) for (let z = -1; z <= 1; z += 1) {
    const nonzero = Number(x !== 0) + Number(y !== 0) + Number(z !== 0);
    if (nonzero === 2) directions.push([x, y, z]);
  }
  return directions;
})());

const directionKey = (direction: PowerVec3) => direction.join(",");
const directionIndex = new Map(OCTREE_POWER_NEIGHBOR_DIRECTIONS.map((direction, index) => [directionKey(direction), index]));

export function validateSameOrFinerPowerDescriptor(descriptor: number): void {
  if (!Number.isSafeInteger(descriptor) || descriptor < 0 || (descriptor & ~OCTREE_POWER_SAME_OR_FINER_MASK) !== 0) {
    throw new RangeError("Same-or-finer power descriptor must be an unsigned 18-bit integer");
  }
}

/** A set bit means the face/edge neighbor remains at anchor resolution. */
export function encodeSameOrFinerPowerDescriptor(sameResolution: readonly boolean[]): number {
  if (sameResolution.length !== OCTREE_POWER_SAME_OR_FINER_BITS) throw new RangeError("Power descriptor needs 18 neighbor states");
  return sameResolution.reduce((word, same, bit) => same ? word | (1 << bit) : word, 0);
}

export function decodeSameOrFinerPowerDescriptor(descriptor: number): readonly boolean[] {
  validateSameOrFinerPowerDescriptor(descriptor);
  return OCTREE_POWER_NEIGHBOR_DIRECTIONS.map((_, bit) => (descriptor & (1 << bit)) !== 0);
}

export function transformSameOrFinerPowerDescriptor(descriptor: number, transform: CubeTransform): number {
  validateSameOrFinerPowerDescriptor(descriptor);
  let result = 0;
  for (let bit = 0; bit < OCTREE_POWER_SAME_OR_FINER_BITS; bit += 1) {
    if ((descriptor & (1 << bit)) === 0) continue;
    const transformed = transformPowerVector(OCTREE_POWER_NEIGHBOR_DIRECTIONS[bit], transform);
    const target = directionIndex.get(directionKey(transformed));
    if (target === undefined) throw new Error("Cube transform did not preserve face/edge directions");
    result |= 1 << target;
  }
  return result;
}

export function canonicalizeSameOrFinerPowerDescriptor(descriptor: number): { descriptor: number; transform: number } {
  validateSameOrFinerPowerDescriptor(descriptor);
  let best = descriptor, transform = 0;
  for (const candidate of OCTREE_CUBE_TRANSFORMS) {
    const word = transformSameOrFinerPowerDescriptor(descriptor, candidate);
    if (word < best || (word === best && candidate.code < transform)) { best = word; transform = candidate.code; }
  }
  return { descriptor: best, transform };
}

/** Exhaustive symmetry quotient of all 2^18 reachable same-or-finer states. */
export function buildCanonicalSameOrFinerPowerDescriptorLookup(): Uint32Array {
  const lookup = new Uint32Array(OCTREE_POWER_SAME_OR_FINER_MASK + 1);
  const visited = new Uint8Array(OCTREE_POWER_SAME_OR_FINER_MASK + 1);
  for (let descriptor = 0; descriptor <= OCTREE_POWER_SAME_OR_FINER_MASK; descriptor += 1) {
    if (visited[descriptor]) continue;
    const orbit = OCTREE_CUBE_TRANSFORMS.map((transform) => transformSameOrFinerPowerDescriptor(descriptor, transform));
    const canonical = Math.min(...orbit);
    orbit.forEach((word) => { visited[word] = 1; lookup[word] = canonical; });
  }
  return lookup;
}

export function enumerateCanonicalSameOrFinerPowerDescriptors(): readonly number[] {
  return [...new Set(buildCanonicalSameOrFinerPowerDescriptorLookup())].sort((a, b) => a - b);
}

/**
 * Reconstructs exactly the paper's 18 same-resolution neighbors or their 48
 * possible finer replacements around a size-two anchor. Duplicate fine sites
 * are keyed by origin and collapsed deterministically.
 */
export function sitesForSameOrFinerPowerDescriptor(descriptor: number): readonly OctreePowerSite[] {
  validateSameOrFinerPowerDescriptor(descriptor);
  const sites = new Map<string, OctreePowerSite>();
  const add = (origin: [number, number, number], size: 1 | 2) => {
    const key = `${origin.join(",")}/${size}`;
    sites.set(key, createOctreePowerSite(key, origin, size));
  };
  add([0, 0, 0], 2);
  OCTREE_POWER_NEIGHBOR_DIRECTIONS.forEach((direction, bit) => {
    const same = (descriptor & (1 << bit)) !== 0;
    if (same) {
      add(direction.map((value) => value * 2) as [number, number, number], 2);
      return;
    }
    const zeroAxes = [0, 1, 2].filter((axis) => direction[axis] === 0);
    const variants = 1 << zeroAxes.length;
    for (let variant = 0; variant < variants; variant += 1) {
      const origin = direction.map((value) => value < 0 ? -1 : value > 0 ? 2 : 0) as [number, number, number];
      zeroAxes.forEach((axis, localBit) => { origin[axis] = (variant >> localBit) & 1; });
      add(origin, 1);
    }
  });
  return [...sites.values()].sort((a, b) => a.key.localeCompare(b.key));
}

export interface SameOrCoarserPowerDescriptor {
  readonly child: readonly [0 | 1, 0 | 1, 0 | 1];
  /** X, Y, Z outward faces followed by XY, XZ, YZ outward edges. */
  readonly coarseNeighbors: readonly [boolean, boolean, boolean, boolean, boolean, boolean];
}

export function encodeSameOrCoarserPowerDescriptor(value: SameOrCoarserPowerDescriptor): number {
  if (value.child.length !== 3 || value.child.some((bit) => bit !== 0 && bit !== 1)
    || value.coarseNeighbors.length !== 6) throw new RangeError("Same-or-coarser descriptor needs three child bits and six neighbor bits");
  let word = OCTREE_POWER_SAME_OR_COARSER_FLAG
    | value.child[0] | (value.child[1] << 1) | (value.child[2] << 2);
  value.coarseNeighbors.forEach((present, bit) => { if (present) word |= 1 << (bit + 3); });
  return word >>> 0;
}

export function decodeSameOrCoarserPowerDescriptor(descriptor: number): SameOrCoarserPowerDescriptor {
  if (!Number.isSafeInteger(descriptor) || descriptor < 0 || descriptor > 0xffff_ffff
    || (descriptor >>> 31) !== 1 || ((descriptor & 0x7fff_ffff) & ~OCTREE_POWER_SAME_OR_COARSER_MASK) !== 0) {
    throw new RangeError("Same-or-coarser power descriptor has invalid bits");
  }
  return {
    child: [descriptor & 1, (descriptor >> 1) & 1, (descriptor >> 2) & 1] as [0 | 1, 0 | 1, 0 | 1],
    coarseNeighbors: [0, 1, 2, 3, 4, 5].map((bit) => (descriptor & (1 << (bit + 3))) !== 0) as [boolean, boolean, boolean, boolean, boolean, boolean],
  };
}

function boxesOverlap(a: OctreePowerSite, b: OctreePowerSite): boolean {
  return [0, 1, 2].every((axis) => Math.min(a.origin[axis] + a.size, b.origin[axis] + b.size) > Math.max(a.origin[axis], b.origin[axis]));
}

/** Reconstructs the paper's parity-constrained six possible coarse sites. */
export function sitesForSameOrCoarserPowerDescriptor(descriptor: number): readonly OctreePowerSite[] {
  const decoded = decodeSameOrCoarserPowerDescriptor(descriptor);
  const outward = decoded.child.map((bit) => bit === 0 ? -1 : 1) as [number, number, number];
  const coarseDirections: PowerVec3[] = [
    [outward[0], 0, 0], [0, outward[1], 0], [0, 0, outward[2]],
    [outward[0], outward[1], 0], [outward[0], 0, outward[2]], [0, outward[1], outward[2]],
  ];
  const coarse: OctreePowerSite[] = [];
  coarseDirections.forEach((direction, index) => {
    if (!decoded.coarseNeighbors[index]) return;
    const origin = direction.map((value) => value < 0 ? -2 : value > 0 ? 2 : 0) as [number, number, number];
    coarse.push(createOctreePowerSite(`coarse:${index}`, origin, 2));
  });
  const anchor = createOctreePowerSite("anchor", decoded.child, 1);
  const sites = [anchor, ...coarse];
  for (const direction of OCTREE_POWER_NEIGHBOR_DIRECTIONS) {
    const origin = anchor.origin.map((value, axis) => value + direction[axis]) as [number, number, number];
    const candidate = createOctreePowerSite(`same:${origin.join(",")}`, origin, 1);
    if (!coarse.some((site) => boxesOverlap(site, candidate))) sites.push(candidate);
  }
  return sites.sort((a, b) => a.key.localeCompare(b.key));
}
