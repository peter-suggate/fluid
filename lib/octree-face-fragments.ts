/**
 * Reference contract for the unified octree MAC face representation.
 *
 * Runtime construction will be GPU-resident.  This CPU implementation is the
 * topology oracle used by unit tests and parity tooling: every physical face
 * patch is published exactly once, coarse/fine interfaces are split by the
 * finer neighbour footprint, and flux signs are shared by divergence and
 * projection.
 */

export const OCTREE_FACE_FRAGMENT_INVALID_LEAF = 0xffff_ffff;
export const OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS = 4;

export type OctreeAxis = 0 | 1 | 2;
export type OctreeCoord = readonly [number, number, number];

export interface OctreeFaceLeaf {
  id: number;
  origin: OctreeCoord;
  size: number;
}

export interface OctreeFaceFragment {
  /** Stable index in canonical axis/origin order. */
  index: number;
  axis: OctreeAxis;
  /** Finest-cell coordinate of the fragment's lower corner. */
  origin: OctreeCoord;
  /** Fragment width along each of the two tangential axes, in finest cells. */
  span: number;
  /** Area measured in finest-cell faces. */
  areaFineFaces: number;
  /** Leaf on the negative side of the oriented face, or the domain boundary. */
  negativeLeaf: number | null;
  /** Leaf on the positive side of the oriented face, or the domain boundary. */
  positiveLeaf: number | null;
}

export interface OctreeFaceIncidence {
  /** CSR offsets in the supplied leaf order; length is leaves.length + 1. */
  offsets: Uint32Array;
  fragmentIndices: Uint32Array;
  /** +1 for the fragment's negative leaf, -1 for its positive leaf. */
  signs: Int8Array;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value <= 0) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function powerOfTwo(value: number): boolean {
  return (value & (value - 1)) === 0;
}

function linearIndex(x: number, y: number, z: number, dims: OctreeCoord): number {
  return x + dims[0] * (y + dims[1] * z);
}

function tangentialAxes(axis: OctreeAxis): readonly [OctreeAxis, OctreeAxis] {
  return [((axis + 1) % 3) as OctreeAxis, ((axis + 2) % 3) as OctreeAxis];
}

type PendingFragment = Omit<OctreeFaceFragment, "index">;

/**
 * Builds a canonical physical-face list for a complete, balanced leaf tiling.
 * Internal faces are visited from their negative side only.  Domain faces are
 * included because the same ABI must eventually carry wall/open-boundary flux.
 */
export function buildCanonicalOctreeFaceFragments(
  leaves: readonly OctreeFaceLeaf[],
  dimensions: OctreeCoord,
): OctreeFaceFragment[] {
  const dims = dimensions.map((value, axis) => positiveInteger(value, `Octree dimension ${axis}`)) as [number, number, number];
  const cellCount = dims[0] * dims[1] * dims[2];
  const owners = new Int32Array(cellCount).fill(-1);
  const ids = new Set<number>();

  for (let leafIndex = 0; leafIndex < leaves.length; leafIndex += 1) {
    const leaf = leaves[leafIndex];
    if (!Number.isInteger(leaf.id) || leaf.id < 0 || ids.has(leaf.id)) throw new RangeError(`Invalid or duplicate octree leaf id ${leaf.id}`);
    ids.add(leaf.id);
    positiveInteger(leaf.size, `Leaf ${leaf.id} size`);
    if (!powerOfTwo(leaf.size)) throw new RangeError(`Leaf ${leaf.id} size must be dyadic`);
    for (let axis = 0; axis < 3; axis += 1) {
      const start = leaf.origin[axis];
      if (!Number.isInteger(start) || start < 0 || start + leaf.size > dims[axis]) {
        throw new RangeError(`Leaf ${leaf.id} lies outside the octree domain`);
      }
      if (start % leaf.size !== 0) throw new RangeError(`Leaf ${leaf.id} origin is not aligned to its size`);
    }
    for (let z = leaf.origin[2]; z < leaf.origin[2] + leaf.size; z += 1) {
      for (let y = leaf.origin[1]; y < leaf.origin[1] + leaf.size; y += 1) {
        for (let x = leaf.origin[0]; x < leaf.origin[0] + leaf.size; x += 1) {
          const cell = linearIndex(x, y, z, dims);
          if (owners[cell] !== -1) throw new RangeError(`Leaf ${leaf.id} overlaps leaf ${leaves[owners[cell]].id}`);
          owners[cell] = leafIndex;
        }
      }
    }
  }
  const missing = owners.indexOf(-1);
  if (missing !== -1) throw new RangeError(`Octree leaves do not cover finest cell ${missing}`);

  const pending: PendingFragment[] = [];
  const add = (fragment: PendingFragment) => pending.push(fragment);
  for (let leafIndex = 0; leafIndex < leaves.length; leafIndex += 1) {
    const leaf = leaves[leafIndex];
    for (let axis = 0 as OctreeAxis; axis < 3; axis = (axis + 1) as OctreeAxis) {
      const [u, v] = tangentialAxes(axis);
      if (leaf.origin[axis] === 0) {
        add({ axis, origin: leaf.origin, span: leaf.size, areaFineFaces: leaf.size ** 2, negativeLeaf: null, positiveLeaf: leaf.id });
      }
      const plane = leaf.origin[axis] + leaf.size;
      if (plane === dims[axis]) {
        const origin = [...leaf.origin] as [number, number, number]; origin[axis] = plane;
        add({ axis, origin, span: leaf.size, areaFineFaces: leaf.size ** 2, negativeLeaf: leaf.id, positiveLeaf: null });
        continue;
      }

      // Group finest face samples by the leaf on the positive side.  Strict
      // 2:1 balance guarantees at most four groups and a square footprint.
      const groups = new Map<number, { minU: number; minV: number; maxU: number; maxV: number; samples: number }>();
      for (let dv = 0; dv < leaf.size; dv += 1) {
        for (let du = 0; du < leaf.size; du += 1) {
          const cell = [...leaf.origin] as [number, number, number];
          cell[axis] = plane;
          cell[u] += du;
          cell[v] += dv;
          const neighborIndex = owners[linearIndex(cell[0], cell[1], cell[2], dims)];
          const group = groups.get(neighborIndex);
          if (group) {
            group.minU = Math.min(group.minU, cell[u]); group.minV = Math.min(group.minV, cell[v]);
            group.maxU = Math.max(group.maxU, cell[u]); group.maxV = Math.max(group.maxV, cell[v]); group.samples += 1;
          } else groups.set(neighborIndex, { minU: cell[u], minV: cell[v], maxU: cell[u], maxV: cell[v], samples: 1 });
        }
      }
      if (groups.size > OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS) {
        throw new RangeError(`Leaf ${leaf.id} face exceeds the 2:1 four-fragment bound`);
      }
      for (const [neighborIndex, group] of groups) {
        const neighbor = leaves[neighborIndex];
        const ratio = Math.max(leaf.size, neighbor.size) / Math.min(leaf.size, neighbor.size);
        if (ratio > 2) throw new RangeError(`Leaves ${leaf.id} and ${neighbor.id} violate 2:1 balance`);
        const widthU = group.maxU - group.minU + 1;
        const widthV = group.maxV - group.minV + 1;
        if (widthU !== widthV || group.samples !== widthU * widthV) {
          throw new RangeError(`Leaves ${leaf.id} and ${neighbor.id} do not form a square face fragment`);
        }
        const origin = [...leaf.origin] as [number, number, number];
        origin[axis] = plane; origin[u] = group.minU; origin[v] = group.minV;
        add({ axis, origin, span: widthU, areaFineFaces: group.samples, negativeLeaf: leaf.id, positiveLeaf: neighbor.id });
      }
    }
  }

  pending.sort((a, b) => a.axis - b.axis
    || a.origin[2] - b.origin[2] || a.origin[1] - b.origin[1] || a.origin[0] - b.origin[0]
    || (a.negativeLeaf ?? -1) - (b.negativeLeaf ?? -1) || (a.positiveLeaf ?? -1) - (b.positiveLeaf ?? -1));
  return pending.map((fragment, index) => ({ index, ...fragment }));
}

/** Area-weighted restriction of fragment-normal velocities onto one coarse face. */
export function restrictOctreeFaceVelocity(
  fragments: readonly OctreeFaceFragment[],
  normalVelocities: readonly number[],
): number {
  if (fragments.length !== normalVelocities.length || fragments.length === 0) {
    throw new RangeError("Face restriction requires one velocity per fragment");
  }
  let area = 0;
  let flux = 0;
  for (let index = 0; index < fragments.length; index += 1) {
    area += fragments[index].areaFineFaces;
    flux += fragments[index].areaFineFaces * normalVelocities[index];
  }
  return flux / area;
}

/**
 * Builds the row-to-face CSR consumed by adaptive divergence and projection.
 * Keeping the sign beside the face index makes both operators use one exact
 * incidence matrix rather than independently rediscovering orientation.
 */
export function buildOctreeFaceIncidence(
  leaves: readonly OctreeFaceLeaf[],
  fragments: readonly OctreeFaceFragment[],
): OctreeFaceIncidence {
  const rowById = new Map<number, number>();
  leaves.forEach((leaf, row) => {
    if (rowById.has(leaf.id)) throw new RangeError(`Duplicate octree leaf id ${leaf.id}`);
    rowById.set(leaf.id, row);
  });
  const rows = leaves.map(() => [] as Array<{ fragment: number; sign: 1 | -1 }>);
  for (const fragment of fragments) {
    if (fragment.negativeLeaf !== null) {
      const row = rowById.get(fragment.negativeLeaf);
      if (row === undefined) throw new RangeError(`Face fragment references missing leaf ${fragment.negativeLeaf}`);
      rows[row].push({ fragment: fragment.index, sign: 1 });
    }
    if (fragment.positiveLeaf !== null) {
      const row = rowById.get(fragment.positiveLeaf);
      if (row === undefined) throw new RangeError(`Face fragment references missing leaf ${fragment.positiveLeaf}`);
      rows[row].push({ fragment: fragment.index, sign: -1 });
    }
  }
  const offsets = new Uint32Array(leaves.length + 1);
  for (let row = 0; row < rows.length; row += 1) {
    rows[row].sort((a, b) => a.fragment - b.fragment);
    offsets[row + 1] = offsets[row] + rows[row].length;
  }
  const fragmentIndices = new Uint32Array(offsets[offsets.length - 1]);
  const signs = new Int8Array(fragmentIndices.length);
  for (let row = 0; row < rows.length; row += 1) {
    rows[row].forEach((entry, local) => {
      const index = offsets[row] + local;
      fragmentIndices[index] = entry.fragment;
      signs[index] = entry.sign;
    });
  }
  return { offsets, fragmentIndices, signs };
}

/** Net outward flux per leaf using the canonical negative-to-positive orientation. */
export function octreeLeafNetFluxes(
  leaves: readonly OctreeFaceLeaf[],
  fragments: readonly OctreeFaceFragment[],
  normalVelocities: readonly number[],
): Map<number, number> {
  if (fragments.length !== normalVelocities.length) throw new RangeError("Flux reduction requires one velocity per fragment");
  const result = new Map(leaves.map((leaf) => [leaf.id, 0]));
  for (let index = 0; index < fragments.length; index += 1) {
    const fragment = fragments[index];
    const flux = fragment.areaFineFaces * normalVelocities[index];
    if (fragment.negativeLeaf !== null) result.set(fragment.negativeLeaf, result.get(fragment.negativeLeaf)! + flux);
    if (fragment.positiveLeaf !== null) result.set(fragment.positiveLeaf, result.get(fragment.positiveLeaf)! - flux);
  }
  return result;
}
