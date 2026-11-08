/**
 * Patch partitions of a cloth mesh for the Tier B patch-resident solve.
 *
 * A partition assigns every vertex to exactly one patch of at most
 * `maxVertices`. One GPU workgroup owns a patch: it keeps the patch's
 * positions in workgroup memory and runs Gauss-Seidel over the patch's
 * colour lists, with every other vertex frozen at the dispatch's start
 * (block Jacobi across patches). Two partitions alternate between dispatches,
 * two-colour multiplicative Schwarz, so a seam of one partition lies inside
 * the patches of the other.
 *
 * The patches are cells of a cubic grid over the rest shape. The second
 * partition's grid is shifted by half a cell on every axis. A cell holding
 * more than `maxVertices` is bisected at the median of its longest extent
 * until every piece fits. This is O(N log N), deterministic, and needs only
 * rest positions, so a cut rebuilds it locally.
 */
import type { PreparedClothTopology } from './topology';

export interface ClothPatchPartition {
  readonly patchCount: number;
  /** Patch p's vertices are vertices[patchOffsets[p] .. patchOffsets[p + 1]), sorted by colour, then index. */
  readonly patchOffsets: Uint32Array;
  readonly vertices: Uint32Array;
  /** Patch p's colour c occupies local slots [colourOffsets[p (C+1) + c], colourOffsets[p (C+1) + c + 1]). */
  readonly colourOffsets: Uint32Array;
  /** Per vertex: (patch << CLOTH_PATCH_SLOT_BITS) | local slot. */
  readonly owner: Uint32Array;
}

export const CLOTH_PATCH_SLOT_BITS = 11;
/** Hard ceiling of a patch (the slot field and the workgroup arrays). */
export const CLOTH_PATCH_MAX_VERTICES = 1 << CLOTH_PATCH_SLOT_BITS;

export interface ClothPatchOptions {
  /** Target vertices per patch (the grid cell is sized for it). Default 512. */
  readonly targetVertices?: number;
  /** Most vertices a patch may hold (workgroup memory). Default 1024. */
  readonly maxVertices?: number;
}

/** The two staggered partitions of `topology` (colours from colourClothVertices). */
export function clothPatchPartitions(topology: PreparedClothTopology, colours: Uint32Array, options: ClothPatchOptions = {}): [ClothPatchPartition, ClothPatchPartition] {
  const target = options.targetVertices ?? 512, max = options.maxVertices ?? 1024;
  if (!(Number.isInteger(max) && max >= 1 && max <= CLOTH_PATCH_MAX_VERTICES)) throw new Error(`Cloth patch capacity must be an integer in [1, ${CLOTH_PATCH_MAX_VERTICES}]`);
  if (!(target >= 1 && target <= max)) throw new Error('Cloth patch target must lie in [1, maxVertices]');
  const n = topology.vertexCount;
  if (colours.length !== n) throw new Error('Cloth patch partition needs a colour per vertex');
  let area = 0;
  for (let f = 0; f < topology.faceCount; f++) area += topology.faceArea_m2[f]!;
  // A flat sheet crossing a cube cell of side s holds about s^2 / (area / n) vertices.
  const cell = Math.sqrt(target * area / n);
  if (!(cell > 0)) throw new Error('Cloth patch partition needs a positive rest area');
  const rest = topology.data.restPositions_m;
  return [partition(rest, n, colours, cell, 0, max), partition(rest, n, colours, cell, 0.5, max)];
}

function partition(rest: ArrayLike<number>, n: number, colours: Uint32Array, cell: number, shift: number, max: number): ClothPatchPartition {
  const lo = [Infinity, Infinity, Infinity];
  for (let v = 0; v < n; v++) for (let a = 0; a < 3; a++) lo[a] = Math.min(lo[a]!, rest[3 * v + a]!);
  const cells = new Map<string, number[]>();
  for (let v = 0; v < n; v++) {
    const key = [0, 1, 2].map(a => Math.floor((rest[3 * v + a]! - lo[a]!) / cell + shift)).join(',');
    const list = cells.get(key); if (list) list.push(v); else cells.set(key, [v]);
  }
  const patches: number[][] = [];
  const split = (list: number[]) => {
    if (list.length <= max) { patches.push(list); return; }
    let axis = 0, best = -1;
    for (let a = 0; a < 3; a++) {
      let mn = Infinity, mx = -Infinity;
      for (const v of list) { mn = Math.min(mn, rest[3 * v + a]!); mx = Math.max(mx, rest[3 * v + a]!); }
      if (mx - mn > best) { best = mx - mn; axis = a; }
    }
    const sorted = [...list].sort((p, q) => rest[3 * p + axis]! - rest[3 * q + axis]! || p - q), half = sorted.length >> 1;
    split(sorted.slice(0, half)); split(sorted.slice(half));
  };
  for (const key of [...cells.keys()].sort()) split(cells.get(key)!);
  const C = colours.reduce((m, c) => Math.max(m, c + 1), 0);
  const patchOffsets = new Uint32Array(patches.length + 1), vertices = new Uint32Array(n), colourOffsets = new Uint32Array(patches.length * (C + 1)), owner = new Uint32Array(n);
  if (patches.length >= 2 ** (32 - CLOTH_PATCH_SLOT_BITS)) throw new Error('Cloth mesh has too many patches for the owner encoding');
  let at = 0;
  patches.forEach((list, p) => {
    const sorted = [...list].sort((a, b) => colours[a]! - colours[b]! || a - b);
    patchOffsets[p] = at;
    let slot = 0;
    for (let c = 0; c <= C; c++) {
      colourOffsets[p * (C + 1) + c] = slot;
      while (c < C && slot < sorted.length && colours[sorted[slot]!] === c) slot++;
    }
    sorted.forEach((v, s) => { vertices[at + s] = v; owner[v] = (p << CLOTH_PATCH_SLOT_BITS) | s; });
    at += sorted.length;
  });
  patchOffsets[patches.length] = at;
  return { patchCount: patches.length, patchOffsets, vertices, colourOffsets, owner };
}
