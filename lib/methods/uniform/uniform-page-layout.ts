import { signedSpatialCoordinateHash } from "../../core/signed-spatial-hash";

/** Experimental fixed-resolution residency planner, independent of solver fields.
 * Adapted from Sparse CM12's signed directory / stable leaf-slot design. A plan
 * is a candidate, not a GPU publication: callers must initialize new slots and
 * retire all users of the previous snapshot before reusing released addresses.
 * No fluid support policy, adaptive resolution, or physical boundaries live here.
 */
export type UniformPageEdge = 16 | 32;
export type UniformPageCoordinate = readonly [number, number, number];
export const UNIFORM_PAGE_MISSING = 0xffff_ffff;
export const UNIFORM_PAGE_NEIGHBORS: readonly UniformPageCoordinate[] = [
  [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
];

export interface UniformPageLayout {
  readonly edge: UniformPageEdge;
  readonly capacity: number;
  readonly generation: number;
  readonly coordinates: readonly (UniformPageCoordinate | undefined)[];
  /** Physical slots, in caller's requested execution order. */
  readonly activeSlots: Uint32Array;
  /** Six slots per physical page: -x,+x,-y,+y,-z,+z. */
  readonly neighbors: Uint32Array;
  /** Four words per hash bucket: signed x,y,z and slot. Missing slot = empty. */
  readonly directory: Uint32Array;
  readonly directoryCapacity: number;
  readonly newSlots: Uint32Array;
  readonly releasedSlots: Uint32Array;
}

function validateCoordinate(q: UniformPageCoordinate): void {
  if (q.length !== 3) throw new RangeError("A uniform page coordinate has three axes");
  signedSpatialCoordinateHash(q);
}
const key = (q: UniformPageCoordinate) => q.join("/");

export function uniformPageCellAddress(cell: UniformPageCoordinate, edge: UniformPageEdge): {
  coordinate: UniformPageCoordinate; local: UniformPageCoordinate; localIndex: number;
} {
  if (edge !== 16 && edge !== 32) throw new RangeError("Uniform page edge must be 16 or 32");
  validateCoordinate(cell);
  const coordinate = cell.map(v => Math.floor(v / edge)) as [number, number, number];
  const local = cell.map((v, i) => v - coordinate[i]! * edge) as [number, number, number];
  return { coordinate, local, localIndex: local[0] + edge * (local[1] + edge * local[2]) };
}

export function lookupUniformPage(layout: UniformPageLayout, q: UniformPageCoordinate): number {
  validateCoordinate(q);
  let bucket = signedSpatialCoordinateHash(q) & (layout.directoryCapacity - 1);
  for (let probe = 0; probe < layout.directoryCapacity; probe++) {
    const at = bucket * 4, slot = layout.directory[at + 3]!;
    if (slot === UNIFORM_PAGE_MISSING) return slot;
    if (q.every((v, axis) => (layout.directory[at + axis]! | 0) === v)) return slot;
    bucket = (bucket + 1) & (layout.directoryCapacity - 1);
  }
  return UNIFORM_PAGE_MISSING;
}

/** All-or-nothing host planning; capacity failure never edits the accepted plan.
 * Slots released by this transition are deliberately NOT reused in the same
 * transition. Reusing earlier holes still requires the caller's GPU lifetime fence.
 */
export function planUniformPages(edge: UniformPageEdge, capacity: number,
  requested: readonly UniformPageCoordinate[], previous?: UniformPageLayout,
): UniformPageLayout {
  if ((edge !== 16 && edge !== 32) || !Number.isSafeInteger(capacity)
    || capacity < 1 || capacity * edge ** 3 >= UNIFORM_PAGE_MISSING) {
    throw new RangeError("Invalid uniform page size/capacity or u32 field address overflow");
  }
  if (previous && (previous.edge !== edge || previous.capacity > capacity)) {
    throw new RangeError("Page transitions cannot change edge or shrink capacity");
  }
  const wanted = new Set<string>();
  for (const q of requested) {
    validateCoordinate(q);
    if (wanted.has(key(q))) throw new Error("Duplicate uniform page coordinate");
    wanted.add(key(q));
  }
  const coordinates: (UniformPageCoordinate | undefined)[] = new Array(capacity);
  const free: number[] = [], released: number[] = [], fresh: number[] = [];
  for (let slot = 0; slot < capacity; slot++) {
    const old = previous?.coordinates[slot];
    if (!old) free.push(slot);
    else if (wanted.has(key(old))) coordinates[slot] = old;
    else released.push(slot);
  }
  const retained = new Map<string, number>();
  coordinates.forEach((q, slot) => { if (q) retained.set(key(q), slot); });
  const newCount = requested.length - retained.size;
  if (newCount > free.length) throw new RangeError(
    `Uniform page capacity exhausted: ${newCount} new pages need ${newCount - free.length} additional slots; retiring slots are not reusable yet`);
  const activeSlots = new Uint32Array(requested.length);
  let next = 0;
  requested.forEach((q, index) => {
    let slot = retained.get(key(q));
    if (slot === undefined) {
      slot = free[next++]!;
      coordinates[slot] = Object.freeze([...q]) as UniformPageCoordinate;
      fresh.push(slot);
    }
    activeSlots[index] = slot;
  });
  const directoryCapacity = 2 ** Math.ceil(Math.log2(Math.max(2, 2 * requested.length)));
  const directory = new Uint32Array(4 * directoryCapacity).fill(UNIFORM_PAGE_MISSING);
  for (const slot of activeSlots) {
    const q = coordinates[slot]!;
    let bucket = signedSpatialCoordinateHash(q) & (directoryCapacity - 1);
    while (directory[4 * bucket + 3] !== UNIFORM_PAGE_MISSING) bucket = (bucket + 1) & (directoryCapacity - 1);
    directory.set([...q.map(v => v >>> 0), slot], 4 * bucket);
  }
  const neighbors = new Uint32Array(6 * capacity).fill(UNIFORM_PAGE_MISSING);
  const layout: UniformPageLayout = Object.freeze({ edge, capacity,
    generation: (previous?.generation ?? -1) + 1, coordinates: Object.freeze(coordinates),
    activeSlots, neighbors, directory, directoryCapacity,
    newSlots: Uint32Array.from(fresh), releasedSlots: Uint32Array.from(released) });
  for (const slot of activeSlots) for (let face = 0; face < 6; face++) {
    const q = coordinates[slot]!.map((v, axis) => v + UNIFORM_PAGE_NEIGHBORS[face]![axis]!) as [number, number, number];
    if (q.some(v => v < -0x8000_0000 || v > 0x7fff_ffff)) continue;
    neighbors[6 * slot + face] = lookupUniformPage(layout, q);
  }
  return layout;
}
