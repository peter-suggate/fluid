/**
 * Pure ABI for the structured positive-air velocity support arena.
 *
 * The producer lives in a coarse-dynamics backend; the coarse summary build and
 * the fine level-set transport are shared engine stages that read the same
 * arena. The layout description therefore sits here, where both directions can
 * reach it without the shared engine naming a lane.
 */

import { OCTREE_CATALOG_EXTENTS } from "./octree-catalog-extents";

export const OCTREE_AIR_SUPPORT_LAYOUT_VERSION = 3;
export const OCTREE_AIR_SUPPORT_VALID = 0x4156_5350;
export const OCTREE_AIR_SUPPORT_INVALID = 0xffff_ffff;
export const OCTREE_AIR_SUPPORT_TAG = 0x8000_0000;
/** Direct selector-indexed row width. The generated catalog proves the exact
 * global table extent; byte packing constrains the ABI ceiling, not storage. */
export const OCTREE_AIR_SUPPORT_SELECTOR_STRIDE =
  OCTREE_CATALOG_EXTENTS.tetrahedronVertexCount;
export const OCTREE_AIR_SUPPORT_REGULAR_STENCIL_SIZE = 27;
export const OCTREE_AIR_SUPPORT_MAXIMUM_CASE_SELECTORS = 36;
export const OCTREE_AIR_SUPPORT_CONTROL_WORDS = 16;

/**
 * The adaptive owner directory's open-addressing ABI, shared by its producer
 * and by every consumer.
 *
 * A record is four words -- `{originCell + 1, size, tag, case | transform}` --
 * placed at `hash(originCell, size)` and resolved by linear probing that stops
 * at the first zero key. The identity is the (origin, size) pair of a dyadic
 * octree leaf, so a lookup walks candidate sizes from the octree's maximum
 * leaf down to one; a size-32 empty leaf costs one record instead of 32^3.
 *
 * These constants live here, not in the shaders, because the producer and the
 * consumers agree only if they mix, probe, and bound identically. When the
 * directory replaced the dense finest-cell map, one consumer
 * (`webgpu-octree-coarse-summary.ts`) kept reading `owners[4 * cell]` and
 * silently resolved every coarse cell's velocity from an unrelated hash slot;
 * `tests/octree-air-support-owner-directory.test.ts` now fails closed on any
 * consumer that does not interpolate the values below.
 */
export const OCTREE_AIR_SUPPORT_OWNER_HASH = Object.freeze({
  originMultiplier: 0x9e37_79b1,
  sizeMultiplier: 0x85eb_ca6b,
  /** Bounded probe run. The <=0.5 load factor makes overflow a rejected
   * publication rather than a silent miss; see `ownerDirectorySlotCapacity`. */
  maximumProbes: 64,
  recordWords: 4,
  /** Largest dyadic identity a lookup must consider. `octreeLeafSize` clamps
   * every authored maximum leaf to 32, so starting a probe here can never skip
   * a published owner; a consumer that knows its own exact maximum may start
   * lower and save probes, never higher. */
  maximumLeafSize: 32,
} as const);

/** WGSL for the shared probe-start mix. Every owner-directory reader emits
 * this rather than repeating the constants. */
export function octreeAirSupportOwnerHashStartWGSL(name: string): string {
  return `fn ${name}(originCell:u32,size:u32,capacity:u32)->u32{`
    + `return ((originCell*${OCTREE_AIR_SUPPORT_OWNER_HASH.originMultiplier}u)`
    + `^(size*${OCTREE_AIR_SUPPORT_OWNER_HASH.sizeMultiplier}u))%capacity;}`;
}

export interface OctreeAirVelocitySupportLayout {
  readonly rowCapacity: number;
  readonly slotCapacity: number;
  readonly supportCapacity: number;
  readonly selectorStride: number;
  readonly transportMetricOffsetBytes: 0;
  readonly transportMetricBytes: number;
  readonly selectorTagOffsetBytes: number;
  readonly selectorTagOffsetWords: number;
  readonly selectorTagBytes: number;
  readonly regularTagOffsetBytes: number;
  readonly regularTagOffsetWords: number;
  readonly regularTagBytes: number;
  readonly controlOffsetBytes: number;
  readonly controlOffsetWords: number;
  readonly controlBytes: number;
  readonly supportVectorOffsetBytes: number;
  readonly supportVectorOffsetWords: number;
  readonly supportVectorBytes: number;
  /** Adaptive owner hash records `{originCell, size, tag, case|transform}`.
   * Fine transport probes dyadic identities from `maximumLeafSize` down to
   * one, so a size-32 empty leaf occupies one record rather than 32^3 dense
   * finest-cell records. This remains coalesced into the already-bound arena
   * so fine transport does not need another storage binding. */
  readonly ownerDirectoryOffsetBytes: number;
  readonly ownerDirectoryOffsetWords: number;
  readonly ownerDirectoryBytes: number;
  readonly ownerDirectorySlotCapacity: number;
  /** Finest-cell extent addressable by the adaptive directory. This is a
   * coordinate bound, not the number of allocated owner records. */
  readonly ownerDirectoryCellCapacity: number;
  readonly totalBytes: number;
}
