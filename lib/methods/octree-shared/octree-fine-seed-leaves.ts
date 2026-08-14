/**
 * Compact leaf records used only to seed and classify the authoritative
 * factor-m fine level set. This ABI contains no row-attached phi pages,
 * directories, lifecycle state, or dense compatibility publication.
 */

export const OCTREE_FINE_SEED_INVALID = 0xffff_ffff;
export const OCTREE_FINE_SEED_LEAF_RECORD_BYTES = 64;
export const OCTREE_FINE_SEED_CANDIDATE_BYTES = 8;

export const OCTREE_FINE_SEED_STATE = Object.freeze({
  core: 1 << 1,
  halo: 1 << 2,
  /** Leaf record belongs to the current compact topology generation. */
  live: 1 << 5,
} as const);

export interface OctreeFineSeedCandidateSource {
  /** Stable, row-sorted, unique `{ leafRow, flags }` records. */
  readonly candidates: GPUBuffer;
  /**
   * `[count, dispatch xyz, generation, published, error, capacity, delta]`.
   * A valid zero count is distinct from an unpublished or rejected generation.
   */
  readonly countAndDispatch: GPUBuffer;
  readonly indirectOffsetBytes?: number;
}

export interface OctreeFineSeedLeafResources {
  /** Array of 64-byte compact FineSeedLeaf records. */
  readonly leaves: GPUBuffer;
  /** Previous compact-row generation used by exact dirty-row publication. */
  readonly previousFineSeedLeaves?: GPUBuffer;
  readonly candidates: OctreeFineSeedCandidateSource;
}
