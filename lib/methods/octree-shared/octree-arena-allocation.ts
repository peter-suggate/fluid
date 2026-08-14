/**
 * Word and byte layouts for the projection's shared scratch arenas.
 *
 * Each planner returns offsets, not merely sizes, because the two ends of an
 * arena are written and read by passes several stages apart -- the frontier's
 * row-delta maps, the compaction arena's per-tile signatures, the cooperative
 * row-classification scratch. Deriving both ends from one function is what
 * stops a pass from writing at one offset and a later pass from decoding at
 * another, a mismatch that yields plausible numbers rather than a validation
 * error.
 *
 * The header splits are deliberate and documented at each planner: active
 * authority is kept apart from the inactive candidate transaction so a
 * candidate published at the tail of one substep cannot move the selector
 * before the coupled boundary commit.
 *
 * Nothing here allocates. A planner is arithmetic over capacities the caller
 * has already fixed.
 */

export interface OctreeLeafFrontierAllocationPlan {
  cellCount: number;
  listCapacity: number;
  /** Third immutable-sort stream used only while merging dirty candidates. */
  candidateOffsetWords: number;
  /** Fixed control header for the exact old/new row-delta transaction. */
  rowDeltaControlOffsetWords: number;
  /** Exact `newRow -> oldRow|INVALID` map. */
  rowDeltaNewToOldOffsetWords: number;
  /** Exact `oldRow -> newRow|INVALID` map. */
  rowDeltaOldToNewOffsetWords: number;
  /** Rows whose exact identity was added this generation. */
  rowDeltaDirtyRowsOffsetWords: number;
  /** Dirty rows plus their exact current/retired one-ring. */
  rowDeltaAffectedRowsOffsetWords: number;
  candidateBytes: number;
  allocatedBytes: number;
}

/** Allocate only the sorted A/B publication, dirty candidate stream, and row delta.
 *
 * The ten-word header deliberately separates active authority (0..3) from the
 * inactive candidate transaction (4..9).  Candidate validation may populate
 * the latter at the tail of substep N, but only the coupled owner/frontier
 * boundary commit may change the active selector/generation at N+1.
 */
export function planOctreeLeafFrontierAllocation(
  cellCount: number,
  rowCapacity: number,
): OctreeLeafFrontierAllocationPlan {
  if (!Number.isSafeInteger(cellCount) || cellCount < 1) throw new Error("Octree frontier cell count must be a positive integer");
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) throw new Error("Octree frontier row capacity must be a positive integer");
  const listCapacity = Math.min(cellCount, rowCapacity);
  const candidateOffsetWords = 10 + 2 * listCapacity;
  const candidateBytes = listCapacity * 4;
  // The persistent frontier owns the row-delta publication because it is the
  // only stage that can still see both exact old and new `(cell,size)`
  // identities.  Sixteen control words are followed by two total maps and two
  // compact worklists.  Downstream descriptor/topology/face stages consume
  // these offsets directly; there is no topology-wide reuse branch.
  const rowDeltaControlOffsetWords = candidateOffsetWords + listCapacity;
  const rowDeltaNewToOldOffsetWords = rowDeltaControlOffsetWords + 16;
  const rowDeltaOldToNewOffsetWords = rowDeltaNewToOldOffsetWords + listCapacity;
  const rowDeltaDirtyRowsOffsetWords = rowDeltaOldToNewOffsetWords + listCapacity;
  const rowDeltaAffectedRowsOffsetWords = rowDeltaDirtyRowsOffsetWords + listCapacity;
  const allocatedBytes = (rowDeltaAffectedRowsOffsetWords + listCapacity) * 4;
  return {
    cellCount,
    listCapacity,
    candidateOffsetWords,
    rowDeltaControlOffsetWords,
    rowDeltaNewToOldOffsetWords,
    rowDeltaOldToNewOffsetWords,
    rowDeltaDirtyRowsOffsetWords,
    rowDeltaAffectedRowsOffsetWords,
    candidateBytes,
    allocatedBytes,
  };
}

/** Exact named-resource sum used by production allocation telemetry. */
export function sumOctreePowerAllocationBreakdown(
  breakdown: Readonly<Record<string, number>>,
): number {
  let total = 0;
  for (const [name, bytes] of Object.entries(breakdown)) {
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new RangeError(`Octree power allocation ${name} must be non-negative safe bytes`);
    }
    total += bytes;
    if (!Number.isSafeInteger(total)) throw new RangeError("Octree power allocation total exceeds safe integer range");
  }
  return total;
}

interface OctreeCompactionAllocationPlan {
  scanBlockCapacity: number;
  candidateBlockCapacity: number;
  scanAndTaskBytes: number;
  activeTileBytes: number;
  /** Stable tail offsets consumed by GPU-only downstream delta publishers. */
  changeStateBaseWords: number;
  tileChangeFlagsOffsetWords: number;
  tileRefinementSignaturesOffsetWords: number;
  tileFrontierSignaturesOffsetWords: number;
  tileSignatureChangedOffsetWords: number;
  tileFrontierChangeFlagsOffsetWords: number;
  frontierTopologyReuseWord: number;
  dirtyFailureOffsetWords: number;
  /** Plain-storage scratch for cooperative row classification and scans. */
  rowDeltaScratchBaseWords: number;
  rowDeltaScratchWords: number;
  allocatedBytes: number;
}

/**
 * Size the shared scan/task arena from the authorities that can actually
 * publish work. Compact pressure owns one scan record per frontier block;
 * active/retired topology tiles own the exact candidate scan records. The
 * resident tile list remains an independent lower bound because it is copied
 * into the same buffer before topology rebuilds.
 */
export function planOctreeCompactionAllocation(
  dims: { nx: number; ny: number; nz: number },
  pressureRowCapacity: number,
  activeTileWorklistBytes: number,
  activeTileCapacity: number,
  topologyTileSize: number,
): OctreeCompactionAllocationPlan {
  if (![dims.nx, dims.ny, dims.nz].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new Error("Octree compaction dimensions must be positive integers");
  }
  if (!Number.isSafeInteger(pressureRowCapacity) || pressureRowCapacity < 1) {
    throw new Error("Octree compaction pressure capacity must be a positive integer");
  }
  if (!Number.isSafeInteger(activeTileWorklistBytes) || activeTileWorklistBytes < 0
    || !Number.isSafeInteger(activeTileCapacity) || activeTileCapacity < 0
    || !Number.isSafeInteger(topologyTileSize) || topologyTileSize < 8 || topologyTileSize % 8 !== 0) {
    throw new Error("Octree compaction active-tile bounds must be non-negative integers");
  }
  const scanBlockCapacity = Math.ceil(pressureRowCapacity / 256);
  const candidateBlocksPerTile = (topologyTileSize / 8) ** 3;
  const candidateBlockCapacity = 2 * activeTileCapacity * candidateBlocksPerTile;
  const scanAndTaskBytes = 4 * (15 + 3 * scanBlockCapacity
    + 2 * candidateBlockCapacity);
  // The recurring state retains generation-stamped dirty membership, the
  // compact dirty list, and independent five-word structural-refinement and
  // wet-frontier signatures per logical topology tile. A one-word bitmask
  // lets the parallel comparison publish both decisions for the compacting
  // singletons without atomics.
  const tileSignatureWords = 5;
  const tileFrontierSignatureWords = 5;
  const tileSignatureChangedWords = 1;
  const tileFrontierChangeFlagWords = 1;
  const rigidSnapshotWords = 2 + 12 * 12;
  const dirtyAuthorityWords = 1;
  const dirtyFailureWords = 8;
  // The fourteen publication words are a last-good row-control snapshot plus
  // an independent exact-topology reuse bit.  The latter survives restoring
  // words 0..11, so downstream topology consumers can distinguish immutable
  // row reuse from a freshly emitted row set without a host readback.
  const activeTileBytes = 4 * ((2 + tileSignatureWords + tileFrontierSignatureWords
    + tileSignatureChangedWords + tileFrontierChangeFlagWords) * activeTileCapacity
    + dirtyAuthorityWords + rigidSnapshotWords + 14 + dirtyFailureWords) + 32;
  const rowDeltaBlockCount = Math.ceil(pressureRowCapacity / 256);
  // Two row-sized streams (flags and exclusive ranks), one block-total stream
  // plus its exact total, and two words per classification block.
  const rowDeltaScratchWords = 2 * pressureRowCapacity + 3 * rowDeltaBlockCount + 1;
  const allocatedBytes = Math.max(60, scanAndTaskBytes, activeTileWorklistBytes)
    + rowDeltaScratchWords * 4 + activeTileBytes;
  const changeStateWords = (2 + tileSignatureWords + tileFrontierSignatureWords
    + tileSignatureChangedWords + tileFrontierChangeFlagWords) * activeTileCapacity
    + dirtyAuthorityWords + rigidSnapshotWords + 14 + dirtyFailureWords;
  const changeStateBaseWords = allocatedBytes / 4 - 8 - changeStateWords;
  const tileChangeFlagsOffsetWords = changeStateBaseWords;
  const tileRefinementSignaturesOffsetWords = changeStateBaseWords + 2 * activeTileCapacity;
  const tileFrontierSignaturesOffsetWords =
    tileRefinementSignaturesOffsetWords + tileSignatureWords * activeTileCapacity;
  const tileSignatureChangedOffsetWords =
    tileFrontierSignaturesOffsetWords + tileFrontierSignatureWords * activeTileCapacity;
  const tileFrontierChangeFlagsOffsetWords =
    tileSignatureChangedOffsetWords + tileSignatureChangedWords * activeTileCapacity;
  const frontierTopologyReuseWord = changeStateBaseWords
    + (2 + tileSignatureWords + tileFrontierSignatureWords
      + tileSignatureChangedWords + tileFrontierChangeFlagWords) * activeTileCapacity
    + dirtyAuthorityWords + rigidSnapshotWords + 13;
  const dirtyFailureOffsetWords = frontierTopologyReuseWord + 1;
  const rowDeltaScratchBaseWords = changeStateBaseWords - rowDeltaScratchWords;
  return { scanBlockCapacity, candidateBlockCapacity, scanAndTaskBytes, activeTileBytes,
    changeStateBaseWords, tileChangeFlagsOffsetWords, tileRefinementSignaturesOffsetWords,
    tileFrontierSignaturesOffsetWords, tileSignatureChangedOffsetWords,
    tileFrontierChangeFlagsOffsetWords, frontierTopologyReuseWord, dirtyFailureOffsetWords,
    rowDeltaScratchBaseWords, rowDeltaScratchWords, allocatedBytes };
}
