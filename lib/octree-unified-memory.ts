export interface UnifiedOctreeMemoryInputs {
  finestCellCount: number;
  leafCapacity: number;
  faceCapacity: number;
  pressureEntryCapacity: number;
  interfacePageCapacity: number;
  interfacePageSamples: number;
}

export interface UnifiedOctreeMemoryPlan {
  denseCompatibilityBytes: number;
  adaptiveLeafBytes: number;
  adaptiveFaceBytes: number;
  adaptiveSurfaceBytes: number;
  adaptiveTotalBytes: number;
  reductionRatio: number;
}

/**
 * Working-set contract for the final unified representation. This deliberately
 * excludes transient diagnostic materializations and renderer targets.
 *
 * Dense compatibility is the current MacCormack octree host footprint:
 * four rgba32 velocity fields, two r32 pressure fields, two dormant r32 VOF
 * fields, one rg32 flux field, and two padded rgba16 transport fields.
 * The adaptive side budgets compact pressure/header adjacency, double-buffered
 * scalar face velocities, canonical topology/incidence, and double-buffered
 * interface phi pages. No term scales directly with the finest box except the
 * comparison baseline.
 */
export function planUnifiedOctreeWorkingSet(input: UnifiedOctreeMemoryInputs): UnifiedOctreeMemoryPlan {
  for (const [name, value] of Object.entries(input)) {
    if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${name} must be a non-negative safe integer`);
  }
  if (input.finestCellCount < 1) throw new RangeError("finestCellCount must be positive");

  const paddedTransportCells = Math.ceil(input.finestCellCount * 1.08);
  const denseCompatibilityBytes = input.finestCellCount * (4 * 16 + 2 * 4 + 2 * 4 + 8)
    + paddedTransportCells * (2 * 8);

  // Per leaf: two pressure scalars (8), header/gradient (48), compact owner
  // record (8). Each pressure entry is neighbor row + coefficient (8).
  const adaptiveLeafBytes = input.leafCapacity * 64 + input.pressureEntryCapacity * 8;
  // Per canonical face: topology/origin/geometry (20), two velocity states
  // (8), signed incidence payload on its two incident rows (8). Row counts are
  // one u32 per leaf.
  const adaptiveFaceBytes = input.faceCapacity * 36 + input.leafCapacity * 4;
  // Page metadata is two u32s; phi current/predicted are two fp32 samples.
  const adaptiveSurfaceBytes = input.interfacePageCapacity * 8
    + input.interfacePageCapacity * input.interfacePageSamples * 8;
  const adaptiveTotalBytes = adaptiveLeafBytes + adaptiveFaceBytes + adaptiveSurfaceBytes;
  return {
    denseCompatibilityBytes,
    adaptiveLeafBytes,
    adaptiveFaceBytes,
    adaptiveSurfaceBytes,
    adaptiveTotalBytes,
    reductionRatio: denseCompatibilityBytes / Math.max(1, adaptiveTotalBytes),
  };
}
