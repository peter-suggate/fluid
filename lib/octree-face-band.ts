/** Shared sizing and phi-summary rules for the Section 5 regular-face band. */

/** Four 2:1 subfaces on each positive axis. */
export const OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW = 12;
/** Six sides times four 2:1 subfaces is the exact regular-face incidence bound. */
export const OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW = 24;
export const OCTREE_REGULAR_BAND_FACE_BYTES = 32;

export interface OctreeFaceBandPhiSummary {
  readonly representativePhi: number;
  readonly minimumPhi: number;
  readonly maximumPhi: number;
}

/** Closest-to-interface representative without losing a mixed-sign interval. */
export function summarizeOctreeFaceBandPhi(values: readonly number[]): OctreeFaceBandPhiSummary {
  if (values.length === 0) throw new RangeError("Face-band phi summary needs at least one sample");
  let representativePhi = Number.POSITIVE_INFINITY, minimumPhi = Number.POSITIVE_INFINITY;
  let maximumPhi = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) throw new RangeError("Face-band phi samples must be finite");
    minimumPhi = Math.min(minimumPhi, value); maximumPhi = Math.max(maximumPhi, value);
    if (Math.abs(value) < Math.abs(representativePhi)
      || (Math.abs(value) === Math.abs(representativePhi) && value < representativePhi)) representativePhi = value;
  }
  return { representativePhi, minimumPhi, maximumPhi };
}

export interface OctreeRegularFaceBandPlan {
  readonly wetRowCapacity: number;
  readonly maximumFineBricks: number;
  readonly ownerCandidatesPerBrick: number;
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly faceBytes: number;
  readonly incidenceBytes: number;
  readonly phiBytes: number;
  readonly allocatedBytes: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

/**
 * Every active fine brick intersects at most ceil(B / m)^3 finest octree
 * cells. Deduplicating their containing owner keys can only reduce the number
 * of dry band rows. Wet pressure rows remain a disjoint, stable prefix.
 */
export function planOctreeRegularFaceBand(
  wetRowCapacityValue: number,
  maximumFineBricksValue: number,
  brickResolutionValue: number,
  fineFactorValue: number,
): OctreeRegularFaceBandPlan {
  const wetRowCapacity = positiveInteger(wetRowCapacityValue, "Wet row capacity");
  const maximumFineBricks = positiveInteger(maximumFineBricksValue, "Fine brick capacity");
  const brickResolution = positiveInteger(brickResolutionValue, "Fine brick resolution");
  const fineFactor = positiveInteger(fineFactorValue, "Fine factor");
  const ownerCandidatesPerAxis = Math.ceil(brickResolution / fineFactor);
  const ownerCandidatesPerBrick = ownerCandidatesPerAxis ** 3;
  const rowCapacity = wetRowCapacity + maximumFineBricks * ownerCandidatesPerBrick;
  if (!Number.isSafeInteger(rowCapacity)) throw new RangeError("Regular face-band row capacity exceeds exact integer range");
  const faceCapacity = rowCapacity * OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW;
  const faceBytes = faceCapacity * OCTREE_REGULAR_BAND_FACE_BYTES;
  const incidenceBytes = rowCapacity * (1 + OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW) * 4;
  const phiBytes = faceCapacity * 4;
  return { wetRowCapacity, maximumFineBricks, ownerCandidatesPerBrick, rowCapacity, faceCapacity,
    faceBytes, incidenceBytes, phiBytes, allocatedBytes: faceBytes + incidenceBytes + phiBytes };
}
