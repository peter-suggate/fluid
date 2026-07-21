/**
 * Compact coarse-octree level-set contracts for the two-resolution surface
 * representation described by Aanjaneya et al. (2017), Section 5.
 *
 * Fine phi owns the interface wherever a global narrow-band brick is valid.
 * This field owns sign and distance everywhere else.  The interval is not a
 * volume fraction: it conservatively records fine signed-distance extrema so
 * restricting a thin sheet cannot erase its zero crossing.
 */

export const OCTREE_COARSE_PHI_BYTES = 16;

export const OCTREE_COARSE_PHI_FLAG = Object.freeze({
  valid: 1 << 0,
  correctedFromFine: 1 << 1,
  containsInterface: 1 << 2,
  finite: 1 << 3,
} as const);

export type OctreeLevelSetPoint = readonly [number, number, number];

export interface OctreeCoarsePhiLeaf {
  readonly row: number;
  readonly origin: OctreeLevelSetPoint;
  readonly size: number;
  readonly phi: number;
  readonly minimumPhi?: number;
  readonly maximumPhi?: number;
  readonly generation?: number;
}

export interface OctreeFinePhiSample {
  readonly point: OctreeLevelSetPoint;
  readonly phi: number;
  /** Optional compact-owner result. Geometry remains authoritative. */
  readonly coarseRow?: number;
}

export interface OctreeCoarsePhiRecord {
  readonly phi: number;
  readonly minimumPhi: number;
  readonly maximumPhi: number;
  readonly flags: number;
  readonly generation: number;
  readonly fineSampleCount: number;
}

export interface OctreeCoarsePhiCorrection {
  readonly rows: ReadonlyMap<number, OctreeCoarsePhiRecord>;
  readonly correctedRows: number;
  readonly interfaceRows: number;
  readonly unownedFineSamples: number;
}

function finite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return value;
}

function u32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be a non-negative u32 integer`);
  }
  return value;
}

function contains(leaf: OctreeCoarsePhiLeaf, point: OctreeLevelSetPoint): boolean {
  return point.every((coordinate, axis) => coordinate >= leaf.origin[axis]
    && coordinate < leaf.origin[axis] + leaf.size);
}

function distanceSquared(a: OctreeLevelSetPoint, b: OctreeLevelSetPoint): number {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
}

function centre(leaf: OctreeCoarsePhiLeaf): OctreeLevelSetPoint {
  const half = leaf.size * 0.5;
  return [leaf.origin[0] + half, leaf.origin[1] + half, leaf.origin[2] + half];
}

function validateLeaf(leaf: OctreeCoarsePhiLeaf): void {
  u32(leaf.row, "Coarse phi row");
  if (!Number.isSafeInteger(leaf.size) || leaf.size < 1) {
    throw new RangeError("Coarse phi leaf size must be a positive integer");
  }
  leaf.origin.forEach((value, axis) => finite(value, `Coarse phi origin axis ${axis}`));
  finite(leaf.phi, "Coarse phi");
  if (leaf.minimumPhi !== undefined) finite(leaf.minimumPhi, "Coarse minimum phi");
  if (leaf.maximumPhi !== undefined) finite(leaf.maximumPhi, "Coarse maximum phi");
  if (leaf.minimumPhi !== undefined && leaf.maximumPhi !== undefined
    && leaf.minimumPhi > leaf.maximumPhi) {
    throw new RangeError("Coarse phi interval must be ordered");
  }
}

/**
 * Restrict valid fine-band samples onto compact live leaves.  The closest
 * fine sample supplies center phi, as permitted by Section 18.8, while all
 * samples contribute to the conservative min/max interval.  Therefore a
 * sign-changing fine set always produces an interface-bearing coarse record.
 */
export function correctCoarsePhiFromFine(
  leaves: readonly OctreeCoarsePhiLeaf[],
  samples: readonly OctreeFinePhiSample[],
  generationValue = 0,
): OctreeCoarsePhiCorrection {
  const generation = u32(generationValue, "Coarse phi generation");
  const byRow = new Map<number, OctreeCoarsePhiLeaf>();
  const owned = new Map<number, OctreeFinePhiSample[]>();
  for (const leaf of leaves) {
    validateLeaf(leaf);
    if (byRow.has(leaf.row)) throw new RangeError(`Duplicate coarse phi row ${leaf.row}`);
    byRow.set(leaf.row, leaf);
    owned.set(leaf.row, []);
  }

  let unownedFineSamples = 0;
  for (const sample of samples) {
    sample.point.forEach((value, axis) => finite(value, `Fine phi point axis ${axis}`));
    finite(sample.phi, "Fine phi sample");
    let owner = sample.coarseRow === undefined ? undefined : byRow.get(sample.coarseRow);
    if (owner && !contains(owner, sample.point)) owner = undefined;
    if (!owner) owner = leaves.find((leaf) => contains(leaf, sample.point));
    if (!owner) {
      unownedFineSamples += 1;
      continue;
    }
    owned.get(owner.row)?.push(sample);
  }

  const rows = new Map<number, OctreeCoarsePhiRecord>();
  let correctedRows = 0;
  let interfaceRows = 0;
  for (const leaf of leaves) {
    const fine = owned.get(leaf.row) ?? [];
    const previousMinimum = leaf.minimumPhi ?? leaf.phi;
    const previousMaximum = leaf.maximumPhi ?? leaf.phi;
    let phi = leaf.phi;
    let minimumPhi = previousMinimum;
    let maximumPhi = previousMaximum;
    let flags = OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite;
    if (fine.length > 0) {
      const leafCentre = centre(leaf);
      let closest = fine[0];
      let closestDistance = distanceSquared(closest.point, leafCentre);
      minimumPhi = closest.phi;
      maximumPhi = closest.phi;
      for (let index = 1; index < fine.length; index += 1) {
        const sample = fine[index];
        minimumPhi = Math.min(minimumPhi, sample.phi);
        maximumPhi = Math.max(maximumPhi, sample.phi);
        const sampleDistance = distanceSquared(sample.point, leafCentre);
        if (sampleDistance < closestDistance
          || (sampleDistance === closestDistance && sample.phi < closest.phi)) {
          closest = sample;
          closestDistance = sampleDistance;
        }
      }
      phi = closest.phi;
      flags |= OCTREE_COARSE_PHI_FLAG.correctedFromFine;
      correctedRows += 1;
    }
    if (minimumPhi <= 0 && maximumPhi >= 0) {
      flags |= OCTREE_COARSE_PHI_FLAG.containsInterface;
      interfaceRows += 1;
    }
    rows.set(leaf.row, { phi, minimumPhi, maximumPhi, flags, generation, fineSampleCount: fine.length });
  }
  return { rows, correctedRows, interfaceRows, unownedFineSamples };
}

/** Resolve fine authority when present and the compact coarse fallback otherwise. */
export function resolveTwoResolutionPhi(finePhi: number | undefined, coarse: OctreeCoarsePhiRecord): number {
  if (finePhi !== undefined) return finite(finePhi, "Resolved fine phi");
  if ((coarse.flags & OCTREE_COARSE_PHI_FLAG.valid) === 0) {
    throw new RangeError("Missing fine phi requires a valid coarse phi fallback");
  }
  return finite(coarse.phi, "Resolved coarse phi");
}

/** Pack the externally visible 16-byte GPU record: phi, min, max, flags. */
export function packOctreeCoarsePhiRecords(
  rowCapacity: number,
  records: ReadonlyMap<number, OctreeCoarsePhiRecord>,
): ArrayBuffer {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) {
    throw new RangeError("Coarse phi row capacity must be a positive integer");
  }
  const buffer = new ArrayBuffer(rowCapacity * OCTREE_COARSE_PHI_BYTES);
  const floats = new Float32Array(buffer);
  const words = new Uint32Array(buffer);
  for (const [row, record] of records) {
    if (u32(row, "Packed coarse phi row") >= rowCapacity) throw new RangeError("Coarse phi row exceeds capacity");
    const base = row * 4;
    floats[base] = finite(record.phi, "Packed coarse phi");
    floats[base + 1] = finite(record.minimumPhi, "Packed coarse minimum phi");
    floats[base + 2] = finite(record.maximumPhi, "Packed coarse maximum phi");
    words[base + 3] = u32(record.flags, "Packed coarse phi flags");
  }
  return buffer;
}
