/** Deterministic bridge from global fine bricks to compact coarse-octree rows. */

import { FINE_LEVELSET_SAMPLE_FLAGS, type FineLevelSetBrickOracle,
  unpackFineLevelSetBrickKey } from "./octree-fine-levelset-bricks";
import type { OctreeCoarsePhiLeaf } from "./octree-coarse-levelset";
import type { OctreeFinePhiContribution } from "./webgpu-octree-coarse-levelset";

export interface FineToCoarsePhiCSR {
  readonly rowOffsets: Uint32Array;
  readonly contributions: readonly OctreeFinePhiContribution[];
  readonly maximumContributionsPerRow: number;
  readonly unownedSamples: number;
}

function contains(leaf: OctreeCoarsePhiLeaf, point: readonly [number, number, number]): boolean {
  return point.every((value, axis) => value >= leaf.origin[axis] && value < leaf.origin[axis] + leaf.size);
}

/**
 * CPU oracle/planner for the GPU owner-page classification pass. Fine brick
 * keys remain independent of the resulting row IDs; row ownership is resolved
 * afresh for the compact generation and emitted as deterministic CSR.
 */
export function buildFineToCoarsePhiCSR(
  oracle: FineLevelSetBrickOracle,
  leaves: readonly OctreeCoarsePhiLeaf[],
  rowCapacity: number,
): FineToCoarsePhiCSR {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) {
    throw new RangeError("Fine-to-coarse row capacity must be a positive integer");
  }
  const byRow = new Map<number, OctreeCoarsePhiLeaf>();
  for (const leaf of leaves) {
    if (!Number.isSafeInteger(leaf.row) || leaf.row < 0 || leaf.row >= rowCapacity) {
      throw new RangeError("Fine-to-coarse leaf row exceeds capacity");
    }
    if (byRow.has(leaf.row)) throw new RangeError(`Duplicate fine-to-coarse row ${leaf.row}`);
    byRow.set(leaf.row, leaf);
  }
  const rows: Array<Array<OctreeFinePhiContribution & { order: number }>> = Array.from({ length: rowCapacity }, () => []);
  let unownedSamples = 0;
  for (const page of [...oracle.residentPages()].sort((a, b) => a.key - b.key)) {
    const brick = unpackFineLevelSetBrickKey(oracle.plan, page.key);
    for (let z = 0; z < oracle.plan.brickResolution; z += 1) for (let y = 0; y < oracle.plan.brickResolution; y += 1) {
      for (let x = 0; x < oracle.plan.brickResolution; x += 1) {
        const local = x + oracle.plan.brickResolution * (y + oracle.plan.brickResolution * z);
        if ((page.flags[local] & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0) continue;
        const q = [brick[0] * oracle.plan.brickResolution + x,
          brick[1] * oracle.plan.brickResolution + y,
          brick[2] * oracle.plan.brickResolution + z] as const;
        if (q.some((value, axis) => value >= oracle.plan.sampleDimensions[axis])) continue;
        const point = q.map((value, axis) => oracle.plan.domainOrigin[axis]
          + (value + 0.5) * oracle.plan.fineCellWidth) as [number, number, number];
        const owner = leaves.find((leaf) => contains(leaf, point));
        if (!owner) { unownedSamples += 1; continue; }
        const centre = owner.origin.map((value) => value + owner.size * 0.5);
        const distanceSquared = point.reduce((sum, value, axis) => sum + (value - centre[axis]) ** 2, 0);
        rows[owner.row].push({ phi: page.phi[local], distanceSquared, valid: true,
          order: page.key * oracle.plan.samplesPerBrick + local });
      }
    }
  }
  const rowOffsets = new Uint32Array(rowCapacity + 1);
  const contributions: OctreeFinePhiContribution[] = [];
  let maximumContributionsPerRow = 0;
  for (let row = 0; row < rowCapacity; row += 1) {
    rowOffsets[row] = contributions.length;
    rows[row].sort((a, b) => a.order - b.order);
    maximumContributionsPerRow = Math.max(maximumContributionsPerRow, rows[row].length);
    contributions.push(...rows[row].map(({ phi, distanceSquared, valid }) => ({ phi, distanceSquared, valid })));
  }
  rowOffsets[rowCapacity] = contributions.length;
  return { rowOffsets, contributions, maximumContributionsPerRow, unownedSamples };
}
