import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_COARSE_PHI_BYTES,
  OCTREE_COARSE_PHI_FLAG,
  correctCoarsePhiFromFine,
  packOctreeCoarsePhiRecords,
  resolveTwoResolutionPhi,
} from "../lib/octree-coarse-levelset";

test("fine restriction preserves a thin-sheet zero crossing in the coarse interval", () => {
  const corrected = correctCoarsePhiFromFine([
    { row: 0, origin: [0, 0, 0], size: 16, phi: 8 },
  ], [
    { point: [7.75, 8, 8], phi: -0.1 },
    { point: [8.25, 8, 8], phi: 0.1 },
  ], 7);
  const row = corrected.rows.get(0);
  assert.ok(row);
  assert.equal(row.minimumPhi, -0.1);
  assert.equal(row.maximumPhi, 0.1);
  assert.ok((row.flags & OCTREE_COARSE_PHI_FLAG.containsInterface) !== 0);
  assert.ok((row.flags & OCTREE_COARSE_PHI_FLAG.correctedFromFine) !== 0);
  assert.equal(row.generation, 7);
});

test("global fine samples resolve geometrically when cached owner rows are stale", () => {
  const corrected = correctCoarsePhiFromFine([
    { row: 4, origin: [0, 0, 0], size: 8, phi: 2 },
    { row: 9, origin: [8, 0, 0], size: 8, phi: 2 },
  ], [{ point: [9, 2, 2], phi: -0.25, coarseRow: 4 }]);
  assert.equal(corrected.rows.get(4)?.fineSampleCount, 0);
  assert.equal(corrected.rows.get(9)?.fineSampleCount, 1);
});

test("fine phi owns valid narrow-band queries and missing bricks use coarse phi", () => {
  const coarse = correctCoarsePhiFromFine([
    { row: 0, origin: [0, 0, 0], size: 4, phi: -3 },
  ], []).rows.get(0);
  assert.ok(coarse);
  assert.equal(resolveTwoResolutionPhi(0.125, coarse), 0.125);
  assert.equal(resolveTwoResolutionPhi(undefined, coarse), -3);
});

test("coarse GPU records have a stable 16-byte row ABI", () => {
  const rows = correctCoarsePhiFromFine([
    { row: 1, origin: [0, 0, 0], size: 4, phi: 2 },
  ], [{ point: [1, 1, 1], phi: -0.5 }]).rows;
  const packed = packOctreeCoarsePhiRecords(3, rows);
  assert.equal(packed.byteLength, 3 * OCTREE_COARSE_PHI_BYTES);
  const floats = new Float32Array(packed);
  const words = new Uint32Array(packed);
  assert.equal(floats[4], -0.5);
  assert.equal(floats[5], -0.5);
  assert.equal(floats[6], -0.5);
  assert.ok((words[7] & OCTREE_COARSE_PHI_FLAG.valid) !== 0);
});

test("non-finite fine samples fail closed", () => {
  assert.throws(() => correctCoarsePhiFromFine([
    { row: 0, origin: [0, 0, 0], size: 4, phi: 1 },
  ], [{ point: [1, 1, 1], phi: Number.NaN }]), /finite/);
});
