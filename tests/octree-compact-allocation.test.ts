import assert from "node:assert/strict";
import test from "node:test";

import { planOctreeCompactAllocation, planOctreeTwoResolutionLevelSetAllocation } from "../lib/octree-compact-allocation";
import { planOctreeFaceMirror } from "../lib/webgpu-octree-face-mirror";
import { planOctreeFaceTopologyTransfer } from "../lib/webgpu-octree-face-transfer";
import { planOctreeSurfacePages } from "../lib/webgpu-octree-surface-pages";

test("production face capacities use the proved 24-candidate bound and omit audit records", () => {
  const rows = 43_520;
  const compactMirror = planOctreeFaceMirror(rows);
  const legacyMirror = planOctreeFaceMirror(rows, 48);
  assert.equal(compactMirror.faceCapacity, 1_044_480);
  assert.equal(legacyMirror.faceCapacity, 2_088_960);
  const compactTransfer = planOctreeFaceTopologyTransfer(compactMirror.faceCapacity);
  const inspectedTransfer = planOctreeFaceTopologyTransfer(compactMirror.faceCapacity, { retainRecords: true });
  assert.equal(compactTransfer.recordBytes, 0);
  assert.equal(inspectedTransfer.recordBytes - compactTransfer.recordBytes, compactMirror.faceCapacity * 24);
  assert.ok(compactMirror.allocatedBytes + compactTransfer.allocatedBytes
    < (legacyMirror.allocatedBytes + planOctreeFaceTopologyTransfer(legacyMirror.faceCapacity, { retainRecords: true }).allocatedBytes) * 0.5);
});

test("surface page backing remains explicitly budgetable without an implicit correctness cap", () => {
  const dimensions = [61, 46, 41] as const;
  const densePhiBytes = dimensions[0] * dimensions[1] * dimensions[2] * 3 * 4;
  const pages = planOctreeSurfacePages(43_520, dimensions, {
    maximumResidentFraction: 0.75,
    maximumArenaBytes: densePhiBytes,
  });
  assert.ok(pages.arenaBytes <= densePhiBytes);
  assert.equal(pages.pageCapacity, 2_276);
});

test("deep-domain compact allocation has an exact surface-to-volume crossover", () => {
  const dam = planOctreeCompactAllocation([61, 46, 41], 16, 4);
  assert.equal(dam.denseSurfaceBytesRemoved, 3_221_288);
  assert.equal(dam.denseSnapshotBytesRemoved, 460_180);
  assert.equal(dam.denseOwnerBytesRemoved, 920_368);
  assert.equal(dam.denseFrontierBytesRemoved, 460_184);
  assert.equal(dam.denseAtlasBytesRemoved, 5_883_648);
  assert.equal(dam.ownerPageBytes, 594_144);
  assert.equal(dam.frontierHashBytes, 524_288);
  assert.equal(dam.denseBytesRemoved, dam.denseHostBytesRemoved + dam.denseSurfaceBytesRemoved
    + dam.denseSnapshotBytesRemoved + dam.denseOwnerBytesRemoved + dam.denseFrontierBytesRemoved
    + dam.denseAtlasBytesRemoved);
  assert.equal(dam.surfacePageBytes, 3_425_360);
  // Includes the face mirror's projected-divergence tail and the full transfer diagnostics arena.
  assert.equal(dam.compactAuxiliaryBytes, 72_386_280);
  assert.equal(dam.netBytes, 48_775_976);
  assert.ok(dam.netBytes > 0, "the proved face bound truthfully exposes the small-scene radix-transfer overhead");

  const ocean = planOctreeCompactAllocation([320, 96, 80], 32, 4);
  assert.equal(ocean.denseSurfaceBytesRemoved, 68_812_800);
  assert.equal(ocean.denseSnapshotBytesRemoved, 9_830_396);
  assert.equal(ocean.denseOwnerBytesRemoved, 19_660_800);
  assert.equal(ocean.denseFrontierBytesRemoved, 9_830_400);
  assert.equal(ocean.denseAtlasBytesRemoved, 98_317_792);
  assert.equal(ocean.ownerPageBytes, 8_279_640);
  assert.equal(ocean.frontierHashBytes, 4_194_304);
  assert.equal(ocean.surfacePageBytes, 28_958_884);
  assert.equal(ocean.compactAuxiliaryBytes, 703_133_876);
  assert.equal(ocean.netBytes, 229_181_124);
  assert.ok(ocean.netBytes > 0, "the 320-wide target records the temporary capacity-sized radix-transfer amplification");

  const target = planOctreeCompactAllocation([640, 192, 160], 32, 4);
  assert.equal(target.denseSurfaceBytesRemoved, 550_502_400);
  assert.equal(target.denseSnapshotBytesRemoved, 78_643_196);
  assert.equal(target.denseOwnerBytesRemoved, 157_286_400);
  assert.equal(target.denseFrontierBytesRemoved, 78_643_200);
  assert.equal(target.denseAtlasBytesRemoved, 786_540_992);
  assert.equal(target.ownerPageBytes, 33_153_340);
  assert.equal(target.frontierHashBytes, 16_777_216);
  assert.equal(target.surfacePageBytes, 115_911_096);
  assert.equal(target.compactAuxiliaryBytes, 2_815_069_356);
  assert.equal(target.netBytes, -968_168_740);
  assert.ok(target.netBytes < -900_000_000, "the large target still saves substantial memory with the proved face bound");
});

test("two-resolution level-set accounting scales with resident bricks and live rows", () => {
  const factor4 = planOctreeTwoResolutionLevelSetAllocation({ dimensions: [320, 96, 80], physicalCellWidth: 0.1,
    rowCapacity: 10_000, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 2_000 });
  const factor8 = planOctreeTwoResolutionLevelSetAllocation({ dimensions: [320, 96, 80], physicalCellWidth: 0.1,
    rowCapacity: 10_000, fineFactor: 8, brickResolution: 4, maximumResidentBricks: 2_000 });
  assert.equal(factor4.coarsePhiBytes, factor8.coarsePhiBytes);
  assert.equal(factor4.finePayloadCapacityBytes, factor8.finePayloadCapacityBytes,
    "equal brick resolution and resident capacity have equal bounded payload despite the larger logical address space");
  assert.equal(factor4.fineRollbackCapacityBytes,
    factor4.maximumResidentBricks * 4 ** 3 * 4);
  assert.equal(factor4.fineParameterBytes, 160);
  assert.equal(factor4.allocatedBytes, factor4.finePayloadCapacityBytes + factor4.fineRollbackCapacityBytes
    + factor4.fineHashAndMetadataBytes + factor4.fineWorklistBytes
    + factor4.fineParameterBytes + factor4.coarsePhiBytes);
});
