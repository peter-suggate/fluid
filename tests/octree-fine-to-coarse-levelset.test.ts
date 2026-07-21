import assert from "node:assert/strict";
import test from "node:test";
import { buildFineToCoarsePhiCSR } from "../lib/octree-fine-to-coarse-levelset";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey, planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";

test("global fine bricks restrict to a new compact row generation without changing brick identity", () => {
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [2, 1, 1],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 2 });
  const fine = new FineLevelSetBrickOracle(plan);
  const key = packFineLevelSetBrickKey(plan, [0, 0, 0]);
  fine.publishInterfaceAndRing([key], ([x]) => x - 1);
  const first = buildFineToCoarsePhiCSR(fine, [
    { row: 0, origin: [0, 0, 0], size: 1, phi: -0.5 },
    { row: 1, origin: [1, 0, 0], size: 1, phi: 0.5 },
  ], 2);
  const second = buildFineToCoarsePhiCSR(fine, [
    { row: 1, origin: [0, 0, 0], size: 1, phi: -0.5 },
    { row: 0, origin: [1, 0, 0], size: 1, phi: 0.5 },
  ], 2);
  assert.equal(fine.pageForKey(key)?.key, key);
  assert.equal(first.rowOffsets[1], second.rowOffsets[1]);
  assert.ok(first.contributions[0].phi < 0);
  assert.ok(second.contributions[0].phi > 0);
  assert.equal(first.unownedSamples, 0);
});
