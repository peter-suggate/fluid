import assert from "node:assert/strict";
import test from "node:test";
import { planUnifiedOctreeWorkingSet } from "../lib/octree-unified-memory";

test("unified octree working set contains no finest-box-sized adaptive term", () => {
  const common = {
    leafCapacity: 80_000,
    faceCapacity: 260_000,
    pressureEntryCapacity: 900_000,
    interfacePageCapacity: 4_000,
    interfacePageSamples: 10 ** 3,
  };
  const shallow = planUnifiedOctreeWorkingSet({ finestCellCount: 320 * 48 * 80, ...common });
  const deep = planUnifiedOctreeWorkingSet({ finestCellCount: 320 * 96 * 80, ...common });
  assert.equal(deep.adaptiveTotalBytes, shallow.adaptiveTotalBytes);
  assert.equal(deep.denseCompatibilityBytes, 2 * shallow.denseCompatibilityBytes);
});

test("deep-ocean target clears the four-times working-set gate", () => {
  const plan = planUnifiedOctreeWorkingSet({
    finestCellCount: 320 * 96 * 80,
    leafCapacity: 80_000,
    faceCapacity: 260_000,
    pressureEntryCapacity: 900_000,
    interfacePageCapacity: 4_000,
    interfacePageSamples: 10 ** 3,
  });
  assert.ok(plan.reductionRatio >= 4, `expected >=4x, got ${plan.reductionRatio.toFixed(2)}x`);
});

test("unified memory planning rejects fractional and negative capacities", () => {
  assert.throws(() => planUnifiedOctreeWorkingSet({
    finestCellCount: 1,
    leafCapacity: -1,
    faceCapacity: 1,
    pressureEntryCapacity: 1,
    interfacePageCapacity: 1,
    interfacePageSamples: 1,
  }), /leafCapacity/);
});
