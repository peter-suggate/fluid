import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandBudgetCapacity } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { particleSlabSpill } from "../lib/methods/uniform/uniform-particle-slab.wgsl";
import { particleDepositStorage } from "../lib/methods/uniform/uniform-particle-deposit-storage";
import { validateScene } from "../lib/core/model";
import { createReservoirSeepScene } from "../lib/core/reservoir-seep-scene";

test("a particle budget's slabs hold its samples, the spill reserve on top", () => {
  for (const budget of [{ fineTiles: 1, perFineCell: 4 }, { fineTiles: 4000, perFineCell: 4 }, { fineTiles: 16000, perFineCell: 2.5 }]) {
    const capacity = narrowBandBudgetCapacity(budget), samples = Math.ceil(64 * budget.fineTiles * budget.perFineCell);
    assert.ok(capacity - particleSlabSpill(capacity) >= samples);
    assert.ok(capacity <= samples + Math.max(4096, Math.ceil(samples / 15)) + 1);
  }
});

test("a budget's deposit still shares the consumed particle bank", () => {
  // 385 deposit words a fine tile against 64 cells of 48-byte records at one a cell.
  const fineTiles = 4000, bank = 48 * narrowBandBudgetCapacity({ fineTiles, perFineCell: 1 });
  assert.equal(particleDepositStorage(1 << 20, fineTiles, 2 ** 31, bank).shared, true);
});

test("fluid.tiers validates its brick count and density", () => {
  const errors = (tiers: object) => { const scene = createReservoirSeepScene(); (scene.fluid as { tiers?: object }).tiers = tiers; return validateScene(scene).filter(e => e.includes("fluid.tiers")); };
  assert.deepEqual(errors({ bricks: 512, particlesPerFineCell: 4 }), []);
  assert.equal(errors({ bricks: 0 }).length, 1);
  assert.equal(errors({ bricks: 1.5 }).length, 1);
  assert.equal(errors({ particlesPerFineCell: 0 }).length, 1);
});
