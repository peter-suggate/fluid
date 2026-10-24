import assert from "node:assert/strict";
import test from "node:test";
import { createNbflipSlopeClearance } from "../lib/core/nbflip-slope-clearance-scene";
import { SOLID_MASK_UNIT_WORDS, SolidOccupancyMask } from "../lib/core/solid-occupancy-mask";
import { fluidSolidWorldForScene, sceneWithSolidStroke } from "../lib/core/solid-world";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformSolidMaskPages } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";

test("a solid mask's slots are the pages partly solid, sealed at a count and refused past it", () => {
  const pages = uniformSolidMaskPages(uniformLatticeAddress([256, 64, 128], false));
  const blocks = (n: number) => Array.from({ length: n }, (_, i) => ({ operation: "fill" as const, minimum: [32 * i + 3, 20, 7] as [number, number, number], maximumExclusive: [32 * i + 5, 21, 9] as [number, number, number] }));
  const mask = new SolidOccupancyMask(pages), base = createNbflipSlopeClearance();
  mask.update(fluidSolidWorldForScene(sceneWithSolidStroke(base, blocks(2))));
  const held = mask.slotsHeld, first = mask.layout.firstSlot;
  assert.ok(held >= 2 && held < pages.slots, "the build's partly solid pages, not the box's");
  assert.ok(mask.words.length < mask.layout.words, "the host image holds the slots handed out, not the slots allowed");
  mask.seal(held + 1);
  assert.equal(mask.layout.slots, held + 1); assert.equal(mask.words.length, (first + held + 1) * SOLID_MASK_UNIT_WORDS);
  assert.throws(() => mask.seal(held - 1), /cannot be sealed/);
  // Every cell as an image sized for the box reads it: the units are taken in the same order.
  const whole = new SolidOccupancyMask(pages); whole.update(fluidSolidWorldForScene(sceneWithSolidStroke(base, blocks(3))));
  mask.update(fluidSolidWorldForScene(sceneWithSolidStroke(base, blocks(3))));
  assert.deepEqual(mask.words.subarray(0, (first + held + 1) * SOLID_MASK_UNIT_WORDS), whole.words.subarray(0, (first + held + 1) * SOLID_MASK_UNIT_WORDS));
  assert.throws(() => mask.update(fluidSolidWorldForScene(sceneWithSolidStroke(base, blocks(4)))), /page slots/);
});
