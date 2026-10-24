import assert from "node:assert/strict";
import test from "node:test";
import { createNbflipSlopeClearance } from "../lib/core/nbflip-slope-clearance-scene";
import { SolidOccupancyMask } from "../lib/core/solid-occupancy-mask";
import { fluidSolidWorldForScene, sceneWithSolidStroke } from "../lib/core/solid-world";
import { UNIFORM_BED_SLOPE_LAYERS, uniformBedSlopeBakeCapacity, uniformBedSlopePartialColumns } from "../lib/methods/uniform/uniform-bed-slope-bake";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { UNIFORM_SOLID_BED_SLOPE_UNITS, uniformSolidBedSlopeAt, uniformSolidBedSlopeEntry, uniformSolidMaskPages, uniformSolidScratchLayout } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { BED_SLOPE_WIDE, bedMask, layoutOf, partialOf, reference, slopeIn, wideBed } from "../tools/bed-slope-bake-reference";

test("the bed-slope lane's page columns are those of the mask's partly solid pages", () => {
  const size = [128, 32, 32] as const;
  // A page made solid whole folds to FULL and is no partly solid page; a small block is one.
  const scene = sceneWithSolidStroke(createNbflipSlopeClearance(), [
    { operation: "fill", minimum: [64, 0, 0], maximumExclusive: [96, 32, 32] }, { operation: "fill", minimum: [8, 20, 16], maximumExclusive: [10, 22, 18] }]);
  const mask = new SolidOccupancyMask(uniformSolidMaskPages(uniformLatticeAddress([...size], false)));
  mask.update(fluidSolidWorldForScene(scene));
  const got = uniformBedSlopePartialColumns(mask.words, size), want = new Uint8Array(4096);
  for (let px = 0; px < 4; px++) {
    let seen = 0;
    for (let z = 0; z < 32; z++) for (let y = 0; y < 32; y++) for (let x = 32 * px; x < 32 * px + 32; x++) seen |= mask.solid(x, y, z) ? 1 : 2;
    want[px] = +(seen === 3);
  }
  assert.deepEqual(got, want);
  assert.equal(got[2], 0, "the full page"); assert.equal(got[0], 1, "the block's page");
  const slots = uniformBedSlopeBakeCapacity(size, got);
  assert.ok(slots.laneSlots <= 4 && slots.windowSlots <= 4, "never more than the lattice's page columns");
});

test("a dense bake's lane is zero past the lane's page columns, and the sparse read is umBedSlope's", () => {
  // The emulation bakes the whole lattice densely: what the sparse bake keeps must be all it writes.
  const mask = bedMask(BED_SLOPE_WIDE, x => wideBed(x)), l = layoutOf(BED_SLOPE_WIDE), partial = partialOf(mask), lane = reference(mask, l).lane, slope = slopeIn(BED_SLOPE_WIDE, lane);
  const near = (px: number, pz: number) => { for (let j = pz - l.laneRadius; j <= pz + l.laneRadius; j++) for (let i = px - l.laneRadius; i <= px + l.laneRadius; i++) if (i >= 0 && j >= 0 && i < 64 && j < 64 && partial[i + 64 * j]) return true; return false; };
  let inside = 0, outside = 0;
  for (let z = 0; z < BED_SLOPE_WIDE[2]; z++) for (let y = 0; y < BED_SLOPE_WIDE[1]; y++) for (let x = 0; x < BED_SLOPE_WIDE[0]; x++) {
    if (slope(x, y, z).every(v => v === 0)) continue;
    if (near(x >> 5, z >> 5)) inside++; else outside++;
  }
  assert.ok(inside > 0); assert.equal(outside, 0);
  assert.deepEqual(Array.from(partial.subarray(0, 16)), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert.deepEqual(uniformBedSlopeBakeCapacity(BED_SLOPE_WIDE, partial), { laneSlots: 32, windowSlots: 32 });

  // The solver's read of a sparse lane: a directory word a page column, then a slot's words a column, lowest floor first.
  const base = uniformSolidScratchLayout().bedSlopeBaseWord, slot = base + 8192, words = new Uint32Array(slot + 1024 * UNIFORM_BED_SLOPE_LAYERS), entry = uniformSolidBedSlopeEntry(-0.05, 0.1);
  words[base + 1 + 64 * 2] = slot; words[slot + UNIFORM_BED_SLOPE_LAYERS * (3 + 32 * 5) + 1] = (17 + 1) << 16 | entry;
  const at = (x: number, y: number, z: number) => uniformSolidBedSlopeAt(words, x, y, z);
  assert.deepEqual(at(35, 17, 69), [(entry << 24 >> 24) / UNIFORM_SOLID_BED_SLOPE_UNITS, (entry << 16 >> 24) / UNIFORM_SOLID_BED_SLOPE_UNITS]);
  assert.deepEqual(at(35, 16, 69), [0, 0]); assert.deepEqual(at(36, 17, 69), [0, 0]); assert.deepEqual(at(3, 17, 5), [0, 0]); assert.deepEqual(at(-1, 17, 69), [0, 0]);
});
