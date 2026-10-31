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
  const got = uniformBedSlopePartialColumns(mask.words, size), want = new Uint8Array(4);
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
  const [pagesX, pagesZ] = l.pages;
  const near = (px: number, pz: number) => { for (let j = pz - l.laneRadius; j <= pz + l.laneRadius; j++) for (let i = px - l.laneRadius; i <= px + l.laneRadius; i++) if (i >= 0 && j >= 0 && i < pagesX && j < pagesZ && partial[i + pagesX * j]) return true; return false; };
  let inside = 0, outside = 0;
  for (let z = 0; z < BED_SLOPE_WIDE[2]; z++) for (let y = 0; y < BED_SLOPE_WIDE[1]; y++) for (let x = 0; x < BED_SLOPE_WIDE[0]; x++) {
    if (slope(x, y, z).every(v => v === 0)) continue;
    if (near(x >> 5, z >> 5)) inside++; else outside++;
  }
  assert.ok(inside > 0); assert.equal(outside, 0);
  assert.deepEqual(Array.from(partial.subarray(0, 16)), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  // 16x2 page columns: four super-block columns, every one of them in the lane's and the window's set.
  assert.deepEqual(uniformBedSlopeBakeCapacity(BED_SLOPE_WIDE, partial), { laneSlots: 32, windowSlots: 32, laneChildren: 4, windowChildren: 4 });

  // The solver's read of a sparse lane: a top word a super-block column, a
  // child word a page column, then a slot's words a column, lowest floor
  // first; past page column 63 as before it.
  const base = uniformSolidScratchLayout().bedSlopeBaseWord, child = base + 8192, slot = child + 64, words = new Uint32Array(slot + 2 * 1024 * UNIFORM_BED_SLOPE_LAYERS), entry = uniformSolidBedSlopeEntry(-0.05, 0.1);
  words[base + 0 + 64 * 0] = child; words[child + 1 + 4 * 2] = slot; words[slot + UNIFORM_BED_SLOPE_LAYERS * (3 + 32 * 5) + 1] = (17 + 1) << 16 | entry;
  // Page column (66, 201): super-block column (16, 50), its child (2, 1).
  const far = slot + 1024 * UNIFORM_BED_SLOPE_LAYERS;
  words[base + 16 + 64 * 50] = child + 16; words[child + 16 + 2 + 4 * 1] = far; words[far + UNIFORM_BED_SLOPE_LAYERS * (3 + 32 * 5) + 0] = (17 + 1) << 16 | entry;
  const at = (x: number, y: number, z: number) => uniformSolidBedSlopeAt(words, x, y, z);
  const want = [(entry << 24 >> 24) / UNIFORM_SOLID_BED_SLOPE_UNITS, (entry << 16 >> 24) / UNIFORM_SOLID_BED_SLOPE_UNITS];
  assert.deepEqual(at(35, 17, 69), want);
  assert.deepEqual(at(35, 16, 69), [0, 0]); assert.deepEqual(at(36, 17, 69), [0, 0]); assert.deepEqual(at(3, 17, 5), [0, 0]); assert.deepEqual(at(-1, 17, 69), [0, 0]);
  assert.deepEqual(at(32 * 66 + 3, 17, 32 * 201 + 5), want);
  assert.deepEqual(at(32 * 67 + 3, 17, 32 * 201 + 5), [0, 0], "the child's next page column has no slot");
  assert.deepEqual(at(32 * 2 + 3, 17, 32 * 9 + 5), [0, 0], "a super-block column with no child");
});
