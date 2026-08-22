import assert from "node:assert/strict";
import test from "node:test";

import { planSvoNodeMipAddresses } from "../lib/svo/svo-node-mip-address-plan";
import { svoOpacityLevelFloor } from "../lib/svo/svo-node-mip-pyramid";
import { WebGpuLiveSvoTetrahedralRadiance } from "../lib/svo/webgpu-svo-tetrahedral-radiance";

/**
 * The radiance atlas has to fit in the slot space left above its offset.
 *
 * `WebGpuLiveSvoTetrahedralRadiance` shares the opacity pyramid's slot
 * numbering: pages below the radiance floor take a slot without taking an atlas
 * page, so the atlas occupies `[radianceSlotOffset, pageCapacity)` and its
 * constructor rejects anything wider. The atlas was sized from the *domain*
 * above the floor and the slot space from today's occupancy plus a fixed byte
 * reserve, and on a partial plan with the floor above level zero the domain
 * figure overflowed the space. The constructor threw, the whole derived-lighting
 * block in `webgpu-svo-sparse-bricks.ts` unwound, and the render panel read
 * `Lighting visibility: EXACT FALLBACK` before the first frame — measured on
 * `twin-dam-collision` at a 25 mm lattice, where the plan held 9489 slots, 6776
 * of them below the floor, and asked for a 9489-page atlas.
 *
 * A domain far larger than its occupancy, so `reservePages` binds and the plan
 * comes back partial: a ground slab across a 25x24x24 page grid plus one box.
 */
const basePageDimensions: [number, number, number] = [25, 24, 24];
const levelCount = 6;

function occupiedPages(): [number, number, number][] {
  const pages: [number, number, number][] = [];
  for (let x = 0; x < basePageDimensions[0]; x += 1) {
    for (let z = 0; z < basePageDimensions[2]; z += 1) pages.push([x, 0, z]);
  }
  for (let x = 8; x < 16; x += 1) {
    for (let y = 0; y < 4; y += 1) {
      for (let z = 8; z < 16; z += 1) pages.push([x, y, z]);
    }
  }
  return pages;
}

function planAt(cellSize_m: number) {
  return planSvoNodeMipAddresses({
    occupiedBasePages: occupiedPages(),
    basePageDimensions,
    levelCount,
    // The directory's own limit on a device that reports the WebGPU minimum.
    addressCapacity: 2_048 * Math.floor(2_048 / 2),
    generation: 1,
    cellSize_m,
    opacityFloorLevel: svoOpacityLevelFloor({ levelCount, cellSize_m }),
  });
}

/**
 * The constructor's precondition, transcribed. Executed rather than restated so
 * a change to either side of it fails here instead of in a frame — and asserted
 * across the whole ladder of radiance floors, because level 0 (the 50 mm
 * lattice this scene ships at) satisfies it trivially and proves nothing.
 */
test("every radiance floor leaves the atlas inside the slot index space", () => {
  for (const cellSize_m of [0.05, 0.025, 0.0125, 0.006_25]) {
    const plan = planAt(cellSize_m);
    assert.equal(plan.total, false, `${cellSize_m} m: a total plan cannot exhibit the fault`);
    assert.ok(plan.radiancePageCapacity >= 1, `${cellSize_m} m: the atlas must hold at least one page`);
    assert.ok(plan.radianceSlotOffset + plan.radiancePageCapacity <= plan.pageCapacity,
      `${cellSize_m} m: atlas of ${plan.radiancePageCapacity} at offset ${plan.radianceSlotOffset}`
      + ` overflows the ${plan.pageCapacity}-slot space`);
    // Still large enough to hold every page the plan actually put above the
    // floor — the other half of the contract, which `prepareGpuUpdate` checks.
    const abovePages = plan.plan.pages.filter((page) => page.key.level >= plan.radianceFloorLevel).length;
    assert.ok(plan.radiancePageCapacity >= abovePages,
      `${cellSize_m} m: atlas of ${plan.radiancePageCapacity} cannot hold ${abovePages} above-floor pages`);
    assert.equal(plan.radianceAtlasPages.reduce((product, value) => product * value, 1) >= plan.radiancePageCapacity, true,
      `${cellSize_m} m: the atlas shape does not contain its declared capacity`);
  }
});

/** The floor has to be above level zero for the fault to exist at all. */
test("a 25 mm lattice puts the radiance floor above level zero", () => {
  assert.equal(planAt(0.05).radianceFloorLevel, 0);
  assert.equal(planAt(0.025).radianceFloorLevel, 1);
  assert.ok(planAt(0.025).radianceSlotOffset > 0);
});

/**
 * The same plan, through the object that rejected it. No device is created:
 * the precondition that threw runs before the first `createTexture`, and
 * reaching it is the whole point — a plan that satisfies it fails on the
 * missing device instead, which is a different message.
 */
test("the tetrahedral radiance constructor accepts the plan it is handed", () => {
  const plan = planAt(0.025);
  assert.throws(() => new WebGpuLiveSvoTetrahedralRadiance(undefined as unknown as GPUDevice, {
    pageCapacity: plan.pageCapacity,
    atlasTexels: plan.radianceAtlasTexels as [number, number, number],
    atlasPageCapacity: plan.radiancePageCapacity,
    slotOffset: plan.radianceSlotOffset,
    radianceFloorLevel: plan.radianceFloorLevel,
  }), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.doesNotMatch(error.message, /slot index space/,
      "the address plan is still overflowing the slot space the atlas has to live in");
    return true;
  });
});
