import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_NODE_MIP_LAYOUT,
  SVO_OPACITY_LEVEL_FLOOR,
  createSvoNodeMipPage,
  createSvoNodeMipPageWithApron,
  decodeSvoNodeMipMorton,
  encodeSvoNodeMipMorton,
  packSvoNodeMipPageKey,
  planSvoNodeMipPyramid,
  publishSvoNodeMipGeneration,
  raiseSvoNodeMipSeedToFloor,
  reduceSvoNodeMipChildren,
  resolveSvoNodeMipVirtualTexel,
  svoNodeMipTexelOffset,
  svoOpacityLevelFloor,
  unpackSvoNodeMipPageKey,
} from "../lib/svo-node-mip-pyramid";
import {
  growSvoNodeMipAddressPlan,
  pagesOutsideSvoNodeMipAddressPlan,
  planSvoNodeMipAddresses,
  svoNodeMipDomainDirectPageTableDimensions,
} from "../lib/svo-node-mip-address-plan";
import {
  integrateSvoNodeMipCone,
  svoNodeMipCoverageOpacity,
  svoNodeMipSamplingWGSL,
} from "../lib/svo-node-mip-sampling";

test("node-mip key ABI round trips 63-bit Morton coordinates", () => {
  const coordinate = [2_097_151, 1_048_579, 77] as const;
  const morton = encodeSvoNodeMipMorton(coordinate);
  assert.deepEqual(decodeSvoNodeMipMorton(morton), coordinate);
  const packed = packSvoNodeMipPageKey({ generation: 17, level: 6, coordinate });
  assert.equal(packed.byteLength, SVO_NODE_MIP_LAYOUT.keyBytes);
  assert.deepEqual(unpackSvoNodeMipPageKey(packed), { generation: 17, level: 6, coordinate });
});

test("pyramid planning deduplicates leaves, inserts ancestors, and reports bounded memory", () => {
  const plan = planSvoNodeMipPyramid({
    generation: 9,
    occupiedPages: [[0, 0, 0], [1, 0, 0], [1, 0, 0], [8, 4, 2]],
    levelCount: 4,
  });
  assert.equal(plan.complete, true);
  assert.equal(plan.requestedPageCount, 9);
  assert.equal(new Set(plan.pages.map((page) => page.keyString)).size, plan.pages.length);
  for (let index = 1; index < plan.pages.length; index += 1) {
    const previous = plan.pages[index - 1].key, current = plan.pages[index].key;
    assert.ok(previous.level < current.level || (previous.level === current.level
      && encodeSvoNodeMipMorton(previous.coordinate) < encodeSvoNodeMipMorton(current.coordinate)), "directory is level/Morton sorted");
  }
  assert.ok(plan.pages.some((page) => page.key.level === 3 && page.key.coordinate.join() === "1,0,0"));
  assert.equal(plan.pagePayloadBytes, plan.residentPageCount * SVO_NODE_MIP_LAYOUT.bytesPerPage);
  assert.equal(plan.directoryBytes, plan.residentPageCount * 32);
  assert.equal(plan.atlasBytes, plan.atlas.capacity * SVO_NODE_MIP_LAYOUT.bytesPerPage);
  assert.equal(plan.allocatedBytes, plan.atlasBytes + plan.directoryBytes);

  const overflow = planSvoNodeMipPyramid({ generation: 9, occupiedPages: [[0, 0, 0], [8, 4, 2]], levelCount: 4, capacity: 3 });
  assert.equal(overflow.complete, false);
  assert.equal(overflow.residentPageCount, 3);
  assert.equal(overflow.overflowPageCount, overflow.requestedPageCount - 3);
});

test("RGBA8 reduction averages mean lanes and conservatively maximizes coverage lanes", () => {
  const children = Array.from({ length: 8 }, (_, i) => [i * 10, i === 3 ? 240 : i, 80 + i, i === 6 ? 251 : 2] as const);
  assert.deepEqual(reduceSvoNodeMipChildren(children), [35, 240, 84, 251]);
});

test("physical page clamps any declared apron to the nearest interior texel", () => {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const apron = SVO_NODE_MIP_LAYOUT.apron;
  const last = SVO_NODE_MIP_LAYOUT.physicalSize - 1;
  const interior = new Uint8Array(n ** 3 * 4);
  for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
    const offset = ((z * n + y) * n + x) * 4;
    interior.set([x, y, z, 255], offset);
  }
  const page = createSvoNodeMipPage(interior);
  assert.equal(page.byteLength, SVO_NODE_MIP_LAYOUT.bytesPerPage);
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(0, 0, 0), svoNodeMipTexelOffset(0, 0, 0) + 4)], [0, 0, 0, 255]);
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(last, last, last), svoNodeMipTexelOffset(last, last, last) + 4)], [7, 7, 7, 255]);
  const interiorTexel = [4, 5, 6].map((value) => value + apron) as [number, number, number];
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(...interiorTexel), svoNodeMipTexelOffset(...interiorTexel) + 4)], [4, 5, 6, 255]);
});

test("apron addressing crosses same-level pages and falls back at the virtual-domain edge", () => {
  assert.deepEqual(resolveSvoNodeMipVirtualTexel([3, 2, 1], [-1, 8, 3]), { page: [2, 3, 1], texel: [7, 0, 3] });
  assert.equal(resolveSvoNodeMipVirtualTexel([0, 0, 0], [-1, 0, 0]), undefined);
  const apron = SVO_NODE_MIP_LAYOUT.apron;
  const last = SVO_NODE_MIP_LAYOUT.physicalSize - 1;
  const interior = new Uint8Array(8 ** 3 * 4).fill(1);
  const page = createSvoNodeMipPageWithApron([3, 2, 1], interior, ({ page: neighbour, texel }) =>
    neighbour.join() === "2,2,1" && texel.join() === "7,0,0" ? [11, 22, 33, 44] : undefined);
  // With the apron at zero the page is all interior: no texel of it belongs to a
  // neighbour, so the sampler is never consulted and the result is the interior
  // copy. That is exactly what the live GPU builder always produced, because its
  // own apron was a clamped replica rather than neighbour data.
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(0, 1, 1), svoNodeMipTexelOffset(0, 1, 1) + 4)],
    apron > 0 ? [11, 22, 33, 44] : [1, 1, 1, 1]);
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(last, 1, 1), svoNodeMipTexelOffset(last, 1, 1) + 4)], [1, 1, 1, 1]);
});

test("publication retains the previous complete generation until every candidate stage completes", () => {
  const oldPlan = planSvoNodeMipPyramid({ generation: 4, occupiedPages: [[0, 0, 0]], levelCount: 1 });
  const nextPlan = planSvoNodeMipPyramid({ generation: 5, occupiedPages: [[0, 0, 0]], levelCount: 1 });
  const visible = { completeGeneration: 4, plan: oldPlan };
  const rejected = publishSvoNodeMipGeneration(visible, {
    generation: 5, plan: nextPlan, directoryComplete: true, payloadComplete: true, apronsComplete: false,
  });
  assert.equal(rejected.published, false);
  assert.equal(rejected.reason, "incomplete-aprons");
  assert.equal(rejected.visible?.completeGeneration, 4);
  const published = publishSvoNodeMipGeneration(visible, {
    generation: 5, plan: nextPlan, directoryComplete: true, payloadComplete: true, apronsComplete: true,
  });
  assert.equal(published.published, true);
  assert.equal(published.visible.completeGeneration, 5);
});

test("CPU cone oracle uses front-to-back opacity, LOD growth, and bounded termination", () => {
  assert.equal(svoNodeMipCoverageOpacity(128, 0), 0);
  assert.ok(Math.abs(svoNodeMipCoverageOpacity(128, 2) - (1 - (1 - 128 / 255) ** 2)) < 1e-12);
  const lods: number[] = [];
  const result = integrateSvoNodeMipCone({
    origin_m: [0, 0, 0], direction: [1, 0, 0], aperture_radians: 0.5,
    minimumVoxelWidth_m: 0.1, maximumDistance_m: 12, maximumSteps: 64,
  }, ({ lod }) => { lods.push(lod); return [96, 255, 0, 0]; });
  assert.equal(result.terminated, "opacity");
  assert.ok(result.opacity >= 0.995);
  assert.ok(result.steps < 64);
  assert.ok(lods.at(-1)! >= lods[0]);

  const clear = integrateSvoNodeMipCone({
    origin_m: [0, 0, 0], direction: [0, 1, 0], aperture_radians: 0,
    minimumVoxelWidth_m: 1, maximumDistance_m: 4,
  }, () => undefined);
  assert.equal(clear.opacity, 0);
  assert.equal(clear.missingSamples, clear.steps);
  assert.equal(clear.terminated, "distance");
});

test("sampling WGSL is binding-free and exposes page, opacity, and LOD helpers", () => {
  assert.doesNotMatch(svoNodeMipSamplingWGSL, /@group|@binding/);
  assert.match(svoNodeMipSamplingWGSL, /fn svoNodeMipSamplePage/);
  assert.match(svoNodeMipSamplingWGSL, /fn svoNodeMipDirectoryEntry/);
  assert.match(svoNodeMipSamplingWGSL, /fn svoNodeMipCompositeOpacity/);
  assert.match(svoNodeMipSamplingWGSL, /fn svoNodeMipLod/);
});

test("opacity level floor is anchored to a world size and bounded by what the build can reduce", () => {
  const levelCount = 10;
  // The reference leaf keeps the base level: every byte and every sample is
  // what shipped, which is why a reference-leaf frame is bit-exact across this.
  assert.equal(svoOpacityLevelFloor({ levelCount, cellSize_m: SVO_OPACITY_LEVEL_FLOOR.targetCellSize_m }), 0);
  // Refining the leaf raises the floor with it, so the finest opacity texel
  // stays at the reference size instead of shrinking with the tree.
  assert.equal(svoOpacityLevelFloor({ levelCount, cellSize_m: SVO_OPACITY_LEVEL_FLOOR.targetCellSize_m / 2 }), 1);
  // Past one level the derived build's absent-child fallback resolves a single
  // leaf for a region spanning many, so a deeper floor is clamped rather than
  // silently wrong. See SVO_OPACITY_LEVEL_FLOOR.
  assert.equal(svoOpacityLevelFloor({ levelCount, cellSize_m: SVO_OPACITY_LEVEL_FLOOR.targetCellSize_m / 8 }), 1);
  assert.equal(svoOpacityLevelFloor({ levelCount, cellSize_m: SVO_OPACITY_LEVEL_FLOOR.targetCellSize_m / 8, override: 4 }), 1);
  assert.equal(svoOpacityLevelFloor({ levelCount: 1, cellSize_m: SVO_OPACITY_LEVEL_FLOOR.targetCellSize_m / 4 }), 0);
  // A coarser leaf than the reference never pushes the base below level zero.
  assert.equal(svoOpacityLevelFloor({ levelCount, cellSize_m: SVO_OPACITY_LEVEL_FLOOR.targetCellSize_m * 4 }), 0);
  assert.equal(svoOpacityLevelFloor({ levelCount }), 0);
  assert.throws(() => svoOpacityLevelFloor({ levelCount: 0 }), RangeError);
});

test("flooring a pyramid removes the base level and leaves every level above it identical", () => {
  // Two neighbouring finest pages plus a coarse leaf, so the seeds arrive at
  // more than one level and the floor has both cases to handle.
  const seeds = [[0, 0, 0], [1, 0, 0], [9, 4, 2], [8, 5, 3], { level: 2, coordinate: [3, 1, 1] as const }] as const;
  const levelCount = 6;
  const base = planSvoNodeMipPyramid({ generation: 3, occupiedPages: seeds as never, levelCount });
  const floored = planSvoNodeMipPyramid({
    generation: 3,
    occupiedPages: seeds.map((seed) => raiseSvoNodeMipSeedToFloor(seed as never, 1)),
    levelCount,
  });
  const keysAtOrAbove = (plan: typeof base, floor: number) => new Set(plan.pages
    .filter((page) => page.key.level >= floor)
    .map((page) => `${page.key.level}:${page.key.coordinate.join()}`));
  // The load-bearing invariant: a raised seed lands on a page the ancestor walk
  // would have inserted anyway, so nothing above the floor moves, appears or
  // disappears — the pyramid simply loses its bottom.
  assert.deepEqual(keysAtOrAbove(floored, 1), keysAtOrAbove(base, 1));
  assert.equal(floored.pages.some((page) => page.key.level === 0), false);
  assert.ok(base.pages.some((page) => page.key.level === 0));
  assert.equal(floored.requestedPageCount, base.requestedPageCount - base.pages.filter((p) => p.key.level === 0).length);

  // Raising is idempotent and never moves a seed already at or above the floor.
  const coarse = { level: 2, coordinate: [3, 1, 1] as const };
  assert.deepEqual(raiseSvoNodeMipSeedToFloor(coarse, 1), coarse);
  assert.deepEqual(raiseSvoNodeMipSeedToFloor(raiseSvoNodeMipSeedToFloor([9, 4, 2], 1), 1),
    raiseSvoNodeMipSeedToFloor([9, 4, 2], 1));
  assert.deepEqual(raiseSvoNodeMipSeedToFloor([9, 4, 2], 1), { level: 1, coordinate: [4, 2, 1] });
  assert.deepEqual(raiseSvoNodeMipSeedToFloor([9, 4, 2], 0), { level: 0, coordinate: [9, 4, 2] });
  assert.throws(() => raiseSvoNodeMipSeedToFloor([1, 1, 1], -1), RangeError);
});

test("floored address plans drop the base level's direct-table slab and floor what an edit activates", () => {
  const basePageDimensions = [40, 12, 24] as const;
  const levelCount = 6;
  const occupied = [[0, 0, 0], [1, 0, 0], [17, 5, 9]] as const;
  const shared = { basePageDimensions, levelCount, addressCapacity: 1_000_000, generation: 1 } as const;
  const unfloored = planSvoNodeMipAddresses({ ...shared, occupiedBasePages: occupied as never });
  const floored = planSvoNodeMipAddresses({ ...shared, occupiedBasePages: occupied as never, opacityFloorLevel: 1 });
  assert.equal(unfloored.opacityFloorLevel, 0);
  assert.equal(floored.opacityFloorLevel, 1);
  assert.equal(floored.plan.pages.some((page) => page.key.level === 0), false);

  // The direct page table is dense over the page grid while the pages in it are
  // sparse, so the base level's slab is the bulk of it.
  const wide = svoNodeMipDomainDirectPageTableDimensions(basePageDimensions, levelCount);
  const narrow = svoNodeMipDomainDirectPageTableDimensions(basePageDimensions, levelCount, 1);
  assert.deepEqual(wide, [40, 12, 24 + 12 + 6 + 3 + 2 + 1]);
  assert.deepEqual(narrow, [20, 6, 12 + 6 + 3 + 2 + 1]);

  // An edit hands the plan a bare finest coordinate. Under a floor it must be
  // raised, or the growth puts a base level back under part of the scene only.
  const activated = [[2, 0, 0], [3, 0, 0]] as const;
  // Both base coordinates name the same level-1 page, so the request collapses
  // to one — the eight-to-one dedup is the whole point of asking at the floor.
  assert.deepEqual(pagesOutsideSvoNodeMipAddressPlan(floored, activated as never), [[2, 0, 0]]);
  assert.deepEqual(pagesOutsideSvoNodeMipAddressPlan(unfloored, activated as never), [[2, 0, 0], [3, 0, 0]]);
  const grown = growSvoNodeMipAddressPlan(floored, activated as never);
  assert.ok(grown);
  assert.equal(grown.plan.plan.pages.some((page) => page.key.level === 0), false);
  // Both name the same level-1 page, so the second is already addressed.
  assert.equal(pagesOutsideSvoNodeMipAddressPlan(grown.plan, [[3, 0, 0]] as never).length, 0);
  // Domain membership is still judged at the finest coordinate the caller named.
  assert.equal(growSvoNodeMipAddressPlan(floored, [[basePageDimensions[0], 0, 0]] as never), undefined);
});
