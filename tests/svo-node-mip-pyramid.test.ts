import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_NODE_MIP_LAYOUT,
  createSvoNodeMipPage,
  createSvoNodeMipPageWithApron,
  decodeSvoNodeMipMorton,
  encodeSvoNodeMipMorton,
  packSvoNodeMipPageKey,
  planSvoNodeMipPyramid,
  publishSvoNodeMipGeneration,
  reduceSvoNodeMipChildren,
  resolveSvoNodeMipVirtualTexel,
  svoNodeMipTexelOffset,
  unpackSvoNodeMipPageKey,
} from "../lib/svo-node-mip-pyramid";
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
  assert.equal(plan.pagePayloadBytes, plan.residentPageCount * 4_000);
  assert.equal(plan.directoryBytes, plan.residentPageCount * 32);
  assert.equal(plan.atlasBytes, plan.atlas.capacity * 4_000);
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

test("physical page clamps its apron to the nearest interior texel", () => {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const interior = new Uint8Array(n ** 3 * 4);
  for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
    const offset = ((z * n + y) * n + x) * 4;
    interior.set([x, y, z, 255], offset);
  }
  const page = createSvoNodeMipPage(interior);
  assert.equal(page.byteLength, 4_000);
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(0, 0, 0), svoNodeMipTexelOffset(0, 0, 0) + 4)], [0, 0, 0, 255]);
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(9, 9, 9), svoNodeMipTexelOffset(9, 9, 9) + 4)], [7, 7, 7, 255]);
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(4, 5, 6), svoNodeMipTexelOffset(4, 5, 6) + 4)], [3, 4, 5, 255]);
});

test("apron addressing crosses same-level pages and falls back at the virtual-domain edge", () => {
  assert.deepEqual(resolveSvoNodeMipVirtualTexel([3, 2, 1], [-1, 8, 3]), { page: [2, 3, 1], texel: [7, 0, 3] });
  assert.equal(resolveSvoNodeMipVirtualTexel([0, 0, 0], [-1, 0, 0]), undefined);
  const interior = new Uint8Array(8 ** 3 * 4).fill(1);
  const page = createSvoNodeMipPageWithApron([3, 2, 1], interior, ({ page: neighbour, texel }) =>
    neighbour.join() === "2,2,1" && texel.join() === "7,0,0" ? [11, 22, 33, 44] : undefined);
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(0, 1, 1), svoNodeMipTexelOffset(0, 1, 1) + 4)], [11, 22, 33, 44]);
  assert.deepEqual([...page.slice(svoNodeMipTexelOffset(9, 1, 1), svoNodeMipTexelOffset(9, 1, 1) + 4)], [1, 1, 1, 1]);
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
