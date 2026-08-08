import assert from "node:assert/strict";
import test from "node:test";
import { scenePresets } from "../lib/scenes";
import {
  coarseSummaryWGSL,
  planOctreeCoarseSummary,
  planOctreeRedistanceSweeps,
} from "../lib/webgpu-octree-coarse-summary";

test("the redistance reach covers the widest distance the ladder asks about", () => {
  // The sweep is min-only fast marching from a one-domain-extent seed, so its
  // reach IS the sweep count: a cell farther than that still reports the seed.
  // `pressureRefinementEvidence` reads phi as a distance out to
  // bandCells + max(0, gradingLayers - 1) * maximumLeafSize, so anything less
  // means the ladder decides how coarse the deep interior may get from a
  // constant. Five sweeps against a width of twenty is what coarsened the
  // dam-break front to the ceiling three cells from the free surface.
  const dam = [24, 18, 16] as const;
  assert.ok(planOctreeRedistanceSweeps(4 + 1 * 8, dam) >= 12,
    "band 4 and grading 1 at an 8-cell ceiling need a 12-cell reach");
  assert.equal(planOctreeRedistanceSweeps(4 + 1 * 8, dam) % 2, 1,
    "the seed lands in the scratch bank, so an odd count finishes in the output");

  // Every count is odd, at least the historical five, and never longer than the
  // lattice -- no sweep can carry information past the domain.
  for (const reach of [0, 1, 5, 6, 12, 20, 36, 4_096, Number.NaN]) {
    for (const dims of [dam, [32, 32, 32] as const, [320, 96, 80] as const]) {
      const sweeps = planOctreeRedistanceSweeps(reach, dims);
      assert.equal(sweeps % 2, 1, `${reach} on ${dims.join("x")} must stay odd`);
      assert.ok(sweeps >= 5, "the historical reach is the floor");
      assert.ok(sweeps <= Math.max(...dims) + 1);
    }
  }
  assert.equal(planOctreeRedistanceSweeps(undefined, dam), 5,
    "a caller that declares no requirement keeps the historical reach");
});

test("factor-one coarse summary uses B4 nodes and bounded sparse entries", () => {
  const plan = planOctreeCoarseSummary([16, 16, 16], 4_096);
  assert.deepEqual(plan.baseDimensions, [4, 4, 4]);
  assert.deepEqual(plan.levelDimensions, [[4, 4, 4], [2, 2, 2], [1, 1, 1]]);
  assert.deepEqual(plan.levelOffsets, [0, 64, 72]);
  assert.equal(plan.hierarchyKeyCapacity, 73);
  assert.equal(plan.entryCapacity, 73);
  assert.ok(plan.directoryWords * 4 < 64 * 1_024,
    "the complete coarse lattice remains far smaller than the former 641 KiB fine band");
});

test("factor-one hierarchy publishes exact owner phase and centre evidence", () => {
  assert.match(coarseSummaryWGSL,
    /let nodeSize=4u<<level;let local=q%vec3u\(nodeSize\);let half=nodeSize\/2u;/,
    "centre coordinates must be relative to the hierarchy node, not the B4 child");
  assert.match(coarseSummaryWGSL,
    /let centreSample=all\(local>=vec3u\(half-1u\)\)&&all\(local<=vec3u\(half\)\)/,
    "the eight dense samples straddling the pressure-cell centre are authoritative");
  assert.match(coarseSummaryWGSL,
    /if\(centreSample\)\{atomicAdd\(&directory\[base\+7u\],bitcast<u32>\(i32\(round\(65536\.\*value\)\)\)\);\}/,
    "B4 entries must retain a signed fixed-point centre sum alongside their masks");
  assert.match(coarseSummaryWGSL,
    /else\{atomicAdd\(&directory\[base\+8u\],1u\);if\(value<0\.0\)\{atomicAdd\(&directory\[base\+9u\],1u\);\}/,
    "parent entries must retain exact valid and negative sample populations");
  assert.match(coarseSummaryWGSL,
    /atomicAdd\(&directory\[base\+10u\],1u\)/,
    "a parent centre is valid only after all eight central samples publish");
});

test("coarse summary entry capacity is bounded by live-row ancestry", () => {
  const plan = planOctreeCoarseSummary([256, 128, 64], 128);
  assert.ok(plan.entryCapacity <= 128 * plan.levelDimensions.length);
  assert.ok(plan.entryCapacity < plan.hierarchyKeyCapacity);
  assert.ok(plan.directoryPageCapacity <= plan.entryCapacity);
});

test("every authored scene fits the coarse-only surface tracker catalog budget", () => {
  const maximumBytes = 256 * 1_024 * 1_024;
  for (const preset of scenePresets) {
    const scene = preset.create();
    const h = scene.voxelDomain.finestCellSize_m;
    const dimensions = [
      Math.max(8, Math.round(scene.container.width_m / h)),
      Math.max(8, Math.round(scene.container.height_m / h)),
      Math.max(8, Math.round(scene.container.depth_m / h)),
    ] as const;
    const domainVolume = dimensions[0] * dimensions[1] * dimensions[2];
    const plan = planOctreeCoarseSummary(dimensions, domainVolume);
    assert.ok(plan.allocatedBytes <= maximumBytes,
      `${preset.id} needs ${(plan.allocatedBytes / 1_048_576).toFixed(1)} MiB for factor-one tracking`);
  }
});
