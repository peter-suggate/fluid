import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { planAdaptiveSparseBrickOctree } from "../lib/adaptive-sparse-brick-plan";
import { planOctreeSvoWideFanout } from "../lib/webgpu-octree-sparse-bricks";

test("octree world derives mixed-level wide terminals from canonical adaptive leaves", () => {
  const canonical = planAdaptiveSparseBrickOctree({
    brickSize: 8,
    solverBricks: [{ x: 0, y: 0, z: 0 }],
    proxyBricks: [{ x: 2, y: 0, z: 0 }],
    maximumDepth: 2,
    maximumEnvironmentCoarseningPower: 1,
  });
  assert.deepEqual(canonical.leaves.map((leaf) => canonical.nodes[leaf.nodeIndex].level), [1, 2]);
  const wide = planOctreeSvoWideFanout(canonical);
  assert.equal(wide.sourceGeneration, 1);
  assert.equal(wide.generation, 1);
  assert.equal(wide.maximumDepth, canonical.maximumDepth);
  const terminals = wide.pages.flatMap((page) => page.descriptors)
    .filter((descriptor) => descriptor.kind === "terminal");
  assert.deepEqual([...new Set(terminals.map((descriptor) => descriptor.sourceLevel))].sort((a, b) => a - b), [1, 2]);
  assert.deepEqual([...new Set(terminals.map((descriptor) => descriptor.kind === "terminal" ? descriptor.sourceNodeIndex : -1))].sort((a, b) => a - b),
    canonical.leaves.map((leaf) => leaf.nodeIndex).sort((a, b) => a - b));
});

test("octree world exposes, accounts, and destroys its optional derived capabilities", () => {
  const producer = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  const sourceAbi = readFileSync(new URL("../lib/webgpu-voxel-debug.ts", import.meta.url), "utf8");
  assert.match(sourceAbi, /wideFanout\?: import\("\.\/webgpu-svo-wide-fanout"\)\.WebGPUSvoWideFanoutSource/);
  assert.match(sourceAbi, /derivedRenderAllocationBytes\?: Readonly<\{ wideFanout: number; nodeMipPyramid\?: number \}>/);
  assert.match(producer, /wideFanout: this\.wideFanout\?\.capability\(\)/);
  assert.match(producer, /derivedRenderAllocationBytes: \{[^]*wideFanout: this\.wideFanout\?\.allocatedBytes \?\? 0,[^]*nodeMipPyramid: this\.nodeMipPyramid\?\.telemetry\(\)\.allocatedBytes \?\? 0/);
  assert.match(producer, /this\.proxyVoxelizer\.allocatedBytes \+ \(this\.wideFanout\?\.allocatedBytes \?\? 0\)[^]*\+ \(this\.nodeMipPyramid\?\.telemetry\(\)\.allocatedBytes \?\? 0\)/);
  assert.match(producer, /this\.wideFanout\?\.destroy\(\)/);
  assert.match(producer, /this\.nodeMipPyramid\?\.destroy\(\)/);
});
