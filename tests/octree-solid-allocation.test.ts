import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { octreeProjectionShader, planOctreeSolidCellAllocation } from "../lib/webgpu-octree";

const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const sparseTreeSource = readFileSync(new URL("../lib/sparse-brick-octree.ts", import.meta.url), "utf8");

test("solid-free ocean keeps only one valid storage record", () => {
  const dims = { nx: 320, ny: 96, nz: 80 };
  const plan = planOctreeSolidCellAllocation(dims, false, 0);
  assert.deepEqual(plan, {
    allocatedBytes: 8,
    denseBytes: 19_660_800,
    savedBytes: 19_660_792,
    hasDenseField: false,
  });
  assert.equal(planOctreeSolidCellAllocation(dims, true, 0).allocatedBytes, plan.denseBytes);
  assert.equal(planOctreeSolidCellAllocation(dims, false, 1).allocatedBytes, plan.denseBytes);
});

test("solid-free topology and publication never index beyond the fallback", () => {
  assert.match(octreeProjectionShader, /word \+ 1u >= arrayLength\(&solidCells\)[\s\S]*SolidCell\(0\.0, -1\)/);
  assert.match(octreeProjectionShader, /let solid = solidAt\(vec3i\(q\)\)\.fraction/);
  assert.match(sparseTreeSource, /if \(dense < arrayLength\(&solidCells\)\) \{ solid = solidCells\[dense\]; \}/);
  assert.match(octreeSource, /if \(this\.hasDenseSolidCells\) \{[\s\S]*dispatch\(this\.rasterizeSolidsPipeline/,
    "the tiny fallback must never receive box-sized rasterization writes");
  assert.match(octreeSource, /writeBuffer\(this\.solidCells, 0, new Int32Array\(\[0, -1\]\)\)/,
    "the fallback must encode zero fraction and no owner for every guarded consumer");
  assert.match(octreeSource, /this\.hasDenseSolidCells \? this\.solidCells : undefined/,
    "surface volume control must disable solid reads in a solid-free scene");
  assert.doesNotMatch(octreeSource, /createCouplingGroup|couplingGroups/,
    "the retired dense coupling shader and its fallback texture views must stay deleted");
});

test("compact surface authority deletes the topology phi snapshot binding and backing code", () => {
  assert.match(octreeSource,
    /allocatedBytes: this\.ownerPages\.allocatedBytes \+ this\.solidCells\.size\s*\+ surfaceStateAllocation\.allocatedBytes/);
  assert.match(octreeSource, /if \(active && !analyticColdBootstrap\) \{[\s\S]*Build exact structural topology-tile delta/,
    "paged power authority must always consume its compact generation-tagged delta");
  assert.doesNotMatch(octreeSource, /changeDrivenEligible/,
    "compact authority must not retain a recurring full-list selector");
  assert.doesNotMatch(octreeSource,
    /phiSnapshot|hasDensePhiSnapshot|planOctreePhiSnapshotAllocation|refreshSnapshot/,
    "the snapshot allocation, binding, pipelines, and switches must stay deleted");
  assert.doesNotMatch(octreeProjectionShader, /@binding\(14\)/,
    "the deleted snapshot must not leave an inert legacy binding");
});
