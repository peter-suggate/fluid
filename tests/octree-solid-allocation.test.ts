import assert from "node:assert/strict";
import { auditWGSLComputeBindingReachability } from "../lib/wgsl-binding-reachability";
import { readFileSync } from "node:fs";
import test from "node:test";
import { OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS, octreeProjectionShader,
  planOctreeSolidCellAllocation } from "../lib/webgpu-octree";

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
  // Binding 14 was the deleted phi snapshot; it now carries the bootstrap
  // level set, which the topology kernels genuinely read. Banning the slot
  // number banned its reuse rather than the snapshot, so the check is now that
  // whatever occupies it is live: declared, named for its current purpose, and
  // reachable from an entry point. An inert binding is the actual defect.
  const binding14 = /@group\(0\)\s*@binding\(14\)\s*var(?:<[^>]*>)?\s*(\w+)/
    .exec(octreeProjectionShader);
  if (binding14) {
    assert.equal(binding14[1], "bootstrapLevelSetIn",
      "binding 14 must be the live bootstrap level set, not a revived snapshot");
    assert.ok(OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS.some((entryPoint) =>
      auditWGSLComputeBindingReachability(octreeProjectionShader, entryPoint)
        .bindings.some(({ binding }) => binding === 14)),
    "a binding no entry point reaches is the inert leftover this guards against");
  }
});
