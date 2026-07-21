import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { octreeProjectionShader, planOctreePhiSnapshotAllocation, planOctreeSolidCellAllocation } from "../lib/webgpu-octree";

const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const sparseTreeSource = readFileSync(new URL("../lib/sparse-brick-octree.ts", import.meta.url), "utf8");
const sparseSurfaceSource = readFileSync(new URL("../lib/webgpu-sparse-surface-band.ts", import.meta.url), "utf8");

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
  assert.match(octreeProjectionShader, /word \+ 1u >= arrayLength\(&solidOrSurface\)[\s\S]*SolidCell\(0\.0, -1\)/);
  assert.match(octreeProjectionShader, /let solid = solidAt\(vec3i\(q\)\)\.fraction/);
  assert.match(sparseTreeSource, /if \(dense < arrayLength\(&solidCells\)\) \{ solid = solidCells\[dense\]; \}/);
  assert.match(sparseSurfaceSource, /if\(i>=arrayLength\(&solidCells\)\)\{return 0\.5\*h;\}/,
    "the legacy sparse-surface A/B must preserve its no-solid value outside the fallback record");
  assert.match(octreeSource, /if \(this\.hasDenseSolidCells\) \{[\s\S]*dispatch\(this\.rasterizeSolidsPipeline/,
    "the tiny fallback must never receive box-sized rasterization writes");
  assert.match(octreeSource, /writeBuffer\(this\.solidCells, 0, new Int32Array\(\[0, -1\]\)\)/,
    "the fallback must encode zero fraction and no owner for every guarded consumer");
  assert.match(octreeSource, /this\.hasDenseSolidCells \? this\.solidCells : undefined/,
    "surface volume control must disable solid reads in a solid-free scene");
  assert.match(octreeSource, /if \(this\.couplingBodyCount > 0 && !this\.solidFaces\)/,
    "the unguarded coupling shader must never run for a solid-free fallback");
});

test("compact surface authority removes the dense topology phi snapshot", () => {
  const dims = { nx: 320, ny: 96, nz: 80 };
  assert.deepEqual(planOctreePhiSnapshotAllocation(dims, true), {
    allocatedBytes: 4,
    denseBytes: 9_830_400,
    savedBytes: 9_830_396,
    hasDenseField: false,
  });
  assert.deepEqual(planOctreePhiSnapshotAllocation(dims, false), {
    allocatedBytes: 9_830_400,
    denseBytes: 9_830_400,
    savedBytes: 0,
    hasDenseField: true,
  });
  assert.match(octreeSource,
    /allocatedBytes: this\.topology\.size \+ \(this\.ownerPages \? 32 : 0\) \+ this\.solidCells\.size \+ phiSnapshotAllocation\.allocatedBytes \+ surfaceStateAllocation\.allocatedBytes/);
  assert.match(octreeSource, /if \(!active && this\.hasDensePhiSnapshot\)/,
    "the compact fallback texel must never receive a box-sized snapshot seed");
  assert.match(octreeSource, /if \(!this\.hasDensePhiSnapshot\) return false;/,
    "change detection must not sample the format-only fallback");
});
