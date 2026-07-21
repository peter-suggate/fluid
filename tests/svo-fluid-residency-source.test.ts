import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  FLUID_BRICK_WORKLIST_HEADER_WORDS,
  FLUID_BRICK_WORKLIST_WORDS,
} from "../lib/webgpu-fluid-brick-residency";
import {
  decodeSparseVoxelFluidResidencyState,
  sparseVoxelFluidResidencyLayout,
  sparseVoxelFluidResidencyWGSL,
  SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
  SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS,
} from "../lib/webgpu-voxel-debug";

test("renderer residency layout exactly matches the producer header and paired lists", () => {
  const capacity = 37;
  const layout = sparseVoxelFluidResidencyLayout(capacity);
  assert.equal(layout.headerBytes, FLUID_BRICK_WORKLIST_HEADER_WORDS * 4);
  assert.equal(layout.activeEntryOffsetBytes, 64);
  assert.equal(layout.entryStrideBytes, 8, "each entry is `(solver brick index, sparse leaf index)`");
  assert.equal(layout.retiredEntryOffsetBytes, 64 + capacity * 8);
  assert.equal(layout.stateStrideBytes, 4);
  assert.equal(layout.worklistByteLength, (16 + capacity * 4) * 4);
  assert.throws(() => sparseVoxelFluidResidencyLayout(0), /positive integer/);
});

test("counter and state-bit semantics remain sourced from GPUFluidBrickResidency", () => {
  assert.equal(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS, FLUID_BRICK_WORKLIST_WORDS);
  assert.deepEqual(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS, {
    activeCount: 0,
    retiredCount: 4,
    coreCount: 8,
    haloCount: 9,
    activatedCount: 10,
    retiredStatsCount: 11,
    generation: 15,
  });
  assert.deepEqual(SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS, {
    resident: 1, core: 2, halo: 4, activated: 8, wasResident: 32,
  });
  const decoded = decodeSparseVoxelFluidResidencyState((9 << 16) | 1 | 2 | 8 | 32);
  assert.deepEqual(decoded, {
    flags: 43,
    dryFrames: 9,
    resident: true,
    core: true,
    halo: false,
    activated: true,
    wasResident: true,
  });
  assert.throws(() => decodeSparseVoxelFluidResidencyState(-1), /uint32/);
});

test("structural source exposes actual producer buffers, filtered views, and ownership fences", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  assert.match(source, /states: \{ buffer: this\.residency\.stateBuffer/);
  assert.match(source, /const residencyWorklistBinding = \{ buffer: this\.residency\.worklist/);
  assert.match(source, /worklist: residencyWorklistBinding/);
  assert.match(source, /active: activeResidencyList/);
  assert.match(source, /requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS\.core/);
  assert.match(source, /requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS\.halo/);
  assert.match(source, /entryOffsetBytes: residencyLayout\.retiredEntryOffsetBytes/);
  assert.match(source, /generation: residencyWord\(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS\.generation\)/);
  assert.match(source, /revision: publicationWord\(SPARSE_VOXEL_PUBLICATION_STATE\.coarseFluidRevision\)/);
  assert.match(source, /owner: "GPUFluidBrickResidency"/);
  assert.doesNotMatch(source, /fluidResidency:[\s\S]{0,500}(?:geometry|materialOwners)/,
    "residency must come from its authoritative state/worklist, not payload inference");
});

test("legacy stats remain published while GPUFluidBrickResidency retains lifecycle ownership", () => {
  const world = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  assert.match(world, /fluidBrickStats: \{ buffer: this\.residency\.worklist \}/);
  const destroy = world.slice(world.indexOf("  destroy(): void {"));
  assert.match(destroy, /this\.residency\.destroy\(\)/);
  assert.doesNotMatch(destroy, /this\.residency\.(?:stateBuffer|worklist|leafStates)\.destroy/);
  const residency = readFileSync(new URL("../lib/webgpu-fluid-brick-residency.ts", import.meta.url), "utf8");
  const residencyDestroy = residency.slice(residency.indexOf("  destroy(): void {"));
  assert.match(residencyDestroy, /this\.states\.destroy\(\)/);
  assert.match(residencyDestroy, /this\.worklist\.destroy\(\)/);
  assert.match(residencyDestroy, /this\.leafStatesBuffer\.destroy\(\)/);
});

test("WGSL decode is binding-free and requires explicit state/list inputs", () => {
  assert.match(sparseVoxelFluidResidencyWGSL, /fn svoResidencyFlags/);
  assert.match(sparseVoxelFluidResidencyWGSL, /fn svoResidencyDryFrames/);
  assert.match(sparseVoxelFluidResidencyWGSL, /fn svoResidencyHas/);
  assert.match(sparseVoxelFluidResidencyWGSL, /fn svoResidencyEntryWord/);
  assert.doesNotMatch(sparseVoxelFluidResidencyWGSL, /@group|@binding|material|geometry|payload/);
});
