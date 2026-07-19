import assert from "node:assert/strict";
import test from "node:test";
import {
  adaptSparseVoxelRenderResidencySource,
  buildSvoRenderResidencyGpuInputs,
  type SvoRenderResidencySourceSnapshot,
} from "../lib/svo-render-residency-source-adapter";
import {
  SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
  SPARSE_VOXEL_VALID_FIELDS,
  sparseVoxelFluidResidencyLayout,
  type SparseVoxelStructuralRenderSource,
} from "../lib/webgpu-voxel-debug";

const fakeBuffer = (size: number) => ({ size } as GPUBuffer);

function fixture(options: {
  generation?: number;
  activeCount?: number;
  coreCount?: number;
  haloCount?: number;
  retiredCount?: number;
} = {}) {
  const capacity = 2;
  const publicationBuffer = fakeBuffer(32);
  const statesBuffer = fakeBuffer(capacity * 4);
  const layout = sparseVoxelFluidResidencyLayout(capacity);
  const worklistBuffer = fakeBuffer(layout.worklistByteLength);
  const publication = { buffer: publicationBuffer, size: 32 };
  const states = { buffer: statesBuffer, size: capacity * 4 };
  const worklist = { buffer: worklistBuffer, size: layout.worklistByteLength };
  const publicationWord = (word: number) => ({ binding: publication, word });
  const worklistWord = (word: number) => ({ binding: worklist, word });
  const activeCount = options.activeCount ?? 1;
  const coreCount = options.coreCount ?? 1;
  const haloCount = options.haloCount ?? activeCount - coreCount;
  const retiredCount = options.retiredCount ?? 1;
  const generation = options.generation ?? 7;
  const structural = {
    capacities: { nodes: 4, leaves: 8, voxels: 1024 },
    domain: {
      worldOrigin_m: [0, 0, 0], cellSize_m: [0.5, 0.5, 0.5],
      dimensionsCells: [32, 8, 8], brickSize: 8, maximumDepth: 2,
    },
    publication: {
      state: publication,
      completeGeneration: publicationWord(0),
      validFields: publicationWord(1),
      revisions: {
        topology: publicationWord(2), staticGeometry: publicationWord(3),
        dynamicSolid: publicationWord(4), coarseFluid: publicationWord(5), fineFluid: publicationWord(6),
      },
    },
    fluidResidency: {
      states,
      worklist,
      domain: { originBricks: [1, 0, 0], dimensionsBricks: [2, 1, 1] },
      stateStrideBytes: 4,
      stateBits: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
      active: { count: worklistWord(0), entryOffsetBytes: layout.activeEntryOffsetBytes, entryStrideBytes: 8, capacity },
      core: { count: worklistWord(8), entryOffsetBytes: layout.activeEntryOffsetBytes, entryStrideBytes: 8, capacity, requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core },
      halo: { count: worklistWord(9), entryOffsetBytes: layout.activeEntryOffsetBytes, entryStrideBytes: 8, capacity, requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.halo },
      retired: { count: worklistWord(4), entryOffsetBytes: layout.retiredEntryOffsetBytes, entryStrideBytes: 8, capacity },
      counters: { activated: worklistWord(10) },
      generation: worklistWord(15),
      revision: publicationWord(5),
      owner: "GPUFluidBrickResidency",
    },
  } as unknown as SparseVoxelStructuralRenderSource;
  const publicationWords = new Uint32Array(8);
  publicationWords[0] = generation;
  publicationWords[1] = SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  publicationWords[5] = generation;
  const worklistWords = new Uint32Array(layout.worklistByteLength / 4);
  worklistWords[0] = activeCount;
  worklistWords[4] = retiredCount;
  worklistWords[8] = coreCount;
  worklistWords[9] = haloCount;
  worklistWords[10] = 1;
  worklistWords[11] = retiredCount;
  worklistWords[15] = generation;
  worklistWords[layout.activeEntryOffsetBytes / 4] = 0;
  worklistWords[layout.activeEntryOffsetBytes / 4 + 1] = 4;
  worklistWords[layout.activeEntryOffsetBytes / 4 + 2] = 1;
  worklistWords[layout.activeEntryOffsetBytes / 4 + 3] = 5;
  worklistWords[layout.retiredEntryOffsetBytes / 4] = 1;
  worklistWords[layout.retiredEntryOffsetBytes / 4 + 1] = 5;
  const stateWords = new Uint32Array([
    SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.resident
      | SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core
      | SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.activated,
    (2 << 16) | SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.wasResident,
  ]);
  return { structural, snapshot: { publicationWords, worklistWords, stateWords } satisfies SvoRenderResidencySourceSnapshot };
}

test("adapter gates a complete snapshot and decodes active/core/halo/retired entries", () => {
  const { structural, snapshot } = fixture();
  const result = adaptSparseVoxelRenderResidencySource({ structural, snapshot, rendererCapacity: 2, coarseCoverageComplete: true });
  if (result.status === "rejected") assert.fail(result.reason);
  assert.equal(result.status, "ready");
  assert.deepEqual(result.activeEntries.map((entry) => [entry.brickIndex, entry.leafIndex, entry.coordinate, entry.state]), [[0, 4, [1, 0, 0], "core"]]);
  assert.equal(result.coreEntries.length, 1);
  assert.equal(result.haloEntries.length, 0);
  assert.deepEqual(result.retiredEntries.map((entry) => [entry.brickIndex, entry.leafIndex, entry.state, entry.dryFrames]), [[1, 5, "retired", 2]]);
  assert.deepEqual(result.desiredRequests.map((request) => [request.key, request.state, request.causes]), [["1,0,0", "core", ["solver-residency-activated"]]]);
  assert.equal(result.dirtyRequests.length, 1);
  assert.equal(result.dirtyRetiredEntries.length, 1);
  assert.equal(result.residency.coverage, "fine-complete");
  assert.equal(result.telemetry.completeGeneration, 7);
});

test("unchanged complete generation retains desired residency but emits no dirty work", () => {
  const { structural, snapshot } = fixture();
  const result = adaptSparseVoxelRenderResidencySource({ structural, snapshot, rendererCapacity: 2, coarseCoverageComplete: true, previousCompleteGeneration: 7 });
  if (result.status === "rejected") assert.fail(result.reason);
  assert.equal(result.status, "unchanged");
  assert.equal(result.desiredRequests.length, 1);
  assert.deepEqual(result.dirtyRequests, []);
  assert.deepEqual(result.dirtyRetiredEntries, []);
  assert.equal(result.telemetry.dirtyRequestCount, 0);
});

test("adapter rejects unpublished, missing, and non-matching publication fences", () => {
  const unpublished = fixture({ generation: 0 });
  assert.equal(adaptSparseVoxelRenderResidencySource({ ...unpublished, rendererCapacity: 2, coarseCoverageComplete: true }).status, "rejected");
  const missing = fixture();
  missing.snapshot.publicationWords[1] = SPARSE_VOXEL_VALID_FIELDS.topology;
  const missingResult = adaptSparseVoxelRenderResidencySource({ ...missing, rendererCapacity: 2, coarseCoverageComplete: true });
  assert.deepEqual([missingResult.status, "reason" in missingResult && missingResult.reason], ["rejected", "missing-fields"]);
  const listStale = fixture();
  listStale.snapshot.worklistWords[15] = 6;
  const staleResult = adaptSparseVoxelRenderResidencySource({ ...listStale, rendererCapacity: 2, coarseCoverageComplete: true });
  assert.deepEqual([staleResult.status, "reason" in staleResult && staleResult.reason], ["rejected", "generation-mismatch"]);
  const revisionStale = fixture();
  revisionStale.snapshot.publicationWords[5] = 6;
  const revisionResult = adaptSparseVoxelRenderResidencySource({ ...revisionStale, rendererCapacity: 2, coarseCoverageComplete: true });
  assert.deepEqual([revisionResult.status, "reason" in revisionResult && revisionResult.reason], ["rejected", "generation-mismatch"]);
});

test("malformed layouts, counters, entries, and undersized snapshots fail closed", () => {
  const badCounters = fixture();
  badCounters.snapshot.worklistWords[9] = 1;
  assert.throws(() => adaptSparseVoxelRenderResidencySource({ ...badCounters, rendererCapacity: 2, coarseCoverageComplete: true }), /counters/);
  const badEntry = fixture();
  badEntry.snapshot.worklistWords[16] = 2;
  assert.throws(() => adaptSparseVoxelRenderResidencySource({ ...badEntry, rendererCapacity: 2, coarseCoverageComplete: true }), /capacity/);
  const short = fixture();
  short.snapshot.stateWords = new Uint32Array(1);
  assert.throws(() => adaptSparseVoxelRenderResidencySource({ ...short, rendererCapacity: 2, coarseCoverageComplete: true }), /readback/);
  const badLayout = fixture();
  badLayout.structural.fluidResidency!.active.entryStrideBytes = 4;
  assert.throws(() => buildSvoRenderResidencyGpuInputs(badLayout.structural), /layout/);
});

test("source and renderer overflow stay publishable only with coarse coverage", () => {
  const overflow = fixture({ activeCount: 3, coreCount: 2, haloCount: 1, retiredCount: 0 });
  overflow.snapshot.stateWords[1] = SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.resident | SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core;
  const safe = adaptSparseVoxelRenderResidencySource({ ...overflow, rendererCapacity: 1, coarseCoverageComplete: true });
  if (safe.status === "rejected") assert.fail(safe.reason);
  assert.equal(safe.status, "ready");
  assert.equal(safe.telemetry.sourceOverflowCount, 1);
  assert.equal(safe.telemetry.rendererOverflowCount, 1);
  assert.equal(safe.residency.overflowCount, 2);
  assert.equal(safe.residency.coverage, "coarse-fallback");
  assert.equal(safe.residency.publishable, true);
  const unsafe = adaptSparseVoxelRenderResidencySource({ ...overflow, rendererCapacity: 1, coarseCoverageComplete: false });
  if (unsafe.status === "rejected") assert.fail(unsafe.reason);
  assert.equal(unsafe.status, "ready");
  assert.equal(unsafe.residency.coverage, "incomplete");
  assert.equal(unsafe.residency.publishable, false);
});

test("GPU inputs expose immutable publication/state/worklist bindings and bounded dispatch metadata", () => {
  const { structural } = fixture();
  const inputs = buildSvoRenderResidencyGpuInputs(structural);
  assert.deepEqual(inputs.bindGroupEntries.map((entry) => entry.binding), [0, 1, 2]);
  assert.equal(inputs.fence.completeGeneration.word, 0);
  assert.equal(inputs.fence.coarseFluidRevision.word, 5);
  assert.equal(inputs.fence.listGeneration.word, 15);
  assert.equal(inputs.lists.active.entryOffsetBytes, 64);
  assert.equal(inputs.lists.retired.entryOffsetBytes, 80);
  assert.equal(inputs.lists.core.requiredStateBit, SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core);
  assert.deepEqual(inputs.dispatch, { workgroupSize: 64, maximumEntryWorkgroups: 1, requiresIndirectPrepare: true });
});
