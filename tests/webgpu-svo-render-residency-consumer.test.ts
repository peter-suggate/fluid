import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS,
  SVO_RENDER_RESIDENCY_CONSUMER_STATUS,
  WebGPUSvoRenderResidencyConsumer,
  referenceSvoRenderResidencyConsumption,
  svoRenderResidencyConsumerLayout,
  svoRenderResidencyConsumerShader,
} from "../lib/webgpu-svo-render-residency-consumer";
import {
  SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
  SPARSE_VOXEL_VALID_FIELDS,
  sparseVoxelFluidResidencyLayout,
  type SparseVoxelStructuralRenderSource,
} from "../lib/webgpu-voxel-debug";
import {
  OctreeOwnerPageLifecycleMirror,
  lookupOctreeOwnerPage,
  packOctreeOwnerPageWord,
  planOctreeOwnerPages,
} from "../lib/webgpu-octree-owner-pages";

const fakeBuffer = (size: number, label = "source") => ({ size, label, destroy() {} } as unknown as GPUBuffer);

function fixture(options: { sourceCapacity?: number; generation?: number } = {}) {
  const sourceCapacity = options.sourceCapacity ?? 3;
  const generation = options.generation ?? 7;
  const sourceLayout = sparseVoxelFluidResidencyLayout(sourceCapacity);
  const publicationBuffer = fakeBuffer(32, "publication");
  const statesBuffer = fakeBuffer(sourceCapacity * 4, "states");
  const worklistBuffer = fakeBuffer(sourceLayout.worklistByteLength, "worklist");
  const publication = { buffer: publicationBuffer, size: 32 };
  const states = { buffer: statesBuffer, size: sourceCapacity * 4 };
  const worklist = { buffer: worklistBuffer, size: sourceLayout.worklistByteLength };
  const publicationWord = (word: number) => ({ binding: publication, word });
  const worklistWord = (word: number) => ({ binding: worklist, word });
  const placeholder = { buffer: fakeBuffer(4096) };
  const structural = {
    control: placeholder, nodes: placeholder, leaves: placeholder, geometry: placeholder,
    velocity: placeholder, materialOwners: placeholder, fluidLeafStates: placeholder,
    capacities: { nodes: 8, leaves: 16, voxels: 512 },
    strides: { control: 4, node: 32, leaf: 16, geometry: 16, velocity: 16, materialOwner: 4 },
    domain: { worldOrigin_m: [0, 0, 0], cellSize_m: [0.1, 0.1, 0.1], dimensionsCells: [24, 8, 8], brickSize: 8, maximumDepth: 3 },
    publication: {
      state: publication,
      completeGeneration: publicationWord(0), validFields: publicationWord(1),
      revisions: {
        topology: publicationWord(2), staticGeometry: publicationWord(3), dynamicSolid: publicationWord(4),
        coarseFluid: publicationWord(5), fineFluid: publicationWord(6),
      },
    },
    fields: {},
    fluidResidency: {
      states, worklist,
      domain: { originBricks: [0, 0, 0], dimensionsBricks: [sourceCapacity, 1, 1] },
      stateStrideBytes: 4,
      stateBits: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
      active: { count: worklistWord(0), entryOffsetBytes: sourceLayout.activeEntryOffsetBytes, entryStrideBytes: 8, capacity: sourceCapacity },
      core: { count: worklistWord(8), entryOffsetBytes: sourceLayout.activeEntryOffsetBytes, entryStrideBytes: 8, capacity: sourceCapacity, requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core },
      halo: { count: worklistWord(9), entryOffsetBytes: sourceLayout.activeEntryOffsetBytes, entryStrideBytes: 8, capacity: sourceCapacity, requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.halo },
      retired: { count: worklistWord(4), entryOffsetBytes: sourceLayout.retiredEntryOffsetBytes, entryStrideBytes: 8, capacity: sourceCapacity },
      counters: { activated: worklistWord(10) }, generation: worklistWord(15), revision: publicationWord(5), owner: "GPUFluidBrickResidency",
    },
  } as unknown as SparseVoxelStructuralRenderSource;
  const publicationWords = new Uint32Array(8);
  publicationWords[0] = generation;
  publicationWords[1] = SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  publicationWords[5] = generation;
  const worklistWords = new Uint32Array(sourceLayout.worklistByteLength / 4);
  worklistWords[0] = 2; worklistWords[4] = 1; worklistWords[8] = 1; worklistWords[9] = 1; worklistWords[11] = 1; worklistWords[15] = generation;
  worklistWords.set([0, 4, 1, 5], sourceLayout.activeEntryOffsetBytes / 4);
  worklistWords.set([2, 6], sourceLayout.retiredEntryOffsetBytes / 4);
  const stateWords = new Uint32Array(sourceCapacity);
  stateWords[0] = SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.resident | SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core;
  stateWords[1] = SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.resident | SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.halo;
  stateWords[2] = SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.wasResident;
  return { structural, publicationWords, worklistWords, stateWords, sourceLayout };
}

test("consumer ABI packs four renderer-owned lists and aligned indirect dispatch records", () => {
  const layout = svoRenderResidencyConsumerLayout(5);
  assert.equal(layout.controlByteLength, 256);
  assert.equal(layout.listStrideBytes, 256);
  assert.equal(layout.entryByteLength, 1024);
  assert.deepEqual(layout.entryOffsetsBytes, { active: 0, core: 256, halo: 512, retired: 768 });
  assert.deepEqual(layout.indirectOffsetsBytes, {
    activeInput: 80, retiredInput: 96, activeOutput: 112, coreOutput: 128, haloOutput: 144, retiredOutput: 160,
  });
  assert.equal(SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.acceptedGeneration, 2);
  assert.throws(() => svoRenderResidencyConsumerLayout(0), /positive uint32/);
});

test("CPU oracle accepts exact fences, splits core/halo, and keeps coarse fallback complete under output pressure", () => {
  const data = fixture();
  const result = referenceSvoRenderResidencyConsumption({ ...data, sourceCapacity: 3, rendererCapacity: 1, leafCapacity: 16 });
  assert.equal(result.acceptedGeneration, 7);
  assert.deepEqual(result.active, [[0, 4]]);
  assert.deepEqual(result.core, [[0, 4]]);
  assert.deepEqual(result.halo, [[1, 5]]);
  assert.deepEqual(result.retired, [[2, 6]]);
  assert.ok((result.status & SVO_RENDER_RESIDENCY_CONSUMER_STATUS.rendererExhausted) !== 0);
  assert.ok((result.status & SVO_RENDER_RESIDENCY_CONSUMER_STATUS.coarseFallback) !== 0);
  assert.deepEqual(result.telemetry, {
    sourceOverflowCount: 0, rendererExhaustedCount: 1, invalidEntryCount: 0, stalePublicationCount: 0, coarseFallbackCount: 1,
  });
});

test("nonzero solver origin remaps active and retired bricks into the structural owner page table", () => {
  const data = fixture();
  const mapping = {
    sourceOriginBricks: [2, 1, 1] as const,
    sourceDimensionsBricks: [3, 1, 1] as const,
    structuralDimensionsBricks: [6, 4, 3] as const,
  };
  const result = referenceSvoRenderResidencyConsumption({
    ...data, ...mapping, sourceCapacity: 3, rendererCapacity: 3, leafCapacity: 16,
  });
  assert.deepEqual(result.active, [[32, 4], [33, 5]]);
  assert.deepEqual(result.core, [[32, 4]]);
  assert.deepEqual(result.halo, [[33, 5]]);
  assert.deepEqual(result.retired, [[34, 6]]);

  const plan = planOctreeOwnerPages([48, 32, 24], { maximumPages: 3 });
  const lifecycle = new OctreeOwnerPageLifecycleMirror(plan.logicalBrickCount, plan.capacity);
  lifecycle.publish(1, [34], []);
  const publication = lifecycle.publish(2, result.active.map(([logical]) => logical), result.retired.map(([logical]) => logical));
  assert.equal(publication.stats.retired, 1);
  assert.equal(lifecycle.slot(34), undefined);
  const arena = new Uint32Array(plan.allocatedWords);
  for (let logical = 0; logical < lifecycle.pageTable.length; logical += 1) {
    const encodedPage = lifecycle.pageTable[logical];
    if (encodedPage === 0) continue;
    let hashSlot = (Math.imul(logical, 0x9e37_79b1) >>> 0) % plan.pageHashCapacity;
    while (arena[plan.pageTableOffsetWords + hashSlot] !== 0) hashSlot = (hashSlot + 1) % plan.pageHashCapacity;
    arena[plan.pageTableOffsetWords + hashSlot] = logical + 1;
    arena[plan.pageTableValueOffsetWords + hashSlot] = encodedPage;
  }
  const cell = [16, 8, 8] as const; // structural brick (2,1,1), logical 32
  const slot = lifecycle.slot(32)!;
  arena[plan.ownerPagesOffsetWords + slot * plan.pageVoxels] = packOctreeOwnerPageWord(cell, cell, 1);
  assert.deepEqual(lookupOctreeOwnerPage(arena, plan, cell, 16), {
    origin: [...cell], size: 1, missing: false, status: 0,
  });
});

test("CPU oracle rejects stale, unchanged, mismatched, and malformed publications without exposing dirty work", () => {
  const stale = fixture();
  const staleResult = referenceSvoRenderResidencyConsumption({ ...stale, sourceCapacity: 3, rendererCapacity: 3, leafCapacity: 16, acceptedGeneration: 8 });
  assert.equal(staleResult.status, SVO_RENDER_RESIDENCY_CONSUMER_STATUS.stale);
  assert.equal(staleResult.telemetry.stalePublicationCount, 1);
  assert.deepEqual(staleResult.active, []);
  const unchangedResult = referenceSvoRenderResidencyConsumption({ ...stale, sourceCapacity: 3, rendererCapacity: 3, leafCapacity: 16, acceptedGeneration: 7 });
  assert.equal(unchangedResult.status, SVO_RENDER_RESIDENCY_CONSUMER_STATUS.unchanged);
  const mismatch = fixture(); mismatch.worklistWords[15] = 6;
  assert.equal(referenceSvoRenderResidencyConsumption({ ...mismatch, sourceCapacity: 3, rendererCapacity: 3, leafCapacity: 16 }).status, SVO_RENDER_RESIDENCY_CONSUMER_STATUS.generationMismatch);
  const duplicate = fixture();
  duplicate.worklistWords[duplicate.sourceLayout.retiredEntryOffsetBytes / 4] = 1;
  duplicate.worklistWords[duplicate.sourceLayout.retiredEntryOffsetBytes / 4 + 1] = 5;
  const invalid = referenceSvoRenderResidencyConsumption({ ...duplicate, sourceCapacity: 3, rendererCapacity: 3, leafCapacity: 16 });
  assert.ok((invalid.status & SVO_RENDER_RESIDENCY_CONSUMER_STATUS.invalidEntry) !== 0);
  assert.equal(invalid.acceptedGeneration, 0);
  assert.equal(invalid.telemetry.coarseFallbackCount, 2);
  assert.deepEqual(invalid.active, []);
});

test("shader validates source fences on GPU, reads source arenas only, and authors bounded output telemetry", () => {
  assert.match(svoRenderResidencyConsumerShader, /publication\[5\]!=generation \|\| sourceWorklist\[15\]!=generation/);
  assert.match(svoRenderResidencyConsumerShader, /coreCount>activeCount \|\| haloCount!=activeCount-coreCount/);
  assert.match(svoRenderResidencyConsumerShader, /atomicExchange\(&seenAttempt\[brick\],attempt\)/);
  assert.match(svoRenderResidencyConsumerShader, /atomicExchange\(&seenAttempt\[leafStamp\],attempt\)/);
  assert.match(svoRenderResidencyConsumerShader, /storeDispatch\(ACTIVE_OUTPUT_DISPATCH,min\(activeWork,params\.rendererCapacity\)\)/);
  assert.match(svoRenderResidencyConsumerShader, /status\|=RENDERER_EXHAUSTED/);
  assert.match(svoRenderResidencyConsumerShader, /status\|=INVALID_ENTRY\|COARSE_FALLBACK/);
  assert.match(svoRenderResidencyConsumerShader, /let scene=local\+params\.sourceOriginBricks\.xyz/);
  assert.match(svoRenderResidencyConsumerShader, /writeEntry\(3u,slot,mapped,leaf\)/,
    "retired source bricks must use the same structural remap as active bricks");
  assert.doesNotMatch(svoRenderResidencyConsumerShader, /var<storage,read_write> (publication|states|sourceWorklist)/);
  assert.doesNotMatch(svoRenderResidencyConsumerShader, /mapAsync|copyBufferToBuffer/);
});

test("WebGPU consumer owns outputs, binds producer arenas read-only, and encodes prepare/compact/finalize without readback", (t) => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, INDIRECT: 8, UNIFORM: 16 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { COMPUTE: 1 } });
  t.after(() => {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage); else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
    if (previousStage) Object.defineProperty(globalThis, "GPUShaderStage", previousStage); else Reflect.deleteProperty(globalThis, "GPUShaderStage");
  });
  const data = fixture();
  const created: Array<{ label?: string; size: number; usage: number; destroyed: boolean }> = [];
  const bindLayouts: GPUBindGroupLayoutEntry[][] = [];
  const pipelines = new Map<string, object>();
  const writes: unknown[][] = [];
  const device = {
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      const buffer = { ...descriptor, destroyed: false, destroy() { buffer.destroyed = true; } };
      created.push(buffer as typeof created[number]); return buffer;
    },
    queue: { writeBuffer: (...args: unknown[]) => writes.push(args) },
    createShaderModule: ({ code }: GPUShaderModuleDescriptor) => ({ code }),
    createBindGroupLayout: ({ entries }: GPUBindGroupLayoutDescriptor) => { bindLayouts.push([...entries]); return {}; },
    createPipelineLayout: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => { const value = { entryPoint: compute.entryPoint }; pipelines.set(compute.entryPoint!, value); return value; },
    createBindGroup: ({ entries }: GPUBindGroupDescriptor) => ({ entries }),
  } as unknown as GPUDevice;
  const consumer = new WebGPUSvoRenderResidencyConsumer(device, data.structural, { capacity: 2, coarseCoverageComplete: true });
  assert.equal(writes.length, 1, "only immutable consumer parameters are uploaded");
  assert.deepEqual(bindLayouts[0].slice(0, 3).map((entry) => entry.buffer?.type), ["read-only-storage", "read-only-storage", "read-only-storage"]);
  assert.equal(consumer.binding("halo").offset, 512);
  assert.equal(consumer.indirectOffset("retired"), 160);
  assert.equal(consumer.allocatedBytes, 256 + 1024 + (3 + 16) * 4 + 80 + 24);
  const calls: Array<readonly unknown[]> = [];
  const encoder = {
    copyBufferToBuffer: (...args: unknown[]) => calls.push(["copy", ...args]),
    beginComputePass: ({ label }: GPUComputePassDescriptor) => ({
      setPipeline: (pipeline: unknown) => calls.push([label, "pipeline", pipeline]),
      setBindGroup: () => {},
      dispatchWorkgroups: (...args: unknown[]) => calls.push([label, "direct", ...args]),
      dispatchWorkgroupsIndirect: (...args: unknown[]) => calls.push([label, "indirect", ...args]),
      end: () => calls.push([label, "end"]),
    }),
  } as unknown as GPUCommandEncoder;
  consumer.encode(encoder);
  assert.deepEqual(calls.filter((call) => call[1] === "direct").map((call) => call[0]), ["SVO renderer residency prepare", "SVO renderer residency finalize"]);
  assert.deepEqual(calls.filter((call) => call[0] === "copy").map((call) => [call[2], call[4], call[5]]), [[80, 0, 12], [96, 12, 12]]);
  assert.deepEqual(calls.filter((call) => call[1] === "indirect").map((call) => call[3]), [0, 12]);
  assert.equal(calls.filter((call) => call[1] === "end").length, 3, "pass boundaries make storage-authored indirect metadata visible");
  consumer.destroy(); consumer.destroy();
  assert.equal(created.filter((buffer) => buffer.destroyed).length, 5);
  assert.equal((data.structural.publication.state.buffer as unknown as { destroyed?: boolean }).destroyed, undefined, "producer buffers remain externally owned");
  assert.throws(() => new WebGPUSvoRenderResidencyConsumer(device, data.structural, { capacity: 2, coarseCoverageComplete: false as true }), /coarse fallback/);
});
