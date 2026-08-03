import assert from "node:assert/strict";
import test from "node:test";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import type { WebGPUFineLevelSetBrickSource } from "../lib/webgpu-octree-fine-levelset-bricks";
import { WebGPUFineLevelSetRedistance } from "../lib/webgpu-octree-fine-levelset-redistance";
import { WebGPUFineLevelSetTopology, planFineLevelSetPageDeltaLayout } from
  "../lib/webgpu-octree-fine-levelset-topology";
import { WebGPUFineLevelSetTransport } from "../lib/webgpu-octree-fine-levelset-transport";
import { WebGPUFineLevelSetVolumeCorrection } from "../lib/webgpu-octree-fine-levelset-volume";

interface FakeGPUCounts {
  shaderModules: number; pipelines: number; asyncPipelines: number;
  bindGroups: number; concurrentCompilations: number; maximumConcurrentCompilations: number;
}

function fakeGPU(): { device: GPUDevice; counts: FakeGPUCounts; buffer(size?: number): GPUBuffer } {
  const counts = { shaderModules: 0, pipelines: 0, asyncPipelines: 0, bindGroups: 0,
    concurrentCompilations: 0, maximumConcurrentCompilations: 0 };
  let bufferId = 0;
  const buffer = (size = 4) => ({ size, id: ++bufferId, destroy() {} }) as unknown as GPUBuffer;
  const pipeline = (label: string | undefined, entryPoint: string) => ({ label, entryPoint,
    getBindGroupLayout: () => ({ label, entryPoint }) });
  const device = {
    limits: {
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 30,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
    queue: { writeBuffer() {}, submit() {} },
    createBuffer: (descriptor: GPUBufferDescriptor) => buffer(Number(descriptor.size)),
    createShaderModule: () => { counts.shaderModules += 1; return {}; },
    createComputePipeline: ({ label, compute }: GPUComputePipelineDescriptor) => {
      counts.pipelines += 1;
      return pipeline(label, compute.entryPoint!);
    },
    createComputePipelineAsync: async ({ label, compute }: GPUComputePipelineDescriptor) => {
      counts.asyncPipelines += 1; counts.concurrentCompilations += 1;
      counts.maximumConcurrentCompilations = Math.max(
        counts.maximumConcurrentCompilations, counts.concurrentCompilations);
      await Promise.resolve();
      counts.concurrentCompilations -= 1;
      return pipeline(label, compute.entryPoint!);
    },
    createBindGroup: () => { counts.bindGroups += 1; return {}; },
    createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish: () => ({}) }),
  } as unknown as GPUDevice;
  return { device, counts, buffer };
}

function source(buffer: (size?: number) => GPUBuffer, generation: number,
  generationSlot: 0 | 1): WebGPUFineLevelSetBrickSource {
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
    finestCellDimensions: [4, 4, 4], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8 });
  return {
    generation, generationSlot, plan,
    params: buffer(64), metadata: buffer(128), worklist: buffer(256),
    samples: buffer(2_048), workA: buffer(2_048), workB: buffer(2_048),
    rollbackSamples: buffer(2_048),
  };
}

function privatePipeline(value: object, name: string): GPUComputePipeline {
  return (value as unknown as Record<string, GPUComputePipeline>)[name]!;
}

test("fine A/B helpers reuse immutable pipelines and retain per-instance bindings", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16, MAP_READ: 32 } });
  try {
    const { device, counts, buffer } = fakeGPU();
    const sourceA = source(buffer, 1, 0); const sourceB = source(buffer, 2, 1);
    // The plan is an immutable value object in production and shared by the
    // generations. Mirror that identity here as well as its equal contents.
    Object.defineProperty(sourceB, "plan", { value: sourceA.plan });

    const coarseWGSL = "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}";
    const topologyA = new WebGPUFineLevelSetTopology(device, sourceA, sourceB, coarseWGSL);
    const topologyPipelineCount = counts.pipelines;
    const topologyB = new WebGPUFineLevelSetTopology(device, sourceB, sourceA, coarseWGSL);
    assert.equal(counts.pipelines, topologyPipelineCount);
    assert.equal(privatePipeline(topologyB, "clearPipeline"),
      privatePipeline(topologyA, "clearPipeline"));
    assert.notEqual((topologyB as unknown as { params: GPUBuffer }).params,
      (topologyA as unknown as { params: GPUBuffer }).params);

    const delta = (pageDelta: GPUBuffer) => ({ pageDelta,
      pageDeltaLayout: planFineLevelSetPageDeltaLayout(sourceA.plan.maximumResidentBricks),
      redistanceDispatches: { buffer: pageDelta, dirtyOffsetBytes: 84 as const,
        supportOffsetBytes: 60 as const } });
    const redistanceA = new WebGPUFineLevelSetRedistance(device, sourceA, delta(buffer(512)));
    const redistancePipelineCount = counts.pipelines;
    const redistanceB = new WebGPUFineLevelSetRedistance(device, sourceB, delta(buffer(512)));
    assert.equal(counts.pipelines, redistancePipelineCount);
    assert.equal(privatePipeline(redistanceB, "jfaSeedPipeline"),
      privatePipeline(redistanceA, "jfaSeedPipeline"));
    assert.notEqual(redistanceB.control, redistanceA.control);

    const coarse = { headers: buffer(16), records: buffer(16), physicalVolumes: buffer(4),
      sampleDirectory: buffer(64), publicationControl: buffer(64), rowCount: buffer(4),
      dimensions: [4, 4, 4] as const, physicalCellSize: 1, maximumLeafSize: 1,
      sampleRowCapacity: 1 };
    const volumeA = new WebGPUFineLevelSetVolumeCorrection(device, sourceA, coarse);
    const volumePipelineCount = counts.pipelines, volumeBindGroups = counts.bindGroups;
    const volumeB = new WebGPUFineLevelSetVolumeCorrection(device, sourceB, coarse, volumeA.control);
    assert.equal(counts.pipelines, volumePipelineCount);
    assert.equal(counts.bindGroups, volumeBindGroups,
      "volume bindings remain lazy until the helper is encoded");
    assert.equal(privatePipeline(volumeB, "finePartialPipeline"),
      privatePipeline(volumeA, "finePartialPipeline"));

    const topologyCatalog = {
      catalogVolumes: buffer(4), catalogFaces: buffer(4),
      catalogTetrahedronHeaders: buffer(4), catalogTetrahedronVertices: buffer(4),
      catalogTetrahedra: buffer(4), metrics: buffer(16), rowTemplateHeaderOffsetBytes: 0,
    };
    const structured = { control: buffer(64), rowVelocities: buffer(64), rowBankStrideWords: 16,
      plan: { rowCapacity: 1, maximumCaseSlots: 1, authorityWords: 16, offsets: {} } };
    const resources = { structured, topology: topologyCatalog,
      dimensions: [4, 4, 4] as const, physicalCellSize: 1, maximumLeafSize: 1 };
    const transportA = new WebGPUFineLevelSetTransport(device, sourceA, resources as never);
    const transportPipelineCount = counts.pipelines, transportBindGroups = counts.bindGroups;
    const transportB = new WebGPUFineLevelSetTransport(device, sourceB, resources as never);
    assert.equal(counts.pipelines, transportPipelineCount);
    assert.equal(counts.bindGroups, transportBindGroups,
      "transport bindings remain lazy until the helper is encoded");
    assert.equal(privatePipeline(transportB, "commitPipeline"),
      privatePipeline(transportA, "commitPipeline"));
    assert.notEqual(transportB.control, transportA.control);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
});

test("deferred topology and redistance compile sequentially and share in-flight A/B bundles", async () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16, MAP_READ: 32 } });
  try {
    const { device, counts, buffer } = fakeGPU();
    const sourceA = source(buffer, 1, 0); const sourceB = source(buffer, 2, 1);
    Object.defineProperty(sourceB, "plan", { value: sourceA.plan });
    const coarseWGSL = "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}";
    const topologyA = new WebGPUFineLevelSetTopology(device, sourceA, sourceB, coarseWGSL, true);
    const topologyB = new WebGPUFineLevelSetTopology(device, sourceB, sourceA, coarseWGSL, true);
    assert.equal(counts.pipelines, 0, "deferred construction must remain allocation-only");
    assert.equal(counts.shaderModules, 0);
    assert.throws(() => topologyA.encode({} as never), /pipelines are not initialized/);
    await Promise.all([topologyA.initializePipelines(), topologyB.initializePipelines()]);
    assert.equal(counts.asyncPipelines, 58);
    assert.equal(counts.shaderModules, 1);
    assert.equal(counts.maximumConcurrentCompilations, 1,
      "a family must never put multiple asynchronous compilations in flight");
    assert.equal(privatePipeline(topologyB, "clearPipeline"),
      privatePipeline(topologyA, "clearPipeline"));
    await topologyA.initializePipelines();
    assert.equal(counts.asyncPipelines, 58, "initialization must be idempotent");

    const delta = (pageDelta: GPUBuffer) => ({ pageDelta,
      pageDeltaLayout: planFineLevelSetPageDeltaLayout(sourceA.plan.maximumResidentBricks),
      redistanceDispatches: { buffer: pageDelta, dirtyOffsetBytes: 84 as const,
        supportOffsetBytes: 60 as const } });
    const redistanceA = new WebGPUFineLevelSetRedistance(
      device, sourceA, delta(buffer(512)), true);
    const redistanceB = new WebGPUFineLevelSetRedistance(
      device, sourceB, delta(buffer(512)), true);
    assert.throws(() => redistanceA.encode({} as never, { bandCells: 8 }),
      /pipelines are not initialized/);
    const beforeRedistance = counts.asyncPipelines;
    await Promise.all([redistanceA.initializePipelines(), redistanceB.initializePipelines()]);
    assert.equal(counts.asyncPipelines - beforeRedistance, 35);
    assert.equal(counts.shaderModules, 2);
    assert.equal(counts.maximumConcurrentCompilations, 1);
    assert.equal(privatePipeline(redistanceB, "jfaSeedPipeline"),
      privatePipeline(redistanceA, "jfaSeedPipeline"));
    await redistanceB.initializePipelines();
    assert.equal(counts.asyncPipelines - beforeRedistance, 35);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
});

test("deferred transport and volume compile sequentially and share in-flight A/B bundles", async () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16, MAP_READ: 32 } });
  try {
    const { device, counts, buffer } = fakeGPU();
    const sourceA = source(buffer, 1, 0); const sourceB = source(buffer, 2, 1);
    Object.defineProperty(sourceB, "plan", { value: sourceA.plan });
    const topologyCatalog = {
      catalogVolumes: buffer(4), catalogFaces: buffer(4),
      catalogTetrahedronHeaders: buffer(4), catalogTetrahedronVertices: buffer(4),
      catalogTetrahedra: buffer(4), metrics: buffer(16), rowTemplateHeaderOffsetBytes: 0,
    };
    const structured = { control: buffer(64), rowVelocities: buffer(64), rowBankStrideWords: 16,
      plan: { rowCapacity: 1, maximumCaseSlots: 1, authorityWords: 16, offsets: {} } };
    const resources = { structured, topology: topologyCatalog,
      dimensions: [4, 4, 4] as const, physicalCellSize: 1, maximumLeafSize: 1 };
    const transportA = new WebGPUFineLevelSetTransport(
      device, sourceA, resources as never, true);
    const transportB = new WebGPUFineLevelSetTransport(
      device, sourceB, resources as never, true);
    assert.equal(counts.pipelines, 0, "deferred construction must remain allocation-only");
    assert.equal(counts.shaderModules, 0);
    assert.throws(() => transportA.encode({} as never, { timestep: 1 / 60 }),
      /pipelines are not initialized/);
    await Promise.all([transportA.initializePipelines(), transportB.initializePipelines()]);
    assert.equal(counts.asyncPipelines, 15);
    assert.equal(counts.shaderModules, 1);
    assert.equal(counts.maximumConcurrentCompilations, 1);
    assert.equal(privatePipeline(transportB, "commitPipeline"),
      privatePipeline(transportA, "commitPipeline"));
    assert.equal(counts.bindGroups, 30,
      "both deferred transport instances must install their own 15 bind groups");
    assert.notEqual(transportB.control, transportA.control);
    await transportB.initializePipelines();
    assert.equal(counts.asyncPipelines, 15, "initialization must be idempotent");

    const coarse = { headers: buffer(16), records: buffer(16), physicalVolumes: buffer(4),
      sampleDirectory: buffer(64), publicationControl: buffer(64), rowCount: buffer(4),
      dimensions: [4, 4, 4] as const, physicalCellSize: 1, maximumLeafSize: 1,
      sampleRowCapacity: 1 };
    const volumeA = new WebGPUFineLevelSetVolumeCorrection(
      device, sourceA, coarse, undefined, true);
    const volumeB = new WebGPUFineLevelSetVolumeCorrection(
      device, sourceB, coarse, volumeA.control, true);
    assert.equal(counts.asyncPipelines, 15);
    assert.equal(counts.shaderModules, 1);
    assert.throws(() => volumeA.encodeMeasurement({} as never),
      /pipelines are not initialized/);
    const beforeVolume = counts.asyncPipelines;
    await Promise.all([volumeA.initializePipelines(), volumeB.initializePipelines()]);
    assert.equal(counts.asyncPipelines - beforeVolume, 11);
    assert.equal(counts.shaderModules, 2);
    assert.equal(counts.maximumConcurrentCompilations, 1);
    assert.equal(privatePipeline(volumeB, "finePartialPipeline"),
      privatePipeline(volumeA, "finePartialPipeline"));
    assert.equal(counts.bindGroups, 52,
      "both deferred volume instances must add their own 11 bind groups");
    await volumeA.initializePipelines();
    assert.equal(counts.asyncPipelines - beforeVolume, 11);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
});
