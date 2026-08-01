import assert from "node:assert/strict";
import test from "node:test";
import { planOctreeOwnerPages } from "../lib/webgpu-octree-owner-pages";
import { planStructuredVelocityGPU, type DirectStructuredVelocitySource } from
  "../lib/webgpu-octree-structured-velocity-gpu";
import type { OctreePowerTopologySource } from "../lib/webgpu-octree-power-topology";
import {
  OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS,
  WebGPUOctreeAirVelocitySupportProducer,
  type OctreeAirVelocitySupportGPUInputs,
} from "../lib/webgpu-octree-air-velocity-support-gpu";
import { planOctreeAirVelocitySupport } from "../lib/webgpu-octree-air-velocity-support";
import {
  WebGPUStructuredVelocityDynamics,
  type StructuredDynamicsResources,
} from "../lib/webgpu-octree-structured-dynamics";

interface FakeGPUCounts {
  shaderModules: number;
  syncPipelines: number;
  asyncPipelines: number;
  bindGroups: number;
  concurrentCompilations: number;
  maximumConcurrentCompilations: number;
}

function fakeGPU(): {
  readonly device: GPUDevice;
  readonly buffer: (size?: number) => GPUBuffer;
  readonly counts: FakeGPUCounts;
} {
  const counts: FakeGPUCounts = { shaderModules: 0, syncPipelines: 0, asyncPipelines: 0,
    bindGroups: 0, concurrentCompilations: 0, maximumConcurrentCompilations: 0 };
  let nextBuffer = 0, nextPipeline = 0;
  const buffer = (size = 1 << 20) => ({ id: ++nextBuffer, size, destroy() {} }) as unknown as GPUBuffer;
  const pipeline = (entryPoint: string) => ({ id: ++nextPipeline, entryPoint,
    getBindGroupLayout: () => ({ entryPoint }) }) as unknown as GPUComputePipeline;
  const device = {
    limits: { minStorageBufferOffsetAlignment: 256,
      maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30,
      maxComputeWorkgroupsPerDimension: 65_535 },
    queue: { writeBuffer() {}, submit() {} },
    createBuffer: (descriptor: GPUBufferDescriptor) => buffer(Number(descriptor.size)),
    createShaderModule: () => { counts.shaderModules += 1; return {}; },
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => {
      counts.syncPipelines += 1;
      return pipeline(compute.entryPoint!);
    },
    createComputePipelineAsync: async ({ compute }: GPUComputePipelineDescriptor) => {
      counts.asyncPipelines += 1;
      counts.concurrentCompilations += 1;
      counts.maximumConcurrentCompilations = Math.max(
        counts.maximumConcurrentCompilations, counts.concurrentCompilations);
      await Promise.resolve();
      counts.concurrentCompilations -= 1;
      return pipeline(compute.entryPoint!);
    },
    createBindGroup: () => { counts.bindGroups += 1; return {}; },
    createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish: () => ({}) }),
  } as unknown as GPUDevice;
  return { device, buffer, counts };
}

function structuredSource(buffer: (size?: number) => GPUBuffer): DirectStructuredVelocitySource {
  const plan = planStructuredVelocityGPU(1);
  return {
    plan,
    params: buffer(), authority: buffer(), control: buffer(), candidateControl: buffer(),
    rowVelocities: buffer(2 * plan.rowCapacity * 16), rowGeometry: buffer(),
    authorityBankStrideWords: plan.authorityWords,
    rowBankStrideWords: plan.rowCapacity,
    worksetBankStrideWords: plan.worksetStrideWords,
    section63: {} as never,
    familyWorksets: {} as never,
    liveRowDispatch: buffer(),
  };
}

function topologySource(buffer: (size?: number) => GPUBuffer): OctreePowerTopologySource {
  return {
    plan: { rowCapacity: 1, entryCount: 1, lookupCount: 1,
      metricBytes: 16, catalogBytes: 16, rowTemplateBytes: 16, allocatedBytes: 48 },
    metrics: buffer(), control: buffer(), catalogEntryHeaders: buffer(),
    catalogVolumes: buffer(16), catalogFaces: buffer(16), catalogCoefficients: buffer(),
    rowTemplateHeaders: buffer(), rowTemplateSlots: buffer(), rowTemplateData: buffer(),
    rowTemplateDiagonals: buffer(), reconstructionData: buffer(16),
    rowTemplateHeaderOffsetBytes: 0, rowTemplateSlotOffsetBytes: 0,
    rowTemplateDataOffsetBytes: 0, rowTemplateDiagonalOffsetBytes: 0,
    reconstructionDataOffsetBytes: 0,
    catalogTetrahedronHeaders: buffer(16), catalogTetrahedra: buffer(16),
    catalogTetrahedronVertices: buffer(16), catalogTetrahedronVertexCount: 1,
    catalogLookup: buffer(), sameOrFinerDirect: buffer(), sameOrCoarserDirect: buffer(),
  };
}

function airInputs(buffer: (size?: number) => GPUBuffer): OctreeAirVelocitySupportGPUInputs {
  const structured = structuredSource(buffer);
  const ownerPlan = planOctreeOwnerPages([1, 1, 1]);
  return {
    structured,
    topology: topologySource(buffer),
    owners: { plan: ownerPlan, arena: buffer(ownerPlan.allocatedBytes) },
    boundaryEpoch: { buffer: buffer(4), offsetWords: 0 },
    liquidMask: buffer(2 * structured.plan.rowCapacity * 4),
    dimensions: [1, 1, 1], maximumLeafSize: 1,
    maximumDisplacementFineCells: 1, closedBoundaryMask: 0,
  };
}

function dynamicsResources(buffer: (size?: number) => GPUBuffer,
  bodyCount = 0): StructuredDynamicsResources {
  const structured = structuredSource(buffer);
  const airSupportLayout = planOctreeAirVelocitySupport(
    structured.plan.rowCapacity, structured.plan.slotCapacity, 256, 1);
  return {
    structured,
    topology: topologySource(buffer),
    pressure: buffer(structured.plan.rowCapacity * 4),
    divergenceRhs: buffer(structured.plan.rowCapacity * 4),
    separationMask: buffer(4),
    liquidMask: buffer(2 * structured.plan.rowCapacity * 4),
    solidNormalVelocities: buffer(2 * structured.plan.slotCapacity * 4),
    rigidBodies: buffer(1 << 20), bodyCount, rigidExchange: buffer(1 << 20),
    boundaryWorksets: buffer(), boundaryControl: buffer(),
    selectorRows: buffer(airSupportLayout.totalBytes),
    selectorStride: airSupportLayout.selectorStride,
    selectorOffsetWords: airSupportLayout.selectorTagOffsetWords,
    airSupportLayout,
    dimensions: [1, 1, 1], physicalCellSize: 1, closedBoundaryMask: 0,
  };
}

function privateValue<T>(instance: object, name: string): T {
  return (instance as unknown as Record<string, T>)[name]!;
}

test("air-support producers reuse completed and in-flight per-device pipeline bundles", async () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    const entryCount = Object.keys(OCTREE_AIR_SUPPORT_GPU_ENTRY_BINDINGS).length;
    const sync = fakeGPU();
    const first = new WebGPUOctreeAirVelocitySupportProducer(
      sync.device, airInputs(sync.buffer));
    const firstBindGroups = sync.counts.bindGroups;
    const second = new WebGPUOctreeAirVelocitySupportProducer(
      sync.device, airInputs(sync.buffer));
    assert.equal(sync.counts.syncPipelines, entryCount);
    assert.equal(privateValue(first, "pipelines"), privateValue(second, "pipelines"));
    assert.ok(sync.counts.bindGroups > firstBindGroups,
      "the second producer must retain its own resource bind groups");
    assert.notEqual(first.arena, second.arena);

    const deferred = fakeGPU();
    const deferredA = new WebGPUOctreeAirVelocitySupportProducer(
      deferred.device, airInputs(deferred.buffer), true);
    const deferredB = new WebGPUOctreeAirVelocitySupportProducer(
      deferred.device, airInputs(deferred.buffer), true);
    await Promise.all([deferredA.initializePipelines(), deferredB.initializePipelines()]);
    assert.equal(deferred.counts.asyncPipelines, entryCount);
    assert.equal(deferred.counts.maximumConcurrentCompilations, 1);
    assert.equal(privateValue(deferredA, "pipelines"), privateValue(deferredB, "pipelines"));
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
});

test("structured dynamics reuses only the exact reachable per-device pipeline roster", async () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousEnergy = process.env.FLUID_STRUCTURED_ENERGY_PROBE;
  const previousFlatten = process.env.FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT;
  const previousAudit = process.env.FLUID_SYMMETRY_STAGE_AUDIT;
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  process.env.FLUID_STRUCTURED_ENERGY_PROBE = "0";
  process.env.FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT = "1";
  delete process.env.FLUID_SYMMETRY_STAGE_AUDIT;
  try {
    const sync = fakeGPU();
    const first = new WebGPUStructuredVelocityDynamics(
      sync.device, dynamicsResources(sync.buffer));
    const rosterSize = privateValue<() => readonly string[]>(first, "pipelineEntryPoints").call(first).length;
    const second = new WebGPUStructuredVelocityDynamics(
      sync.device, dynamicsResources(sync.buffer));
    assert.equal(sync.counts.syncPipelines, rosterSize);
    assert.equal(privateValue(first, "prepare"), privateValue(second, "prepare"));
    assert.notEqual(first.catalog, second.catalog,
      "immutable pipelines must not merge per-instance catalog resources");
    new WebGPUStructuredVelocityDynamics(sync.device, dynamicsResources(sync.buffer, 1));
    assert.equal(sync.counts.syncPipelines, rosterSize + rosterSize + 4,
      "adding body coupling must compile the distinct four-entry larger roster");

    const deferred = fakeGPU();
    const deferredA = new WebGPUStructuredVelocityDynamics(
      deferred.device, dynamicsResources(deferred.buffer), true);
    const deferredB = new WebGPUStructuredVelocityDynamics(
      deferred.device, dynamicsResources(deferred.buffer), true);
    const progressA: string[] = [], progressB: string[] = [];
    await Promise.all([
      deferredA.initializePipelines((entry, completed) => progressA.push(`${entry}:${completed}`)),
      deferredB.initializePipelines((entry, completed) => progressB.push(`${entry}:${completed}`)),
    ]);
    assert.equal(deferred.counts.asyncPipelines, rosterSize);
    assert.equal(deferred.counts.maximumConcurrentCompilations, 1);
    assert.equal(privateValue(deferredA, "prepare"), privateValue(deferredB, "prepare"));
    assert.deepEqual(progressB, progressA,
      "every concurrent caller retains the deferred API's per-entry progress stream");
  } finally {
    if (previousEnergy === undefined) delete process.env.FLUID_STRUCTURED_ENERGY_PROBE;
    else process.env.FLUID_STRUCTURED_ENERGY_PROBE = previousEnergy;
    if (previousFlatten === undefined) delete process.env.FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT;
    else process.env.FLUID_STRUCTURED_BOUNDARY_ADVECT_FLAT = previousFlatten;
    if (previousAudit === undefined) delete process.env.FLUID_SYMMETRY_STAGE_AUDIT;
    else process.env.FLUID_SYMMETRY_STAGE_AUDIT = previousAudit;
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
});
