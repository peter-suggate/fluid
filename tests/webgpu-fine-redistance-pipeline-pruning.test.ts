import assert from "node:assert/strict";
import test from "node:test";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import type { WebGPUFineLevelSetBrickSource } from
  "../lib/webgpu-octree-fine-levelset-bricks";
import {
  maximumFineLevelSetJFAStride,
  WebGPUFineLevelSetRedistance,
} from "../lib/webgpu-octree-fine-levelset-redistance";
import { planFineLevelSetPageDeltaLayout } from
  "../lib/webgpu-octree-fine-levelset-topology";

interface PipelineRecord {
  readonly entryPoint: string;
  readonly stride?: number;
}

function fakeGPU(): {
  readonly device: GPUDevice;
  readonly buffer: (size?: number) => GPUBuffer;
  readonly syncPipelines: PipelineRecord[];
  readonly asyncPipelines: PipelineRecord[];
  readonly shaderModuleCount: () => number;
} {
  const syncPipelines: PipelineRecord[] = [];
  const asyncPipelines: PipelineRecord[] = [];
  let shaderModules = 0;
  const buffer = (size = 4) => ({ size, destroy() {} }) as unknown as GPUBuffer;
  const record = (descriptor: GPUComputePipelineDescriptor): PipelineRecord => ({
    entryPoint: descriptor.compute.entryPoint!,
    ...((descriptor.compute.constants?.JFA_STRIDE as number | undefined) === undefined ? {} : {
      stride: descriptor.compute.constants!.JFA_STRIDE as number,
    }),
  });
  const pipeline = (descriptor: GPUComputePipelineDescriptor) => ({
    ...record(descriptor),
    getBindGroupLayout: () => ({}),
  });
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 },
    queue: { writeBuffer() {} },
    createBuffer: (descriptor: GPUBufferDescriptor) => buffer(Number(descriptor.size)),
    createShaderModule: () => { shaderModules += 1; return {}; },
    createComputePipeline: (descriptor: GPUComputePipelineDescriptor) => {
      syncPipelines.push(record(descriptor));
      return pipeline(descriptor);
    },
    createComputePipelineAsync: async (descriptor: GPUComputePipelineDescriptor) => {
      asyncPipelines.push(record(descriptor));
      await Promise.resolve();
      return pipeline(descriptor);
    },
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  return { device, buffer, syncPipelines, asyncPipelines,
    shaderModuleCount: () => shaderModules };
}

function source(buffer: (size?: number) => GPUBuffer, generation: number,
  generationSlot: 0 | 1): WebGPUFineLevelSetBrickSource {
  const plan = planFineLevelSetBricks({
    domainOrigin: [0, 0, 0], finestCellDimensions: [4, 4, 4], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8,
  });
  return {
    generation, generationSlot, plan,
    params: buffer(64), metadata: buffer(320), worklist: buffer(256),
    flags: buffer(2_048), phi: buffer(2_048), workA: buffer(2_048), workB: buffer(2_048),
    rollbackPhi: buffer(2_048),
  };
}

function delta(buffer: GPUBuffer, pageCapacity: number) {
  return {
    pageDelta: buffer,
    pageDeltaLayout: planFineLevelSetPageDeltaLayout(pageCapacity),
    redistanceDispatches: {
      buffer, dirtyOffsetBytes: 84 as const, supportOffsetBytes: 60 as const,
    },
  };
}

function jfaStrides(records: readonly PipelineRecord[]): number[] {
  return records
    .filter(({ entryPoint }) => entryPoint === "jumpFloodAToB")
    .map(({ stride }) => stride!);
}

test("fine redistance derives the smallest reachable compile-time JFA stride", () => {
  assert.equal(maximumFineLevelSetJFAStride(1), 1);
  assert.equal(maximumFineLevelSetJFAStride(8), 8);
  assert.equal(maximumFineLevelSetJFAStride(23), 16);
  assert.equal(maximumFineLevelSetJFAStride(256), 256);
  assert.throws(() => maximumFineLevelSetJFAStride(0), /bandCells/);
});

test("fine redistance compiles only reachable specialized strides and shares the bounded bundle", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    const { device, buffer, syncPipelines, shaderModuleCount } = fakeGPU();
    const sourceA = source(buffer, 1, 0);
    const sourceB = source(buffer, 2, 1);
    Object.defineProperty(sourceB, "plan", { value: sourceA.plan });
    const options = { maximumRequiredJfaStride: maximumFineLevelSetJFAStride(23) } as const;
    const redistanceA = new WebGPUFineLevelSetRedistance(
      device, sourceA, delta(buffer(512), 8), options);
    const redistanceB = new WebGPUFineLevelSetRedistance(
      device, sourceB, delta(buffer(512), 8), options);

    assert.deepEqual(jfaStrides(syncPipelines), [1, 2, 4, 8, 16]);
    assert.equal(syncPipelines.length, 27,
      "five A/B stride pairs plus 17 unspecialized pipelines replace the old 35-pipeline bundle");
    assert.equal(shaderModuleCount(), 1);
    assert.equal(
      (redistanceA as unknown as { jfaABPipelines: unknown }).jfaABPipelines,
      (redistanceB as unknown as { jfaABPipelines: unknown }).jfaABPipelines,
      "A/B generations with the same immutable reach reuse one bundle",
    );
    assert.throws(() => new WebGPUFineLevelSetRedistance(
      device, sourceA, delta(buffer(512), 8), { maximumRequiredJfaStride: 12 }),
    /power of two/);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
});

test("deferred fine redistance shares an in-flight bounded compilation and rejects wider schedules", async () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    const { device, buffer, asyncPipelines, shaderModuleCount } = fakeGPU();
    const sourceA = source(buffer, 1, 0);
    const sourceB = source(buffer, 2, 1);
    Object.defineProperty(sourceB, "plan", { value: sourceA.plan });
    const options = { deferPipelineCompilation: true,
      maximumRequiredJfaStride: 8 } as const;
    const redistanceA = new WebGPUFineLevelSetRedistance(
      device, sourceA, delta(buffer(512), 8), options);
    const redistanceB = new WebGPUFineLevelSetRedistance(
      device, sourceB, delta(buffer(512), 8), options);
    await Promise.all([redistanceA.initializePipelines(), redistanceB.initializePipelines()]);

    assert.deepEqual(jfaStrides(asyncPipelines), [1, 2, 4, 8]);
    assert.equal(asyncPipelines.length, 25);
    assert.equal(shaderModuleCount(), 1,
      "A/B initialization must join the same bounded in-flight compilation");
    assert.throws(() => redistanceA.encode({} as never, { bandCells: 23 }),
      /requires stride 16, above the immutable compiled maximum 8/);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
});
