import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  decodeGeneratedOctreePowerCatalog,
} from "../lib/generated/octree-power-catalog";
import { WebGPUOctreePowerDescriptor } from "../lib/webgpu-octree-power-descriptor";
import { WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";
import { WebGPUStructuredBoundaryCoefficients } from
  "../lib/webgpu-octree-structured-boundary";
import {
  planStructuredVelocityGPU,
  WebGPUDirectStructuredVelocityAuthority,
} from "../lib/webgpu-octree-structured-velocity-gpu";

interface FakeGPUCounts {
  shaderModules: number;
  pipelines: number;
  bindGroupLayouts: number;
  pipelineLayouts: number;
  bindGroups: number;
}

function fakeGPU(): { device: GPUDevice; counts: FakeGPUCounts; buffer(size?: number): GPUBuffer } {
  const counts = { shaderModules: 0, pipelines: 0, bindGroupLayouts: 0,
    pipelineLayouts: 0, bindGroups: 0 };
  let bufferId = 0;
  const buffer = (size = 4) => {
    const storage = new ArrayBuffer(size);
    return { size, id: ++bufferId, destroy() {}, getMappedRange: () => storage, unmap() {} };
  };
  const device = {
    limits: {
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 30,
      minStorageBufferOffsetAlignment: 256,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
    queue: { writeBuffer() {}, submit() {} },
    createBuffer: (descriptor: GPUBufferDescriptor) => buffer(Number(descriptor.size)),
    createShaderModule: () => { counts.shaderModules += 1; return {}; },
    createBindGroupLayout: () => { counts.bindGroupLayouts += 1; return {}; },
    createPipelineLayout: () => { counts.pipelineLayouts += 1; return {}; },
    createComputePipeline: ({ label, compute }: GPUComputePipelineDescriptor) => {
      counts.pipelines += 1;
      return { label, entryPoint: compute.entryPoint,
        getBindGroupLayout: () => ({ label, entryPoint: compute.entryPoint }) };
    },
    createBindGroup: ({ layout }: GPUBindGroupDescriptor) => {
      counts.bindGroups += 1;
      return { layout, id: counts.bindGroups };
    },
  } as unknown as GPUDevice;
  return { device, counts, buffer: buffer as unknown as (size?: number) => GPUBuffer };
}

function privatePipeline(value: object, name: string): GPUComputePipeline {
  return (value as unknown as Record<string, GPUComputePipeline>)[name]!;
}

function catalogViews() {
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  return decodeGeneratedOctreePowerCatalog(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

test("remaining synchronous WebGPU families reuse per-device immutable pipelines", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true,
    value: { COMPUTE: 1 } });
  try {
    const { device, counts, buffer } = fakeGPU();

    const descriptorA = new WebGPUOctreePowerDescriptor(device, 1);
    assert.equal(counts.pipelines, 7);
    const descriptorB = new WebGPUOctreePowerDescriptor(device, 2);
    assert.equal(counts.pipelines, 7, "descriptor row capacity is runtime data");
    assert.equal(privatePipeline(descriptorB, "preparePipeline"),
      privatePipeline(descriptorA, "preparePipeline"));
    assert.notEqual(descriptorB.control, descriptorA.control);

    const catalog = catalogViews();
    const topologyA = new WebGPUOctreePowerTopology(device, 1, catalog);
    assert.equal(counts.pipelines, 14);
    const topologyB = new WebGPUOctreePowerTopology(device, 2, catalog);
    assert.equal(counts.pipelines, 14, "topology catalog dimensions are runtime buffer data");
    assert.equal(privatePipeline(topologyB, "resolvePipeline"),
      privatePipeline(topologyA, "resolvePipeline"));
    assert.notEqual(topologyB.control, topologyA.control);

    const velocityInputs = {
      leafHeaders: buffer(48),
      topology: {
        plan: { rowCapacity: 1, entryCount: 1 },
        rowTemplateHeaderOffsetBytes: 0,
        rowTemplateSlotOffsetBytes: 0,
        rowTemplateDataOffsetBytes: 0,
        reconstructionDataOffsetBytes: 0,
        catalogFaces: buffer(48),
      },
      rowDelta: { rows: buffer(64), controlOffsetWords: 0, newToOldOffsetWords: 0,
        oldToNewOffsetWords: 0, affectedRowsOffsetWords: 0 },
      dimensions: [1, 1, 1] as const,
      physicalCellSize: 1,
      closedBoundaryMask: 0,
    };
    const velocityA = new WebGPUDirectStructuredVelocityAuthority(
      device, velocityInputs as never);
    assert.equal(counts.pipelines, 23);
    const velocityB = new WebGPUDirectStructuredVelocityAuthority(
      device, velocityInputs as never);
    assert.equal(counts.pipelines, 23);
    assert.equal(privatePipeline(velocityB, "classifyPipeline"),
      privatePipeline(velocityA, "classifyPipeline"));
    assert.notEqual(velocityB.control, velocityA.control);

    const structuredPlan = planStructuredVelocityGPU(1);
    const structured = {
      plan: structuredPlan,
      params: buffer(256), authority: buffer(2 * structuredPlan.authorityBytes),
      control: buffer(128), candidateControl: buffer(128),
      rowVelocities: buffer(32), rowGeometry: buffer(32),
      authorityBankStrideWords: structuredPlan.authorityWords,
      rowBankStrideWords: 4, worksetBankStrideWords: structuredPlan.worksetBytes / 4,
      section63: { coefficientBankStrideWords: 19, coefficients: buffer(152) },
    };
    const boundaryResources = {
      structured,
      // `delta` is the coarse publisher's exact value/phase receipt, bound so
      // the boundary's exact row carry has a phi-side signal to gate on.
      coarse: { directory: buffer(64), rowCapacity: 1, delta: buffer(64) },
      separationMask: buffer(4), rigidBodies: buffer(12 * 8 * 16), bodyCount: 0,
      dimensions: [1, 1, 1] as const, physicalCellSize: 1, closedBoundaryMask: 0,
    };
    const bindGroupsBeforeBoundary = counts.bindGroups;
    const boundaryA = new WebGPUStructuredBoundaryCoefficients(
      device, boundaryResources as never);
    assert.equal(counts.pipelines, 36);
    const boundaryB = new WebGPUStructuredBoundaryCoefficients(
      device, boundaryResources as never);
    assert.equal(counts.pipelines, 36);
    assert.equal(privatePipeline(boundaryB, "accept"), privatePipeline(boundaryA, "accept"));
    assert.notEqual(boundaryB.control, boundaryA.control);
    assert.equal(counts.bindGroups - bindGroupsBeforeBoundary, 2,
      "each boundary instance must bind its own candidate and accepted controls");

    assert.equal(counts.shaderModules, 4);
    assert.equal(counts.bindGroupLayouts, 2,
      "explicit descriptor/topology layouts are shared with their pipelines");
    assert.equal(counts.pipelineLayouts, 2);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
    if (previousStage) Object.defineProperty(globalThis, "GPUShaderStage", previousStage);
    else Reflect.deleteProperty(globalThis, "GPUShaderStage");
  }
});
