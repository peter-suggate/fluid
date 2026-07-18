import assert from "node:assert/strict";
import test from "node:test";
import {
  SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE,
  SPARSE_VOXEL_DEBUG_RECORD_STRIDE,
  SparseVoxelDebugRenderer,
  voxelDebugComputeShader,
  voxelDebugPlan,
  voxelDebugRenderShader
} from "../lib/webgpu-voxel-debug";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";

test("public sparse voxel render source is structural and exposes only inspection modes", () => {
  const external = {} as GPUBuffer;
  const binding = { buffer: external };
  const source = {
    voxelRecords: binding, voxelCount: binding, brickRecords: binding, brickCount: binding, materials: binding,
    voxelCapacity: 64, brickCapacity: 8, materialCount: 3, revision: 7
  } satisfies SparseVoxelRenderSource;
  const modes = ["raw-voxels", "brick-grid"] as const;
  assert.deepEqual(modes.map((mode) => voxelDebugPlan(mode, source).recordKind), ["voxels", "bricks"]);
});

test("voxel inspection plans raw voxels and brick grids from one source", () => {
  const source = { voxelCapacity: 129, brickCapacity: 65 };
  assert.deepEqual(voxelDebugPlan("raw-voxels", source), {
    enabled: true, recordKind: "voxels", capacity: 129, computeWorkgroups: 3, verticesPerInstance: 36, topology: "triangle-list"
  });
  assert.deepEqual(voxelDebugPlan("brick-grid", source), {
    enabled: true, recordKind: "bricks", capacity: 65, computeWorkgroups: 2, verticesPerInstance: 24, topology: "line-list"
  });
  assert.equal(voxelDebugPlan("brick-grid", { voxelCapacity: 1, brickCapacity: 0 }).enabled, false);
});

test("voxel debug ABI and shaders retain GPU material color and indirect instance production", () => {
  assert.equal(SPARSE_VOXEL_DEBUG_RECORD_STRIDE, 48);
  assert.equal(SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE, 32);
  assert.match(voxelDebugComputeShader, /atomicAdd\(&drawArguments\.instanceCount/);
  assert.match(voxelDebugComputeShader, /drawArguments\.vertexCount = 36u/);
  assert.match(voxelDebugComputeShader, /drawArguments\.vertexCount = 24u/);
  assert.match(voxelDebugComputeShader, /materialAndFlags\.y & ACTIVE/);
  assert.match(voxelDebugComputeShader, /compactSettings\.capacity/);
  assert.match(voxelDebugRenderShader, /let material = materials\[/);
  assert.match(voxelDebugRenderShader, /material\.baseColor\.a <= 0\.001\) \{ discard; \}/);
  assert.match(voxelDebugRenderShader, /shadeUnifiedSurface\(closure, lighting\)/);
  assert.match(voxelDebugRenderShader, /input\.level & 1u/);
  assert.match(voxelDebugRenderShader, /array<vec3f, 24>/);
  assert.doesNotMatch(voxelDebugComputeShader + voxelDebugRenderShader, /textureLoad|mapAsync|readBuffer/);
});

test("voxel debug rendering uses indirect draws and destroys only owned buffers once", async (t) => {
  const previousBufferUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousShaderStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, INDIRECT: 8 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } });
  t.after(() => {
    for (const [name, descriptor] of [["GPUBufferUsage", previousBufferUsage], ["GPUShaderStage", previousShaderStage]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });

  const destroyed: string[] = [];
  const writes: unknown[] = [];
  const renderDescriptors: GPURenderPassDescriptor[] = [];
  let indirectDraws = 0;
  const paneDraws: unknown[][] = [];
  const pipeline = {} as GPUComputePipeline & GPURenderPipeline;
  const device = {
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipelineAsync: async () => pipeline,
    createRenderPipelineAsync: async () => pipeline,
    createBuffer: ({ label }: GPUBufferDescriptor) => ({ destroy: () => destroyed.push(label ?? "unlabelled") }),
    createBindGroup: () => ({}),
    queue: { writeBuffer: (...args: unknown[]) => writes.push(args) }
  } as unknown as GPUDevice;
  const computePass = { setBindGroup() {}, setPipeline() {}, dispatchWorkgroups() {}, end() {} };
  const renderPass = { setBindGroup() {}, setPipeline() {}, draw: (...args: unknown[]) => { paneDraws.push(args); }, drawIndirect: () => { indirectDraws += 1; }, end() {} };
  const encoder = {
    beginComputePass: () => computePass,
    beginRenderPass: (descriptor: GPURenderPassDescriptor) => { renderDescriptors.push(descriptor); return renderPass; }
  } as unknown as GPUCommandEncoder;
  let externalDestroyCount = 0;
  const external = { destroy: () => { externalDestroyCount += 1; } } as unknown as GPUBuffer;
  const binding = { buffer: external };
  const renderer = new SparseVoxelDebugRenderer(device, { colorFormat: "rgba8unorm" });
  await renderer.initialize();
  renderer.setSource({
    voxelRecords: binding, voxelCount: binding, brickRecords: binding, brickCount: binding, materials: binding,
    voxelCapacity: 80, brickCapacity: 20, materialCount: 2, revision: 1
  });
  const common = {
    colorTarget: {} as GPUTextureView, depthTarget: {} as GPUTextureView,
    viewProjection: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    cameraPosition: [0, 0, 4] as const,
    containerBounds: { min: [-1, 0, -1] as const, max: [1, 2, 1] as const },
    containerClosedTop: false
  };
  assert.equal(renderer.encode(encoder, { ...common, mode: "raw-voxels", depthLoadOp: "clear", colorLoadOp: "clear" }), true);
  assert.equal(renderer.encode(encoder, { ...common, mode: "brick-grid" }), true);
  assert.equal(indirectDraws, 2);
  assert.deepEqual(paneDraws, [
    [6, 1, 0, 3], [6, 1, 0, 0], [6, 1, 0, 1], [6, 1, 0, 2], [6, 1, 0, 4]
  ], "open tank panes draw back-to-front after opaque voxels");
  assert.equal(renderDescriptors[0].depthStencilAttachment?.depthClearValue, 1);
  const firstColorAttachment = Array.from(renderDescriptors[0].colorAttachments)[0];
  assert.equal(firstColorAttachment?.loadOp, "clear");
  assert.deepEqual(firstColorAttachment?.clearValue, { r: 0.008, g: 0.012, b: 0.018, a: 1 });
  assert.equal(writes.length, 4, "each voxel view uploads only view and declared capacity");

  renderer.destroy();
  renderer.destroy();
  assert.equal(externalDestroyCount, 0, "source buffers remain owned by the sparse representation");
  assert.deepEqual(destroyed.sort(), [
    "Sparse voxel debug compaction settings",
    "Sparse voxel debug indirect draw",
    "Sparse voxel debug instances (80)",
    "Sparse voxel debug view"
  ]);
});
