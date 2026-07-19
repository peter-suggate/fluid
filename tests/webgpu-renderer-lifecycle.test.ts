import assert from "node:assert/strict";
import test from "node:test";
import { canQueuePreparedGPUAdvance, FluidLabRenderer, presentationHasPhysicsSlack, presentationPhysicsQueueDepth, presentationPriorityDue, submitNextPreparedGPUAdvance, type GPUStatus } from "../lib/webgpu-renderer";

test("presentation takes queue priority once a 60 Hz deadline has elapsed", () => {
  assert.equal(presentationPriorityDue(-Infinity, 0), true);
  assert.equal(presentationPriorityDue(100, 108), false);
  assert.equal(presentationPriorityDue(100, 116.2), true);
});

test("physics admission preserves the measured presentation deadline", () => {
  assert.equal(presentationHasPhysicsSlack(-Infinity, 0, 2, 1), false);
  assert.equal(presentationHasPhysicsSlack(100, 105, 4, 2), true);
  assert.equal(presentationHasPhysicsSlack(100, 112, 4, 2), false);
  assert.equal(presentationHasPhysicsSlack(100, 105, 20, 2), false);
});

test("GPU submission advances only once toward prepared simulation debt", () => {
  let submittedTime_s = 0;
  let advances = 0;
  const fluid = {
    info: { submittedTime_s },
    advanceTo(this: { info: { submittedTime_s: number } }, time_s: number) {
      advances += 1;
      submittedTime_s = Math.min(time_s, submittedTime_s + 0.008);
      this.info.submittedTime_s = submittedTime_s;
      return true;
    }
  } as unknown as Parameters<typeof submitNextPreparedGPUAdvance>[0];

  const result = submitNextPreparedGPUAdvance(fluid, 0.1, []);
  assert.equal(result.previousSubmittedTime, 0);
  assert.equal(result.submittedTime, 0.008);
  assert.equal(advances, 1);
});

test("GPU queue stays dense around presentation without admitting a physics burst", () => {
  assert.equal(presentationPhysicsQueueDepth(undefined, 1), 1);
  assert.equal(presentationPhysicsQueueDepth(35, 1), 1);
  assert.equal(presentationPhysicsQueueDepth(3.4, 1), 5, "one whole advance may overshoot the remaining budget");
  assert.equal(canQueuePreparedGPUAdvance(0, 4), true);
  assert.equal(canQueuePreparedGPUAdvance(3, 4), true);
  assert.equal(canQueuePreparedGPUAdvance(4, 4), false);
});

test("renderer stops submitting frames and disposes its device after WebGPU loss", async (t) => {
  let resolveDeviceLost!: (info: GPUDeviceLostInfo) => void;
  const lost = new Promise<GPUDeviceLostInfo>((resolve) => { resolveDeviceLost = resolve; });
  let deviceDestroyCount = 0;
  let submitCount = 0;
  let requestedDescriptor: GPUDeviceDescriptor | undefined;
  const destroyable = () => ({ destroy() {} });
  const texture = () => ({ ...destroyable(), width: 1, height: 1, createView: () => ({}) });
  const pipeline = () => ({ getBindGroupLayout: () => ({}) });
  const device = {
    features: new Set<GPUFeatureName>(),
    lost,
    addEventListener() {},
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createRenderPipeline: pipeline,
    createRenderPipelineAsync: async () => pipeline(),
    createComputePipeline: () => ({}),
    createComputePipelineAsync: async () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createSampler: () => ({}),
    createBuffer: destroyable,
    createTexture: texture,
    createBindGroup: () => ({}),
    queue: { submit: () => { submitCount += 1; } },
    destroy: () => { deviceDestroyCount += 1; }
  } as unknown as GPUDevice;
  const adapter = {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxStorageBuffersPerShaderStage: 10,
      maxStorageBufferBindingSize: 512 * 1024 * 1024,
      maxBufferSize: 1024 * 1024 * 1024,
      maxTextureDimension3D: 2048,
    },
    requestDevice: async (descriptor: GPUDeviceDescriptor) => { requestedDescriptor = descriptor; return device; },
    info: { vendor: "test" }
  } as unknown as GPUAdapter;
  const context = { configure() {} } as unknown as GPUCanvasContext;
  const canvas = { getContext: () => context } as unknown as HTMLCanvasElement;
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const previousBufferUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousTextureUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUTextureUsage");
  const previousShaderStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { requestAdapter: async () => adapter, getPreferredCanvasFormat: () => "bgra8unorm" } } });
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, QUERY_RESOLVE: 8, COPY_SRC: 16, INDIRECT: 32 } });
  Object.defineProperty(globalThis, "GPUTextureUsage", { configurable: true, value: { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } });
  t.after(() => {
    for (const [name, descriptor] of [["navigator", previousNavigator], ["GPUBufferUsage", previousBufferUsage], ["GPUTextureUsage", previousTextureUsage], ["GPUShaderStage", previousShaderStage]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });

  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer(canvas, (status) => statuses.push(status));
  await renderer.initialize();
  assert.equal(statuses.at(-1)?.state, "ready");
  assert.deepEqual(requestedDescriptor?.requiredLimits, {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 512 * 1024 * 1024,
    maxBufferSize: 1024 * 1024 * 1024,
    maxTextureDimension3D: 2048,
  });

  resolveDeviceLost({ reason: "unknown", message: "test device loss" } as GPUDeviceLostInfo);
  await lost;
  await Promise.resolve();
  assert.deepEqual(statuses.at(-1), { state: "lost", label: "GPU device lost: test device loss" });

  const metrics = renderer.draw(0, {} as never, {} as never, [], undefined, undefined, "webgpu", { methodId: "tall-cell", quality: "balanced", values: {} });
  assert.deepEqual(metrics, { cpuFrame_ms: 0, cpuPhysicsSubmit_ms: 0, cpuDataUpload_ms: 0, cpuRenderEncode_ms: 0 });
  assert.equal(submitCount, 0, "a lost device must never receive another queue submission");

  renderer.destroy();
  renderer.destroy();
  assert.equal(deviceDestroyCount, 1, "renderer cleanup must be idempotent across hot reload");
});
