import assert from "node:assert/strict";
import test from "node:test";
import { FluidLabRenderer, type GPUStatus } from "../lib/webgpu-renderer";

test("renderer stops submitting frames and disposes its device after WebGPU loss", async (t) => {
  let resolveDeviceLost!: (info: GPUDeviceLostInfo) => void;
  const lost = new Promise<GPUDeviceLostInfo>((resolve) => { resolveDeviceLost = resolve; });
  let deviceDestroyCount = 0;
  let submitCount = 0;
  const destroyable = () => ({ destroy() {} });
  const texture = () => ({ ...destroyable(), width: 1, height: 1, createView: () => ({}) });
  const pipeline = () => ({ getBindGroupLayout: () => ({}) });
  const device = {
    features: new Set<GPUFeatureName>(),
    lost,
    addEventListener() {},
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createRenderPipeline: pipeline,
    createSampler: () => ({}),
    createBuffer: destroyable,
    createTexture: texture,
    createBindGroup: () => ({}),
    queue: { submit: () => { submitCount += 1; } },
    destroy: () => { deviceDestroyCount += 1; }
  } as unknown as GPUDevice;
  const adapter = {
    features: new Set<GPUFeatureName>(),
    requestDevice: async () => device,
    info: { vendor: "test" }
  } as unknown as GPUAdapter;
  const context = { configure() {} } as unknown as GPUCanvasContext;
  const canvas = { getContext: () => context } as unknown as HTMLCanvasElement;
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const previousBufferUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousTextureUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUTextureUsage");
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { requestAdapter: async () => adapter, getPreferredCanvasFormat: () => "bgra8unorm" } } });
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, QUERY_RESOLVE: 8, COPY_SRC: 16 } });
  Object.defineProperty(globalThis, "GPUTextureUsage", { configurable: true, value: { TEXTURE_BINDING: 1, COPY_DST: 2 } });
  t.after(() => {
    for (const [name, descriptor] of [["navigator", previousNavigator], ["GPUBufferUsage", previousBufferUsage], ["GPUTextureUsage", previousTextureUsage]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });

  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer(canvas, (status) => statuses.push(status));
  await renderer.initialize();
  assert.equal(statuses.at(-1)?.state, "ready");

  resolveDeviceLost({ reason: "unknown", message: "test device loss" } as GPUDeviceLostInfo);
  await lost;
  await Promise.resolve();
  assert.deepEqual(statuses.at(-1), { state: "lost", label: "GPU device lost: test device loss" });

  const metrics = renderer.draw(0, {} as never, {} as never, "scientific", [], undefined, undefined, "webgpu", { methodId: "tall-cell", quality: "medium", values: {} });
  assert.deepEqual(metrics, { cpuFrame_ms: 0, cpuPhysicsSubmit_ms: 0, cpuDataUpload_ms: 0, cpuRenderEncode_ms: 0 });
  assert.equal(submitCount, 0, "a lost device must never receive another queue submission");

  renderer.destroy();
  renderer.destroy();
  assert.equal(deviceDestroyCount, 1, "renderer cleanup must be idempotent across hot reload");
});
