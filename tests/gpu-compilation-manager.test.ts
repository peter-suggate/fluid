import assert from "node:assert/strict";
import test from "node:test";
import {
  defineGPUCompilationManifest,
  gpuCompilationManagerFor,
  GPUCompilationInvalidatedError,
  GPUCompilationManager,
  managedGPUDevice,
  type GPUCompilationManifest,
} from "../lib/gpu-compilation-manager";

interface FakeGPU {
  readonly device: GPUDevice;
  readonly calls: {
    modules: number;
    syncCompute: number;
    syncRender: number;
    asyncCompute: number;
    asyncRender: number;
    concurrent: number;
    maximumConcurrent: number;
    labels: string[];
  };
}

function fakeGPU(options: {
  beforePipeline?: (label: string) => Promise<void>;
} = {}): FakeGPU {
  const calls = {
    modules: 0,
    syncCompute: 0,
    syncRender: 0,
    asyncCompute: 0,
    asyncRender: 0,
    concurrent: 0,
    maximumConcurrent: 0,
    labels: [] as string[],
  };
  const compile = async (descriptor: GPUComputePipelineDescriptor | GPURenderPipelineDescriptor) => {
    const label = descriptor.label ?? "unlabelled";
    calls.labels.push(label);
    calls.concurrent += 1;
    calls.maximumConcurrent = Math.max(calls.maximumConcurrent, calls.concurrent);
    await options.beforePipeline?.(label);
    await Promise.resolve();
    calls.concurrent -= 1;
    return { label, getBindGroupLayout: () => ({}) };
  };
  const device = {
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) => {
      calls.modules += 1;
      return { label: descriptor.label, code: descriptor.code };
    },
    createComputePipeline: () => {
      calls.syncCompute += 1;
      throw new Error("synchronous compute compilation is forbidden");
    },
    createRenderPipeline: () => {
      calls.syncRender += 1;
      throw new Error("synchronous render compilation is forbidden");
    },
    createComputePipelineAsync: async (descriptor: GPUComputePipelineDescriptor) => {
      calls.asyncCompute += 1;
      return await compile(descriptor) as unknown as GPUComputePipeline;
    },
    createRenderPipelineAsync: async (descriptor: GPURenderPipelineDescriptor) => {
      calls.asyncRender += 1;
      return await compile(descriptor) as unknown as GPURenderPipeline;
    },
  } as unknown as GPUDevice;
  return { device, calls };
}

const representativeManifest = defineGPUCompilationManifest({
  id: "representative-family",
  revision: 1,
  label: "Representative family",
  modules: {
    compute: { label: "Compute module", source: "@compute @workgroup_size(1) fn main() {}" },
    raster: {
      label: "Raster module",
      source: "@vertex fn vertexMain()->@builtin(position) vec4f{return vec4f();}",
    },
  },
  compute: {
    prepare: {
      label: "Prepare pipeline",
      layout: "auto",
      compute: { module: "compute", entryPoint: "main" },
    },
    finish: {
      label: "Finish pipeline",
      layout: "auto",
      compute: { module: "compute", entryPoint: "main" },
    },
  },
  render: {
    present: {
      label: "Present pipeline",
      layout: "auto",
      vertex: { module: "raster", entryPoint: "vertexMain" },
      primitive: { topology: "triangle-list" },
    },
  },
});

function onePipelineManifest(
  id: string,
  label: string,
): GPUCompilationManifest<
  { shader: { source: string } },
  { main: { label: string; layout: "auto"; compute: { module: "shader"; entryPoint: string } } },
  Record<string, never>
> {
  return defineGPUCompilationManifest({
    id,
    revision: 1,
    modules: { shader: { source: "@compute @workgroup_size(1) fn main() {}" } },
    compute: {
      main: { label, layout: "auto", compute: { module: "shader", entryPoint: "main" } },
    },
    render: {},
  });
}

test("the compilation manager uses only async pipeline APIs and publishes a complete typed bundle", async () => {
  const gpu = fakeGPU();
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const snapshots: ReturnType<GPUCompilationManager["snapshot"]>[] = [];
  const unsubscribe = manager.subscribe((snapshot) => snapshots.push(snapshot));

  const pending = manager.acquire(representativeManifest, { priority: "critical" });
  assert.equal(gpu.calls.modules, 0, "acquire must return before even module creation starts");
  assert.equal(gpu.calls.asyncCompute, 0);

  const bundle = await pending;
  unsubscribe();

  assert.equal(gpu.calls.modules, 2);
  assert.equal(gpu.calls.syncCompute, 0);
  assert.equal(gpu.calls.syncRender, 0);
  assert.equal(gpu.calls.asyncCompute, 2);
  assert.equal(gpu.calls.asyncRender, 1);
  assert.equal(gpu.calls.maximumConcurrent, 1, "one family must compile sequentially");
  assert.equal(bundle.compute.prepare.label, "Prepare pipeline");
  assert.equal(bundle.compute.finish.label, "Finish pipeline");
  assert.equal(bundle.render.present.label, "Present pipeline");
  assert.equal(manager.ready(representativeManifest), bundle);
  assert.equal(manager.snapshot().cached, 1);
  assert.ok(snapshots.some(({ state, progress }) =>
    state === "compiling" && progress?.current === "Compile compute pipeline: prepare"));
});

test("completed and in-flight bundles are shared by manifest identity", async () => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const gpu = fakeGPU({ beforePipeline: () => gate });
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const manifest = onePipelineManifest("shared", "Shared pipeline");

  const first = manager.acquire(manifest);
  const second = manager.acquire(manifest);
  await Promise.resolve();
  assert.equal(gpu.calls.asyncCompute, 1);
  release();
  const [firstBundle, secondBundle] = await Promise.all([first, second]);
  assert.equal(firstBundle, secondBundle);

  const cached = await manager.acquire(manifest);
  assert.equal(cached, firstBundle);
  assert.equal(gpu.calls.modules, 1);
  assert.equal(gpu.calls.asyncCompute, 1);
});

test("queued bundles are priority ordered and driver pressure is bounded", async () => {
  const gpu = fakeGPU();
  const manager = new GPUCompilationManager(gpu.device, {
    requireWorkerRealm: false,
    maximumConcurrentBundles: 1,
  });
  const background = manager.acquire(onePipelineManifest("background", "Background"), {
    priority: "background",
  });
  const visible = manager.acquire(onePipelineManifest("visible", "Visible"), {
    priority: "visible",
  });
  const critical = manager.acquire(onePipelineManifest("critical", "Critical"), {
    priority: "critical",
  });

  await Promise.all([background, visible, critical]);
  assert.deepEqual(gpu.calls.labels, ["Critical", "Visible", "Background"]);
  assert.equal(gpu.calls.maximumConcurrent, 1);
});

test("a later critical consumer promotes shared queued work", async () => {
  const gpu = fakeGPU();
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const sharedManifest = onePipelineManifest("promoted", "Promoted");
  const sharedBackground = manager.acquire(sharedManifest, { priority: "background" });
  const visible = manager.acquire(onePipelineManifest("promotion-visible", "Visible"), {
    priority: "visible",
  });
  const sharedCritical = manager.acquire(sharedManifest, { priority: "critical" });

  const [backgroundBundle, , criticalBundle] = await Promise.all([
    sharedBackground,
    visible,
    sharedCritical,
  ]);
  assert.deepEqual(gpu.calls.labels, ["Promoted", "Visible"]);
  assert.equal(backgroundBundle, criticalBundle);
});

test("aborting one caller does not cancel shared compilation", async () => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const gpu = fakeGPU({ beforePipeline: () => gate });
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const manifest = onePipelineManifest("caller-cancellation", "Caller cancellation");
  const controller = new AbortController();
  const cancelled = manager.acquire(manifest, { signal: controller.signal });
  const retained = manager.acquire(manifest);

  controller.abort();
  await assert.rejects(cancelled, (error: unknown) =>
    error instanceof DOMException && error.name === "AbortError");
  release();
  const bundle = await retained;
  assert.equal(manager.ready(manifest), bundle);
  assert.equal(gpu.calls.asyncCompute, 1);
});

test("device invalidation rejects active work and permanently retires the manager", async () => {
  let release!: () => void;
  let started!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const pipelineStarted = new Promise<void>((resolve) => { started = resolve; });
  const gpu = fakeGPU({ beforePipeline: async () => { started(); await gate; } });
  const manager = new GPUCompilationManager(gpu.device, {
    requireWorkerRealm: false,
    generation: 7,
  });
  const manifest = onePipelineManifest("retired", "Retired");
  const pending = manager.acquire(manifest);
  await pipelineStarted;

  manager.invalidate("test device lost");
  release();
  await assert.rejects(pending, GPUCompilationInvalidatedError);
  assert.deepEqual(manager.snapshot(), {
    state: "invalidated",
    generation: 7,
    queued: 0,
    active: 0,
    cached: 0,
    invalidationReason: "test device lost",
  });
  await assert.rejects(() => manager.acquire(manifest), GPUCompilationInvalidatedError);
});

test("production construction fails closed outside a worker realm", () => {
  const gpu = fakeGPU();
  assert.throws(() => new GPUCompilationManager(gpu.device), /must be created in a worker realm/);
});

test("manifest definition rejects references to missing modules", () => {
  assert.throws(() => defineGPUCompilationManifest({
    id: "invalid",
    revision: 1,
    modules: {},
    compute: {
      main: {
        layout: "auto",
        compute: { module: "missing", entryPoint: "main" },
      },
    },
    render: {},
  } as never), /references missing module missing/);
});

test("managed devices disable immediate compilation and route async work through one authority", async () => {
  const gpu = fakeGPU();
  const device = managedGPUDevice(gpu.device, { requireWorkerRealm: false });

  assert.throws(() => device.createComputePipeline({} as GPUComputePipelineDescriptor),
    /createComputePipeline is disabled/);
  assert.throws(() => device.createRenderPipeline({} as GPURenderPipelineDescriptor),
    /createRenderPipeline is disabled/);

  const shaderModule = device.createShaderModule({ code: "@compute @workgroup_size(1) fn main() {}" });
  const pipeline = await device.createComputePipelineAsync({
    label: "managed direct compute", layout: "auto", compute: { module: shaderModule, entryPoint: "main" },
  });
  assert.equal(pipeline.label, "managed direct compute");
  assert.equal(gpu.calls.syncCompute, 0);
  assert.equal(gpu.calls.asyncCompute, 1);
  assert.equal(gpuCompilationManagerFor(device), gpuCompilationManagerFor(gpu.device));
});

test("managed direct compilations share the bounded priority scheduler", async () => {
  const gpu = fakeGPU();
  const manager = new GPUCompilationManager(gpu.device, {
    requireWorkerRealm: false,
    maximumConcurrentBundles: 1,
  });
  const shaderModule = manager.createShaderModule({ code: "@compute @workgroup_size(1) fn main() {}" });
  const background = manager.compileComputePipeline({
    label: "direct background", layout: "auto", compute: { module: shaderModule, entryPoint: "main" },
  }, { priority: "background" });
  const critical = manager.compileComputePipeline({
    label: "direct critical", layout: "auto", compute: { module: shaderModule, entryPoint: "main" },
  }, { priority: "critical" });

  await Promise.all([background, critical]);
  assert.deepEqual(gpu.calls.labels, ["direct critical", "direct background"]);
  assert.equal(gpu.calls.maximumConcurrent, 1);
});
