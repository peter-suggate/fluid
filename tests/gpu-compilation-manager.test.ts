import assert from "node:assert/strict";
import test from "node:test";
import {
  defineGPUCompilationManifest,
  gpuCompilationManagerFor,
  GPUCompilationInvalidatedError,
  GPUCompilationManager,
  managedGPUDevice,
  type GPUCompilationManifest,
} from "../lib/core/gpu-compilation-manager";

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
    createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ ...descriptor }),
    createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ ...descriptor }),
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
    modules: { shader: { source: `// ${id}\n@compute @workgroup_size(1) fn main() {}` } },
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
  assert.equal(gpu.calls.asyncCompute, 1);
  assert.equal(gpu.calls.asyncRender, 1);
  assert.equal(gpu.calls.maximumConcurrent, 1, "one family must compile sequentially");
  assert.equal(bundle.compute.prepare.label, "Prepare pipeline");
  assert.equal(bundle.compute.finish, bundle.compute.prepare, "labels do not split identical pipelines");
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
    cacheStats: manager.snapshot().cacheStats,
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
    label: "direct critical", layout: "auto", compute: { module: shaderModule, entryPoint: "another" },
  }, { priority: "critical" });

  await Promise.all([background, critical]);
  assert.deepEqual(gpu.calls.labels, ["direct critical", "direct background"]);
  assert.equal(gpu.calls.maximumConcurrent, 1);
});

// Managed device entry must use the same bounded scheduler as direct lookup.
// Otherwise every solver shader is serialized despite the production policy.
test("managed devices retain production compilation concurrency and explicit overrides",async()=>{
 const raw=fakeGPU(),wrapped=fakeGPU(),serial=fakeGPU();
 const device=managedGPUDevice(wrapped.device,{requireWorkerRealm:false});
 const serialDevice=managedGPUDevice(serial.device,{requireWorkerRealm:false,maximumConcurrentBundles:1});
 const build=async(device:GPUDevice)=>{
  const manager=gpuCompilationManagerFor(device);
  const module=manager.createShaderModule({code:"@compute @workgroup_size(1) fn main(){}"});
  await Promise.all(Array.from({length:6},(_,i)=>manager.compileComputePipeline({layout:"auto",compute:{module,entryPoint:"main",constants:{SLOT:i}}})));
 };
 await build(raw.device);await build(device);await build(serialDevice);
 assert.equal(wrapped.calls.maximumConcurrent,raw.calls.maximumConcurrent);
 assert.equal(serial.calls.maximumConcurrent,1);
});

test("cancelling a scene removes queued direct pipelines but drains native work", async () => {
  let release!: () => void;
  let started!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const running = new Promise<void>(resolve => { started = resolve; });
  const gpu = fakeGPU({ beforePipeline: async () => { started(); await gate; } });
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const controller = new AbortController();
  const compile = (label: string, signal?: AbortSignal) => manager.compileComputePipeline({
    label, layout: "auto", compute: { module: {} as GPUShaderModule, entryPoint: "main" },
  }, { signal });
  let settled = false;
  const active = assert.rejects(compile("active", controller.signal), { name: "AbortError" }).then(() => { settled = true; });
  await running;
  const queued = assert.rejects(compile("abandoned", controller.signal), { name: "AbortError" });
  const replacement = compile("replacement");
  controller.abort();
  await queued;
  assert.equal(settled, false, "cannot clean up resources under a native compilation");
  assert.equal(manager.snapshot().queued, 1);
  release();
  await Promise.all([active, replacement, manager.whenIdle()]);
  assert.deepEqual(gpu.calls.labels, ["active", "replacement"]);
});

test("new scene instances reuse modules, structural layouts and complete pipelines", async () => {
  const gpu = fakeGPU();
  const device = managedGPUDevice(gpu.device, { requireWorkerRealm: false });
  const manager = gpuCompilationManagerFor(device);
  const build = (label: string, explicitDefaults: boolean) => {
    const module = device.createShaderModule({ label, code: "@compute @workgroup_size(1) fn main() {}" });
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: 1, visibility: 4, buffer: { type: "storage" } },
      { binding: 0, visibility: 4, buffer: explicitDefaults ? { type: "uniform", hasDynamicOffset: false, minBindingSize: 0 } : {} },
    ];
    const bindLayout = device.createBindGroupLayout({ label, entries: explicitDefaults ? entries.reverse() : entries });
    const layout = device.createPipelineLayout({ label, bindGroupLayouts: [bindLayout] });
    return device.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: "main" } });
  };
  const [first, concurrent] = await Promise.all([build("scene A", false), build("scene B", true)]);
  const warm = await build("scene C", false);
  assert.equal(first, concurrent);
  assert.equal(first, warm);
  assert.equal(gpu.calls.modules, 1);
  assert.equal(gpu.calls.asyncCompute, 1);
  assert.equal(manager.snapshot().cacheStats.inFlightHits, 1);
  assert.equal(manager.snapshot().cacheStats.pipelineHits, 1);
  assert.equal(manager.snapshot().cacheStats.bindGroupLayouts, 1);
  assert.equal(manager.snapshot().cacheStats.pipelineLayouts, 1);
});

test("manifest and direct pipelines share one cache and one scheduler slot without deadlock", async () => {
  const gpu = fakeGPU();
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false, maximumConcurrentBundles: 1 });
  const module = manager.createShaderModule({ code: representativeManifest.modules.compute.source });
  const direct = manager.compileComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const manifest = manager.acquire(representativeManifest);
  const [pipeline, bundle] = await Promise.all([direct, manifest, manager.whenIdle()]);
  assert.equal(pipeline, bundle.compute.prepare);
  assert.equal(gpu.calls.asyncCompute, 1);
  assert.equal(gpu.calls.maximumConcurrent, 1);
  assert.equal(manager.snapshot().state, "idle");
});

test("pipeline keys preserve constants, entry points, shader source and layout differences", async () => {
  const gpu = fakeGPU();
  const device = managedGPUDevice(gpu.device, { requireWorkerRealm: false });
  const module = device.createShaderModule({ code: "source A" });
  const build = (compute: GPUProgrammableStage, layout: GPUPipelineLayout | "auto" = "auto") =>
    device.createComputePipelineAsync({ layout, compute });
  const first = await build({ module, entryPoint: "main", constants: { label: 1, b: 2 } });
  assert.equal(await build({ module, entryPoint: "main", constants: { b: 2, label: 1 } }), first);
  const variants = await Promise.all([
    build({ module, entryPoint: "main", constants: { label: 2, b: 2 } }),
    build({ module, entryPoint: "other", constants: { label: 1, b: 2 } }),
    build({ module: device.createShaderModule({ code: "source B" }), entryPoint: "main", constants: { label: 1, b: 2 } }),
    build({ module, entryPoint: "main", constants: { label: 1, b: 2 } }, device.createPipelineLayout({ bindGroupLayouts: [] })),
    build({ module: {} as GPUShaderModule, entryPoint: "main" }),
    build({ module: {} as GPUShaderModule, entryPoint: "main" }),
  ]);
  assert.equal(new Set([first, ...variants]).size, 7);
  assert.equal(gpu.calls.asyncCompute, 7);
});

test("render pipeline keys retain fixed-function state and consume one-shot sequences at request time", async () => {
  const seen: GPURenderPipelineDescriptor[] = [];
  const gpu = fakeGPU();
  const original = gpu.device.createRenderPipelineAsync;
  gpu.device.createRenderPipelineAsync = descriptor => { seen.push(descriptor); return original(descriptor); };
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const module = manager.createShaderModule({ code: "raster" });
  const constants = { x: 1 };
  const buffers: GPUVertexBufferLayout[] = [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }];
  const descriptor: GPURenderPipelineDescriptor = { layout: "auto",
    vertex: { module, entryPoint: "vs", constants, buffers: buffers.values() },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" }, multisample: { count: 1 },
  };
  const pending = manager.compileRenderPipeline(descriptor);
  constants.x = 2; buffers[0].arrayStride = 16;
  const first = await pending;
  assert.equal(seen[0].vertex.constants!.x, 1);
  assert.equal(Array.from(seen[0].vertex.buffers!)[0]!.arrayStride, 8);
  assert.equal(await manager.compileRenderPipeline({ ...seen[0], label: "another scene" }), first);
  const variants = await Promise.all([
    manager.compileRenderPipeline({ ...seen[0], primitive: { topology: "line-list" } }),
    manager.compileRenderPipeline({ ...seen[0], multisample: { count: 4 } }),
    manager.compileRenderPipeline({ ...seen[0], fragment: { module, entryPoint: "fs", targets: [{ format: "bgra8unorm" }] } }),
    manager.compileRenderPipeline({ ...seen[0], depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" } }),
  ]);
  assert.equal(new Set([first, ...variants]).size, 5);
});

test("cancelling a queued shared caller preserves other callers and priority promotion", async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const gpu = fakeGPU({ beforePipeline: () => gate });
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const module = manager.createShaderModule({ code: "shader" });
  const build = (entryPoint: string, signal?: AbortSignal, priority: "critical" | "visible" | "background" = "visible") =>
    manager.compileComputePipeline({ label: entryPoint, layout: "auto", compute: { module, entryPoint } }, { signal, priority });
  const active = build("active");
  await Promise.resolve();
  const controller = new AbortController();
  const abandoned = assert.rejects(build("shared", controller.signal, "background"), { name: "AbortError" });
  const visible = build("visible");
  const retained = build("shared", undefined, "critical");
  controller.abort();
  await abandoned;
  assert.equal(manager.snapshot().queued, 2);
  release();
  await Promise.all([active, visible, retained]);
  assert.deepEqual(gpu.calls.labels, ["active", "shared", "visible"]);
});

test("an aborted active consumer drains while a replacement shares and retains its pipeline", async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const gpu = fakeGPU({ beforePipeline: () => gate });
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const module = manager.createShaderModule({ code: "shader" });
  const descriptor: GPUComputePipelineDescriptor = { layout: "auto", compute: { module } };
  const controller = new AbortController();
  let settled = false;
  const aborted = assert.rejects(manager.compileComputePipeline(descriptor, { signal: controller.signal }), { name: "AbortError" }).then(() => { settled = true; });
  await Promise.resolve();
  controller.abort();
  const replacement = manager.compileComputePipeline(descriptor);
  await Promise.resolve();
  assert.equal(settled, false);
  release();
  const [, pipeline] = await Promise.all([aborted, replacement]);
  assert.equal(await manager.compileComputePipeline(descriptor), pipeline);
  assert.equal(gpu.calls.asyncCompute, 1);
});

test("failed compilation is evicted so the same request can be retried", async () => {
  let fail = true;
  const gpu = fakeGPU({ beforePipeline: async () => { if (fail) throw new Error("compiler failed"); } });
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const module = manager.createShaderModule({ code: "shader" });
  const descriptor: GPUComputePipelineDescriptor = { layout: "auto", compute: { module } };
  await assert.rejects(manager.compileComputePipeline(descriptor), /compiler failed/);
  assert.equal(manager.snapshot().cacheStats.pipelines, 0);
  fail = false;
  await manager.compileComputePipeline(descriptor);
  assert.equal(gpu.calls.asyncCompute, 2);
});

test("raw, managed and repeatedly wrapped devices keep one authority; devices never share caches", async () => {
  const gpu = fakeGPU();
  const original = gpuCompilationManagerFor(gpu.device);
  const managed = managedGPUDevice(gpu.device, { requireWorkerRealm: false });
  assert.equal(gpuCompilationManagerFor(managed), original);
  assert.equal(managedGPUDevice(managed), managed);
  assert.equal(managedGPUDevice(gpu.device), managed);
  const second = new GPUCompilationManager(fakeGPU().device, { requireWorkerRealm: false });
  assert.notEqual(original.createShaderModule({ code: "same" }), second.createShaderModule({ code: "same" }));
  original.invalidate();
  const stats = original.snapshot().cacheStats;
  assert.equal(stats.modules + stats.bindGroupLayouts + stats.pipelineLayouts + stats.pipelines, 0);
  assert.throws(() => original.createShaderModule({ code: "same" }), GPUCompilationInvalidatedError);
});

test("signed zero specialization values stay distinct", async () => {
  const gpu = fakeGPU();
  const manager = new GPUCompilationManager(gpu.device, { requireWorkerRealm: false });
  const module = manager.createShaderModule({ code: "shader" });
  const compile = (value: number) => manager.compileComputePipeline({ layout: "auto", compute: { module, constants: { value } } });
  assert.notEqual(await compile(0), await compile(-0));
});

test("harness method decorators observe managed calls without replacing native methods", async () => {
  const gpu = fakeGPU();
  const device = managedGPUDevice(gpu.device, { requireWorkerRealm: false });
  const native = gpu.device.createShaderModule;
  const original = device.createShaderModule.bind(device);
  let observations = 0;
  device.createShaderModule = descriptor => { observations++; return original(descriptor); };
  const module = device.createShaderModule({ code: "shader" });
  assert.equal(device.createShaderModule({ code: "shader" }), module);
  assert.equal(observations, 2);
  assert.equal(gpu.calls.modules, 1);
  assert.equal(gpu.device.createShaderModule, native);
});

test("device loss clears every cached resource and drains an active native call", async () => {
  let lose!: (info: GPUDeviceLostInfo) => void;
  let release!: () => void;
  const lost = new Promise<GPUDeviceLostInfo>(resolve => { lose = resolve; });
  const gate = new Promise<void>(resolve => { release = resolve; });
  const gpu = fakeGPU({ beforePipeline: () => gate });
  Object.defineProperty(gpu.device, "lost", { value: lost });
  const device = managedGPUDevice(gpu.device, { requireWorkerRealm: false });
  const manager = gpuCompilationManagerFor(device);
  const module = device.createShaderModule({ code: "shader" });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries: [] })] });
  const pending = assert.rejects(device.createComputePipelineAsync({ layout, compute: { module } }), GPUCompilationInvalidatedError);
  await Promise.resolve();
  lose({ reason: "unknown", message: "lost during compilation" } as GPUDeviceLostInfo);
  await Promise.resolve();
  assert.equal(manager.snapshot().state, "invalidated");
  assert.equal(manager.snapshot().active, 1);
  const stats = manager.snapshot().cacheStats;
  assert.equal(stats.modules + stats.bindGroupLayouts + stats.pipelineLayouts + stats.pipelines, 0);
  release();
  await Promise.all([pending, manager.whenIdle()]);
  assert.equal(manager.snapshot().active, 0);
});
