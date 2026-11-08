import assert from "node:assert/strict";
import test from "node:test";
import { GPUBackgroundPipelines, joinPendingPipelines } from "../lib/core/gpu-background-pipelines";
import { gpuCompilationManagerFor, holdGPUBackgroundCompilation, managedGPUDevice, whenGPUCompilationIdle } from "../lib/core/gpu-compilation-manager";

/** A raw device whose pipeline compiles finish only when the test says so. */
function fakeCompiler() {
  const started: string[] = [];
  const pending = new Map<string, { resolve(): void; reject(error: Error): void }>();
  const raw = {
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({ label: descriptor.label, code: descriptor.code }),
    createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ ...descriptor }),
    createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ ...descriptor }),
    createComputePipelineAsync: (descriptor: GPUComputePipelineDescriptor) => {
      const label = descriptor.label ?? "unlabelled";
      started.push(label);
      return new Promise<GPUComputePipeline>((resolve, reject) => {
        pending.set(label, { resolve: () => resolve({ label } as unknown as GPUComputePipeline), reject });
      });
    },
  } as unknown as GPUDevice;
  const device = managedGPUDevice(raw, { requireWorkerRealm: false, maximumConcurrentBundles: 1 });
  const module = device.createShaderModule({ label: "stages", code: "shader" });
  const request = (label: string) => device.createComputePipelineAsync({ label, layout: "auto", compute: { module, entryPoint: label } });
  const settle = async () => { for (let i = 0; i < 4; i++) await new Promise(resolve => setImmediate(resolve)); };
  const finish = async (label: string) => { const job = pending.get(label); assert.ok(job, `${label} was not started`); pending.delete(label); job.resolve(); await settle(); };
  const fail = async (label: string) => { const job = pending.get(label)!; pending.delete(label); job.reject(new Error(`${label}: driver failure`)); await settle(); };
  return { device, manager: gpuCompilationManagerFor(device), request, started, settle, finish, fail };
}

test("background requests wait behind construction's hold and every critical request", async t => {
  const gpu = fakeCompiler();
  const release = holdGPUBackgroundCompilation(); t.after(release);
  const background = new GPUBackgroundPipelines();
  background.add("air", () => gpu.request("air"));
  const critical = [gpu.request("move"), gpu.request("transfer")];
  await gpu.settle();
  assert.deepEqual(gpu.started, ["move"]);
  await gpu.finish("move");
  await gpu.finish("transfer");
  await Promise.all(critical);
  // Construction is still running: its slot stays free for a late critical request.
  assert.deepEqual(gpu.started, ["move", "transfer"]);
  assert.equal(gpu.manager.snapshot().batch?.total, 2, "a held background job is not counted in construction's progress");
  const late = gpu.request("late");
  await gpu.settle();
  assert.deepEqual(gpu.started, ["move", "transfer", "late"]);
  await gpu.finish("late");
  await late;
  release();
  await gpu.settle();
  assert.deepEqual(gpu.started, ["move", "transfer", "late", "air"]);
  await gpu.finish("air");
  assert.ok(background.allReady);
});

test("a device seen through a forwarding wrapper still compiles at background priority, held and raised", async t => {
  const gpu = fakeCompiler();
  // A tool's Proxy (a kernel timer, an allocation audit): not registered with the manager.
  const wrapped = new Proxy(gpu.device, { get(target, key) {
    if (key === "createComputePipelineAsync") return async (descriptor: GPUComputePipelineDescriptor) => await target.createComputePipelineAsync(descriptor);
    const value = Reflect.get(target, key) as unknown;
    return typeof value === "function" ? value.bind(target) : value;
  } });
  const module = wrapped.createShaderModule({ label: "wrapped", code: "shader" });
  const release = holdGPUBackgroundCompilation(); t.after(release);
  const background = new GPUBackgroundPipelines();
  background.add("air", () => wrapped.createComputePipelineAsync({ label: "air", layout: "auto", compute: { module, entryPoint: "air" } }));
  background.add("gate", () => wrapped.createComputePipelineAsync({ label: "gate", layout: "auto", compute: { module, entryPoint: "gate" } }));
  const critical = gpu.request("move");
  await gpu.settle();
  await gpu.finish("move");
  await critical;
  assert.deepEqual(gpu.started, ["move"], "held, though requested through the wrapper");
  const wait = background.require(["gate"]);
  assert.ok(wait);
  await gpu.settle();
  assert.deepEqual(gpu.started, ["move", "gate"], "raised to critical, so the hold no longer keeps it");
  await gpu.finish("gate");
  await wait;
  release();
  await gpu.settle();
  assert.deepEqual(gpu.started, ["move", "gate", "air"]);
  // A timing probe's wait for the background compiles, which reaches the manager past the wrapper.
  let idle = false;
  void whenGPUCompilationIdle().then(() => { idle = true; });
  await gpu.settle();
  assert.equal(idle, false, "a background compile is still running");
  await gpu.finish("air");
  assert.ok(background.allReady);
  assert.equal(idle, true);
});

test("a frame waits for a pending background pipeline, raised to critical, before its first queue write", async () => {
  const gpu = fakeCompiler();
  const background = new GPUBackgroundPipelines();
  // The owner: one stage pipeline only some frames bind (here, every frame
  // with `traces` set), requested at background priority at initialize.
  background.add("diagnose", () => gpu.request("diagnose"));
  let traces = false;
  const pipelinesPending = () => background.allReady ? undefined : background.require(traces ? ["diagnose"] : []);
  // The host: the reference's admission (advanceTo) and the frame's encode.
  const queue: string[] = [];
  const advance = async (frame: number) => {
    const compiling = pipelinesPending();
    if (compiling) { queue.push(`frame ${frame} waits`); await compiling; }
    queue.push(`frame ${frame} queue write`);
    if (traces) queue.push(`frame ${frame} binds ${(background.get("diagnose") as unknown as { label: string }).label}`);
  };
  // Another owner's visible request is already queued behind the slot.
  const busy = gpu.request("busy");
  const visible = gpu.request("visible");
  await gpu.settle();
  assert.deepEqual(gpu.started, ["busy"]);
  // A frame that cannot bind it does not wait.
  await advance(1);
  assert.deepEqual(queue, ["frame 1 queue write"]);
  // A frame that may bind it waits, and its request jumps the visible one.
  traces = true;
  assert.throws(() => background.get("diagnose"), /still compiling/, "binding before the wait fails loudly");
  const frame2 = advance(2);
  await gpu.settle();
  assert.deepEqual(queue, ["frame 1 queue write", "frame 2 waits"], "no queue write while the pipeline compiles");
  await gpu.finish("busy");
  await busy;
  assert.deepEqual(gpu.started, ["busy", "diagnose"], "the waited-for request is raised above visible work");
  assert.deepEqual(queue, ["frame 1 queue write", "frame 2 waits"]);
  await gpu.finish("diagnose");
  await frame2;
  assert.deepEqual(queue, ["frame 1 queue write", "frame 2 waits", "frame 2 queue write", "frame 2 binds diagnose"]);
  // Compiled: no further frame waits, and the predicate allocates nothing.
  assert.equal(pipelinesPending(), undefined);
  await advance(3);
  assert.deepEqual(queue.slice(4), ["frame 3 queue write", "frame 3 binds diagnose"]);
  await gpu.finish("visible");
  await visible;
});

test("a failed background compile fails the next frame explicitly, whether or not it binds it", async () => {
  const gpu = fakeCompiler();
  const background = new GPUBackgroundPipelines();
  background.add("air", () => gpu.request("air"));
  background.add("diagnose", () => gpu.request("diagnose"));
  await gpu.settle();
  await gpu.fail("air");
  await assert.rejects(background.require([])!, /Background pipeline air failed to compile/);
  assert.throws(() => background.get("air"), /driver failure/);
  await gpu.finish("diagnose");
  assert.equal(background.allReady, false);
  await assert.rejects(joinPendingPipelines(undefined, background.require(["diagnose"]))!, /air failed to compile/);
});

test("pipelines requested without a manager compile at once and are still awaited", async () => {
  let resolve!: (pipeline: GPUComputePipeline) => void;
  const raw = { createComputePipelineAsync: () => new Promise<GPUComputePipeline>(accept => { resolve = accept; }) } as unknown as GPUDevice;
  const background = new GPUBackgroundPipelines();
  background.add("air", () => raw.createComputePipelineAsync({} as GPUComputePipelineDescriptor));
  const wait = background.require(["air"]);
  assert.ok(wait);
  resolve({ label: "air" } as unknown as GPUComputePipeline);
  await wait;
  assert.equal(background.require(["air"]), undefined);
  assert.equal(joinPendingPipelines(undefined, undefined), undefined);
});
