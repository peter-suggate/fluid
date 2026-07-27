import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import {
  GPU_LOGICAL_ACTIVITY_BIND_GROUP,
  GPU_LOGICAL_ACTIVITY_BINDING,
  createGPULogicalActivityAdoptionContext,
  gpuLogicalActivityTaskDescriptions,
  stableGPULogicalActivityId,
} from "../lib/gpu-logical-activity-adoption";

test("module-owned task descriptions are isolated by shader generation", () => {
  const generation = 0x6a31;
  const activity = createGPULogicalActivityAdoptionContext({
    moduleId: "octree/test-catalog",
    profile: { enabled: true, generation },
  });
  const taskId = activity.describeTask("advance", {
    id: "gpu.physics.test-catalog.advance",
    label: "Test catalog advance",
    phaseId: "pressure-solve",
  });
  assert.deepEqual(gpuLogicalActivityTaskDescriptions(generation)[taskId], {
    id: "gpu.physics.test-catalog.advance",
    label: "Test catalog advance",
    phaseId: "pressure-solve",
  });
  assert.equal(gpuLogicalActivityTaskDescriptions(generation + 1)[taskId], undefined);

  const duplicate = createGPULogicalActivityAdoptionContext({
    moduleId: "octree/test-catalog",
    profile: { enabled: true, generation },
  });
  assert.equal(duplicate.describeTask("advance", {
    id: "gpu.physics.test-catalog.advance",
    label: "Test catalog advance",
    phaseId: "pressure-solve",
  }), taskId);
  assert.throws(() => duplicate.describeTask("advance", {
    id: "gpu.physics.test-catalog.advance",
    label: "Conflicting label",
  }), /conflicts/);
});

function typeScriptFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) return typeScriptFiles(path);
    return entry.name.endsWith(".ts") ? [path] : [];
  });
}

test("disabled adoption snapshots production mode with byte-identical WGSL and no allocation", () => {
  const profile = { enabled: false, generation: 8 };
  const activity = createGPULogicalActivityAdoptionContext({ moduleId: "solver/advection", profile });
  profile.enabled = true;
  profile.generation = 9;
  const source = "@compute @workgroup_size(64) fn advect() {}\n";
  const variant = activity.module(source);
  assert.equal(activity.enabled, false);
  assert.equal(activity.generation, 8);
  assert.equal(activity.preludeWGSL, "");
  assert.equal(activity.workgroup("advect", "enter", { workgroupLaneCount: 64 }), "");
  assert.equal(activity.subgroup("advect", "enter"), "");
  assert.equal(variant.code, source);
  assert.doesNotMatch(variant.code, /fluidGpu|atomic<|@group\(3\)|@binding\(0\)/);
  const hostileDevice = new Proxy({}, { get() { throw new Error("disabled recorder touched device"); } }) as GPUDevice;
  assert.equal(activity.recorder(hostileDevice, { capacity: 4, captureId: 1 }), undefined);
});

test("storage-saturated shaders fail closed to byte-identical timestamp-only coverage", () => {
  const source = "@compute @workgroup_size(1) fn main() {}";
  const activity = createGPULogicalActivityAdoptionContext({
    moduleId: "saturated",
    profile: { enabled: true, generation: 12 },
    support: "timestamp-only",
  });
  assert.equal(activity.requested, true);
  assert.equal(activity.enabled, false);
  assert.equal(activity.timestampOnly, true);
  assert.equal(activity.module(source).code, source);
  assert.equal(activity.module(source).cacheKey,
    "saturated|production|gpu-logical-activity:v2:disabled");
  assert.equal(activity.recorder({} as GPUDevice, { capacity: 4, captureId: 1 }), undefined);
});

test("literal production WGSL leaves the activity bind group reserved", () => {
  const library = fileURLToPath(new URL("../lib", import.meta.url));
  const instrumentationOwners = ["/gpu-logical-activity.ts", "/gpu-logical-activity-adoption.ts"];
  for (const file of typeScriptFiles(library)) {
    if (instrumentationOwners.some((owner) => file.endsWith(owner))) continue;
    assert.doesNotMatch(readFileSync(file, "utf8"), /@group\(3\)/,
      `activity bind group 3 is already occupied by ${file}`);
  }
});

test("enabled adoption reserves group 3 binding 0 and emits concise named checkpoints", () => {
  assert.equal(GPU_LOGICAL_ACTIVITY_BIND_GROUP, 3);
  assert.equal(GPU_LOGICAL_ACTIVITY_BINDING, 0);
  const activity = createGPULogicalActivityAdoptionContext({
    moduleId: "solver/advection", profile: { enabled: true, generation: 12 },
  });
  const heartbeat = activity.workgroup("advect", "enter", {
    tick: "params.step", workgroupLaneCount: 64, numWorkgroups: "numWorkgroups",
    recordWhen: "hasMeaningfulWork",
  });
  const body = `@compute @workgroup_size(64) fn advect(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) numWorkgroups: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32,
) { ${heartbeat} }`;
  const variant = activity.module(body, "advection-pipeline");
  assert.match(activity.preludeWGSL, /@group\(3\) @binding\(0\)/);
  assert.match(heartbeat, /fluidGpuLogicalActivityWorkgroup\(\d+u, \d+u, params\.step/);
  assert.match(heartbeat, /^if \(hasMeaningfulWork\)/,
    "an additional uniform predicate can exclude semantically empty work");
  assert.match(heartbeat, /workgroupId, numWorkgroups, localInvocationIndex/);
  assert.equal(variant.code.indexOf(activity.preludeWGSL), 0);
  assert.equal(variant.code.match(/struct FluidGpuLogicalActivityBuffer/g)?.length, 1);
  assert.match(variant.cacheKey, /activity-generation:12/);
  assert.match(variant.cacheKey, /workgroup:g3:b0/);
});

test("subgroup adoption supplies logical defaults and stable IDs", () => {
  const first = createGPULogicalActivityAdoptionContext({
    moduleId: "pressure/reduce", profile: { enabled: true, generation: 3 }, identity: "subgroup",
  });
  const second = createGPULogicalActivityAdoptionContext({
    moduleId: "pressure/reduce", profile: { enabled: true, generation: 3 }, identity: "subgroup",
  });
  assert.equal(first.taskId("residual"), second.taskId("residual"));
  assert.equal(first.checkpointId("residual", "merge"), second.checkpointId("residual", "merge"));
  assert.notEqual(first.taskId("residual"), first.checkpointId("residual", "merge"));
  assert.equal(stableGPULogicalActivityId("abc"), stableGPULogicalActivityId("abc"));
  assert.notEqual(stableGPULogicalActivityId("abc"), stableGPULogicalActivityId("abd"));
  assert.notEqual(stableGPULogicalActivityId("abc"), 0xffffffff);
  assert.match(first.preludeWGSL, /subgroupBallot/);
  assert.match(first.subgroup("residual", "merge", { active: "ownsRow" }),
    /subgroupId, false, subgroupLane, subgroupSize, ownsRow/);
});

test("enabled adoption creates exactly one bounded recorder allocation", () => {
  const previousBufferUsage = globalThis.GPUBufferUsage;
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 } });
  const allocations: GPUBufferDescriptor[] = [];
  const writes: unknown[][] = [];
  const buffer = { destroy() {} } as GPUBuffer;
  const device = {
    queue: { writeBuffer: (...args: unknown[]) => writes.push(args) },
    createBuffer: (descriptor: GPUBufferDescriptor) => { allocations.push(descriptor); return buffer; },
  } as unknown as GPUDevice;
  try {
    const activity = createGPULogicalActivityAdoptionContext({
      moduleId: "solver/project", profile: { enabled: true, generation: 1 },
    });
    const recorder = activity.recorder(device, { capacity: 5, captureId: 99 });
    assert.ok(recorder);
    assert.equal(allocations.length, 1);
    assert.equal(writes.length, 1);
    assert.equal(recorder.capacity, 5);
    assert.equal(recorder.captureId, 99);
    recorder.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage });
  }
});

function bindingHarness() {
  const previousBufferUsage = globalThis.GPUBufferUsage;
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 } });
  const calls = {
    buffers: 0, writes: 0, bindGroups: 0, beginPasses: 0, copies: 0,
    pipelines: [] as GPUComputePipeline[], bindings: [] as Array<[number, GPUBindGroup]>,
    requestedLayouts: [] as number[], dispatchArguments: [] as unknown[][],
  };
  const buffer = { destroy() {} } as GPUBuffer;
  const device = {
    queue: { writeBuffer: () => { calls.writes += 1; } },
    createBuffer: () => { calls.buffers += 1; return buffer; },
    createBindGroup: () => { calls.bindGroups += 1; return { id: calls.bindGroups } as unknown as GPUBindGroup; },
  } as unknown as GPUDevice;
  const pass = {
    setPipeline: (pipeline: GPUComputePipeline) => calls.pipelines.push(pipeline),
    setBindGroup: (group: number, bindGroup: GPUBindGroup) => calls.bindings.push([group, bindGroup]),
    dispatchWorkgroups: (...args: unknown[]) => calls.dispatchArguments.push(args),
    dispatchWorkgroupsIndirect() {}, end() {},
  } as unknown as GPUComputePassEncoder;
  const encoder = {
    beginComputePass: () => { calls.beginPasses += 1; return pass; },
    copyBufferToBuffer: () => { calls.copies += 1; },
  } as unknown as GPUCommandEncoder;
  const pipeline = {
    getBindGroupLayout: (group: number) => { calls.requestedLayouts.push(group); return {} as GPUBindGroupLayout; },
  } as GPUComputePipeline;
  return {
    device, encoder, pipeline, calls,
    restore: () => Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage }),
  };
}

test("disabled binding session returns the original encoder and allocates nothing", () => {
  const harness = bindingHarness();
  try {
    const activity = createGPULogicalActivityAdoptionContext({
      moduleId: "disabled", profile: { enabled: false, generation: 4 },
    });
    activity.registerPipeline(harness.pipeline);
    const session = activity.bindingSession(harness.device, harness.encoder, { capacity: 4, captureId: 1 });
    assert.equal(session.encoder, harness.encoder);
    assert.equal(session.recorder, undefined);
    session.finish();
    assert.deepEqual(harness.calls, {
      buffers: 0, writes: 0, bindGroups: 0, beginPasses: 0, copies: 0,
      pipelines: [], bindings: [], requestedLayouts: [], dispatchArguments: [],
    });
  } finally { harness.restore(); }
});

test("binding session binds registered pipelines at dispatch and caches their group", () => {
  const harness = bindingHarness();
  try {
    const activity = createGPULogicalActivityAdoptionContext({
      moduleId: "auto-layout", profile: { enabled: true, generation: 5 },
    });
    assert.equal(activity.registerPipeline(harness.pipeline), harness.pipeline);
    const session = activity.bindingSession(harness.device, harness.encoder, { capacity: 4, captureId: 2 });
    const pass = session.encoder.beginComputePass();
    pass.setPipeline(harness.pipeline);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(harness.pipeline);
    pass.dispatchWorkgroups(2, 3);
    pass.setPipeline(harness.pipeline);
    pass.dispatchWorkgroups(4, 5, 6);
    pass.setPipeline(harness.pipeline);
    pass.dispatchWorkgroupsIndirect({} as GPUBuffer, 0);
    assert.equal(harness.calls.bindGroups, 1, "one bind group is cached per pipeline and frame capture");
    assert.deepEqual(harness.calls.requestedLayouts, [3]);
    assert.deepEqual(harness.calls.bindings.map(([group]) => group), [3, 3, 3, 3]);
    assert.deepEqual(harness.calls.dispatchArguments, [[1], [2, 3], [4, 5, 6]],
      "the proxy must preserve optional WebIDL argument arity for native Dawn bindings");
    assert.deepEqual(session.diagnostics, {
      computeDispatchCount: 4,
      instrumentedComputeDispatchCount: 4,
      unregisteredComputeDispatchCount: 0,
      unregisteredComputePipelineCount: 0,
      unregisteredComputePipelineLabels: [],
    });
    session.finish();
    assert.equal(harness.calls.copies, 1);
    session.destroy();
  } finally { harness.restore(); }
});

test("binding session reports every dispatch through unregistered compute pipelines", () => {
  const harness = bindingHarness();
  try {
    const activity = createGPULogicalActivityAdoptionContext({
      moduleId: "coverage", profile: { enabled: true, generation: 5 },
    });
    const unregistered = {
      label: "Legacy unprofiled pressure",
      getBindGroupLayout: () => ({} as GPUBindGroupLayout),
    } as unknown as GPUComputePipeline;
    const session = activity.bindingSession(harness.device, harness.encoder, {
      capacity: 4,
      captureId: 22,
    });
    const pass = session.encoder.beginComputePass();
    pass.setPipeline(unregistered);
    pass.dispatchWorkgroups(1);
    pass.dispatchWorkgroupsIndirect({} as GPUBuffer, 0);
    assert.deepEqual(session.diagnostics, {
      computeDispatchCount: 2,
      instrumentedComputeDispatchCount: 0,
      unregisteredComputeDispatchCount: 2,
      unregisteredComputePipelineCount: 1,
      unregisteredComputePipelineLabels: ["Legacy unprofiled pressure"],
    });
    session.destroy();
  } finally { harness.restore(); }
});

test("shared-recorder binding sessions do not stage, read, or destroy their recorder", async () => {
  const harness = bindingHarness();
  try {
    const activity = createGPULogicalActivityAdoptionContext({
      moduleId: "shared-discard", profile: { enabled: true, generation: 5 },
    });
    activity.registerPipeline(harness.pipeline);
    const sharedRecorder = activity.recorder(harness.device, { capacity: 1, captureId: 0 });
    assert.ok(sharedRecorder);
    const session = activity.bindingSession(harness.device, harness.encoder, { sharedRecorder });
    const pass = session.encoder.beginComputePass();
    pass.setPipeline(harness.pipeline);
    pass.dispatchWorkgroups(1);
    session.finish();
    assert.equal(harness.calls.copies, 0, "a shared discard recorder has no per-frame readback");
    assert.equal(await session.read(), undefined);
    session.destroy();
    assert.equal(sharedRecorder.state, "recording", "the session does not own the shared recorder");
    sharedRecorder.destroy();
    assert.equal(sharedRecorder.state, "destroyed");
  } finally { harness.restore(); }
});

test("binding proxy stacks on either side of an existing timestamp encoder proxy", () => {
  const run = (activityOutside: boolean) => {
    const harness = bindingHarness();
    let timestampPasses = 0;
    const timestampWrap = (encoder: GPUCommandEncoder) => new Proxy(encoder, {
      get(target, property) {
        if (property === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
          timestampPasses += 1;
          return target.beginComputePass(descriptor);
        };
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as GPUCommandEncoder;
    try {
      const activity = createGPULogicalActivityAdoptionContext({
        moduleId: `stack-${activityOutside}`, profile: { enabled: true, generation: 6 },
      });
      activity.registerPipeline(harness.pipeline);
      const inner = activityOutside ? timestampWrap(harness.encoder) : harness.encoder;
      const session = activity.bindingSession(harness.device, inner, { capacity: 2, captureId: 3 });
      const outer = activityOutside ? session.encoder : timestampWrap(session.encoder);
      const pass = outer.beginComputePass({ label: "stacked" });
      pass.setPipeline(harness.pipeline);
      pass.dispatchWorkgroups(1);
      assert.equal(timestampPasses, 1);
      assert.deepEqual(harness.calls.bindings.map(([group]) => group), [3]);
      session.destroy();
    } finally { harness.restore(); }
  };
  run(true);
  run(false);
});

test("binding session rejects a registered pipeline from another shader generation", () => {
  const harness = bindingHarness();
  try {
    const oldActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "old", profile: { enabled: true, generation: 1 },
    });
    const currentActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "current", profile: { enabled: true, generation: 2 },
    });
    oldActivity.registerPipeline(harness.pipeline);
    const session = currentActivity.bindingSession(harness.device, harness.encoder, { capacity: 2, captureId: 4 });
    const pass = session.encoder.beginComputePass();
    assert.throws(() => pass.setPipeline(harness.pipeline), /pipeline generation 1 used in capture generation 2/);
    session.destroy();
  } finally { harness.restore(); }
});

test("binding session round-trips group 3 activity through Dawn", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU binding validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const adapter = await create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const activity = createGPULogicalActivityAdoptionContext({
    moduleId: "dawn/group-3", profile: { enabled: true, generation: 1 },
  });
  const record = activity.workgroup("main", "enter", {
    workgroupId: "workgroupId", localInvocationIndex: "lane", workgroupLaneCount: 4,
  });
  const variant = activity.module(`
@group(0) @binding(0) var<uniform> enabled: u32;

@compute @workgroup_size(4)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) lane: u32) {
  if (enabled == 0u) { return; }
  ${record}
}`);
  const shaderModule = device.createShaderModule({ code: variant.code });
  const info = await shaderModule.getCompilationInfo();
  assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
  // Match the registered production power-volume path. node-webgpu 0.4.0's
  // async pipeline entry point crashes natively for sparse auto-layout groups.
  const pipeline = activity.registerPipeline(device.createComputePipeline({
    layout: "auto", compute: { module: shaderModule, entryPoint: "main" },
  }));
  const uniform = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniform, 0, new Uint32Array([1]));
  const uniformGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniform } }],
  });
  const rawEncoder = device.createCommandEncoder();
  const session = activity.bindingSession(device, rawEncoder, { capacity: 4, captureId: 17 });
  const pass = session.encoder.beginComputePass();
  pass.setPipeline(pipeline);
  // Setting a lower-numbered group after the pipeline can disturb group 3.
  // The activity proxy therefore binds group 3 immediately before dispatch.
  pass.setBindGroup(0, uniformGroup);
  pass.dispatchWorkgroups(2);
  pass.end();
  session.finish();
  device.queue.submit([session.encoder.finish()]);
  const result = await session.read();
  assert.equal(result?.ok, true);
  if (result?.ok) assert.equal(result.capture.events.length, 2);
  session.destroy();
  uniform.destroy();
  device.destroy();
});
