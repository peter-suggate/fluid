import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";
import {
  combineMainThreadPerformanceTraces,
  CPUPerformanceTrace,
  decodeGPUTimestampPartition,
  GPUQueueWallPerformanceTraceRecorder,
  GPUStageTimestampRecorder,
  averagePerformanceTraces,
  partitionPerformanceTrace,
  performanceTraceAccounting,
  performanceTraceAccounted_ms,
  performanceTraceClosureError_ms,
  performanceTraceIsExact,
  performanceTraceMatchesLane,
} from "../lib/performance-trace";

test("a disjoint interval chain retains gaps as measured other work", () => {
  const trace = partitionPerformanceTrace({
    sampleId: 1,
    domain: "gpu",
    lane: "physics",
    context: "octree",
    start_ms: 10,
    end_ms: 20,
    intervals: [
      { id: "coarse-grid", label: "Grid adaptation", start_ms: 10, end_ms: 12 },
      { id: "pressure-solve", label: "Pressure solve", start_ms: 14, end_ms: 19 },
    ],
  });
  assert.equal(trace.total_ms, 10);
  assert.equal(performanceTraceAccounted_ms(trace), 10);
  assert.equal(trace.phases.find((phase) => phase.id === "other")?.duration_ms, 3);
  assert.deepEqual(performanceTraceAccounting(trace), {
    observedTotal_ms: 10,
    accounted_ms: 10,
    closureError_ms: 0,
    exact: true,
  });
  assert.equal(performanceTraceIsExact(trace), true);
});

test("overlapping intervals are rejected instead of double counted", () => {
  assert.throws(() => partitionPerformanceTrace({
    sampleId: 1,
    domain: "cpu",
    lane: "main-thread",
    context: "frame",
    start_ms: 0,
    end_ms: 10,
    intervals: [
      { id: "frame-control", label: "Control", start_ms: 0, end_ms: 6 },
      { id: "command-encoding", label: "Encode", start_ms: 5, end_ms: 10 },
    ],
  }), /overlap/);
});

test("GPU adjacent boundaries sum exactly to their root", () => {
  const trace = decodeGPUTimestampPartition({
    sampleId: 3,
    lane: "physics",
    context: "octree",
    timestamps: new BigUint64Array([1_000_000n, 3_000_000n, 8_000_000n, 10_000_000n]),
    phases: [
      { id: "coarse-grid", label: "Grid adaptation" },
      { id: "pressure-solve", label: "Pressure solve" },
      { id: "adaptive-publication", label: "Publication" },
    ],
  });
  assert.ok(trace);
  assert.equal(trace.total_ms, 9);
  assert.deepEqual(trace.phases.map((phase) => phase.duration_ms), [2, 5, 2]);
  assert.equal(performanceTraceClosureError_ms(trace), 0);
  assert.equal(performanceTraceMatchesLane(trace, "gpu", "physics"), true);
  assert.equal(performanceTraceMatchesLane(trace, "gpu", "presentation"), false);
  assert.equal(performanceTraceIsExact(trace), true);
});

test("one missing or reversed GPU boundary rejects the whole sample", () => {
  const phases = [{ id: "coarse-grid" as const, label: "Grid" }];
  assert.equal(decodeGPUTimestampPartition({
    sampleId: 1, lane: "physics", context: "octree",
    timestamps: new BigUint64Array([0n, 2n]), phases,
  }), undefined);
  assert.equal(decodeGPUTimestampPartition({
    sampleId: 1, lane: "physics", context: "octree",
    timestamps: new BigUint64Array([2n, 1n]), phases,
  }), undefined);
});

test("CPU transitions share one boundary with no gaps", () => {
  const times = [1, 3, 8, 10];
  const cpu = new CPUPerformanceTrace(4, "frame", { id: "frame-control", label: "Control" }, () => times.shift()!);
  cpu.transition({ id: "scene-upload", label: "Upload" });
  cpu.transition({ id: "command-encoding", label: "Encode" });
  const trace = cpu.finish();
  assert.deepEqual(trace.phases.map((phase) => phase.duration_ms), [2, 5, 2]);
  assert.equal(performanceTraceClosureError_ms(trace), 0);
  assert.equal(performanceTraceMatchesLane(trace, "cpu", "main-thread"), true);
  assert.equal(performanceTraceIsExact(trace), true);
});

test("disjoint controller and renderer callbacks close one active CPU ledger", () => {
  const controller = partitionPerformanceTrace({
    sampleId: 7,
    domain: "cpu",
    lane: "main-thread",
    context: "octree:medium",
    start_ms: 0,
    end_ms: 2,
    intervals: [{ id: "frame-control", label: "Simulation clock", start_ms: 0, end_ms: 2 }],
  });
  const renderer = partitionPerformanceTrace({
    sampleId: 8,
    domain: "cpu",
    lane: "main-thread",
    context: "octree:medium",
    start_ms: 10,
    end_ms: 15,
    intervals: [{ id: "command-encoding", label: "Presentation encoding", start_ms: 10, end_ms: 15 }],
  });
  const combined = combineMainThreadPerformanceTraces([controller, renderer]);
  assert.ok(combined);
  assert.equal(combined.total_ms, 7, "idle wall time between callbacks is not CPU work");
  assert.equal(performanceTraceAccounted_ms(combined), 7);
  assert.equal(performanceTraceClosureError_ms(combined), 0);
  assert.throws(
    () => combineMainThreadPerformanceTraces([renderer, renderer]),
    /double count/,
  );
  assert.throws(
    () => combineMainThreadPerformanceTraces([controller, {
      ...renderer,
      domain: "gpu",
      lane: "presentation",
    }]),
    /Only exact CPU main-thread/,
  );
});

test("GPU queue-wall fallback publishes an exact sample when timestamps are unavailable", async () => {
  const times = [10, 14];
  const recorder = new GPUQueueWallPerformanceTraceRecorder(
    12,
    "presentation",
    "octree:balanced",
    () => times.shift()!,
  );
  recorder.begin();
  const trace = await recorder.read({ onSubmittedWorkDone: async () => undefined });
  assert.equal(trace.measurementSource, "gpu-queue-wall");
  assert.equal(trace.total_ms, 4);
  assert.equal(trace.phases.length, 1);
  assert.match(trace.phases[0].label, /hardware timestamps unavailable/);
  assert.equal(performanceTraceClosureError_ms(trace), 0);
  assert.equal(performanceTraceMatchesLane(trace, "gpu", "presentation"), true);
});

/** Minimal recording device: enough for the stage recorder's query set, its
 * staging buffers, the marker pipeline, and a scripted resolved timestamp set. */
function stageTimestampHarness(resolvedTimestamps_ns: readonly bigint[]) {
  const scope = globalThis as Record<string, unknown>;
  scope.GPUBufferUsage ??= { QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 };
  scope.GPUMapMode ??= { READ: 1 };
  const passes: Array<{ label?: string; timestampWrites?: { beginningOfPassWriteIndex?: number } }> = [];
  const encoderBreaks: number[] = [];
  const resolves: Array<{ first: number; count: number }> = [];
  const pass = { setPipeline() {}, dispatchWorkgroups() {}, end() {} };
  const buffer = () => ({
    destroy() {},
    mapState: "unmapped" as GPUBufferMapState,
    mapAsync: async () => undefined,
    getMappedRange: () => BigUint64Array.from(resolvedTimestamps_ns).buffer,
    unmap() {},
  });
  const device = {
    features: new Set(["timestamp-query"]),
    createQuerySet: () => ({ destroy() {} }),
    createBuffer: buffer,
    createShaderModule: () => ({}),
    createComputePipeline: () => ({}),
  } as unknown as GPUDevice;
  const encoder = {
    beginComputePass(descriptor?: GPUComputePassDescriptor) { passes.push({ ...descriptor }); return pass; },
    beginRenderPass(descriptor: GPURenderPassDescriptor) { passes.push({ ...descriptor }); return pass; },
    copyBufferToBuffer(_source: GPUBuffer, _sourceOffset: number, _destination: GPUBuffer, _destinationOffset: number, size: number) {
      encoderBreaks.push(size);
    },
    resolveQuerySet(_set: GPUQuerySet, first: number, count: number) { resolves.push({ first, count }); },
  } as unknown as GPUCommandEncoder;
  return { device, encoder, passes, encoderBreaks, resolves };
}

test("stage boundaries ride the frame's own passes and add one marker in total", async () => {
  const harness = stageTimestampHarness([1_000_000n, 4_000_000n, 9_000_000n]);
  const recorder = new GPUStageTimestampRecorder(harness.device, 14, "physics", "octree:balanced");
  const encoder = recorder.instrument(harness.encoder);
  recorder.begin();
  encoder.beginComputePass({ label: "Structured advection" }).end();
  recorder.completePhase(encoder, { id: "velocity-advection", label: "Velocity advection" });
  // A stage that encoded nothing must not consume a boundary of its own.
  recorder.completePhase(encoder, { id: "pressure-system", label: "Row assembly" });
  encoder.beginComputePass({ label: "MGPCG" }).end();
  recorder.completePhase(encoder, { id: "pressure-solve", label: "Pressure solve" });
  recorder.resolve(encoder);

  assert.deepEqual(harness.passes.map((entry) => entry.label),
    ["Structured advection", "MGPCG", "GPU stage trace close"],
    "only the closing boundary needs a pass of its own");
  assert.deepEqual(harness.passes.map((entry) => entry.timestampWrites?.beginningOfPassWriteIndex), [0, 1, 2]);
  assert.deepEqual(harness.encoderBreaks, [4, 4, 4, 24],
    "one 4-byte blit forces each boundary's encoder break, then one staging copy of the resolved set");
  assert.deepEqual(harness.resolves, [{ first: 0, count: 3 }]);

  const trace = await recorder.read();
  assert.ok(trace);
  assert.equal(trace.measurementSource, "gpu-hardware-timestamp");
  assert.equal(trace.total_ms, 8);
  assert.deepEqual(trace.phases.map((phase) => [phase.id, phase.duration_ms]),
    [["velocity-advection", 3], ["pressure-solve", 5]],
    "the empty stage closes at exactly zero and is dropped from the partition");
  assert.equal(performanceTraceClosureError_ms(trace), 0);
  assert.equal(performanceTraceMatchesLane(trace, "gpu", "physics"), true);
});

test("a pass that already carries timestamp writes is never displaced", async () => {
  const harness = stageTimestampHarness([2_000_000n, 5_000_000n]);
  const recorder = new GPUStageTimestampRecorder(harness.device, 15, "presentation", "svo");
  const encoder = recorder.instrument(harness.encoder);
  recorder.begin();
  const foreign = { querySet: undefined as unknown as GPUQuerySet, beginningOfPassWriteIndex: 7 };
  encoder.beginComputePass({ label: "Owned elsewhere", timestampWrites: foreign }).end();
  encoder.beginComputePass({ label: "Dry scene" }).end();
  recorder.completePhase(encoder, { id: "dry-scene", label: "Dry scene" });
  recorder.resolve(encoder);
  assert.equal(harness.passes[0].timestampWrites?.beginningOfPassWriteIndex, 7);
  assert.equal(harness.passes[1].timestampWrites?.beginningOfPassWriteIndex, 0);
  const trace = await recorder.read();
  assert.equal(trace?.total_ms, 3);
});

test("an unusable hardware sample yields no trace instead of a wrong one", async () => {
  const harness = stageTimestampHarness([0n, 5_000_000n]);
  const recorder = new GPUStageTimestampRecorder(harness.device, 16, "physics", "octree");
  const encoder = recorder.instrument(harness.encoder);
  recorder.begin();
  encoder.beginComputePass({ label: "Folded to nothing" }).end();
  recorder.completePhase(encoder, { id: "pressure-solve", label: "Pressure solve" });
  recorder.resolve(encoder);
  assert.equal(await recorder.read(), undefined);
});

const webgpuModulePath = process.env.WEBGPU_NODE_MODULE;
test("stage timestamp recorder resolves real Metal timestamps", {
  skip: !webgpuModulePath && "set WEBGPU_NODE_MODULE for native timestamp validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(webgpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  assert.equal(adapter.features.has("timestamp-query"), true);
  const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  const recorder = new GPUStageTimestampRecorder(device, 17, "physics", "native-stage-recorder");
  const encoder = recorder.instrument(device.createCommandEncoder());
  const shaderModule = device.createShaderModule({ code: "@compute @workgroup_size(1) fn main() {}" });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModule, entryPoint: "main" },
  });
  recorder.begin();
  for (const [id, label] of [["velocity-advection", "First"], ["pressure-solve", "Second"]] as const) {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(1);
    pass.end();
    recorder.completePhase(encoder, { id, label });
  }
  recorder.resolve(encoder);
  device.queue.submit([encoder.finish()]);
  const trace = await recorder.read();
  assert.ok(trace);
  assert.equal(trace.measurementSource, "gpu-hardware-timestamp");
  assert.equal(trace.phases.length, 2);
  device.destroy();
});

test("averaging never invents a negative phase from floating-point dust", () => {
  const trace = {
    sampleId: 1,
    domain: "gpu" as const,
    lane: "presentation" as const,
    context: "octree",
    capturedAt_ms: 1,
    measurementSource: "gpu-hardware-timestamp" as const,
    total_ms: 0.3,
    phases: [
      { id: "surface-extraction" as const, label: "Surface", duration_ms: 0.1 },
      { id: "present" as const, label: "Present", duration_ms: 0.2 },
    ],
  };
  const averaged = averagePerformanceTraces([trace, { ...trace, sampleId: 2, capturedAt_ms: 2 }]);
  assert.ok(averaged);
  assert.equal(averaged.phases.some((phase) => phase.duration_ms < 0), false);
  assert.equal(performanceTraceIsExact(averaged), true);
});
