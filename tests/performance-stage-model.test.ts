import assert from "node:assert/strict";
import test from "node:test";
import {
  combineMainThreadPerformanceTraces,
  CPUPerformanceTrace,
  decodeGPUTimestampPartition,
  GPUQueueWallPerformanceTraceRecorder,
  GPUSegmentedQueueWallPerformanceTraceRecorder,
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

test("segmented queue-wall fallback measures every semantic phase", async () => {
  let encoderId = 0;
  const commandBuffers: Array<{ id: number }> = [];
  const makeEncoder = () => {
    const id = ++encoderId;
    return { finish: () => ({ id }) } as unknown as GPUCommandEncoder;
  };
  const device = {
    createCommandEncoder: () => makeEncoder(),
  } as unknown as GPUDevice;
  const times = [10, 13, 18];
  const recorder = new GPUSegmentedQueueWallPerformanceTraceRecorder(
    device,
    14,
    "physics",
    "octree:balanced",
    () => times.shift()!,
  );
  let encoder = recorder.completePhase(makeEncoder(), {
    id: "coarse-grid",
    label: "Adaptive coarse-grid topology",
  });
  encoder = recorder.completePhase(encoder, {
    id: "pressure-solve",
    label: "Pressure solve",
  });
  const trace = await recorder.read({
    submit: ([commandBuffer]) => {
      commandBuffers.push(commandBuffer as unknown as { id: number });
    },
    onSubmittedWorkDone: async () => undefined,
  });
  assert.equal(trace.measurementSource, "gpu-segmented-queue-wall");
  assert.equal(trace.total_ms, 8);
  assert.deepEqual(trace.phases.map((phase) => phase.duration_ms), [3, 5]);
  assert.equal(commandBuffers.length, 2);
  assert.equal(performanceTraceClosureError_ms(trace), 0);
  assert.equal(performanceTraceMatchesLane(trace, "gpu", "physics"), true);
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
