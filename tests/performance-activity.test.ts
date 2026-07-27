import assert from "node:assert/strict";
import test from "node:test";
import {
  cpuPerformanceClockSnapshot,
  gpuPhysicsPerformanceActivityFrameId,
  mergePerformanceActivityCaptureDiagnostics,
  mergePerformanceActivityFrame,
  performanceActivityTaskColor,
  slicePerformanceActivityRows,
  synthesizePerformanceActivityFrame,
  type ActivityResource,
  type ActivitySpan,
} from "../lib/performance-activity";
import { createPerformanceActivityStore } from "../lib/stores/performance-activity-store";
import type { PerformanceTrace } from "../lib/performance-trace";

const cpuTrace: PerformanceTrace = {
  sampleId: 7,
  domain: "cpu",
  lane: "main-thread",
  context: "octree:balanced",
  capturedAt_ms: 15,
  measurementSource: "cpu-active-wall",
  total_ms: 3,
  phases: [
    { id: "frame-control", label: "Frame control", duration_ms: 1.25 },
    { id: "command-encoding", label: "Command encoding", duration_ms: 1.75 },
  ],
};

const gpuTrace: PerformanceTrace = {
  sampleId: 9,
  domain: "gpu",
  lane: "physics",
  context: "octree:sim-1",
  capturedAt_ms: 25,
  measurementSource: "gpu-hardware-timestamp",
  total_ms: 2.5,
  phases: [
    { id: "fine-sdf-advection", label: "Fine transport", duration_ms: 1 },
    { id: "pressure-solve", label: "Pressure solve", duration_ms: 1.5 },
  ],
};

test("CPU performance clock exposes the exact time-origin alignment", () => {
  const snapshot = cpuPerformanceClockSnapshot({ timeOrigin: 1_700_000_000_000, now: () => 12.5 });
  assert.deepEqual(snapshot, { now_ms: 12.5, timeOrigin_ms: 1_700_000_000_000, epoch_ms: 1_700_000_000_012.5 });
});

test("GPU physics frame identity is stable across asynchronous completion times", () => {
  const laterTrace: PerformanceTrace = { ...gpuTrace, capturedAt_ms: 9_999 };
  assert.equal(
    gpuPhysicsPerformanceActivityFrameId({ sampleId: 19, context: "octree:balanced:sim-4" }),
    "gpu-physics:octree%3Abalanced%3Asim-4:19",
  );
  assert.equal(
    gpuPhysicsPerformanceActivityFrameId(gpuTrace),
    gpuPhysicsPerformanceActivityFrameId(laterTrace),
  );
});

test("one-millisecond slicing distinguishes measured, reconstructed, idle, and unknown evidence", () => {
  const resource: ActivityResource = {
    id: "cpu.main", label: "main", kind: "cpu-main", lane: "cpu-main", clockDomain: "cpu-performance",
  };
  const identity = { frameId: "f", generation: 1 };
  const spans: ActivitySpan[] = [
    { id: "coverage", kind: "observation", resourceId: resource.id, clockDomain: resource.clockDomain,
      start_ms: 0, end_ms: 3, evidence: "measured", identity },
    { id: "direct", kind: "task", taskId: "direct", resourceId: resource.id, clockDomain: resource.clockDomain,
      start_ms: 0, end_ms: 1, evidence: "measured", identity },
    { id: "derived", kind: "task", taskId: "derived", resourceId: resource.id, clockDomain: resource.clockDomain,
      start_ms: 1, end_ms: 2, evidence: "reconstructed", identity },
  ];
  const [row] = slicePerformanceActivityRows({ resources: [resource], spans, windows: [{ resourceId: resource.id, start_ms: 0, end_ms: 4 }] });
  assert.deepEqual(row.slices.map((slice) => slice.evidence), ["measured", "reconstructed", "idle", "unknown"]);
  assert.ok(row.slices.every((slice) => slice.end_ms - slice.start_ms === 1));
});

test("a heartbeat marks observed progress in its millisecond without inventing continuous activity", () => {
  const resource: ActivityResource = {
    id: "gpu.logical.0", label: "logical capacity 0", kind: "gpu-logical-capacity",
    lane: "gpu-physics", clockDomain: "gpu-physics-timestamp",
  };
  const identity = { frameId: "heartbeat", generation: 2 };
  const [row] = slicePerformanceActivityRows({
    resources: [resource],
    spans: [],
    events: [{
      id: "heartbeat.0", kind: "instant", taskId: "gpu.task", resourceId: resource.id,
      clockDomain: resource.clockDomain, at_ms: 1.4, evidence: "measured", identity,
    }],
    windows: [{ resourceId: resource.id, start_ms: 0, end_ms: 3 }],
  });
  assert.deepEqual(row.slices.map((slice) => slice.evidence), ["unknown", "measured", "unknown"]);
  assert.equal(row.slices[1].active_ms, 0, "an instant observation is not a full millisecond of occupancy");
  assert.deepEqual(row.slices[1].eventIds, ["heartbeat.0"]);
});

test("legacy lane synthesis preserves raw evidence and refuses to invent GPU clock synchronization", () => {
  const frame = synthesizePerformanceActivityFrame({
    identity: { frameId: "frame-42", generation: 3 },
    context: "octree:balanced",
    cpu: cpuTrace,
    physics: gpuTrace,
    cpuTimeOrigin_ms: 1_000,
    physicsSubmissionId: "queue-physics-12",
    physicsPublicationId: "fluid-generation-8",
    gpuLogicalCapacityRows: { physics: 3 },
    workerRows: [{ id: "quadtree-topology", label: "Quadtree topology worker", timeOrigin_ms: 1_002 }],
  });
  assert.equal(frame.capturedAt_epoch_ms, 1_025);
  assert.equal(frame.spans.filter((span) => span.kind === "observation").length, 2);
  assert.equal(frame.spans.filter((span) => span.kind === "task").length, 4);
  assert.ok(frame.events.length >= 6);
  assert.equal(frame.clocks.find((clock) => clock.id === "cpu-performance")?.status.state, "reference");
  assert.equal(frame.clocks.find((clock) => clock.id === "gpu-physics-timestamp")?.status.state, "unsynchronized");
  assert.deepEqual(frame.clocks.find((clock) => clock.id === "cpu-worker:quadtree-topology:performance")?.status, {
    state: "synchronized", to: "cpu-performance", offset_ms: 2, uncertainty_ms: 0, source: "shared-time-origin",
  });
  assert.equal(frame.resources.filter((resource) => resource.kind === "gpu-logical-capacity").length, 3);
  assert.equal(frame.rows.find((row) => row.resource.id === "gpu.physics.logical.1")?.slices[0].evidence, "unknown");
  const physicsSpan = frame.spans.find((span) => span.identity.submissionId === "queue-physics-12" && span.kind === "task");
  assert.equal(physicsSpan?.identity.publicationId, "fluid-generation-8");
  assert.equal(physicsSpan?.evidence, "reconstructed");
  assert.ok(frame.events.some((event) => event.kind === "publish"
    && event.identity.publicationId === "fluid-generation-8"));
});

test("a supplied GPU calibration is explicit and leaves raw timestamps unchanged", () => {
  const frame = synthesizePerformanceActivityFrame({
    identity: { frameId: "aligned", generation: 1 }, context: "test", physics: gpuTrace,
    cpuTimeOrigin_ms: 0,
    alignments: [{ from: "gpu-physics-timestamp", to: "cpu-performance", offset_ms: 42,
      uncertainty_ms: 0.2, source: "calibrated" }],
  });
  assert.deepEqual(frame.clocks.find((clock) => clock.id === "gpu-physics-timestamp")?.status, {
    state: "synchronized", to: "cpu-performance", offset_ms: 42, uncertainty_ms: 0.2, source: "calibrated",
  });
  assert.equal(frame.spans.find((span) => span.clockDomain === "gpu-physics-timestamp")?.start_ms, 0);
});

test("detailed evidence can be merged without losing the synthesized frame", () => {
  const frame = synthesizePerformanceActivityFrame({
    identity: { frameId: "merged", generation: 4 }, context: "test", cpu: cpuTrace,
    cpuTimeOrigin_ms: 0,
  });
  const identity = { frameId: "merged", generation: 4, submissionId: "cpu:direct" };
  const detailed = mergePerformanceActivityFrame(frame, {
    tasks: [{ id: "cpu.main.direct", label: "Direct CPU stage", color: "#ffffff", lane: "cpu-main" }],
    spans: [{
      id: "cpu.main.direct.0", kind: "task", taskId: "cpu.main.direct", resourceId: "cpu.main",
      clockDomain: "cpu-performance", start_ms: 10, end_ms: 11, evidence: "measured", identity,
    }],
    events: [{
      id: "cpu.main.direct.begin", kind: "begin", taskId: "cpu.main.direct", resourceId: "cpu.main",
      clockDomain: "cpu-performance", at_ms: 10, evidence: "measured", identity,
    }],
  });
  assert.equal(detailed.spans.length, frame.spans.length + 1);
  assert.equal(detailed.tasks.filter((task) => task.id === "cpu.main.direct").length, 1);
  const cpuRow = detailed.rows.find((row) => row.resource.id === "cpu.main");
  assert.equal(cpuRow?.windowStart_ms, 10, "detailed evidence expands an existing synthesized row");
  const directSlice = cpuRow?.slices.find((slice) => slice.start_ms === 10);
  assert.equal(directSlice?.evidence, "measured");
  assert.equal(directSlice?.taskId, "cpu.main.direct");
  assert.throws(() => mergePerformanceActivityFrame(frame, {
    events: [{ ...detailed.events.at(-1)!, identity: { frameId: "other", generation: 4 } }],
  }), /Cannot merge activity/);
});

test("capture completeness reasons and counts merge idempotently", () => {
  const first = {
    reasons: ["recorder-overflow" as const],
    recorderOverflowed: true,
    droppedEventCount: 19,
    unprofiledDispatchCount: 3,
    unprofiledPipelineLabels: ["transport"],
  };
  const merged = mergePerformanceActivityCaptureDiagnostics(first, {
    reasons: ["recorder-overflow", "missing-frame-end"],
    droppedEventCount: 19,
    unprojectedEventCount: 2,
    unprofiledDispatchCount: 3,
    unprofiledPipelineLabels: ["transport", "projection"],
  });
  assert.deepEqual(merged, {
    reasons: ["recorder-overflow", "missing-frame-end"],
    recorderOverflowed: true,
    droppedEventCount: 19,
    droppedRowCount: 0,
    unprojectedEventCount: 2,
    unprofiledDispatchCount: 3,
    unprofiledPipelineLabels: ["transport", "projection"],
  });
});

test("task colors are deterministic and independent of display labels", () => {
  assert.equal(performanceActivityTaskColor("gpu.physics.pressure-solve"), performanceActivityTaskColor("gpu.physics.pressure-solve"));
  assert.match(performanceActivityTaskColor("gpu.physics.pressure-solve"), /^#[0-9a-f]{6}$/);
});

test("activity history is bounded and rejects disabled, old-generation, and pre-enable publications", () => {
  const store = createPerformanceActivityStore(2);
  const base = synthesizePerformanceActivityFrame({
    identity: { frameId: "disabled", generation: 0 }, context: "test", cpu: cpuTrace,
    cpuTimeOrigin_ms: performance.timeOrigin,
  });
  assert.equal(store.getState().publish(base), false);
  store.getState().setEnabled(true);
  const generation = store.getState().generation;
  const enabledAt = store.getState().enabledAt_epoch_ms;
  const makeFrame = (frameId: string, capturedAt_epoch_ms: number, requestedGeneration = generation) => ({
    ...base,
    identity: { frameId, generation: requestedGeneration },
    capturedAt_epoch_ms,
  });
  assert.equal(store.getState().publish(makeFrame("stale", enabledAt + 1, generation - 1)), false);
  assert.equal(store.getState().publish(makeFrame("old", enabledAt - 1)), false);
  assert.equal(store.getState().publish(makeFrame("one", enabledAt + 1)), true);
  assert.equal(store.getState().publish(makeFrame("two", enabledAt + 2)), true);
  store.getState().pinReference("one");
  assert.equal(store.getState().publish(makeFrame("three", enabledAt + 3)), true);
  assert.deepEqual(store.getState().history.map((frame) => frame.identity.frameId), ["one", "three"]);
  assert.equal(store.getState().selectedFrameId, "three");
  assert.equal(store.getState().referenceFrameId, "one");
  store.getState().selectFrame("one");
  assert.equal(store.getState().selectedFrameId, "one");
  assert.equal(store.getState().publish(makeFrame("four", enabledAt + 4)), true);
  assert.equal(store.getState().selectedFrameId, "one", "manual selection must not jump as live captures arrive");
  const nextGeneration = store.getState().beginGeneration();
  assert.equal(nextGeneration, generation + 1);
  assert.equal(store.getState().history.length, 0);
  assert.equal(store.getState().selectedFrameId, undefined);
  assert.equal(store.getState().referenceFrameId, undefined);
  assert.equal(store.getState().publish(makeFrame("late", store.getState().enabledAt_epoch_ms + 1, generation)), false);
});

test("a custom one-frame store remains bounded and releases an evicted reference", () => {
  const store = createPerformanceActivityStore(1);
  store.getState().setEnabled(true);
  const generation = store.getState().generation;
  const enabledAt = store.getState().enabledAt_epoch_ms;
  const frame = (frameId: string, offset: number) => ({
    ...synthesizePerformanceActivityFrame({
      identity: { frameId, generation }, context: "test", cpu: cpuTrace,
      cpuTimeOrigin_ms: performance.timeOrigin,
    }),
    capturedAt_epoch_ms: enabledAt + offset,
  });
  assert.equal(store.getState().historyLimit, 1);
  store.getState().publish(frame("reference", 1));
  store.getState().pinReference("reference");
  store.getState().publish(frame("current", 2));
  assert.deepEqual(store.getState().history.map((candidate) => candidate.identity.frameId), ["current"]);
  assert.equal(store.getState().referenceFrameId, undefined);
});

test("asynchronous heartbeat evidence can update a retained frame without creating another capture", () => {
  const store = createPerformanceActivityStore(2);
  store.getState().setEnabled(true);
  const generation = store.getState().generation;
  const frame = {
    ...synthesizePerformanceActivityFrame({
      identity: { frameId: "async", generation }, context: "test", cpu: cpuTrace,
      cpuTimeOrigin_ms: performance.timeOrigin,
    }),
    capturedAt_epoch_ms: store.getState().enabledAt_epoch_ms + 1,
  };
  assert.equal(store.getState().publish(frame), true);
  assert.equal(store.getState().mergeEvidence("async", {
    tasks: [{ id: "gpu.task", label: "GPU task", color: "#ffffff", lane: "gpu-physics" }],
  }), true);
  assert.equal(store.getState().history.length, 1);
  assert.equal(store.getState().latest?.tasks.some((task) => task.id === "gpu.task"), true);
  assert.equal(store.getState().mergeEvidence("missing", {}), false);
});

test("evidence arriving before its base frame is generation-checked, buffered, and merged in order", () => {
  const store = createPerformanceActivityStore(3);
  store.getState().setEnabled(true);
  const generation = store.getState().generation;
  const identity = { frameId: "gpu-before-base", generation };
  const resource = (slot: number): ActivityResource => ({
    id: `gpu.physics.logical.${slot}`,
    label: `WG ${slot},0,0`,
    kind: "gpu-logical-capacity",
    lane: "gpu-physics",
    clockDomain: "gpu-physics-timestamp",
    capacitySlot: slot,
  });
  const addition = (slot: number, at_ms: number) => ({
    resources: [resource(slot)],
    tasks: [{ id: `gpu.task.${slot}`, label: `GPU task ${slot}`, color: "#ffffff", lane: "gpu-physics" as const }],
    events: [{
      id: `heartbeat.${slot}`,
      kind: "instant" as const,
      taskId: `gpu.task.${slot}`,
      resourceId: resource(slot).id,
      clockDomain: "gpu-physics-timestamp",
      at_ms,
      evidence: "measured" as const,
      identity,
    }],
    windows: [{ resourceId: resource(slot).id, start_ms: 0, end_ms: 8 }],
  });

  assert.equal(store.getState().ingestEvidence(identity, addition(0, 1.25)), "buffered");
  assert.equal(store.getState().ingestEvidence(identity, addition(1, 6.25)), "buffered");
  assert.equal(store.getState().pendingEvidence.length, 1);
  assert.equal(store.getState().pendingEvidence[0].additions.length, 2);
  assert.equal(store.getState().ingestEvidence({ ...identity, generation: generation - 1 }, addition(2, 2.25)), "rejected");
  assert.equal(store.getState().ingestEvidence(identity, {
    ...addition(2, 2.25),
    events: [{ ...addition(2, 2.25).events[0], identity: { frameId: "wrong", generation } }],
  }), "rejected");

  const base = {
    ...synthesizePerformanceActivityFrame({ identity, context: "test", physics: gpuTrace }),
    capturedAt_epoch_ms: store.getState().enabledAt_epoch_ms + 1,
  };
  assert.equal(store.getState().publish(base), true);
  assert.equal(store.getState().history.length, 1);
  assert.equal(store.getState().pendingEvidence.length, 0);
  const logicalRows = store.getState().latest?.rows.filter((row) => row.resource.kind === "gpu-logical-capacity") ?? [];
  assert.equal(logicalRows.length, 2);
  assert.deepEqual(logicalRows.map((row) => [row.windowStart_ms, row.windowEnd_ms, row.slices.length]), [
    [0, 8, 8],
    [0, 8, 8],
  ]);
  assert.deepEqual(store.getState().latest?.events.filter((event) => event.id.startsWith("heartbeat.")).map((event) => event.id), [
    "heartbeat.0",
    "heartbeat.1",
  ]);

  assert.equal(store.getState().ingestEvidence(identity, addition(2, 3.25)), "merged");
  assert.equal(store.getState().history.length, 1, "late evidence updates the retained frame in place");
  assert.equal(store.getState().latest?.rows.find((row) => row.resource.id === resource(2).id)?.windowEnd_ms, 8);
});
