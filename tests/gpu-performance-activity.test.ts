import assert from "node:assert/strict";
import test from "node:test";

import {
  gpuLogicalActivityMatrixAddition,
  publishDecodedGPULogicalActivity,
} from "../lib/gpu-performance-activity";
import type { GPULogicalActivityCapture } from "../lib/gpu-logical-activity";
import {
  mergePerformanceActivityFrame,
  performanceActivityTaskColor,
  synthesizePerformanceActivityFrame,
} from "../lib/performance-activity";
import { createPerformanceActivityStore } from "../lib/stores/performance-activity-store";

const capture: GPULogicalActivityCapture = {
  captureId: 19,
  capacity: 8,
  overflowed: false,
  droppedEventCount: 0,
  events: [
    {
      sequence: 0, taskId: 7, checkpointId: 1, tick: 100, workgroupId: [0, 0, 0],
      workgroupEvidence: "measured", subgroupId: 0, subgroupEvidence: "measured", laneId: 0,
      logicalLaneCount: 32, logicalLaneCountEvidence: "measured", activeLaneCount: 23,
      activeLaneMask: [0x007fffff, 0, 0, 0], activeLaneEvidence: "measured",
    },
    {
      sequence: 1, taskId: 8, checkpointId: 1, tick: 101, workgroupId: [1, 0, 0],
      workgroupEvidence: "measured", subgroupId: 0, subgroupEvidence: "measured", laneId: 0,
      logicalLaneCount: 32, logicalLaneCountEvidence: "measured", activeLaneCount: 12,
      activeLaneMask: [0x00000fff, 0, 0, 0], activeLaneEvidence: "measured",
    },
    {
      sequence: 2, taskId: 7, checkpointId: 2, workgroupId: [2, 0, 0],
      workgroupEvidence: "measured", subgroupEvidence: "unknown",
      logicalLaneCountEvidence: "unknown", activeLaneEvidence: "unknown",
    },
    {
      sequence: 3, taskId: 7, checkpointId: 2, tick: 102, workgroupId: [0, 0, 0],
      workgroupEvidence: "measured", subgroupId: 0, subgroupEvidence: "measured", laneId: 0,
      logicalLaneCount: 32, logicalLaneCountEvidence: "measured", activeLaneCount: 23,
      activeLaneMask: [0x007fffff, 0, 0, 0], activeLaneEvidence: "measured",
    },
  ],
};

test("GPU heartbeat capture becomes real workgroup/subgroup rows over the horizontal time axis", () => {
  const identity = { frameId: "gpu-matrix", generation: 2, submissionId: "physics-4" };
  const addition = gpuLogicalActivityMatrixAddition({
    capture,
    identity,
    lane: "gpu-physics",
    clockDomain: "gpu-physics-timestamp",
    windowStart_ms: 0,
    windowEnd_ms: 4,
    locateTime: (event) => event.tick === undefined ? undefined : ({
      time_ms: (event.tick - 100) + 0.25,
      evidence: "reconstructed",
    }),
    tasks: {
      7: {
        id: "gpu.physics.advect", label: "Velocity advection", color: "#238cff",
        checkpoints: { enter: 1, exit: 2 },
      },
      8: { id: "gpu.physics.pressure", label: "Pressure solve", color: "#9b4fe0" },
      9: {
        id: "gpu.physics.registered-only",
        label: "Registered but not observed",
        phaseId: "pressure-system",
      },
    },
  });

  assert.equal(addition.rowCount, 3, "one logical row exists for every observed subgroup/workgroup identity");
  assert.equal(addition.events?.length, 3, "events without a time projection never get invented positions");
  assert.equal(addition.spans?.length, 1, "explicit enter/exit semantics create an occupied interval");
  assert.equal(addition.unknownTimeEventCount, 1);
  assert.equal(addition.captureOverflowed, false);
  assert.equal(addition.droppedEventCount, 0);
  assert.deepEqual(addition.captureDiagnostics, {
    reasons: ["unprojected-event"],
    recorderOverflowed: false,
    droppedEventCount: 0,
    droppedRowCount: 0,
    unprojectedEventCount: 1,
  });
  assert.deepEqual(addition.tasks?.find((task) => task.id === "gpu.physics.registered-only"), {
    id: "gpu.physics.registered-only",
    label: "Registered but not observed",
    color: performanceActivityTaskColor("gpu.physics.registered-only"),
    lane: "gpu-physics",
    stageId: "pressure-system",
  });
  assert.deepEqual(addition.windows?.map((window) => [window.start_ms, window.end_ms]),
    [[0, 4], [0, 4], [0, 4]]);

  const frame = synthesizePerformanceActivityFrame({
    identity: { frameId: identity.frameId, generation: identity.generation },
    context: "test",
    capturedAt_cpu_ms: 4,
    cpuTimeOrigin_ms: 1_000,
  });
  const merged = mergePerformanceActivityFrame(frame, addition);
  assert.deepEqual(merged.captureDiagnostics, addition.captureDiagnostics,
    "capture completeness diagnostics survive retention");
  const rows = merged.rows.filter((row) => row.resource.kind === "gpu-logical-capacity");
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((row) => row.slices.length), [4, 4, 4]);
  assert.equal(rows[0].slices[0].taskId, "gpu.physics.advect");
  assert.equal(rows[0].slices[0].evidence, "reconstructed");
  assert.equal(rows[0].slices[1].taskId, "gpu.physics.advect");
  assert.equal(rows[1].slices[1].taskId, "gpu.physics.pressure");
  assert.ok(rows[2].slices.every((slice) => slice.evidence === "unknown"));
});

test("overflow and display truncation remain explicit capture-level failures", () => {
  const addition = gpuLogicalActivityMatrixAddition({
    capture: { ...capture, overflowed: true, droppedEventCount: 17 },
    identity: { frameId: "overflow", generation: 3 },
    lane: "gpu-physics",
    clockDomain: "gpu-physics-timestamp",
    windowStart_ms: 0,
    windowEnd_ms: 4,
    maximumRows: 1,
    locateTime: () => undefined,
  });

  assert.equal(addition.captureDiagnostics?.recorderOverflowed, true);
  assert.equal(addition.captureDiagnostics?.droppedEventCount, 17);
  assert.ok((addition.captureDiagnostics?.droppedRowCount ?? 0) > 0);
  assert.deepEqual(addition.captureDiagnostics?.reasons,
    ["recorder-overflow", "row-limit", "unprojected-event"]);
});

test("solver-facing decoded capture ingestion correlates before and after base publication", () => {
  const store = createPerformanceActivityStore(2);
  store.getState().setEnabled(true);
  const generation = store.getState().generation;
  const frameIdentity = { frameId: "physics-sample-19", generation };
  const publish = (decoded: GPULogicalActivityCapture) => publishDecodedGPULogicalActivity({
    sink: store.getState(),
    capture: decoded,
    identity: { ...frameIdentity, submissionId: "physics-19" },
    lane: "gpu-physics",
    clockDomain: "gpu-physics-timestamp",
    windowStart_ms: 0,
    windowEnd_ms: 6,
    locateTime: (event) => event.tick === undefined ? undefined : ({
      time_ms: (event.tick - 100) + 0.25,
      evidence: "reconstructed",
    }),
    tasks: {
      7: {
        id: "gpu.physics.advect", label: "Velocity advection", color: "#238cff",
        checkpoints: { enter: 1, exit: 2 },
      },
      8: { id: "gpu.physics.pressure", label: "Pressure solve", color: "#9b4fe0" },
    },
  });

  const early = publish(capture);
  assert.equal(early.result, "buffered");
  assert.equal(early.addition.rowCount, 3);
  assert.equal(store.getState().history.length, 0);

  const base = {
    ...synthesizePerformanceActivityFrame({
      identity: frameIdentity,
      context: "octree:balanced",
      capturedAt_cpu_ms: 6,
      cpuTimeOrigin_ms: performance.timeOrigin,
    }),
    capturedAt_epoch_ms: store.getState().enabledAt_epoch_ms + 1,
  };
  assert.equal(store.getState().publish(base), true);
  const earlyRows = store.getState().latest?.rows.filter((row) => row.resource.kind === "gpu-logical-capacity") ?? [];
  assert.equal(earlyRows.length, 3);
  assert.ok(earlyRows.every((row) => row.windowStart_ms === 0 && row.windowEnd_ms === 6 && row.slices.length === 6));

  const lateCapture: GPULogicalActivityCapture = {
    ...capture,
    captureId: 20,
    events: [{ ...capture.events[0], sequence: 0, tick: 104, workgroupId: [3, 0, 0] }],
  };
  const late = publish(lateCapture);
  assert.equal(late.result, "merged");
  assert.equal(store.getState().history.length, 1);
  const lateRow = store.getState().latest?.rows.find((row) => row.resource.label === "WG 3,0,0 · SG 0");
  assert.deepEqual([lateRow?.windowStart_ms, lateRow?.windowEnd_ms, lateRow?.slices.length], [0, 6, 6]);
  assert.equal(lateRow?.slices[4].taskId, "gpu.physics.advect");

  assert.equal(store.getState().publish(base), true,
    "a repeated controller publication for the same GPU frame is idempotent");
  assert.equal(store.getState().history.length, 1);
  assert.ok(store.getState().latest?.rows.some((row) => row.resource.kind === "gpu-logical-capacity"),
    "idempotent base publication cannot erase asynchronously merged rows");

  const rejected = publishDecodedGPULogicalActivity({
    sink: store.getState(),
    capture: lateCapture,
    identity: { ...frameIdentity, generation: generation - 1 },
    lane: "gpu-physics",
    clockDomain: "gpu-physics-timestamp",
    windowStart_ms: 0,
    windowEnd_ms: 6,
    locateTime: () => ({ time_ms: 1, evidence: "measured" }),
  });
  assert.equal(rejected.result, "rejected");
});
