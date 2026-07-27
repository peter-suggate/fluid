import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  PHYSICS_ACTIVITY_FRAME_BEGIN_CHECKPOINT_ID,
  PHYSICS_ACTIVITY_FRAME_END_CHECKPOINT_ID,
  PHYSICS_ACTIVITY_PHASE_BOUNDARY_CHECKPOINT_ID,
  PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID,
  PHYSICS_ACTIVITY_POWER_VOLUME_TASK_ID,
  maximumPhysicsLogicalActivityCaptureCapacity,
  physicsLogicalActivityCaptureCapacity,
  physicsPhaseBoundaryTimeProjection,
  validateGPUPhysicsLogicalActivityCapture,
} from "../lib/webgpu-uniform-eulerian";
import { stableGPULogicalActivityId } from "../lib/gpu-logical-activity-adoption";
import type { GPULogicalActivityEvent } from "../lib/gpu-logical-activity";
import type { PerformanceTrace } from "../lib/performance-trace";

const event = (tick: number | undefined, sequence: number): GPULogicalActivityEvent => ({
  sequence,
  taskId: PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID,
  checkpointId: PHYSICS_ACTIVITY_PHASE_BOUNDARY_CHECKPOINT_ID,
  ...(tick === undefined ? {} : { tick }),
  workgroupId: [0, 0, 0],
  workgroupEvidence: "measured",
  sampleIndex: 0,
  sampleCount: 1,
  dispatchWorkgroupCount: 1,
  laneId: 0,
  logicalLaneCount: 1,
  logicalLaneCountEvidence: "reconstructed",
  activeLaneEvidence: "unknown",
  subgroupEvidence: "unknown",
});

const frameSentinel = (
  checkpointId: number,
  sequence: number,
): GPULogicalActivityEvent => ({
  ...event(undefined, sequence),
  checkpointId,
});

test("physics phase ticks project through measured durations without using append sequence", () => {
  const trace: PerformanceTrace = {
    sampleId: 9,
    domain: "gpu",
    lane: "physics",
    context: "octree:sim-1",
    capturedAt_ms: 50,
    measurementSource: "gpu-hardware-timestamp",
    total_ms: 3,
    phases: [
      { id: "other", label: "Setup", duration_ms: 1.25 },
      { id: "pressure-solve", label: "Solve", duration_ms: 1.75 },
    ],
  };
  const projection = physicsPhaseBoundaryTimeProjection(trace);
  assert.deepEqual(projection.phaseBoundaries_ms, [1.25, 3]);
  assert.deepEqual(projection.locateTime(event(0, 999)), {
    time_ms: 1.25,
    evidence: "reconstructed",
  });
  assert.deepEqual(projection.locateTime(event(1, 0)), {
    time_ms: 3,
    evidence: "reconstructed",
  });
  assert.equal(projection.locateTime(event(undefined, 1)), undefined);
  assert.equal(projection.locateTime(event(2, 2)), undefined);
  assert.equal(projection.locateTime({ ...event(0, 3), taskId: 123 }), undefined,
    "task-local shader ticks are not global phase indices");
  assert.deepEqual(projection.locateTime({
    ...event(1, 4), taskId: PHYSICS_ACTIVITY_POWER_VOLUME_TASK_ID,
  }), { time_ms: 3, evidence: "reconstructed" },
  "validated fine-split power-volume ticks share the phase-boundary coordinate");
});

test("frame sentinels project to the exact sampled physics window endpoints", () => {
  const trace: PerformanceTrace = {
    sampleId: 10,
    domain: "gpu",
    lane: "physics",
    context: "octree:sentinels",
    capturedAt_ms: 60,
    measurementSource: "gpu-hardware-timestamp",
    total_ms: 7.5,
    phases: [{ id: "other", label: "Frame", duration_ms: 7.5 }],
  };
  const projection = physicsPhaseBoundaryTimeProjection(trace);
  assert.deepEqual(projection.locateTime(frameSentinel(
    PHYSICS_ACTIVITY_FRAME_BEGIN_CHECKPOINT_ID,
    0,
  )), { time_ms: 0, evidence: "reconstructed" });
  assert.deepEqual(projection.locateTime(frameSentinel(
    PHYSICS_ACTIVITY_FRAME_END_CHECKPOINT_ID,
    1,
  )), { time_ms: 7.5, evidence: "reconstructed" });
});

test("known shader heartbeats become reconstructed points inside their measured semantic phase", () => {
  const trace: PerformanceTrace = {
    sampleId: 10,
    domain: "gpu",
    lane: "physics",
    context: "octree:sim-2",
    capturedAt_ms: 60,
    measurementSource: "gpu-hardware-timestamp",
    total_ms: 9,
    phases: [
      { id: "coarse-grid", label: "Topology", duration_ms: 2 },
      { id: "fine-sdf-redistance", label: "Fine redistance", duration_ms: 6 },
      { id: "adaptive-publication", label: "Publication", duration_ms: 1 },
    ],
  };
  const taskId = stableGPULogicalActivityId(
    "task\0octree/fine-redistance-jfa\0jump-flood-a-to-b",
  );
  const heartbeat = { ...event(undefined, 10), taskId };
  const capture = {
    captureId: 1,
    capacity: 4,
    overflowed: false,
    droppedEventCount: 0,
    events: [event(0, 5), heartbeat, event(1, 15)],
  };
  const projection = physicsPhaseBoundaryTimeProjection(trace, capture);

  assert.deepEqual(projection.locateTime(heartbeat), {
    time_ms: 5,
    evidence: "reconstructed",
  });
  assert.equal(physicsPhaseBoundaryTimeProjection(trace).locateTime({
    ...heartbeat,
    taskId: 0x12345678,
  }), undefined);
});

test("module-owned task descriptors place newly adopted shaders in their measured parent phase", () => {
  const trace: PerformanceTrace = {
    sampleId: 12,
    domain: "gpu",
    lane: "physics",
    context: "octree:dynamic-task",
    capturedAt_ms: 80,
    measurementSource: "gpu-hardware-timestamp",
    total_ms: 8,
    phases: [
      { id: "fine-sdf-advection", label: "Fine transport", duration_ms: 3 },
      { id: "adaptive-publication", label: "Fine restriction", duration_ms: 5 },
    ],
  };
  const taskId = stableGPULogicalActivityId(
    "task\0octree/fine-to-coarse-levelset\0restrict-coarse-rows",
  );
  const projection = physicsPhaseBoundaryTimeProjection(trace, undefined, {
    [taskId]: {
      id: "gpu.physics.fine-restriction.rows",
      label: "Fine restriction · restrict coarse rows",
      phaseId: "adaptive-publication",
    },
  });
  assert.deepEqual(projection.locateTime({ ...event(undefined, 1), taskId }), {
    time_ms: 5.5,
    evidence: "reconstructed",
  });
});

test("queue-wall fallback reconstructs shader points from explicit command-ordered phase markers", () => {
  const trace: PerformanceTrace = {
    sampleId: 11,
    domain: "gpu",
    lane: "physics",
    context: "octree:fallback",
    capturedAt_ms: 70,
    measurementSource: "gpu-queue-wall",
    total_ms: 9,
    phases: [{ id: "other", label: "Queue completion", duration_ms: 9 }],
  };
  const heartbeat = { ...event(undefined, 10), taskId: 0x12345678 };
  const capture = {
    captureId: 2,
    capacity: 5,
    overflowed: false,
    droppedEventCount: 0,
    events: [event(0, 5), heartbeat, event(1, 15), event(2, 20)],
  };
  const projection = physicsPhaseBoundaryTimeProjection(trace, capture);

  assert.deepEqual(projection.locateTime(heartbeat), {
    time_ms: 4.5,
    evidence: "reconstructed",
  });
  assert.deepEqual(projection.locateTime(event(2, 20)), {
    time_ms: 9,
    evidence: "reconstructed",
  });
});

test("physics activity capacity is bounded by both the sampled budget and adapter limits", () => {
  assert.equal(physicsLogicalActivityCaptureCapacity({
    maxStorageBufferBindingSize: 1 << 30,
    maxBufferSize: 1 << 30,
  } as GPUSupportedLimits), 4_096);
  assert.equal(physicsLogicalActivityCaptureCapacity({
    maxStorageBufferBindingSize: 65_536,
    maxBufferSize: 1 << 30,
  } as GPUSupportedLimits), 909);
  assert.equal(maximumPhysicsLogicalActivityCaptureCapacity({
    maxStorageBufferBindingSize: 1 << 30,
    maxBufferSize: 1 << 30,
  } as GPUSupportedLimits), 1_000_000);
  assert.equal(maximumPhysicsLogicalActivityCaptureCapacity({
    maxStorageBufferBindingSize: 65_536,
    maxBufferSize: 1 << 30,
  } as GPUSupportedLimits), 909);
});

test("whole-frame validation requires ordered sentinels, no overflow, registered dispatches, and hardware time", () => {
  const begin = frameSentinel(PHYSICS_ACTIVITY_FRAME_BEGIN_CHECKPOINT_ID, 0);
  const end = frameSentinel(PHYSICS_ACTIVITY_FRAME_END_CHECKPOINT_ID, 2);
  const trace = {
    measurementSource: "gpu-hardware-timestamp" as const,
    phases: [{ id: "other", label: "Frame", duration_ms: 1 }] as PerformanceTrace["phases"],
  };
  const complete = validateGPUPhysicsLogicalActivityCapture({
    captureId: 41,
    capacity: 8,
    overflowed: false,
    droppedEventCount: 0,
    events: [begin, event(0, 1), end],
  }, trace, {
    computeDispatchCount: 3,
    instrumentedComputeDispatchCount: 3,
    unregisteredComputeDispatchCount: 0,
    unregisteredComputePipelineCount: 0,
    unregisteredComputePipelineLabels: [],
  });
  assert.equal(complete.complete, true);
  assert.deepEqual(complete.reasons, []);

  const incomplete = validateGPUPhysicsLogicalActivityCapture({
    captureId: 42,
    capacity: 2,
    overflowed: true,
    droppedEventCount: 5,
    events: [{ ...event(0, 0), taskId: 123 }],
  }, {
    measurementSource: "gpu-queue-wall",
    phases: [
      { id: "other", label: "One", duration_ms: 1 },
      { id: "other", label: "Two", duration_ms: 1 },
    ],
  }, {
    computeDispatchCount: 4,
    instrumentedComputeDispatchCount: 1,
    unregisteredComputeDispatchCount: 3,
    unregisteredComputePipelineCount: 1,
    unregisteredComputePipelineLabels: ["Legacy pressure"],
  });
  assert.equal(incomplete.complete, false);
  assert.deepEqual(incomplete.reasons, [
    "recorder-overflow",
    "missing-frame-begin",
    "missing-frame-end",
    "phase-marker-mismatch",
    "unprofiled-dispatch",
    "timestamp-fallback",
  ]);
  assert.equal(incomplete.attemptedEventCount, 6);
  assert.deepEqual(incomplete.unregisteredComputePipelineLabels, ["Legacy pressure"]);
});

test("sampled physics owns one submission bracketed by logical frame sentinels and readback", () => {
  const source = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const advanceStart = source.indexOf("  advanceTo(time_s: number");
  const advanceEnd = source.indexOf("\n  async readStats()", advanceStart);
  assert.ok(advanceStart >= 0 && advanceEnd > advanceStart);
  const advance = source.slice(advanceStart, advanceEnd);
  assert.equal((advance.match(/this\.device\.queue\.submit\(/g) ?? []).length, 1,
    "one physics advance must own exactly one queue submission");
  const begin = advance.indexOf("this.encodeLogicalActivityFrameBegin(encoder)");
  const end = advance.indexOf("this.encodeLogicalActivityFrameEnd(encoder)");
  const finish = advance.indexOf("logicalActivitySession?.finish()");
  const submit = advance.indexOf("this.device.queue.submit(");
  assert.ok(begin >= 0 && begin < end && end < finish && finish < submit,
    "logical begin/end and same-encoder readback must bracket the sole submission");
});

test("adopted shaders delegate whole-dispatch stratification to the shared ABI", () => {
  const sources = [
    "../lib/webgpu-octree-fine-levelset-redistance.ts",
    "../lib/webgpu-octree-fine-levelset-volume.ts",
    "../lib/webgpu-octree-structured-velocity-gpu.ts",
    "../lib/webgpu-octree.ts",
  ].map((path) => readFileSync(new URL(path, import.meta.url), "utf8"));

  for (const source of sources) {
    assert.match(source, /numWorkgroups:/);
    assert.doesNotMatch(source, /GPU_LOGICAL_ACTIVITY_DEFAULT_WORKGROUP_SAMPLE_LIMIT/);
    assert.doesNotMatch(source, /\.y == 0u && .*\.z == 0u/);
  }
  assert.match(sources[2], /@builtin\(num_workgroups\)activityNumWorkgroups:vec3u/);
  assert.match(sources[3], /@builtin\(num_workgroups\) activityNumWorkgroups:vec3u/);
});
