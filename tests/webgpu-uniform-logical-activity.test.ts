import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID,
  PHYSICS_ACTIVITY_POWER_VOLUME_TASK_ID,
  physicsLogicalActivityCaptureCapacity,
  physicsPhaseBoundaryTimeProjection,
} from "../lib/webgpu-uniform-eulerian";
import { stableGPULogicalActivityId } from "../lib/gpu-logical-activity-adoption";
import type { GPULogicalActivityEvent } from "../lib/gpu-logical-activity";
import type { PerformanceTrace } from "../lib/performance-trace";

const event = (tick: number | undefined, sequence: number): GPULogicalActivityEvent => ({
  sequence,
  taskId: PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID,
  checkpointId: 2,
  ...(tick === undefined ? {} : { tick }),
  workgroupId: [0, 0, 0],
  workgroupEvidence: "measured",
  laneId: 0,
  logicalLaneCount: 1,
  logicalLaneCountEvidence: "reconstructed",
  activeLaneEvidence: "unknown",
  subgroupEvidence: "unknown",
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
  } as GPUSupportedLimits), 1_023);
});

test("adopted shader samples flatten multidimensional workgroup grids", () => {
  const sources = [
    "../lib/webgpu-octree-fine-levelset-redistance.ts",
    "../lib/webgpu-octree-fine-levelset-volume.ts",
    "../lib/webgpu-octree-structured-velocity-gpu.ts",
    "../lib/webgpu-octree.ts",
  ].map((path) => readFileSync(new URL(path, import.meta.url), "utf8"));

  for (const source of sources) {
    assert.match(source, /\.x \+ .*\.x \* \(.*\.y \+ .*\.y \* .*\.z\) < \$\{GPU_LOGICAL_ACTIVITY_DEFAULT_WORKGROUP_SAMPLE_LIMIT\}u/);
    assert.doesNotMatch(source, /\.y == 0u && .*\.z == 0u/);
  }
  assert.match(sources[2], /@builtin\(num_workgroups\)activityNumWorkgroups:vec3u/);
  assert.match(sources[3], /@builtin\(num_workgroups\) activityNumWorkgroups:vec3u/);
});
