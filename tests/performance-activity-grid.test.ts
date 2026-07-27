import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  buildPerformanceActivityView,
  DEFAULT_ACTIVITY_MILLISECONDS_WIDTH_PX,
  GPU_ACTIVITY_DISPLAY_BAND_LIMIT,
  PerformanceActivityGrid,
  performanceActivityTaskLegend,
  performanceActivityTicks,
  projectGpuStageEnvelopes,
} from "../components/PerformanceActivityGrid";
import type { ActivityEvidence, PerformanceActivityFrame } from "../lib/performance-activity";
import type { PerformanceTrace } from "../lib/performance-trace";

const trace = (patch: Partial<PerformanceTrace> & Pick<PerformanceTrace, "lane" | "total_ms" | "phases">): PerformanceTrace => ({
  sampleId: 7,
  domain: patch.lane === "main-thread" ? "cpu" : "gpu",
  context: "octree:balanced",
  capturedAt_ms: 100,
  measurementSource: patch.lane === "main-thread" ? "cpu-active-wall" : "gpu-hardware-timestamp",
  ...patch,
});

const activityFrame = (
  duration_ms: number,
  evidence: readonly ActivityEvidence[] = ["measured", "reconstructed", "idle", "unknown"],
): PerformanceActivityFrame => ({
  identity: { frameId: `frame-${duration_ms}`, generation: 1 },
  context: "octree:balanced",
  capturedAt_cpu_ms: duration_ms,
  capturedAt_epoch_ms: 1_000 + duration_ms,
  cpuTimeOrigin_ms: 1_000,
  clocks: [{ id: "cpu-performance", label: "CPU performance clock", status: { state: "reference" } }],
  tasks: [{ id: "solve", label: "Solve", color: "#e8bf5e", lane: "cpu-main" }],
  resources: [{ id: "cpu.main", label: "CPU · main thread", kind: "cpu-main", lane: "cpu-main", clockDomain: "cpu-performance" }],
  spans: [{
    id: "solve-span", kind: "task", taskId: "solve", resourceId: "cpu.main", clockDomain: "cpu-performance",
    start_ms: 2, end_ms: Math.min(7, duration_ms), evidence: "measured", identity: { frameId: `frame-${duration_ms}`, generation: 1 },
  }],
  events: [],
  captureDiagnostics: { reasons: [] },
  rows: [{
    resource: { id: "cpu.main", label: "CPU · main thread", kind: "cpu-main", lane: "cpu-main", clockDomain: "cpu-performance" },
    windowStart_ms: 0,
    windowEnd_ms: duration_ms,
    slice_ms: 1,
    slices: evidence.map((state, index) => ({
      index,
      start_ms: index,
      end_ms: Math.min(duration_ms, index + 1),
      evidence: state,
      ...(state === "measured" || state === "reconstructed" ? { taskId: "solve" } : {}),
      active_ms: state === "measured" || state === "reconstructed" ? 1 : 0,
      spanIds: state === "measured" || state === "reconstructed" ? ["solve-span"] : [],
      eventIds: [],
    })),
  }],
});

const activityFrameWithGpuRows = (rowCount: number, duration_ms = 4): PerformanceActivityFrame => {
  const base = activityFrame(duration_ms, Array.from({ length: duration_ms }, () => "idle"));
  const gpuTask = { id: "gpu-solve", label: "GPU solve", color: "#32a9b8", lane: "gpu-physics" as const };
  const aggregateResource: PerformanceActivityFrame["resources"][number] = {
    id: "gpu.physics.aggregate",
    label: "GPU · physics queue aggregate",
    kind: "gpu-aggregate",
    lane: "gpu-physics",
    clockDomain: "cpu-performance",
  };
  const aggregateRow: PerformanceActivityFrame["rows"][number] = {
    resource: aggregateResource,
    windowStart_ms: 0,
    windowEnd_ms: duration_ms,
    slice_ms: 1,
    slices: Array.from({ length: duration_ms }, (_, index) => ({
      index,
      start_ms: index,
      end_ms: index + 1,
      evidence: "reconstructed" as const,
      taskId: gpuTask.id,
      active_ms: 1,
      spanIds: ["gpu-stage-span"],
      eventIds: [],
    })),
  };
  const gpuRows = Array.from({ length: rowCount }, (_, capacitySlot) => {
    const resource: PerformanceActivityFrame["resources"][number] = {
      id: `gpu.wg.${capacitySlot}`,
      label: `Logical workgroup ${capacitySlot}`,
      kind: "gpu-logical-capacity",
      lane: "gpu-physics",
      clockDomain: "cpu-performance",
      capacitySlot,
    };
    return {
      resource,
      windowStart_ms: 0,
      windowEnd_ms: duration_ms,
      slice_ms: 1 as const,
      slices: Array.from({ length: duration_ms }, (_, index) => {
        const measured = capacitySlot < (index + 1) * 4;
        return {
          index,
          start_ms: index,
          end_ms: index + 1,
          evidence: measured ? "measured" as const : "idle" as const,
          ...(measured ? { taskId: gpuTask.id } : {}),
          active_ms: 0,
          spanIds: [],
          eventIds: measured ? [`heartbeat-${capacitySlot}-${index}`] : [],
        };
      }),
    };
  });
  return {
    ...base,
    tasks: [...base.tasks, gpuTask],
    resources: [...base.resources, aggregateResource, ...gpuRows.map((row) => row.resource)],
    spans: [...base.spans, {
      id: "gpu-observation",
      kind: "observation",
      resourceId: aggregateResource.id,
      clockDomain: aggregateResource.clockDomain,
      start_ms: 0,
      end_ms: duration_ms,
      evidence: "measured",
      identity: base.identity,
    }],
    rows: [...base.rows, aggregateRow, ...gpuRows],
  };
};

test("existing exact traces become duration-comparable local-origin task lanes", () => {
  const cpu = trace({
    lane: "main-thread",
    total_ms: 5,
    phases: [
      { id: "frame-control", label: "Frame control", duration_ms: 2 },
      { id: "command-encoding", label: "Encode presentation", duration_ms: 3 },
    ],
  });
  const physics = trace({
    lane: "physics",
    total_ms: 8,
    phases: [
      { id: "coarse-grid", label: "Grid topology", duration_ms: 3 },
      { id: "pressure-solve", label: "Pressure solve", duration_ms: 5 },
    ],
  });

  const view = buildPerformanceActivityView({ cpu, physics });
  assert.equal(view.synchronized, false);
  assert.equal(view.duration_ms, 8);
  assert.deepEqual(view.lanes.map((lane) => lane.clockOrigin), ["local", "local"]);
  assert.deepEqual(view.lanes[0].tasks.map((task) => [task.start_ms, task.end_ms]), [[0, 2], [2, 5]]);
  assert.deepEqual(view.lanes[1].tasks.map((task) => [task.start_ms, task.end_ms]), [[0, 3], [3, 8]]);
  assert.ok(view.lanes.every((lane) => lane.tasks.every((task) => task.evidence === "reconstructed")));
  assert.ok(view.resources.every((resource) => resource.segments.every((segment) => segment.evidence === "reconstructed")));
  assert.match(view.resources[1].segments[0].detail ?? "", /not a physical core measurement/);
});

test("queue-wall fallback is not presented as measured semantic activity", () => {
  const presentation = trace({
    lane: "presentation",
    total_ms: 4,
    measurementSource: "gpu-queue-wall",
    phases: [{ id: "other", label: "GPU queue completion", duration_ms: 4 }],
  });
  const view = buildPerformanceActivityView({ presentation });
  assert.equal(view.lanes[0].tasks[0].evidence, "unknown");
  assert.equal(view.resources[0].segments[0].evidence, "unknown");
});

test("a correlated activity frame preserves absolute starts and all evidence states", () => {
  const frame = activityFrame(10);
  const view = buildPerformanceActivityView({ frame });
  assert.equal(view.synchronized, true);
  assert.deepEqual([view.lanes[0].tasks[0].start_ms, view.lanes[0].tasks[0].end_ms], [2, 7]);
  assert.deepEqual(view.resources[0].segments.map((segment) => segment.evidence), ["measured-progress", "reconstructed", "idle", "unknown"]);
  assert.ok(view.resources[0].segments.every((segment) => segment.end_ms - segment.start_ms === 1));
});

test("logical GPU workgroups are projected into at most sixteen utilization bands", () => {
  const frame = activityFrameWithGpuRows(128);
  const view = buildPerformanceActivityView({ frame });
  const gpuBands = view.resources.filter((resource) => resource.matrixKind === "gpu-logical");

  assert.equal(frame.rows.length, 130, "the exhaustive source frame remains unchanged");
  assert.equal(view.logicalGpuResourceCount, 128);
  assert.equal(gpuBands.length, GPU_ACTIVITY_DISPLAY_BAND_LIMIT);
  assert.deepEqual(gpuBands.map((band) => band.sourceResourceCount), Array(16).fill(8));
  assert.match(gpuBands[0].label, /Stratified WG bin 01 · strata 1–8/);
  assert.equal(gpuBands[0].segments[0].activeResourceCount, 4);
  assert.equal(gpuBands[0].segments[0].resourceCount, 8);
  assert.equal(gpuBands[0].segments[0].activeFraction, 0.5);
  assert.equal(gpuBands[0].segments[0].evidence, "measured-progress");
  assert.equal(gpuBands[0].segments[0].color, "#32a9b8");
  assert.match(gpuBands[0].segments[0].detail ?? "", /not a physical execution unit/);
  assert.equal(view.resources.filter((resource) => resource.matrixKind === "cpu-context").length, 1);
});

test("active-lane ballots drive logical workgroup intensity instead of binary sample coverage", () => {
  const base = activityFrameWithGpuRows(1, 1);
  const logicalResource = base.resources.find((resource) => resource.kind === "gpu-logical-capacity");
  assert.ok(logicalResource);
  const frame: PerformanceActivityFrame = {
    ...base,
    events: [{
      id: "heartbeat-0-0",
      kind: "instant",
      taskId: "gpu-solve",
      resourceId: logicalResource.id,
      clockDomain: logicalResource.clockDomain,
      at_ms: 0.5,
      evidence: "measured",
      identity: base.identity,
      metadata: { activeLaneCount: 8, logicalLaneCount: 32 },
    }],
  };
  const view = buildPerformanceActivityView({ frame });
  const segment = view.resources.find((resource) => resource.matrixKind === "gpu-logical")?.segments[0];
  assert.ok(segment);
  assert.equal(segment.activeLaneCount, 8);
  assert.equal(segment.logicalLaneCount, 32);
  assert.equal(segment.activeFraction, 0.25);
  assert.match(segment.detail ?? "", /25\.0% sampled logical utilization/);

  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, { frame }));
  assert.match(markup, /data-active-lanes="8"/);
  assert.match(markup, /data-logical-lanes="32"/);
  assert.match(markup, /ACTIVE-LANE INTENSITY/);
});

test("timestamped stages bridge sparse shader checkpoints only on rows which sampled that stage", () => {
  const aggregateTask = {
    id: "gpu.physics.pressure-solve",
    label: "Pressure solve",
    color: "#e8bf5e",
    lane: "gpu-physics" as const,
    stageId: "pressure-solve",
  };
  const shaderTask = {
    id: "gpu.physics.persistent-mgpcg.whole-solve",
    label: "Persistent MGPCG · whole solve",
    color: "#32a9b8",
    lane: "gpu-physics" as const,
    stageId: "pressure-solve",
  };
  const unknown = (index: number) => ({
    start_ms: index,
    end_ms: index + 1,
    evidence: "unknown" as const,
    activeFraction: 0,
  });
  const resources = projectGpuStageEnvelopes({
    taskById: new Map([[aggregateTask.id, aggregateTask], [shaderTask.id, shaderTask]]),
    resources: [{
      id: "aggregate",
      label: "GPU physics queue",
      group: "gpu",
      matrixKind: "aggregate",
      segments: [{
        start_ms: 0,
        end_ms: 5,
        evidence: "reconstructed",
        taskId: aggregateTask.id,
        stageId: aggregateTask.stageId,
        color: aggregateTask.color,
        activeFraction: 1,
        label: aggregateTask.label,
      }],
    }, {
      id: "sampled-band",
      label: "Sampled band",
      group: "gpu",
      matrixKind: "gpu-logical",
      segments: [unknown(0), {
        start_ms: 1,
        end_ms: 2,
        evidence: "measured-progress",
        taskId: shaderTask.id,
        stageId: shaderTask.stageId,
        color: shaderTask.color,
        activeFraction: 0.25,
        activeLaneCount: 8,
        logicalLaneCount: 32,
        label: shaderTask.label,
      }, unknown(2), unknown(3), unknown(4)],
    }, {
      id: "unsampled-band",
      label: "Unsampled band",
      group: "gpu",
      matrixKind: "gpu-logical",
      segments: [unknown(0), unknown(1), unknown(2), unknown(3), unknown(4)],
    }],
  });

  const sampled = resources.find((resource) => resource.id === "sampled-band");
  const unsampled = resources.find((resource) => resource.id === "unsampled-band");
  assert.ok(sampled);
  assert.ok(unsampled);
  assert.deepEqual(sampled.segments.map((segment) => [segment.start_ms, segment.end_ms, segment.evidence]), [
    [0, 1, "reconstructed"],
    [1, 2, "measured-progress"],
    [2, 5, "reconstructed"],
  ]);
  assert.equal(sampled.segments[0].projection, "stage-envelope");
  assert.equal(sampled.segments[0].activeFraction, 0.25);
  assert.match(sampled.segments[0].detail ?? "", /inferred time, excluded from utilization, not hardware occupancy/);
  assert.ok(unsampled.segments.every((segment) => segment.evidence === "unknown"));
});

test("rendered GPU matrix output is bounded by display bands rather than raw workgroup rows", () => {
  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, {
    frame: activityFrameWithGpuRows(128),
  }));
  assert.equal((markup.match(/class="activity-cell"/g) ?? []).length, 4 + 4 + 16 * 4);
  assert.match(markup, /GPU LOGICAL WORKGROUP BINS/);
  assert.match(markup, /128 STRATIFIED SAMPLES/);
  assert.match(markup, /128 \/ 16/);
  assert.match(markup, /data-active-resources="4"/);
});

test("task legend is populated from the registered catalog and distinguishes observed tasks", () => {
  const base = activityFrameWithGpuRows(8);
  const frame: PerformanceActivityFrame = {
    ...base,
    tasks: [
      ...base.tasks.map((task) => task.id === "gpu-solve"
        ? { ...task, stageId: "pressure-solve" }
        : { ...task, stageId: "frame-control" }),
      {
        id: "gpu-future",
        label: "Registered future stage",
        color: "#a079be",
        lane: "gpu-physics",
        stageId: "adaptive-publication",
      },
    ],
  };
  const legend = performanceActivityTaskLegend(frame);
  const entries = legend.flatMap((group) => group.entries);
  assert.equal(entries.find((entry) => entry.id === "gpu-solve")?.observed, true);
  assert.equal(entries.find((entry) => entry.id === "gpu-future")?.observed, false);
  assert.ok(legend.some((group) => group.id === "pressure-solve"));

  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, { frame }));
  assert.match(markup, /Dynamically registered task and stage legend/);
  assert.match(markup, /TASK \/ STAGE LEGEND/);
  assert.match(markup, /2 OBSERVED · 3 REGISTERED/);
  assert.match(markup, /Registered future stage/);
  assert.match(markup, /data-observed="false"/);
});

test("the matrix defaults to a compact adjustable horizontal millisecond scale", () => {
  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, {
    frame: activityFrame(250),
  }));

  assert.equal(DEFAULT_ACTIVITY_MILLISECONDS_WIDTH_PX, 2);
  assert.match(markup, /aria-label="Matrix horizontal scale in pixels per millisecond"/);
  assert.match(markup, /min="1"/);
  assert.match(markup, /max="8"/);
  assert.match(markup, /2 PX\/MS/);
  assert.match(markup, /--activity-ms-width:2px/);
});

test("the shared millisecond ruler uses stable nice ticks", () => {
  assert.deepEqual(performanceActivityTicks(8), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual(performanceActivityTicks(27), [0, 5, 10, 15, 20, 25]);
});

test("retained capture metadata renders a compact reference delta", () => {
  const cpu = trace({
    lane: "main-thread",
    total_ms: 5,
    phases: [{ id: "frame-control", label: "Frame control", duration_ms: 5 }],
  });
  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, {
    cpu,
    captureLabel: "Frame 18",
    referenceFrame: activityFrame(7),
    referenceLabel: "Frame 12",
  }));
  assert.match(markup, /Frame 18/);
  assert.match(markup, /Frame 12/);
  assert.match(markup, /−2\.00 ms/);
  assert.match(markup, /activity-capture-comparison/);
});

test("the component labels clock and resource evidence honestly", () => {
  const component = readFileSync(new URL("../components/PerformanceActivityGrid.tsx", import.meta.url), "utf8");
  const css = readFileSync(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(component, /INDEPENDENT LANE ORIGINS/);
  assert.match(component, /data-clock-alignment=/);
  assert.match(component, /data-queue-wall-fallback=/);
  assert.match(component, /Measured progress/);
  assert.match(component, /Reconstructed/);
  assert.match(component, /Measured idle/);
  assert.match(component, /Unknown/);
  assert.match(component, /Resource × 1 ms activity matrix/);
  assert.doesNotMatch(component, /Task accounting/);
  assert.match(css, /\.activity-resource-row \{ min-height: 13px/);
  assert.match(css, /\.activity-task-legend \{ height: 114px; min-height: 114px;/,
    "the dynamic task catalog must not vertically reflow the matrix between captures");
  assert.match(css, /\.activity-task-legend-groups \{ height: 92px;/);
  for (const state of ["measured-progress", "reconstructed", "idle", "unknown"]) {
    assert.match(css, new RegExp(`data-evidence=\\"${state}\\"`));
  }
  assert.doesNotMatch(component, /GPU core [0-9]/);
  assert.doesNotMatch(component, /<div className="activity-task-lanes">/);
  assert.match(component, /UNIFIED FRAME ACCOUNTING/);
  assert.match(component, /ACCOUNTING LEDGERS/);
  assert.doesNotMatch(component, /activity-clock-notice/);
});

test("the task legend retains its fixed slot while the catalog is awaiting capture", () => {
  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, {}));
  assert.match(markup, /class="activity-task-legend" data-empty="true"/);
  assert.match(markup, /TASK CATALOG AWAITING CAPTURE/);
});

test("an incomplete retained capture withholds utilization and never renders idle as proven", () => {
  const base = activityFrame(4, ["measured", "idle", "idle", "unknown"]);
  const frame: PerformanceActivityFrame = {
    ...base,
    captureDiagnostics: {
      reasons: ["recorder-overflow", "missing-frame-end", "unprofiled-dispatch"],
      recorderOverflowed: true,
      droppedEventCount: 23,
      unprofiledDispatchCount: 37,
      unprofiledPipelineLabels: ["fine transport", "pressure apply"],
    },
  };
  const view = buildPerformanceActivityView({ frame });
  assert.equal(view.captureState, "incomplete");
  assert.deepEqual(view.resources[0].segments.map((segment) => segment.evidence),
    ["measured-progress", "unknown", "unknown", "unknown"]);

  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, { frame }));
  assert.match(markup, /data-capture-completeness="incomplete"/);
  assert.match(markup, /CAPTURE INCOMPLETE · 23 DROPPED/);
  assert.match(markup, /RECORDER OVERFLOW · MISSING FRAME END · UNPROFILED DISPATCH/);
  assert.match(markup, /37 unprofiled dispatches · fine transport, pressure apply/);
  assert.match(markup, /UTILIZATION/);
  assert.match(markup, /WITHHELD/);
  assert.match(markup, /IDLE AND UTILIZATION NOT ASSERTED/);
  assert.equal((markup.match(/class="activity-cell"/g) ?? []).length, 1,
    "unknown buckets use the row's fail-closed grid paint instead of inert DOM nodes");
});

test("the rendered central view is matrix-first with individually addressable time cells", () => {
  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, {
    frame: activityFrameWithGpuRows(1),
  }));
  assert.match(markup, /Resource × 1 ms activity matrix/);
  assert.equal((markup.match(/class="activity-cell"/g) ?? []).length, 12,
    "the measured stage envelope remains visible alongside logical workgroup cells");
  assert.match(markup, /--activity-cell-color:#32a9b8/);
  assert.match(markup, /GPU LOGICAL CAPACITY|CPU SOFTWARE RESOURCES/);
  assert.match(markup, /Resource activity matrix/);
  assert.match(markup, /TIMED TASK \/ STAGE ENVELOPES/);
  assert.match(markup, /1\/1 CLOSED/);
  assert.doesNotMatch(markup, /AGGREGATE FALLBACK/);
  assert.doesNotMatch(markup, /Task map/);
});
