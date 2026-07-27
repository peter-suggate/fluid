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
  performanceActivityTicks,
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
    resources: [...base.resources, ...gpuRows.map((row) => row.resource)],
    rows: [...base.rows, ...gpuRows],
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

  assert.equal(frame.rows.length, 129, "the exhaustive source frame remains unchanged");
  assert.equal(view.logicalGpuResourceCount, 128);
  assert.equal(gpuBands.length, GPU_ACTIVITY_DISPLAY_BAND_LIMIT);
  assert.deepEqual(gpuBands.map((band) => band.sourceResourceCount), Array(16).fill(8));
  assert.match(gpuBands[0].label, /Logical WG bin 01 · slots 0–7/);
  assert.equal(gpuBands[0].segments[0].activeResourceCount, 4);
  assert.equal(gpuBands[0].segments[0].resourceCount, 8);
  assert.equal(gpuBands[0].segments[0].activeFraction, 0.5);
  assert.equal(gpuBands[0].segments[0].evidence, "measured-progress");
  assert.equal(gpuBands[0].segments[0].color, "#32a9b8");
  assert.match(gpuBands[0].segments[0].detail ?? "", /display bin, not a physical execution unit/);
  assert.equal(view.resources.filter((resource) => resource.matrixKind === "cpu-context").length, 1);
});

test("rendered GPU matrix output is bounded by display bands rather than raw workgroup rows", () => {
  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, {
    frame: activityFrameWithGpuRows(128),
  }));
  assert.equal((markup.match(/class="activity-cell"/g) ?? []).length, 4 + 16 * 4);
  assert.match(markup, /GPU LOGICAL WORKGROUP BINS/);
  assert.match(markup, /128 WORKGROUPS/);
  assert.match(markup, /128 \/ 16/);
  assert.match(markup, /data-active-resources="4"/);
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
  for (const state of ["measured-progress", "reconstructed", "idle", "unknown"]) {
    assert.match(css, new RegExp(`data-evidence=\\"${state}\\"`));
  }
  assert.doesNotMatch(component, /GPU core [0-9]/);
  assert.doesNotMatch(component, /<div className="activity-task-lanes">/);
  assert.match(component, /UNIFIED FRAME ACCOUNTING/);
  assert.match(component, /ACCOUNTING LEDGERS/);
  assert.doesNotMatch(component, /activity-clock-notice/);
});

test("the rendered central view is matrix-first with individually addressable time cells", () => {
  const markup = renderToStaticMarkup(createElement(PerformanceActivityGrid, {
    frame: activityFrame(4),
  }));
  assert.match(markup, /Resource × 1 ms activity matrix/);
  assert.equal((markup.match(/class="activity-cell"/g) ?? []).length, 4);
  assert.match(markup, /--activity-cell-color:#e8bf5e/);
  assert.match(markup, /GPU LOGICAL CAPACITY|CPU SOFTWARE RESOURCES/);
  assert.match(markup, /Resource activity matrix/);
  assert.doesNotMatch(markup, /AGGREGATE FALLBACK/);
  assert.doesNotMatch(markup, /Task map/);
});
