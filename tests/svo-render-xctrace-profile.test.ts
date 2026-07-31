import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { detectFrames, renderFrameReportHtml, type FrameReport } from "../tools/xctrace-frame-report";

const profiler = readFileSync(new URL("../tools/profile-svo-render-xctrace.ts", import.meta.url), "utf8");

test("render xctrace profiler captures external Metal counters and anchors real render passes", () => {
  assert.match(profiler, /"Metal GPU Counters"/);
  assert.match(profiler, /"ALU Utilization"|meanAlu/);
  assert.match(profiler, /"Sparse voxel cone-lighting prepass"/);
  assert.match(profiler, /"Sparse voxel dry scene"/);
  assert.match(profiler, /occupancyCounterName: "Fragment Occupancy"/);
  assert.match(profiler, /--reuse-trace/);
  assert.match(profiler, /--reuse-tables/);
  assert.match(profiler, /CounterRowSelector/);
  assert.match(profiler, /FLUID_SVO_DRY_FRAME_ISOLATE_PASS_ENCODERS: timingOnly \? "0" : "1"/);
});

test("render profiles preserve enough variant and source identity for controlled A/Bs", () => {
  assert.match(profiler, /--variant=baseline/);
  assert.match(profiler, /--traversal=hybrid\|canonical\|canonical-parametric\|compact\|wide/);
  assert.match(profiler, /FLUID_SVO_DRY_FRAME_TRAVERSAL: traversal/);
  assert.match(profiler, /sourceProvenance\(\)/);
  assert.match(profiler, /createHash\("sha256"\)/);
  assert.match(profiler, /fingerprint:/);
});

test("frame detection prefers the render prepass over an equally periodic interior pass", () => {
  const intervals = Array.from({ length: 8 }, (_, frame) => [
    { start: frame * 50_000, duration: 8_000, label: "Sparse voxel cone-lighting prepass",
      encoders: ["Sparse voxel cone-lighting prepass"], channel: "Fragment", merged: false },
    { start: frame * 50_000 + 8_000, duration: 34_000, label: "Sparse voxel dry scene",
      encoders: ["Sparse voxel dry scene"], channel: "Fragment", merged: false },
  ]).flat();
  const detected = detectFrames(intervals,
    ["Sparse voxel cone-lighting prepass", "Sparse voxel dry scene"]);
  assert.equal(detected.anchor, "Sparse voxel cone-lighting prepass");
  assert.equal(detected.boundaries.length, 8);
});

test("frame detection tolerates one externally delayed render frame", () => {
  const intervals = Array.from({ length: 80 }, (_, frame) => {
    const delayedStart = frame >= 40 ? 75_000 : 0;
    return {
      start: frame * 20_000 + delayedStart,
      duration: 8_000,
      label: "Sparse voxel primary visibility",
      encoders: ["Sparse voxel primary visibility"],
      channel: "Fragment",
      merged: false,
    };
  });
  const detected = detectFrames(intervals, ["Sparse voxel primary visibility"]);
  assert.equal(detected.anchor, "Sparse voxel primary visibility");
  assert.equal(detected.boundaries.length, 80);
});

test("frame reports render with rendering vocabulary", () => {
  const html = renderFrameReportHtml({
    workUnit: "frame", title: "SVO rendering — GPU frame profile",
  } as FrameReport);
  assert.match(html, /SVO rendering — GPU frame profile/);
  assert.match(html, /One frame, end to end/);
  assert.doesNotMatch(html, /One advance, end to end/);
});
