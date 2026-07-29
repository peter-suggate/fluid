import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildFrameReport } from "../tools/xctrace-frame-report";

/**
 * The report's per-advance figures are only meaningful if a "frame" really is
 * one advance, so the numbers that prove it -- encoders and passes per frame,
 * and the per-frame series they come from -- are pinned here against a stream
 * whose true segmentation is known by construction.
 */

/** Instruments formats a timestamp as mm:ss.mmm.uuu and a duration in µs. */
const stamp = (microseconds: number): string => {
  const totalMs = Math.floor(microseconds / 1000);
  const mm = String(Math.floor(totalMs / 60000)).padStart(2, "0");
  const ss = String(Math.floor((totalMs % 60000) / 1000)).padStart(2, "0");
  const ms = String(totalMs % 1000).padStart(3, "0");
  const us = String(Math.round(microseconds % 1000)).padStart(3, "0");
  return `${mm}:${ss}.${ms}.${us}`;
};

/** One advance = three encoders; `extra` adds a fourth to that advance only. */
const TASKS = ["Advect", "Solve pressure", "Publish"];

const writeTrace = (options: {
  frames: number; frameUs: number; extraOn?: readonly number[];
  spacingUs?: number; intervalUs?: number; tasks?: readonly string[];
  slicesPerEncoder?: number;
}): Record<string, string> => {
  const directory = mkdtempSync(join(tmpdir(), "xctrace-seg-"));
  const intervals: string[] = [];
  const encoders: string[] = [];
  const base = options.tasks ?? TASKS;
  let id = 0;
  for (let frame = 0; frame < options.frames; frame += 1) {
    const labels = options.extraOn?.includes(frame) ? [...base, "Rebuild topology"] : base;
    labels.forEach((label, slot) => {
      id += 1;
      const encoderId = `0x${id.toString(16)}`;
      const start = 1_000_000 + frame * options.frameUs + slot * (options.spacingUs ?? 1000);
      encoders.push(JSON.stringify({
        "encoder-id": encoderId, "encoder-label": `Command Buffer 0:${label}`,
        process: "node (4242)",
      }));
      for (let slice = 0; slice < (options.slicesPerEncoder ?? 1); slice += 1) {
        intervals.push(JSON.stringify({
          "encoder-id": encoderId, start: stamp(start + slice * 250),
          duration: `${(options.intervalUs ?? 500).toFixed(2)} µs`,
          "channel-name": "Compute", process: "node (4242)",
          "event-label": `${label}      ( node (4242) )  0x1`,
        }));
      }
    });
  }
  writeFileSync(join(directory, "intervals.ndjson"), `${intervals.join("\n")}\n`);
  writeFileSync(join(directory, "encoders.ndjson"), `${encoders.join("\n")}\n`);
  return {
    "gpu-intervals": join(directory, "intervals.ndjson"),
    encoders: join(directory, "encoders.ndjson"),
  };
};

const build = async (tables: Record<string, string>, steps?: number,
  environment: Record<string, string> = {}, singleFrame = false, firstFrame = false) => buildFrameReport({
  tables, lane: "mini", environment, tracedPid: 4242,
  traced: steps === undefined ? undefined : { steps, simulationWall_ms: steps * 50 },
  singleFrame, firstFrame,
});

test("each detected frame is one advance and carries that advance's work", async () => {
  const report = await build(writeTrace({ frames: 12, frameUs: 50_000 }), 12);
  // The first and last anchors are clipped, so 12 advances yield 9 analysed.
  assert.equal(report.frames.count, report.frames.samples.length);
  assert.ok(report.frames.count >= 8, `expected most advances analysed, got ${report.frames.count}`);
  for (const frame of report.frames.samples) {
    assert.equal(frame.encoders, TASKS.length, `frame ${frame.index} encoder count`);
    assert.equal(frame.passes, TASKS.length, `frame ${frame.index} pass count`);
    assert.ok(Math.abs(frame.durationMs - 50) < 0.01, `frame ${frame.index} duration`);
  }
});

test("the per-frame series reconciles with the per-advance averages", async () => {
  const report = await build(writeTrace({ frames: 12, frameUs: 50_000 }), 12);
  const samples = report.frames.samples;
  const mean = (pick: (frame: typeof samples[number]) => number): number => samples
    .reduce((sum, frame) => sum + pick(frame), 0) / samples.length;
  // Whatever the report says an advance costs must be what the advances cost.
  assert.ok(Math.abs(mean((frame) => frame.busyMs) - report.gpu.busyMsPerFrame) < 1e-6);
  assert.ok(Math.abs(mean((frame) => frame.durationMs) - report.gpu.wallMsPerFrame) < 1e-6);
  assert.ok(Math.abs(mean((frame) => frame.passes) - report.gpu.passesPerFrame) < 1e-6);
  for (const frame of samples) {
    // Busy + idle must account for the whole advance, with nothing invented.
    assert.ok(Math.abs(frame.busyMs + frame.gapMs - frame.durationMs) < 1e-9);
    const tasks = frame.tasks.reduce((sum, value) => sum + value, 0);
    assert.ok(Math.abs(tasks - frame.busyMs) < 1e-9, "per-task ms must sum to the frame's busy ms");
  }
});

test("overlapping Metal interval records are counted once as GPU busy time", async () => {
  const report = await build(writeTrace({
    frames: 12, frameUs: 50_000, spacingUs: 1000, intervalUs: 1500,
  }), 12);
  assert.ok(Math.abs(report.gpu.intervalMsPerFrame - 4.5) < 1e-9,
    "three attributed 1.5 ms records remain visible as interval time");
  assert.ok(Math.abs(report.gpu.busyMsPerFrame - 3.5) < 1e-9,
    "the union of overlapping records is the actual busy span");
  assert.ok(Math.abs(report.gpu.overlapMsPerFrame - 1) < 1e-9);
  assert.ok(report.frames.samples.every((frame) =>
    Math.abs(frame.busyMs + frame.gapMs - frame.durationMs) < 1e-9));
});

test("one representative frame coalesces resumed encoder slices without losing their cost", async () => {
  const report = await build(writeTrace({
    frames: 12, frameUs: 50_000, intervalUs: 200, slicesPerEncoder: 2,
  }), 12, { FLUID_GPU_ISOLATE_PASS_LABELS: "1" }, true);
  assert.equal(report.frames.count, 1);
  assert.equal(report.frames.samples.length, 1);
  assert.equal(report.frames.captures.length, 1);
  assert.equal(report.frames.samples[0].encoders, TASKS.length,
    "resumed slices are not additional encoder calls");
  assert.equal(report.frames.captures[0].intervals.length, TASKS.length,
    "the timeline contains one item per Metal encoder");
  assert.equal(new Set(report.timeline.intervals.map((interval) => interval.encoderId)).size,
    report.timeline.intervals.length);
  const advect = report.passes.find((pass) => pass.label === "Advect");
  assert.equal(advect?.callsPerFrame, 1);
  assert.ok(Math.abs((advect?.gpuMsPerFrame ?? 0) - 0.4) < 1e-9,
    "both 0.2 ms execution slices contribute to stage GPU time");
});

test("literal first-frame mode selects advance 1 and uses advance 2 as its exact boundary", async () => {
  const report = await build(writeTrace({ frames: 2, frameUs: 50_000,
    tasks: ["Open coupled topology ready-commit gate", "Advect", "Solve pressure"] }), 2,
    { FLUID_GPU_ISOLATE_PASS_LABELS: "1" }, false, true);
  assert.equal(report.frames.count, 1);
  assert.equal(report.frames.firstAdvance, 1);
  assert.equal(report.frames.samples[0].durationMs, 50);
  assert.equal(report.frames.captures.length, 1);
  assert.match(report.console[0], /literal advance 1 only/);
});

test("encoder-isolation blits are overhead, not duplicate solver stages", async () => {
  const report = await build(writeTrace({
    frames: 12, frameUs: 50_000,
    tasks: ["Blit Command 42", "Open coupled topology ready-commit gate", "Solve pressure"],
  }), 12, {
    FLUID_GPU_ISOLATE_PASS_LABELS: "1",
    FLUID_GPU_ISOLATE_PASS_ENCODERS: "1",
  }, true);
  assert.deepEqual(report.passes.map((pass) => pass.label).sort(),
    ["Open coupled topology ready-commit gate", "Solve pressure"]);
  assert.equal(report.frames.anchor, "Open coupled topology ready-commit gate");
  assert.equal(report.timeline.intervals[0].label,
    "Open coupled topology ready-commit gate");
  assert.equal(report.timeline.intervals[0].start, 1000,
    "time zero is the preceding setup blit, one millisecond before first compute in this fixture");
  assert.ok(report.timeline.intervals.every((interval) =>
    !interval.label.startsWith("Blit Command")));
  assert.equal(report.gpu.diagnosticBlitsPerFrame, 1);
  assert.ok(Math.abs(report.gpu.diagnosticBlitMsPerFrame - 0.5) < 1e-9);
  assert.equal(report.frames.samples[0].encoders, 2);
});

test("an advance doing extra work is reported as such, not as a broken boundary", async () => {
  const report = await build(writeTrace({ frames: 14, frameUs: 50_000, extraOn: [5, 9] }), 14);
  const shapes = new Set(report.frames.samples.map((frame) => `${frame.encoders}/${frame.passes}`));
  assert.equal(shapes.size, 2, "the odd advances must show up as a second shape");
  const boundary = report.console.find((line) => line.startsWith("frame boundary:"));
  const shape = report.console.find((line) => line.startsWith("frame shape:"));
  assert.ok(boundary?.includes("one per advance"),
    `boundary should validate against the step count, got: ${boundary}`);
  assert.ok(!boundary?.includes("NOT AN ADVANCE"));
  assert.ok(shape?.includes("do more or less work"), `shape census should report it, got: ${shape}`);
});

test("a capture covering part of the run says so instead of claiming the whole run", async () => {
  const report = await build(writeTrace({ frames: 12, frameUs: 50_000 }), 400);
  const boundary = report.console.find((line) => line.startsWith("frame boundary:"));
  assert.ok(boundary?.includes("% of the run"), `expected a coverage note, got: ${boundary}`);
  assert.equal(report.frames.firstAdvance, undefined,
    "advance numbers are unknowable when the capture is a window");
});

/**
 * A partially scoped capture still prints a full pass table, and every row in
 * it reads like a kernel. On the mini lane 2026-07-28 that let a 3.55 ms row
 * named `Structured boundary worksets - count row classes` -- in truth the
 * SPGrid candidate hierarchy rebuild sharing that pass -- be quoted as the
 * classify kernel's cost. The report must refuse to present it that way.
 */
const SCOPED = {
  FLUID_GPU_ISOLATE_PASS_LABELS: "1",
  FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES: "Fine JFA ·",
};
const MIXED = ["Fine JFA - flood", "Structured boundary worksets - count row classes", "Publish"];

test("a partially scoped capture reports its unscoped rows as composite, by name", async () => {
  const report = await build(
    writeTrace({ frames: 12, frameUs: 50_000, tasks: MIXED }), 12, SCOPED);
  assert.equal(report.attribution.mode, "scoped");
  // The broker normalises "·" to "-" before matching, so the report must too:
  // a prefix that isolates on the GPU and fails to match here would demote a
  // real stage to a composite bucket without anyone noticing.
  assert.deepEqual(report.attribution.isolatedPrefixes, ["Fine JFA -"]);
  const byLabel = new Map(report.passes.map((pass) => [pass.label, pass]));
  const exact = byLabel.get("Fine JFA - flood");
  assert.equal(exact?.exactAttribution, true, "the isolated prefix is an exact stage");
  assert.equal(exact?.compositeReason, undefined);
  for (const label of ["Structured boundary worksets - count row classes", "Publish"]) {
    const pass = byLabel.get(label);
    assert.equal(pass?.exactAttribution, false, `${label} was never isolated`);
    assert.ok(pass?.compositeReason?.includes("next pass boundary"),
      `${label} must say what its number actually covers: ${pass?.compositeReason}`);
  }
  assert.equal(report.attribution.exactBuckets, 1);
  assert.equal(report.attribution.compositeBuckets, 2);
  assert.ok(report.attribution.largestComposites
    .some((entry) => entry.label === "Structured boundary worksets - count row classes"));
  // The caveat has to be where a reader lands, not buried under the table.
  const banner = report.console.find((line) => line.startsWith("attribution:"));
  assert.ok(banner?.includes("PARTIAL"), `expected a scope warning, got: ${banner}`);
  assert.ok(report.console.some((line) => line.includes("COMPOSITE ROW IS NOT")));
  assert.ok(report.console.some((line) => line.includes("largest COMPOSITE buckets")));
});

test("with isolation off every bucket is composite, and with it on none is by default", async () => {
  const off = await build(writeTrace({ frames: 12, frameUs: 50_000, tasks: MIXED }), 12);
  assert.equal(off.attribution.mode, "off");
  assert.equal(off.attribution.exactBuckets, 0);
  assert.equal(off.attribution.compositeBuckets, MIXED.length);
  assert.ok(off.passes.every((pass) => pass.compositeReason?.includes("label isolation was off")));
  assert.ok(off.console.some((line) => line.includes("label isolation OFF")));

  const full = await build(writeTrace({ frames: 12, frameUs: 50_000, tasks: MIXED }), 12,
    { FLUID_GPU_ISOLATE_PASS_LABELS: "1" });
  assert.equal(full.attribution.mode, "full");
  assert.equal(full.attribution.compositeBuckets, 0);
  assert.ok(full.passes.every((pass) => pass.exactAttribution));
});

test("an Instruments-merged encoder is never an exact stage, however isolated", async () => {
  // Instruments joins the encoders it drew as one interval with " & ". Label
  // isolation cannot make such a bucket a stage: the interval spans them all.
  const directory = mkdtempSync(join(tmpdir(), "xctrace-merge-"));
  const intervals: string[] = [];
  const encoders: string[] = [];
  for (let frame = 0; frame < 12; frame += 1) {
    ["Fine JFA - flood & Fine JFA - resolve", "Fine JFA - seed"].forEach((label, slot) => {
      const encoderId = `0x${(frame * 8 + slot + 1).toString(16)}`;
      encoders.push(JSON.stringify({
        "encoder-id": encoderId, "encoder-label": `Command Buffer 0:${label}`,
        process: "node (4242)",
      }));
      intervals.push(JSON.stringify({
        "encoder-id": encoderId, start: stamp(1_000_000 + frame * 50_000 + slot * 1000),
        duration: "500.00 µs", "channel-name": "Compute", process: "node (4242)",
      }));
    });
  }
  writeFileSync(join(directory, "intervals.ndjson"), `${intervals.join("\n")}\n`);
  writeFileSync(join(directory, "encoders.ndjson"), `${encoders.join("\n")}\n`);
  const report = await build({
    "gpu-intervals": join(directory, "intervals.ndjson"),
    encoders: join(directory, "encoders.ndjson"),
  }, 12, { FLUID_GPU_ISOLATE_PASS_LABELS: "1" });
  const merged = report.passes.find((pass) => pass.merged);
  assert.ok(merged, "the two-encoder interval must survive as a merged bucket");
  assert.equal(merged.exactAttribution, false);
  assert.ok(merged.compositeReason?.includes("merged"), merged.compositeReason);
  assert.equal(report.passes.find((pass) => pass.label === "Fine JFA - seed")?.exactAttribution,
    true, "the single-encoder stage beside it stays exact");
});

test("frames are retained in full so any of them can be redrawn", async () => {
  const report = await build(writeTrace({ frames: 12, frameUs: 50_000 }), 12);
  assert.ok(report.frames.captures.length > 0);
  assert.ok(report.frames.captures.some((capture) => capture.index === 0), "the first frame");
  assert.ok(report.frames.captures.some(
    (capture) => capture.index === report.frames.samples.length - 1), "the last frame");
  const representative = report.frames.captures[report.frames.representative];
  assert.ok(representative !== undefined, "the default frame must be retained");
  for (const capture of report.frames.captures) {
    assert.equal(capture.intervals.length, TASKS.length);
    // Intervals are frame-relative, so a frame can be drawn on its own clock.
    assert.ok(capture.intervals.every((interval) => interval.start >= 0
      && interval.start < capture.durationUs));
  }
});
