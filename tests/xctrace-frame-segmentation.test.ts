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
}): Record<string, string> => {
  const directory = mkdtempSync(join(tmpdir(), "xctrace-seg-"));
  const intervals: string[] = [];
  const encoders: string[] = [];
  let id = 0;
  for (let frame = 0; frame < options.frames; frame += 1) {
    const labels = options.extraOn?.includes(frame) ? [...TASKS, "Rebuild topology"] : TASKS;
    labels.forEach((label, slot) => {
      id += 1;
      const encoderId = `0x${id.toString(16)}`;
      const start = 1_000_000 + frame * options.frameUs + slot * 1000;
      encoders.push(JSON.stringify({
        "encoder-id": encoderId, "encoder-label": `Command Buffer 0:${label}`,
        process: "node (4242)",
      }));
      intervals.push(JSON.stringify({
        "encoder-id": encoderId, start: stamp(start), duration: "500.00 µs",
        "channel-name": "Compute", process: "node (4242)",
        "event-label": `${label}      ( node (4242) )  0x1`,
      }));
    });
  }
  writeFileSync(join(directory, "intervals.ndjson"), `${intervals.join("\n")}\n`);
  writeFileSync(join(directory, "encoders.ndjson"), `${encoders.join("\n")}\n`);
  return {
    "gpu-intervals": join(directory, "intervals.ndjson"),
    encoders: join(directory, "encoders.ndjson"),
  };
};

const build = async (tables: Record<string, string>, steps?: number) => buildFrameReport({
  tables, lane: "mini", environment: {}, tracedPid: 4242,
  traced: steps === undefined ? undefined : { steps, simulationWall_ms: steps * 50 },
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
