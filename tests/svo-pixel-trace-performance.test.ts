import assert from "node:assert/strict";
import test from "node:test";
import { SVO_BRICK_RASTER_CONTRACT } from "../lib/svo/webgpu-svo-brick-raster";
import {
  createSvoBrickRasterProbeWGSL,
  svoBrickRasterProbeBindGroupLayoutEntries,
} from "../lib/svo/webgpu-svo-brick-raster-probe";
import { containerDecorationSpace } from "../lib/core/visualization-decorations";
import {
  assembleDecorations,
  decorationAssemblyKey,
  decorationVisualization,
} from "../lib/core/visualization-registry";
import {
  SVO_PIXEL_TRACE_LIVE_PROBE_INTERVAL_MS,
  svoPixelTraceProbeDue,
} from "../lib/core/webgpu-renderer";
import {
  buildSvoPixelTraceGeometry,
  resolveSvoPixelTracePinnedFrame,
  SVO_PIXEL_TRACE_FLAGS,
  SVO_PIXEL_TRACE_KINDS,
  SVO_PIXEL_TRACE_VISUALIZATION_MAXIMUM_DENSE_BOXES,
  type SvoPixelTrace,
} from "../lib/svo/svo-pixel-trace";

const probeOptions = {
  fragmentDepthWritten: true,
  paramsWordCount: 28,
  payloadLaneWordOffset: 20,
};

test("raster pixel tracing reuses the shipping per-pixel coverage candidates", () => {
  const shader = createSvoBrickRasterProbeWGSL({
    ...probeOptions,
    coverageAccelerated: true,
  });
  assert.match(shader, /publishedCount=select\(0u,svoProbeCoverageCounts\[probePixel\]/);
  assert.match(shader, new RegExp(`coverageCapacity=${SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel}u`));
  assert.match(shader, /candidate<candidates/);
  assert.match(shader, /if\(useCoverage\)\{index=svoProbeCoverageCandidates\[probePixel\*coverageCapacity\+candidate\];\}/);
  assert.match(shader, /let candidates=select\(drawn,publishedCount,useCoverage\)/,
    "overflow must retain the all-instance correctness fallback");
});

test("direct raster experiments retain the whole-instance probe fallback", () => {
  const shader = createSvoBrickRasterProbeWGSL(probeOptions);
  assert.doesNotMatch(shader, /publishedCount=svoProbeCoverageCounts/);
  assert.match(shader, /let candidates=drawn/);
});

test("the raster probe binds both halves of the existing coverage index", () => {
  Object.assign(globalThis, { GPUShaderStage: { COMPUTE: 4 } });
  const bindings = svoBrickRasterProbeBindGroupLayoutEntries().map((entry) => entry.binding);
  assert.ok(bindings.includes(7), "coverage count binding");
  assert.ok(bindings.includes(8), "coverage candidate binding");
});

test("decoration identity can be resolved without rebuilding plugin geometry", () => {
  let builds = 0;
  const definition = decorationVisualization<{ revision: number }>({
    kind: "decoration",
    id: "test/expensive",
    pass: "test",
    label: "Expensive",
    description: "fixture",
    accepts: (subject): subject is { revision: number } =>
      typeof subject === "object" && subject !== null && "revision" in subject,
    key: (subject) => String(subject.revision),
    build: (_subject, _context, into) => {
      builds += 1;
      into.worldSegment([0, 0, 0], [1, 1, 1], {
        colorLinear: [1, 1, 1], width_px: 1,
      });
    },
  });
  const options = {
    definitions: [definition],
    subjects: [{ revision: 3 }],
    space: containerDecorationSpace([1, 1, 1], [1, 1, 1]),
  };
  const key = decorationAssemblyKey(options);
  assert.equal(builds, 0);
  const assembled = assembleDecorations(options);
  assert.equal(assembled.key, key);
  assert.equal(builds, 1);
});

test("live ray probes are paced while a pin remains a one-shot frozen query", () => {
  const previous = "old";
  const lastAt = 1_000;
  assert.equal(svoPixelTraceProbeDue(
    "new", { pinned: false }, previous, lastAt,
    lastAt + SVO_PIXEL_TRACE_LIVE_PROBE_INTERVAL_MS - 1,
  ), false);
  assert.equal(svoPixelTraceProbeDue(
    "new", { pinned: false }, previous, lastAt,
    lastAt + SVO_PIXEL_TRACE_LIVE_PROBE_INTERVAL_MS,
  ), true);
  assert.equal(svoPixelTraceProbeDue(
    "new", { pinned: false, settled: false }, previous, lastAt,
    lastAt + SVO_PIXEL_TRACE_LIVE_PROBE_INTERVAL_MS * 10,
  ), false, "pointer motion must not queue expensive traces that are stale before readback");
  assert.equal(svoPixelTraceProbeDue(
    "new", { pinned: false, urgent: true }, previous, lastAt, lastAt + 1,
  ), true);
  assert.equal(svoPixelTraceProbeDue(
    "new", { pinned: true }, previous, lastAt, lastAt + 1,
  ), false, "animated-scene staleness must not turn a pin into a probe loop");
  assert.equal(svoPixelTraceProbeDue(
    "new", { pinned: true }, previous, lastAt, lastAt + 1,
  ), false);
  assert.equal(svoPixelTraceProbeDue(
    previous, { pinned: true }, previous, lastAt, lastAt + 1,
  ), false, "an in-flight stale-pin answer must not be encoded every frame");
});

test("animated scene epochs mark a pinned trace stale without refreshing it", () => {
  assert.deepEqual(resolveSvoPixelTracePinnedFrame({
    pinned: true,
    sceneChanged: true,
  }), { stale: true });
  assert.deepEqual(resolveSvoPixelTracePinnedFrame({
    pinned: false,
    sceneChanged: true,
  }), { stale: false });
});

test("dense trace visualization is sampled without changing the shader record set", () => {
  const records = Array.from({ length: 100 }, (_, order) => ({
    kind: SVO_PIXEL_TRACE_KINDS.brickCell,
    order,
    level: 0,
    detail: 0,
    flags: SVO_PIXEL_TRACE_FLAGS.tagged,
    a: [order, 0, 0],
    b: [order + 1, 1, 1],
    tEnter_m: order,
    tExit_m: order + 1,
  }));
  const trace = {
    primaryMode: "traced",
    ray: { origin_m: [0, 0, 0], direction: [1, 0, 0] },
    counters: { maximumDepth: 8 },
    records,
  } as unknown as SvoPixelTrace;
  const complete = buildSvoPixelTraceGeometry(trace, { layers: ["cells"] });
  const sampled = buildSvoPixelTraceGeometry(trace, {
    layers: ["cells"],
    maximumDenseBoxesPerLayer: SVO_PIXEL_TRACE_VISUALIZATION_MAXIMUM_DENSE_BOXES,
  });
  assert.equal(records.length, 100, "sampling is presentation-only");
  assert.equal(complete.segmentCount, 100 * 12);
  assert.equal(sampled.segmentCount, SVO_PIXEL_TRACE_VISUALIZATION_MAXIMUM_DENSE_BOXES * 12);
});
