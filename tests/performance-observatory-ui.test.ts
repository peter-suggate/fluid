import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("the performance UI has no legacy profiler or FPS visualization surface", () => {
  const css = source("../app/globals.css");
  const diagnostics = source("../components/DiagnosticsPanel.tsx");
  const fluidLab = source("../components/FluidLab.tsx");
  const viewport = source("../components/WebGPUViewport.tsx");

  assert.doesNotMatch(css, /\.perf-[\w-]+|\.sparkline\b|\.frame-rate-counter\b/);
  assert.doesNotMatch(diagnostics, /Sparkline|Render encode|GPU step|GPU completion cadence/);
  assert.doesNotMatch(fluidLab + viewport, /FrameRateCounter|presentation-frame-rate|recordPresentedFrame/);
  assert.equal(existsSync(new URL("../components/FrameRateCounter.tsx", import.meta.url)), false);
  assert.equal(existsSync(new URL("../lib/presentation-frame-rate.ts", import.meta.url)), false);
});

test("the observatory exposes paper fields with exact axis, slice, and legend controls", () => {
  const panel = source("../components/PerformancePanel.tsx");

  for (const mode of [
    "fine-band-lifecycle",
    "resolution",
    "global-fine-phi",
    "power-cells",
    "octree-lifecycle",
    "pressure",
    "speed",
    "projection",
    "divergence",
  ]) assert.match(panel, new RegExp(`mode: "${mode}"`), mode);

  assert.match(panel, /paper-view-controls/);
  assert.match(panel, /paper-field-legend/);
  assert.match(panel, /\(\["x", "y", "z"\] as const\)/);
  assert.match(panel, />VOLUME<\/button>/);
  assert.match(panel, />HIDE<\/button>/);
  assert.match(panel, /type="range"/);
  assert.match(panel, /source:/);
  assert.match(panel, /traces\.length === 3 && traces\.every\(performanceTraceIsExact\)/);
  assert.match(panel, /OBSERVED TOTAL/);
  assert.match(panel, /ACCOUNTED PHASE SUM/);
  assert.match(panel, /CLOSURE ERROR/);
  assert.match(panel, /performanceTraceAccounting\(trace\)/);
  assert.match(panel, /data-observed-total-ms=\{accounting\.observedTotal_ms\}/);
  assert.match(panel, /data-accounted-ms=\{accounting\.accounted_ms\}/);
  assert.match(panel, /data-closure-error-ms=\{accounting\.closureError_ms\}/);
  assert.match(panel, /QUEUE COMPLETION FALLBACK/);
  assert.match(panel, /requires a valid timestamp sample/);
  assert.match(panel, /const PHASE_LAYOUT/);
  assert.match(panel, /const SVO_CONE_PRESENTATION_PHASE_LAYOUT/);
  assert.match(panel, /trace\.context\.includes\(":lighting-cone:smooth:svo:"\)/);
  for (const phase of [
    "water-caustics",
    "svo-cone-lighting",
    "svo-primary",
    "svo-temporal",
    "water-front-interface",
    "water-back-interface",
  ]) assert.match(panel, new RegExp(`\\["${phase}"`), phase);
  assert.match(panel, /const averagedLane/);
  assert.match(panel, /new Map\(traces\.map/);
  assert.match(panel, /latest\.capturedAt_ms - latestHardware\.capturedAt_ms <= 2_000/);
  assert.match(panel, /AVG \{sampleCount\} · LATEST/);
  assert.match(panel, /runState !== "paused"/);
  assert.match(panel, /Holding the last completed measurements/);
  assert.match(panel, /CPU and GPU are independent ledgers and are never added together/);
  assert.doesNotMatch(panel, /cpu\.total_ms\s*\+\s*physics\.total_ms|physics\.total_ms\s*\+\s*presentation\.total_ms/);
});

test("performance lane geometry is stable under long and changing measurements", () => {
  const css = source("../app/globals.css");

  assert.match(css, /\.trace-title-block\s*\{[^}]*min-width:\s*0[^}]*overflow:\s*hidden/s);
  assert.match(css, /\.trace-meta code\s*\{[^}]*text-overflow:\s*ellipsis[^}]*white-space:\s*nowrap/s);
  assert.match(css, /\.trace-row > span\s*\{[^}]*text-overflow:\s*ellipsis[^}]*white-space:\s*nowrap/s);
  assert.match(css, /font-variant-numeric:\s*tabular-nums/);
  assert.match(css, /\.trace-row\s*\{[^}]*minmax\(0,1\.35fr\)/s);
});
