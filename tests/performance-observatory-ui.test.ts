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
  const grid = source("../components/PerformanceActivityGrid.tsx");

  for (const mode of [
    "fine-band-lifecycle",
    "resolution",
    "global-fine-phi",
    "power-cells",
    "octree-lifecycle",
    "pressure",
    "evaluated-velocity",
    "projection-update",
    "divergence-closure",
    "structured-velocity",
  ]) assert.match(panel, new RegExp(`mode: "${mode}"`), mode);

  assert.equal((panel.match(/axis: "volume"/g) ?? []).length, 13,
    "every observatory card defaults to its volume presentation");
  assert.match(panel, /if \(overlayAxis === "off"\) \{[\s\S]*setOverlayAxis\(view\.axis\)/,
    "a card should use its volume default only when no presentation is active");
  assert.doesNotMatch(panel, /setOverlayMode\(view\.mode\);\s*setOverlayAxis\(view\.axis\);/,
    "switching fields must preserve the user's selected view plane or volume");
  assert.match(panel,
    /if \(overlayMode === view\.mode && overlayAxis !== "off"\) \{\s*setOverlayAxis\("off"\);\s*return;/,
    "clicking the selected observatory card should unselect and hide it");
  assert.match(panel, /const active = overlayAxis !== "off" && overlayMode === view\.mode/,
    "a hidden field must not leave its observatory card selected");

  assert.match(panel, /paper-view-controls/);
  assert.match(panel, /paper-field-legend/);
  assert.match(panel, /\(\["x", "y", "z"\] as const\)/);
  assert.match(panel, />VOLUME<\/button>/);
  assert.match(panel, />HIDE<\/button>/);
  assert.match(panel, /type="range"/);
  assert.match(panel, /source:/);
  assert.match(panel, /traces\.length === 3 && traces\.every\(performanceTraceIsExact\)/);
  assert.match(grid, /performanceTraceAccounting\(trace\)\.exact/);
  assert.match(grid, /ACCOUNTING LEDGERS/);
  assert.match(grid, /data-queue-wall-fallback=/);
  assert.doesNotMatch(grid, /activity-clock-notice/);
  assert.match(panel, /const PHASE_LAYOUT/);
  assert.match(panel, /const SVO_CONE_PRESENTATION_PHASE_LAYOUT/);
  assert.match(panel, /:gi:smooth:/);
  for (const phase of [
    "water-caustics",
    "svo-cone-lighting",
    "svo-environment-gi",
    "svo-primary",
    "svo-rigid",
    "svo-glass",
    "water-front-interface",
    "water-back-interface",
  ]) assert.match(panel, new RegExp(`\\["${phase}"`), phase);
  assert.match(panel, /const averagedTrace/);
  assert.match(panel, /new Map\(traces\.map/);
  assert.match(panel, /latest\.capturedAt_ms - latestHardware\.capturedAt_ms <= 2_000/);
  assert.match(panel, /PERFORMANCE_AVERAGE_WINDOW/);
  assert.match(panel, /runState !== "paused"/);
  assert.match(panel, /PAUSED · LAST SETTLED CAPTURE/);
  assert.match(panel, /activityHistory\.filter\(performanceActivityFrameHasSettledEvidence\)/);
  assert.match(panel, /WAITING FOR COMPLETE CAPTURE/);
  assert.match(panel, /instrumentationMode === "timeline" \? cpu : undefined/);
  assert.match(panel, /role="group" aria-label="Performance capture mode"/);
  assert.match(panel, /mode: "timeline", label: "TIMELINE"/);
  assert.match(panel, /mode: "activity", label: "DETAILED"/);
  assert.doesNotMatch(panel, /<select[\s\S]*Performance capture mode/);
  assert.match(grid, /data-accounting-ledgers="cpu-gpu-independent"/);
  assert.doesNotMatch(panel, /cpu\.total_ms\s*\+\s*physics\.total_ms|physics\.total_ms\s*\+\s*presentation\.total_ms/);
});

test("performance matrix geometry is stable under long and changing measurements", () => {
  const css = source("../app/globals.css");

  assert.match(css, /\.activity-scrollport\s*\{[^}]*overflow-x:\s*auto/s);
  assert.match(css, /\.activity-resource-label > span\s*\{[^}]*text-overflow:\s*ellipsis[^}]*white-space:\s*nowrap/s);
  assert.match(css, /\.activity-cell::after\s*\{[^}]*width:\s*100%/s);
  assert.match(css, /font-variant-numeric:\s*tabular-nums/);
  assert.match(css, /var\(--activity-slice-count\)/);
  assert.match(css, /var\(--activity-ms-width\)/);
  assert.match(css, /\.activity-cell\s*\{[^}]*border:\s*0/s);
});
