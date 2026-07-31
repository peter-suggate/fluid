import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("cone-traced SVO presentation records every encoded render-path item", () => {
  const renderer = source("../lib/webgpu-renderer.ts");
  const water = source("../lib/webgpu-water-pipeline.ts");
  const dry = source("../lib/webgpu-svo-dry-scene.ts");

  assert.match(renderer, /const traceDetailedSvoRenderPath = true/);
  assert.doesNotMatch(renderer, /svoRenderMode|svoLightingMode/);
  assert.match(renderer, /new GPUStageTimestampRecorder\(/);
  assert.match(renderer, /const detailedPresentationTrace = traceDetailedSvoRenderPath \? presentationTrace : undefined/);
  assert.match(renderer, /completeDetailedPresentationPhase\(\{ id: "inspection-overlay"/);
  assert.match(renderer, /completeDetailedPresentationPhase\(\{ id: "present"/);

  for (const phase of [
    "surface-extraction",
    "water-caustics",
    "water-front-interface",
    "water-back-interface",
    "optical-composite",
  ]) assert.match(water, new RegExp(`tracePhase\\?\\.\\(\\{ id: "${phase}"`), phase);

  for (const phase of ["svo-cone-lighting", "svo-environment-gi", "svo-primary", "svo-rigid", "svo-glass"]) {
    assert.match(dry, new RegExp(`tracePhase\\?\\.\\(\\{ id: "${phase}"`), phase);
  }
  assert.doesNotMatch(dry, /svo-temporal|temporal resolve/i);
});

test("fallback instrumentation retains the compact fixed trace helper", () => {
  const renderer = source("../lib/webgpu-renderer.ts");

  assert.match(renderer, /if \(phase && !traceDetailedSvoRenderPath\) presentationTrace\?\.completePhase/);
  assert.match(renderer, /PRESENTATION_TRACE_PHASES\[fixedPresentationPhase\]/);
});

test("presentation boundaries ride the frame's passes and add no measurement fence", () => {
  const renderer = source("../lib/webgpu-renderer.ts");

  assert.match(renderer, /const encoder = presentationTrace\?\.instrument\(rawEncoder\) \?\? rawEncoder/,
    "the presentation encoder must be the instrumented one so boundaries can attach to real passes");
  assert.doesNotMatch(renderer, /presentationTrace\?\.boundary\(/,
    "marker passes are no longer how a presentation boundary is recorded");
});
