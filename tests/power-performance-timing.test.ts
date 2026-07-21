import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const solver = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const panel = readFileSync(new URL("../components/PerformancePanel.tsx", import.meta.url), "utf8");
const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");

const ordered = (source: string, labels: string[]) => {
  let cursor = -1;
  for (const label of labels) {
    const next = source.indexOf(label, cursor + 1);
    assert.ok(next > cursor, `expected ${label} after byte ${cursor}`);
    cursor = next;
  }
};

test("authoritative power timings cover the solve and complete velocity publication", () => {
  ordered(octree, [
    "Power pressure timing start",
    "this.mgpcg!.encode(encoder, pressureIn, pressureOut)",
    "Power pressure timing end",
    "Power projection timing start",
    "this.encodePowerProjectionMirror",
    "this.encodePowerVelocityPublication",
    "this.encodeGlobalFineFaceBand",
    "Power projection timing end",
  ]);
});

test("topology, every CFL rebuild, and adaptive surface publication have complete outer ranges", () => {
  assert.match(solver, /Octree topology timing start[^]*encodeInlineRebuild\(encoder\)[^]*Octree topology timing end/);
  assert.match(solver, /Octree substep topology timing start[^]*encodeInlineRebuild\(encoder\)[^]*Octree substep topology timing end/);
  assert.match(solver, /Adaptive surface timing start[^]*adaptiveProjection\.encodeSurface\([^]*Adaptive surface timing end/);
  assert.match(octree, /beginRange\("Fluid brick residency"[^]*endRange\("Fluid brick residency"[^]*beginRange\("Sparse scene publication"[^]*endRange\("Sparse scene publication"/);
});

test("performance graph exposes every submitted presentation pass and effective SVO mode", () => {
  for (const key of ["caustics", "dry-scene", "svo-temporal", "front-interface", "back-interface", "composite", "overlays", "upscale"]) {
    assert.match(panel, new RegExp(`key: "${key}"`));
  }
  assert.match(panel, /snapshot\.effectiveRenderMode === "svo"/);
  assert.match(panel, /SVO traversal \+ dry shading/);
});

test("production SVO traversal and shading execute for every submitted presentation", () => {
  assert.doesNotMatch(renderer, /drySceneReuseKey|Sparse voxel dry scene reuse timestamp/);
  assert.match(renderer, /SVO visibility and shading are presentation work[^]*svoDryScenePipeline\?\.encode\(replacementEncoder, target, timestampWrites, temporalFrame, temporalTimestampWrites\)/);
  assert.match(viewport, /continuousPerformancePresentation[^]*ui\.rightPanel === "performance"[^]*!continuousPerformancePresentation/,
    "the performance panel must keep a paused static dry scene submitting completion-gated presentations");
});
