import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { initialRasterPresentationReadiness, requiresFencedInitialRasterPresentation } from "../lib/gpu-t0-presentation";
import type { WaterRenderDiagnostics } from "../lib/webgpu-water-pipeline";

const base = {
  solverAttached: true,
  initialSparseAuthorityReady: true,
  globalFineAttached: true,
  surfaceSourceAttached: true,
  surfaceExtractionSubmitted: true,
  presentationFenceCompleted: true,
  diagnosticsRequired: false,
} as const;

function diagnostic(
  surfaceGeometrySource: WaterRenderDiagnostics["surfaceGeometrySource"],
  vertexCount: number,
): WaterRenderDiagnostics {
  return {
    vertexCount, activeCubeCount: vertexCount > 0 ? 1 : 0,
    vertexAllocator: 0xffff_ffff,
    globalFineAuthorityLatch: surfaceGeometrySource === "global-fine-coarse" || surfaceGeometrySource === "compact-coarse" ? 1 : 0,
    surfaceGeometrySource,
    globalFineAttached: surfaceGeometrySource !== "compact-coarse",
    globalFineCrossingPublished: surfaceGeometrySource === "global-fine-coarse",
    presentationFallbackActive: surfaceGeometrySource === "retained-previous",
  };
}

test("only the power octree requires the sparse t=0 raster fence", () => {
  assert.equal(requiresFencedInitialRasterPresentation("octree"), true);
  for (const method of ["tall-cell", "quadtree-tall-cell", "uniform", "cpu-reference"]) {
    assert.equal(requiresFencedInitialRasterPresentation(method), false, method);
  }
});

test("paused t=0 stays locked until every solver, source, extraction, and fence prerequisite completes", () => {
  for (const key of [
    "solverAttached", "initialSparseAuthorityReady", "globalFineAttached",
    "surfaceSourceAttached", "surfaceExtractionSubmitted", "presentationFenceCompleted",
  ] as const) {
    const result = initialRasterPresentationReadiness({ ...base, [key]: false });
    assert.equal(result.ready, false, key);
    assert.equal(result.state, "pending", key);
  }
  assert.deepEqual(initialRasterPresentationReadiness(base), {
    ready: true,
    state: "gpu-authoritative",
    label: "WebGPU t=0 ready · GPU raster publication fenced",
  });
});

test("diagnostics mode confirms a current fine/coarse crossing or factor-one compact publication", () => {
  const fine = initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true,
    diagnostics: diagnostic("global-fine-coarse", 12) });
  assert.equal(fine.ready, true); assert.equal(fine.state, "crossing-confirmed");

  const compact = initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true,
    diagnostics: diagnostic("compact-coarse", 12) });
  assert.equal(compact.ready, true); assert.equal(compact.state, "compact-confirmed");

  const volume = initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true,
    diagnostics: diagnostic("volume", 6) });
  assert.equal(volume.ready, false); assert.equal(volume.state, "failed-closed");

  for (const source of ["empty", "retained-previous"] as const) {
    const failed = initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true,
      diagnostics: diagnostic(source, source === "retained-previous" ? 6 : 0) });
    assert.equal(failed.ready, false); assert.equal(failed.state, "failed-closed");
  }
  assert.equal(initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true }).state, "pending");
});

test("the retired coarse-only presentation switch and backing path stay deleted", () => {
  assert.deepEqual(initialRasterPresentationReadiness({ ...base, globalFineAttached: false }), {
    ready: false,
    state: "pending",
    label: "Waiting for global-fine renderer source",
  });
  const source = readFileSync(new URL("../lib/gpu-t0-presentation.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /globalFineRequired|coarse-only mode|coarse-octree raster/);
});

test("renderer publishes ready only after first raster submission completion and controller retains both locks", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  const transport = readFileSync(new URL("../components/TransportBar.tsx", import.meta.url), "utf8");
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");

  assert.doesNotMatch(renderer, /gpuInfoCallback\?\.\(solver\.info\);this\.onStatus\(\{state:"ready",label:"WebGPU solver ready"/);
  assert.match(renderer, /Warmed solver attached; publishing fenced t=0 raster surface/);
  assert.match(renderer, /fencedInitialRaster=requiresFencedInitialRasterPresentation\(config\.methodId\)/);
  assert.match(renderer, /WebGPU direct-field solver ready/);
  assert.match(renderer, /initialRasterSubmission[\s\S]*queue\.onSubmittedWorkDone\(\)\.then\(async\(\)=>[\s\S]*settleInitialRasterPresentation/);
  assert.match(renderer, /initialDiagnostics=await surfaceDiagnosticsCompletion[\s\S]*settleInitialRasterPresentation/,
    "diagnostics mode must settle from the readback belonging to the fenced t=0 submission");
  assert.match(renderer, /state: "blocked", label: outcome\.label/,
    "an empty or retained raster diagnostic must remain attached but fail transport closed");
  assert.doesNotMatch(renderer, /initialRasterGlobalFineRequired|globalFineRequired/);
  assert.match(renderer, /readyGPUFluid\.initialSparseAuthorityReady === true[\s\S]*Boolean\(readyGPUFluid\.globalFineLevelSetSource \|\| readyGPUFluid\.coarseLevelSetSource\)/);
  assert.match(renderer, /this\.globalFineWaterAttached[\s\S]*rasterResult\.surfaceUpdated/);
  assert.match(controller, /initialSparseAuthorityReady === true[\s\S]*initialRasterSurfaceReady === true/);
  assert.match(transport, /initialSparseAuthorityReady === true[\s\S]*initialRasterSurfaceReady === true/);
  assert.match(viewport, /status\.state === "lost" \|\| status\.state === "unavailable"/);
  assert.doesNotMatch(viewport, /status\.state === "blocked"[\s\S]*stopGPU/,
    "a fail-closed raster stays visually inspectable and must not destroy the device");
  assert.match(viewport, /startupMode\(\) === "manual" \|\| startupMode\(\) === "safe"/,
    "manual and safe startup must remain explicit");
});
