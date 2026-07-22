import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { initialRasterPresentationReadiness, requiresFencedInitialRasterPresentation } from "../lib/gpu-t0-presentation";
import type { AdaptiveWaterRenderDiagnostics } from "../lib/webgpu-water-pipeline";

const base = {
  solverAttached: true,
  initialSparseAuthorityReady: true,
  globalFineAttached: true,
  adaptiveSurfaceAttached: true,
  surfaceExtractionSubmitted: true,
  presentationFenceCompleted: true,
  diagnosticsRequired: false,
} as const;

function diagnostic(
  surfaceGeometrySource: AdaptiveWaterRenderDiagnostics["surfaceGeometrySource"],
  vertexCount: number,
): AdaptiveWaterRenderDiagnostics {
  return {
    leafCapacity: 1, pageCapacity: 1, pageResolution: 4, samplesPerPage: 64,
    surfaceFreePages: 0, surfaceAllocatedPages: 1, surfaceCandidatePages: 1, surfaceActivePages: 1,
    surfaceOverflow: 0, finestResidentPages: 1, coarseResidentPages: 0, maximumResidentLeafSize: 1,
    surfaceDispatch: [1, 1, 1], vertexCount, activeCubeCount: vertexCount > 0 ? 1 : 0,
    vertexAllocator: surfaceGeometrySource === "adaptive-fallback" ? vertexCount : 0xffff_ffff,
    globalFineAuthorityLatch: surfaceGeometrySource === "global-fine-coarse" ? 1 : 0,
    surfaceGeometrySource,
    globalFineAttached: true,
    globalFineCrossingPublished: surfaceGeometrySource === "global-fine-coarse",
    presentationFallbackActive: surfaceGeometrySource === "adaptive-fallback" || surfaceGeometrySource === "retained-previous",
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
    "adaptiveSurfaceAttached", "surfaceExtractionSubmitted", "presentationFenceCompleted",
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

test("diagnostics mode confirms only a current fine/coarse crossing", () => {
  const fine = initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true,
    diagnostics: diagnostic("global-fine-coarse", 12) });
  assert.equal(fine.ready, true); assert.equal(fine.state, "crossing-confirmed");

  const fallback = initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true,
    diagnostics: diagnostic("adaptive-fallback", 6) });
  assert.equal(fallback.ready, false); assert.equal(fallback.state, "failed-closed");

  for (const source of ["empty", "retained-previous"] as const) {
    const failed = initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true,
      diagnostics: diagnostic(source, source === "retained-previous" ? 6 : 0) });
    assert.equal(failed.ready, false); assert.equal(failed.state, "failed-closed");
  }
  assert.equal(initialRasterPresentationReadiness({ ...base, diagnosticsRequired: true }).state, "pending");
});

test("coarse-only octree reaches t=0 readiness without allocating a global-fine source", () => {
  const coarse = { ...base, globalFineRequired: false, globalFineAttached: false } as const;
  assert.deepEqual(initialRasterPresentationReadiness(coarse), {
    ready: true,
    state: "gpu-authoritative",
    label: "WebGPU t=0 ready · coarse-octree raster publication fenced",
  });

  const confirmed = initialRasterPresentationReadiness({
    ...coarse,
    diagnosticsRequired: true,
    diagnostics: diagnostic("adaptive-octree", 12),
  });
  assert.equal(confirmed.ready, true);
  assert.equal(confirmed.state, "crossing-confirmed");
  assert.match(confirmed.label, /coarse-octree raster crossing confirmed/);

  const empty = initialRasterPresentationReadiness({
    ...coarse,
    diagnosticsRequired: true,
    diagnostics: diagnostic("empty", 0),
  });
  assert.equal(empty.ready, false);
  assert.equal(empty.state, "failed-closed");
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
  assert.match(renderer, /adaptiveDiagnosticsCompletion=await adaptiveDiagnosticsCompletion[\s\S]*settleInitialRasterPresentation|initialDiagnostics=await adaptiveDiagnosticsCompletion[\s\S]*settleInitialRasterPresentation/,
    "diagnostics mode must settle from the readback belonging to the fenced t=0 submission");
  assert.match(renderer, /state: "blocked", label: outcome\.label/,
    "an empty or retained raster diagnostic must remain attached but fail transport closed");
  assert.match(renderer, /initialRasterGlobalFineRequired[\s\S]*readyGPUFluid\.globalFineLevelSetSource/);
  assert.match(renderer, /this\.adaptiveWaterAttached[\s\S]*rasterResult\.surfaceUpdated/);
  assert.match(controller, /initialSparseAuthorityReady === true[\s\S]*initialRasterSurfaceReady === true/);
  assert.match(transport, /initialSparseAuthorityReady === true[\s\S]*initialRasterSurfaceReady === true/);
  assert.match(viewport, /status\.state === "lost" \|\| status\.state === "unavailable"/);
  assert.doesNotMatch(viewport, /status\.state === "blocked"[\s\S]*stopGPU/,
    "a fail-closed raster stays visually inspectable and must not destroy the device");
  assert.match(viewport, /startupMode\(\) === "manual" \|\| startupMode\(\) === "safe"/,
    "manual and safe startup must remain explicit");
});
