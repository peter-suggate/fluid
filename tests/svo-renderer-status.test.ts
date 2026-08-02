import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { useDiagnosticsStore } from "../lib/stores/diagnostics-store";
import { resolveEffectiveRendererStatus, type EffectiveRendererConditions } from "../lib/webgpu-renderer";

const ready: EffectiveRendererConditions = {
  pipelineAvailable: true,
  sourceAvailable: true,
  terrainSupported: true,
  glassSupported: true,
  materialsSupported: true,
  lightingSupported: true,
  svoEncoded: true,
};

test("effective renderer status reports the GLOBAL SVO lifecycle", () => {
  assert.deepEqual(resolveEffectiveRendererStatus(ready), {
    state: "active",
  });
  assert.deepEqual(useDiagnosticsStore.getInitialState().effectiveRendererStatus, {
    state: "pending",
    failureReason: "missing-source",
  });
});

test("effective renderer status distinguishes pending and failed-closed SVO states", () => {
  assert.deepEqual(resolveEffectiveRendererStatus({ ...ready, sourceAvailable: false, svoEncoded: false }), {
    state: "pending", failureReason: "missing-source",
  });
  assert.deepEqual(resolveEffectiveRendererStatus({ ...ready, terrainSupported: false, svoEncoded: false }), {
    state: "failed", failureReason: "unsupported-terrain",
  });
  assert.deepEqual(resolveEffectiveRendererStatus({ ...ready, glassSupported: false, svoEncoded: false }), {
    state: "failed", failureReason: "unsupported-glass-cutout",
  });
  assert.deepEqual(resolveEffectiveRendererStatus({ ...ready, materialsSupported: false, svoEncoded: false }), {
    state: "failed", failureReason: "missing-pbr-materials",
  });
  assert.deepEqual(resolveEffectiveRendererStatus({ ...ready, lightingSupported: false, svoEncoded: false }), {
    state: "failed", failureReason: "missing-lighting-publications",
  });
  assert.deepEqual(resolveEffectiveRendererStatus({ ...ready, pipelineAvailable: false, svoEncoded: false }), {
    state: "failed", failureReason: "pipeline-compile-failure",
  });
  assert.deepEqual(resolveEffectiveRendererStatus({ ...ready, svoEncoded: false }), {
    state: "failed", failureReason: "frame-rejected", detail: "live SVO renderer declined the frame",
  });
  assert.deepEqual(resolveEffectiveRendererStatus({
    ...ready,
    sourceAvailable: false,
    svoEncoded: false,
    contractFailure: "material-owner payload field is unavailable",
  }), {
    state: "failed",
    failureReason: "frame-rejected",
    detail: "material-owner payload field is unavailable",
  });
});

test("supported analytic garden terrain can report effective SVO", () => {
  assert.deepEqual(resolveEffectiveRendererStatus({ ...ready, terrainSupported: true, svoEncoded: true }), {
    state: "active",
  });
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(renderer, /terrainMaterialId: scenePrimitives\.analyticTerrain\?\.materialId/);
  assert.match(renderer, /!sceneHasTerrain\(scene\) \|\| Boolean\(scenePrimitives\.analyticTerrain\)/);
  assert.match(renderer, /terrainSupported: this\.svoTerrainSupported/);
  assert.doesNotMatch(renderer, /terrainSupported: this\.svoTerrainSupported && !sceneHasTerrain/);
});

test("renderer publishes effective status through the viewport diagnostics bridge", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(renderer, /publishEffectiveRendererStatus\(resolveEffectiveRendererStatus\(\{/);
  assert.match(renderer, /const replacementResult = this\.svoDryScenePipeline\?\.encode/);
  assert.match(renderer, /svoEncoded = Boolean\(replacementResult\)/);
  assert.match(renderer, /sparseVoxelDrySceneContractFailure\(source, publication\)/);
  assert.match(renderer, /publishScene\(publication\)/,
    "the fixed renderer arenas receive the new exact scene publication without a source rebuild");
  assert.match(viewport, /effectiveRendererStatus\) => useDiagnosticsStore\.getState\(\)\.set\(\{ effectiveRendererStatus \}\)/);
  assert.match(panel, /data-testid="effective-renderer-status"/);
  assert.match(panel, /effectiveRendererStatus\.detail/);
  for (const reason of ["missing-source", "unsupported-terrain", "unsupported-glass-cutout", "missing-pbr-materials", "missing-lighting-publications", "pipeline-compile-failure", "frame-rejected"]) assert.ok(panel.includes(`"${reason}"`));
  assert.doesNotMatch(panel, /RASTER FALLBACK|SVO fallback/);
});

test("a pipeline that is rebuilding reports as compiling rather than as a failure", () => {
  // Startup and a primary-traversal swap both retire the pipeline while its
  // replacement compiles. Reporting that as a compile failure tells the user
  // their renderer broke when it is only busy, and the two states are told
  // apart solely by whether a compile is still in flight.
  assert.deepEqual(resolveEffectiveRendererStatus({
    ...ready, pipelineAvailable: false, pipelineCompiling: true, svoEncoded: false,
  }), { state: "pending", failureReason: "pipeline-compiling" });
  assert.deepEqual(resolveEffectiveRendererStatus({
    ...ready, pipelineAvailable: false, pipelineCompiling: false, svoEncoded: false,
  }), { state: "failed", failureReason: "pipeline-compile-failure" });
  assert.deepEqual(resolveEffectiveRendererStatus({
    ...ready, pipelineCompiling: true, pipelinePending: "Compiling requested SVO cone bundle at scale 0.5", svoEncoded: false,
  }), {
    state: "pending", failureReason: "pipeline-compiling",
    detail: "Compiling requested SVO cone bundle at scale 0.5",
  });
  assert.deepEqual(resolveEffectiveRendererStatus({
    ...ready, pipelineFailure: "Requested SVO split bundle failed: compiler rejected WGSL", svoEncoded: false,
  }), {
    state: "failed", failureReason: "pipeline-compile-failure",
    detail: "Requested SVO split bundle failed: compiler rejected WGSL",
  });
});
