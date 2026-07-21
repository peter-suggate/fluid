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
  primitiveCandidatesSupported: true,
  lightingSupported: true,
  inspectionMode: false,
  svoEncoded: true,
};

test("effective renderer status preserves raster default and reports successful SVO", () => {
  assert.deepEqual(resolveEffectiveRendererStatus("raster", {
    pipelineAvailable: false,
    sourceAvailable: false,
    terrainSupported: false,
    inspectionMode: false,
    svoEncoded: false,
  }), { requestedMode: "raster", effectiveMode: "raster" });
  assert.deepEqual(resolveEffectiveRendererStatus("svo", ready), {
    requestedMode: "svo",
    effectiveMode: "svo",
  });
  assert.deepEqual(useDiagnosticsStore.getInitialState().effectiveRendererStatus, {
    requestedMode: "raster",
    effectiveMode: "raster",
  });
});

test("effective renderer status distinguishes required SVO fallbacks", () => {
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, sourceAvailable: false, svoEncoded: false }), {
    requestedMode: "svo", effectiveMode: "raster", fallbackReason: "missing-source",
  });
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, terrainSupported: false, svoEncoded: false }), {
    requestedMode: "svo", effectiveMode: "raster", fallbackReason: "unsupported-terrain",
  });
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, glassSupported: false, svoEncoded: false }), {
    requestedMode: "svo", effectiveMode: "raster", fallbackReason: "unsupported-glass-cutout",
  });
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, materialsSupported: false, svoEncoded: false }), {
    requestedMode: "svo", effectiveMode: "raster", fallbackReason: "missing-pbr-materials",
  });
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, primitiveCandidatesSupported: false, svoEncoded: false }), {
    requestedMode: "svo", effectiveMode: "raster", fallbackReason: "missing-primitive-candidates",
  });
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, lightingSupported: false, svoEncoded: false }), {
    requestedMode: "svo", effectiveMode: "raster", fallbackReason: "missing-lighting-publications",
  });
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, pipelineAvailable: false, svoEncoded: false }), {
    requestedMode: "svo", effectiveMode: "raster", fallbackReason: "pipeline-compile-failure",
  });
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, inspectionMode: true, svoEncoded: false }), {
    requestedMode: "svo", effectiveMode: "raster", fallbackReason: "inspection-mode",
  });
});

test("supported analytic garden terrain can report effective SVO", () => {
  assert.deepEqual(resolveEffectiveRendererStatus("svo", { ...ready, terrainSupported: true, svoEncoded: true }), {
    requestedMode: "svo", effectiveMode: "svo",
  });
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(renderer, /terrainMaterialId:scenePrimitives\.analyticTerrain\?\.materialId/);
  assert.match(renderer, /!sceneHasTerrain\(scene\)\|\|Boolean\(scenePrimitives\.analyticTerrain\)/);
  assert.match(renderer, /terrainSupported: this\.svoTerrainSupported/);
  assert.doesNotMatch(renderer, /terrainSupported: this\.svoTerrainSupported && !sceneHasTerrain/);
});

test("renderer publishes effective status through the viewport diagnostics bridge", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(renderer, /publishEffectiveRendererStatus\(resolveEffectiveRendererStatus\(svoRenderMode/);
  assert.match(renderer, /const replacementResult = this\.svoDryScenePipeline\?\.encode/);
  assert.match(renderer, /svoEncoded = Boolean\(replacementResult\)/);
  assert.match(renderer, /canEncodeSparseVoxelDryScene\(sparseSceneSource,drySceneData\)/);
  assert.match(viewport, /effectiveRendererStatus\) => useDiagnosticsStore\.getState\(\)\.set\(\{ effectiveRendererStatus \}\)/);
  assert.match(panel, /data-testid="effective-renderer-status"/);
  for (const reason of ["missing-source", "unsupported-terrain", "unsupported-glass-cutout", "missing-pbr-materials", "missing-primitive-candidates", "missing-lighting-publications", "pipeline-compile-failure"]) assert.ok(panel.includes(`"${reason}"`));
});
