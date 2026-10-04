import assert from "node:assert/strict";
import test from "node:test";
import { lightingQuery, lightingTuningQuery } from "../lib/svo/features/lighting-visibility/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { createUIStore } from "../lib/core/stores/ui-store";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";

test("raster AO is the default, cones remain saved explicitly, and selecting raster AO prepares the mesh", () => {
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode, "raster-ao");
  assert.equal(lightingQuery.read(new URLSearchParams()).svoConeTracingMode, "raster-ao");
  const saved = lightingQuery.read(new URLSearchParams("svoCones=cones"));
  assert.equal(saved.svoConeTracingMode, "cones");
  const query = new URLSearchParams(); lightingQuery.write(query, saved);
  assert.equal(query.get("svoCones"), "cones");
  assert.equal(lightingTuningQuery.read(new URLSearchParams()).rasterCoarseAoStrength, 0.6);
  assert.equal(lightingTuningQuery.read(new URLSearchParams("svoCoarseAO=0")).rasterCoarseAoStrength, 0);
  const ui = createUIStore(); ui.getState().setSvoPrimaryTraversal("traced");
  ui.getState().setSvoConeTracingMode("raster-ao");
  assert.equal(ui.getState().svoPrimaryTraversal, "mesh");
  assert.equal(ui.getState().svoConeTracingMode, "raster-ao");
  ui.getState().setSvoConeTracingMode("cones");
  assert.equal(ui.getState().svoConeTracingMode, "cones");
});

test("switching the lighting backend during compilation retires the prior renderer once published", () => {
  const renderer = Object.create(FluidLabRenderer.prototype);
  let destroyed = 0;
  Object.assign(renderer, { requestedPrimaryTraversal: "mesh", requestedPrimaryWorkMap: false,
    requestedSunlightCache: false, requestedWaterShadows: false, requestedRasterAo: false,
    optionalPipelineTasks: new Map([["svo-dry-scene", Promise.resolve()]]),
    failedOptionalPipelines: new Set(), optionalPipelineFailures: new Map(),
    svoDryScenePipeline: { destroy: () => destroyed++ } });
  const request = () => renderer.applyPrimaryTraversalRequest("traced", {}, false, false, false, true);
  request(); assert.equal(destroyed, 0); assert.equal(renderer.requestedRasterAo, false);
  renderer.optionalPipelineTasks.clear(); request();
  assert.equal(destroyed, 1); assert.equal(renderer.requestedRasterAo, true);
  assert.equal(renderer.requestedPrimaryTraversal, "mesh", "Raster + AO selects mesh even for an old saved traced-primary preference");
  request(); assert.equal(destroyed, 1);
});
