import assert from "node:assert/strict";
import test from "node:test";
import { lightingQuery } from "../lib/svo/features/lighting-visibility/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { createUIStore } from "../lib/core/stores/ui-store";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";

test("raster AO is an explicit saved opt-in and selecting it prepares the required mesh", () => {
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode, "cones");
  assert.equal(lightingQuery.read(new URLSearchParams()).svoConeTracingMode, "cones");
  const saved = lightingQuery.read(new URLSearchParams("svoCones=raster-ao"));
  assert.equal(saved.svoConeTracingMode, "raster-ao");
  const query = new URLSearchParams(); lightingQuery.write(query, saved);
  assert.equal(query.get("svoCones"), "raster-ao");
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
  assert.equal(renderer.requestedPrimaryTraversal, "mesh", "preview selects mesh even for an old saved traced-primary preference");
  request(); assert.equal(destroyed, 1);
});
