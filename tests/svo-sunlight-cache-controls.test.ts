import assert from "node:assert/strict";
import test from "node:test";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, svoRenderTuningKey } from "../lib/svo/pipeline/svo-render-tuning";

test("sunlight caching is opt-in and participates in the saved tuning key", () => {
  const off = normalizeSvoRenderTuning(DEFAULT_SVO_RENDER_TUNING);
  const on = normalizeSvoRenderTuning({ ...off, sunlightCacheEnabled: true });
  assert.equal(off.sunlightCacheEnabled, false);
  assert.equal(on.sunlightCacheEnabled, true);
  assert.notEqual(svoRenderTuningKey(off), svoRenderTuningKey(on));
});

test("a cache toggle during compilation retires the old bundle after publication", () => {
  // Exercise the request lifecycle without constructing a GPU device. The
  // pending compiler owns the old key until it has published its candidate.
  const renderer = Object.create(FluidLabRenderer.prototype);
  let destroyed = 0;
  Object.assign(renderer, {
    requestedPrimaryTraversal: "traced", requestedPrimaryWorkMap: false,
    requestedSunlightCache: false,
    optionalPipelineTasks: new Map([["svo-dry-scene", Promise.resolve()]]),
    failedOptionalPipelines: new Set(), optionalPipelineFailures: new Map(),
    svoDryScenePipeline: { destroy: () => { destroyed += 1; } },
  });
  const request = () => renderer.applyPrimaryTraversalRequest("traced",
    { leafBricks: 1, targetPixels: 368000, environmentRefinementDepth: 0 }, false, true);
  request();
  assert.equal(renderer.requestedSunlightCache, false);
  assert.equal(destroyed, 0);
  renderer.optionalPipelineTasks.clear();
  request();
  assert.equal(renderer.requestedSunlightCache, true);
  assert.equal(destroyed, 1);
  assert.equal(renderer.svoDryScenePipeline, undefined);
  request();
  assert.equal(destroyed, 1);
});
