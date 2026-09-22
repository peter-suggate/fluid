import assert from "node:assert/strict";
import test from "node:test";
import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";

// Control native compilation completion to exercise the race independently of
// compiler caches. The browser integration additionally checks the frame gate.
function fixture() {
  const pipeline = Object.create(RasterWaterPipeline.prototype) as RasterWaterPipeline;
  const state = pipeline as unknown as Record<string, unknown>;
  state.globalFineLevelSet = {};
  return { pipeline, state };
}

test("sparse water waits for the entire deferred chain, starting it only once", async () => {
  const { pipeline, state } = fixture();
  let starts = 0;
  let complete!: () => void;
  state.startGlobalSurfaceCompilation = () => {
    starts++;
    return new Promise<void>(resolve => { complete = resolve; });
  };
  assert.equal(pipeline.prepareSurfacePipelines(), false);
  state.polygoniseGlobalFineScanPipeline = {};
  state.polygoniseGlobalFineEmitPipeline = {};
  assert.equal(pipeline.prepareSurfacePipelines(), false, "classifier must also finish");
  state.extractGlobalFinePipeline = {};
  for (const stage of ["prepareClassifyPipeline", "prepareSurfaceScanPipeline", "countSurfaceBlocksPipeline", "addSurfaceBlockOffsetsPipeline"]) {
    assert.equal(pipeline.prepareSurfacePipelines(), false, `${stage} must finish before presentation`);
    state[stage] = {};
  }
  complete();
  await state.extractGlobalFinePipelinePromise;
  assert.equal(pipeline.prepareSurfacePipelines(), true);
  assert.equal(starts, 1);
});

test("deferred water failures propagate instead of waiting forever", async () => {
  const { pipeline, state } = fixture();
  const failure = new Error("native classifier compilation failed");
  state.startGlobalSurfaceCompilation = () => Promise.reject(failure);
  assert.equal(pipeline.prepareSurfacePipelines(), false);
  await state.extractGlobalFinePipelinePromise;
  assert.throws(() => pipeline.prepareSurfacePipelines(), error => error === failure);
});

test("dry frames need no adaptive specialization; coarse failures remain fatal", () => {
  const { pipeline, state } = fixture();
  state.globalFineLevelSet = undefined;
  assert.equal(pipeline.prepareSurfacePipelines(), true);
  state.coarseLevelSet = {};
  state.extractGlobalCoarsePipelineFailed = true;
  state.globalCoarseCompilationError = new Error("coarse classifier failed");
  assert.throws(() => pipeline.prepareSurfacePipelines(), /coarse classifier failed/);
});
