import assert from "node:assert/strict";
import test from "node:test";

import "../lib/methods";
import { defaultCamera, defaultScene } from "../lib/core/model";
import { FluidLabRenderer, gpuSceneStructuralKey, type SimulationRunConfig } from "../lib/core/webgpu-renderer";

const config: SimulationRunConfig = {
  methodId: "uniform-volume", quality: "balanced", values: {}, simulationEpoch: 1,
};

for (const mode of ["fluid-only", "full-scene"] as const) {
  for (const transition of ["cold load", "reset", "scene switch"] as const) {
    test(`${mode} holds presentation during ${transition} until the matching source is ready`, () => {
      const resumed = new Error("ready source reached presentation");
      const solver = { info: { nx: 64, ny: 64, nz: 64, surfaceRevision: 0 } };
      let ready = false;
      let sourceRequests = 0;
      const renderer = Object.assign(Object.create(FluidLabRenderer.prototype), {
        device: {}, context: {}, uniformBuffer: {}, bodyBuffer: {},
        waterPipeline: {
          setFluidDomain: () => assert.fail("must not alter retained water bindings"),
          encode: () => assert.fail("must not extract the old field with fallback dimensions"),
        },
        gpuFluid: transition === "cold load" ? undefined : solver,
        reseedInFlight: transition === "reset",
        presentationTexture: {}, upscalePipeline: {}, upscaleBindGroup: {},
        surfaceTuningCache: { resolve: () => ({ resolutionScale: 1 }), key: "test" },
        resize: () => {},
        currentGPUFluid: () => { sourceRequests++; return ready ? solver : undefined; },
        sparseSceneProducer: () => undefined,
        publishRenderScene: () => { throw resumed; },
        ensureRequestedOptionalPipelines: () => { throw resumed; },
      });
      const draw = () => renderer.draw(0, defaultScene, defaultCamera, [], undefined,
        config, undefined, undefined, mode);

      assert.equal(draw().presentationSubmitted, false);
      assert.equal(draw().presentationSubmitted, false);
      assert.equal(sourceRequests, 2, "keep pumping source initialization while presentation is held");
      ready = true;
      assert.throws(draw, error => error === resumed,
        "resume presentation as soon as the matching source is available");
    });
  }
}

test("an in-place reseed invalidates the mesh even with unchanged textures and revision zero", async () => {
  let finish!: (value: boolean) => void;
  let invalidations = 0;
  const solver = { info: { surfaceRevision: 0 }, reseed: () => new Promise<boolean>(resolve => { finish = resolve; }) };
  const renderer = Object.assign(Object.create(FluidLabRenderer.prototype), {
    device: {}, gpuFluid: solver, gpuFluidGeneration: 1, gpuFluidRequestGeneration: 1,
    attachedStructuralKey: gpuSceneStructuralKey(defaultScene, config),
    waterPipeline: { invalidateSurface: () => { invalidations++; } },
    onStatus: () => {}, resetGPUQueueTracking: () => {}, pausedPresentationRevision: 0,
  });

  assert.equal(renderer.tryReseedGPUFluid(defaultScene, config, "new seed", "fluid-only"), true);
  assert.equal(invalidations, 0, "retain the mesh while reseeding is pending");
  finish(true);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(renderer.gpuFluid, solver);
  assert.equal(solver.info.surfaceRevision, 0);
  assert.equal(invalidations, 1);
  assert.equal(renderer.pausedPresentationRevision, 1, "wake the paused renderer to extract the new seed");
});
