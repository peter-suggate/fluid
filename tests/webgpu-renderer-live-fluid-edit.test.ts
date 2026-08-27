import assert from "node:assert/strict";
import test from "node:test";

import { FluidLabRenderer } from "../lib/core/webgpu-renderer";

test("a live liquid edit retires the retained water mesh", () => {
  let injected = 0;
  let invalidated = 0;
  const renderer = Object.create(FluidLabRenderer.prototype) as {
    disposed: boolean;
    deviceLost: boolean;
    gpuFluid: {
      simulationReady: boolean;
      injectLiquidBall: () => void;
    };
    waterPipeline: { invalidateSurface: () => void };
    pausedPresentationRevision: number;
  };
  renderer.disposed = false;
  renderer.deviceLost = false;
  renderer.gpuFluid = {
    simulationReady: true,
    injectLiquidBall: () => { injected += 1; },
  };
  renderer.waterPipeline = {
    invalidateSurface: () => { invalidated += 1; },
  };
  renderer.pausedPresentationRevision = 7;

  assert.equal(FluidLabRenderer.prototype.injectLiquidBall.call(
    renderer as unknown as FluidLabRenderer, {
    centre_m: { x: 0, y: 0.8, z: 0 },
    radius_m: 0.1,
  }), true);
  assert.equal(injected, 1, "the edit must reach the attached solver once");
  assert.equal(invalidated, 1, "the next frame must re-extract the water surface");
  assert.equal((renderer as unknown as FluidLabRenderer).presentationRevision, 8,
    "a paused viewport must be woken for the replacement mesh");
});
