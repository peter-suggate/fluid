import "../lib/methods";
import test from "node:test";
import assert from "node:assert/strict";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
import type { GPUStatus } from "../lib/core/gpu-status";
import { defaultCamera, defaultScene } from "../lib/core/model";

test("a runtime failure is terminal, preserves its first cause, and cannot restart", async () => {
  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, (status) => statuses.push(status));
  renderer.stopAfterFailure(new Error("Invalid bind group in pressure solve"));
  renderer.stopAfterFailure(new Error("Cleanup failed"));
  renderer.setSimulationRunning(true);
  renderer.resetSimulationTimeline();
  await assert.rejects(renderer.initialize(), /Invalid bind group in pressure solve/);
  const metrics = renderer.draw(1 / 30, defaultScene, defaultCamera, [], undefined,
    { methodId: "uniform-volume", quality: "balanced", values: {} });
  assert.equal(metrics.presentationSubmitted, false);
  assert.deepEqual(statuses.map(({ state, label }) => ({ state, label })), [
    { state: "unavailable", label: "Invalid bind group in pressure solve" },
  ]);
});

test("a lens-bloom compile failure is reported once and ends the renderer", async () => {
  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, (status) => statuses.push(status));
  const pipeline = {
    lensBloomOptions: { enabled: true }, lensBloomReady: false,
    ensureLensBloomReady: () => Promise.reject(new Error("bloom shader")),
  };
  Object.assign(renderer, { waterPipeline: pipeline });
  const access = renderer as unknown as { prepareLensBloom(pipeline: unknown): void; lensBloomPreparation?: Promise<void> };
  access.prepareLensBloom(pipeline);
  await access.lensBloomPreparation;
  renderer.stopAfterFailure(new Error("device lost later"));
  assert.deepEqual(statuses.map(({ state, label }) => ({ state, label })), [
    { state: "unavailable", label: "Lens bloom preparation failed: bloom shader" },
  ]);
  await assert.rejects(renderer.initialize(), /Lens bloom preparation failed: bloom shader/);
});

test("destroy after a failure still releases resources exactly once", () => {
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  let destroyed = 0;
  Object.assign(renderer, { waterPipeline: { destroy: () => { destroyed += 1; } } });
  renderer.stopAfterFailure(new Error("validation"));
  renderer.destroy();
  renderer.destroy();
  assert.equal(destroyed, 1);
});
