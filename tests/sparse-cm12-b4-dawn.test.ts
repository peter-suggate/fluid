import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("B4 advances a mixed-resolution dam with conservative mass and healthy topology", { timeout: 180_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "b4-dam");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    assert.ok(device);
    const errors: string[] = [];
    device!.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    solver = await WebGPUAdaptiveMassSolver.createAsync(device!,
      sceneDocument(getSceneDefinition("minimal-power-dam-break-32")), "balanced", undefined,
      adaptiveMassSolverOptions({ brickFineResolution: "4", selectorMode: "coarse-first" }), () => {});
    await solver.waitForSimulationReady();
    const initial = (await solver.readAcceptedGeometricVolumeQA()).volumeFine3;
    for (let step = 1; step <= 8; step++) {
      while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion();
    }
    const activity = await solver.readGPUActivityPolicy();
    const fields = await solver.readDiagnosticFields();
    assert.equal(activity.faultFlags, 0, JSON.stringify(activity));
    assert.equal(activity.commitFailed, false);
    assert.ok(activity.bricks.filter(b => b.active).every(b => b.acceptedResolution <= 4));
    assert.ok(fields.density.every(Number.isFinite));
    const volume = await solver.readAcceptedGeometricVolumeQA();
    assert.equal(volume.outsideAuthoredVolumeFine3, 0, "B4 wall reachability must not leak through the closed tank");
    assert.equal(volume.nonfiniteCells, 0);
    assert.equal(volume.invalidCells, 0);
    assert.ok(Math.abs(volume.volumeFine3 / initial - 1) < 0.005);
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});

(dawnModule ? test : test.skip)("B4 surface-distance production preserves a still waterline and coarse submerged cells", { timeout: 180_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "b4-hydrostatic-surface-distance");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    assert.ok(device);
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const options = adaptiveMassSolverOptions({});
    assert.equal(options.brickFineResolution, 4);
    assert.equal(options.activityPolicy?.activitySignals, false);
    solver = await WebGPUAdaptiveMassSolver.createAsync(device,
      sceneDocument(getSceneDefinition("hydrostatic-power-large-offset")), "balanced", undefined, options, () => {});
    await solver.waitForSimulationReady();
    const initialVolume = (await solver.readAcceptedGeometricVolumeQA()).volumeFine3;
    for (let step = 1; step <= 8; step++) {
      while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion();
      await solver.assertSimulationHealthy();
    }
    const activity = await solver.readGPUActivityPolicy();
    const active = activity.bricks.filter(b => b.active);
    const surface = active.filter(b => (b.reasons & 1) !== 0);
    assert.ok(surface.length > 0, "the offset free surface must be classified");
    assert.ok(surface.every(b => b.acceptedResolution === 4), "every classified interface must be finest");
    assert.ok(active.some(b => b.acceptedResolution < 4 && (b.reasons & 64) !== 0), "submerged liquid must remain adaptive");
    assert.equal(activity.faultFlags, 0);
    assert.equal(activity.commitFailed, false);
    const volume = await solver.readAcceptedGeometricVolumeQA();
    assert.ok(Math.abs(volume.volumeFine3 / initialVolume - 1) < 1e-6, "a still pool conserves volume");
    const qa = await solver.readAdaptiveLevelSetQA(true);
    const vertices = qa.vertices ?? [];
    const byPosition = new Map(vertices.map(v => [v.positionFine.join("/"), v]));
    let crossings = 0, maximumHeightError = 0;
    for (const vertex of vertices) {
      const [x, y, z] = vertex.positionFine;
      if (x < 2 || x > solver.info.nx - 2 || z < 2 || z > solver.info.nz - 2) continue;
      const next = byPosition.get(`${x}/${y + 1}/${z}`);
      if (!next || vertex.support === 0 || next.support === 0 || vertex.phiFine >= 0 || next.phiFine < 0) continue;
      const height = y - vertex.phiFine / (next.phiFine - vertex.phiFine);
      maximumHeightError = Math.max(maximumHeightError, Math.abs(height - 15.25));
      crossings++;
    }
    assert.ok(crossings > 0, "measure the represented waterline, not only the density census");
    assert.ok(maximumHeightError < 0.1, `still-water height error ${maximumHeightError}`);
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
