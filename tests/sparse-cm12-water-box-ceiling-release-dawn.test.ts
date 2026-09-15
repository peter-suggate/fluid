import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function densityMass(density: Float32Array) {
  let raw = 0, positive = 0;
  for (const value of density) {
    raw += value;
    positive += Math.max(0, value);
  }
  return { raw, positive };
}

function topStripVolume(density: Float32Array,
  dimensions: readonly [number, number, number]): number {
  const [nx, ny, nz] = dimensions;
  let volume = 0;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    volume += Math.max(0, density[x + nx * (ny - 1 + ny * z)]!);
  }
  return volume;
}

function percentile(values: readonly number[], fraction: number): number {
  assert.ok(values.length > 0, "phi sample set must be nonempty");
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1,
    Math.max(0, Math.floor(fraction * (sorted.length - 1))))]!;
}

dawnTest("water-box dam break releases from closed walls through frames 30/60/90", {
  timeout: 240_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test",
    "tests/sparse-cm12-water-box-ceiling-release-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [
      `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
      "enable-dawn-features=disable_blob_cache",
    ]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", event => {
      event.preventDefault(); validationErrors.push(event.error.message);
    });

    const scene = getScenePreset("water-box-dam-break").create();
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
      resolveMethodValues(adaptiveMassMethod, "balanced", { timeStep: "paper" }),
      undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [24, 16, 16]);

    const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
    const initialFields = await solver.readDiagnosticFields(true);
    const initialMass = densityMass(initialFields.density);
    const initialTopStripVolume = topStripVolume(initialFields.density, dimensions);
    const initialVolume = await solver.readAcceptedGeometricVolumeQA();
    const topologyGenerations = new Set<number>();
    let frame30TopStrip = Number.NaN, frame90TopStrip = Number.NaN;
    let frame30CeilingWet = -1, frame90CeilingWet = -1;

    for (let frame = 1; frame <= 90; frame += 1) {
      while (!solver.advanceTo(frame * CM12_PAPER_DT_S, [])) {
        await new Promise<void>(setImmediate);
      }
      await solver.awaitFrameCompletion?.();
      if (frame % 10 !== 0) continue;
      await device.queue.onSubmittedWorkDone();
      await solver.assertSimulationHealthy();
      const [fields, levelSet, volume, activity] = await Promise.all([
        solver.readDiagnosticFields(true),
        solver.readAdaptiveLevelSetQA(true),
        solver.readAcceptedGeometricVolumeQA(),
        solver.readGPUActivityPolicy(),
      ]);
      topologyGenerations.add(activity.acceptedTopologyGeneration);
      const vertices = levelSet.vertices ?? [];
      const ceilingPhi = vertices.filter(vertex =>
        vertex.positionFine[1] === dimensions[1])
        .map(vertex => vertex.phiFine);
      assert.ok(ceilingPhi.length >= 4, `frame ${frame} lacks physical-ceiling samples`);
      const ceiling = {
        minimum: Math.min(...ceilingPhi), median: percentile(ceilingPhi, 0.5),
        maximum: Math.max(...ceilingPhi),
        wet: ceilingPhi.filter(phi => phi <= 0).length,
        count: ceilingPhi.length,
      };
      // Keep the old reservoir-top plane in the trace because it distinguishes
      // ordinary column collapse from liquid retained at the physical ceiling.
      const reservoirTopPhi = vertices.filter(vertex => vertex.positionFine[1] === 14)
        .map(vertex => vertex.phiFine);

      const mass = densityMass(fields.density);
      const topStrip = topStripVolume(fields.density, dimensions);
      if (frame === 30) {
        frame30TopStrip = topStrip; frame30CeilingWet = ceiling.wet;
      }
      if (frame === 90) {
        frame90TopStrip = topStrip; frame90CeilingWet = ceiling.wet;
      }
      const sample = {
        frame, mass,
        relativeRawMassError: Math.abs(mass.raw - initialMass.raw)
          / Math.max(Math.abs(initialMass.raw), 1e-12),
        geometricVolumeFine3: volume.volumeFine3,
        relativeGeometricVolumeError: Math.abs(volume.volumeFine3
          - initialVolume.volumeFine3) / Math.max(initialVolume.volumeFine3, 1e-12),
        topStripVolume: topStrip, initialTopStripVolume, ceiling,
        reservoirTop: reservoirTopPhi.length === 0 ? undefined : {
          minimum: Math.min(...reservoirTopPhi),
          median: percentile(reservoirTopPhi, 0.5),
          maximum: Math.max(...reservoirTopPhi),
        },
        topologyGeneration: activity.acceptedTopologyGeneration,

      };
      if (process.env.FLUID_CEILING_RELEASE_TRACE === "1") {
        process.stderr.write(`[cm12-water-box-ceiling-release] ${JSON.stringify(sample)}\n`);
      }

      // Ninety f32 transport/reduction steps accumulate a few parts per million.
      // Check a symmetric bound: creating volume is as invalid as losing it.
      assert.ok(sample.relativeRawMassError <= 5e-6, JSON.stringify(sample));
      assert.ok(sample.relativeGeometricVolumeError <= 5e-6, JSON.stringify(sample));
    }

    assert.ok(frame30TopStrip > initialTopStripVolume,
      `the frame-30 splash never reached the ceiling strip: ${JSON.stringify({
        initialTopStripVolume, frame30TopStrip })}`);
    assert.ok(frame90TopStrip < frame30TopStrip,
      `ceiling-strip volume did not drain: ${JSON.stringify({
        frame30TopStrip, frame90TopStrip })}`);
    assert.equal(frame90CeilingWet, 0,
      `ceiling phi remained wet at frame 90: ${JSON.stringify({
        frame30CeilingWet, frame90CeilingWet })}`);
    assert.ok(topologyGenerations.size > 1,
      `adaptive topology did not transition: ${[...topologyGenerations].join(",")}`);
    assert.deepEqual(validationErrors, []);
  } finally {
    try { solver?.destroy(); device?.destroy(); }
    finally { await releaseWebGPUExclusiveLock(); }
  }
});
