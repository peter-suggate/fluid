import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createAdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { commitSliceSourceLedger, planSliceDynamicRemap } from
  "../lib/methods/adaptive-volume/advance-slice/slice-dynamic-remap";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("2-D source ledger transactions match a production GPU receipt",
  { timeout: 120_000 }, async t => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/advance-slice-source-ledger-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU; globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      t.diagnostic(`Dawn backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}; adapter=${JSON.stringify(adapter.info)}`);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const definition = getSceneDefinition("garden-hose");
      const scene = sceneDocument(definition);
      const profile = definition.methodProfile?.methodId === "adaptive-volume"
        ? definition.methodProfile : undefined;
      const values = resolveMethodValues(adaptiveMassMethod,
        profile?.quality ?? "balanced", profile?.overrides ?? {});
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene,
        profile?.quality ?? "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      while (!solver.advanceTo(scene.numerics.fixedDt_s, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion?.();
      await device.queue.onSubmittedWorkDone();
      const receipt = await solver.readGeometricVolumeTransportReceiptQA();
      const gpuLedger = receipt.hoseSourceLedger as Record<string, number>;
      assert.ok(gpuLedger.eventRequested! > 0);
      assert.ok(receipt.executedSubsteps > 0);
      // Before the storage-boundary fix this exact Metal fixture produced
      // pending=14.455704689025879 and both compensation lanes were zero.
      // Preserve the observed failure fingerprint as regression provenance.
      assert.equal(receipt.executedSubsteps, 11);
      assert.equal(gpuLedger.pending, 14.455714225769043);
      assert.equal(gpuLedger.pendingCompensation, 0);
      assert.equal(gpuLedger.emittedCompensation, 0.00000286102294921875);

      // Feed the GPU-authored event, availability, factor and two-level rate
      // reduction into the CPU transaction. This isolates the persistent
      // ledger algebra from the intentionally 2-D source geometry.
      const slice = createAdvanceSlice(productionSceneSliceSeedById("garden-hose"));
      const count = slice.numericalTopology.cells.length;
      const sourceRate = new Float32Array(count);
      sourceRate[0] = gpuLedger.continuousPlannedRate!;
      const plan = planSliceDynamicRemap({ topology: slice.numericalTopology,
        density: slice.fields.density,
        capacityBefore: slice.fields.capacity,
        capacityAfter: slice.fields.capacity,
        capacityRate: new Float32Array(count), sourceRate,
        // A unit planning interval preserves the already-rounded requested
        // event exactly; transport commits retain the GPU's physical dt below.
        dt: 1, pendingSourceAreaFine: 0,
        requestedSourceAreaRateFine: gpuLedger.eventRequested,
        sourceAvailableAreaFine: gpuLedger.available,
        sourceFactor: gpuLedger.factor,
        continuousPlannedRate: gpuLedger.continuousPlannedRate });
      let ledger = plan.ledger;
      for (let step = 0; step < receipt.executedSubsteps; step += 1) {
        ledger = commitSliceSourceLedger(ledger, sourceRate, receipt.substepDt_s).ledger;
      }
      const differences: string[] = [];
      for (const lane of ["pending", "requested", "emitted", "available", "factor",
        "eventRequested", "eventEmitted", "fault", "requestedCompensation",
        "emittedCompensation", "eventBalanceResidual", "eventPendingBefore",
        "pendingCompensation", "continuousPlannedRate"] as const) {
        if (ledger[lane] !== gpuLedger[lane]) differences.push(
          `${lane}: GPU=${gpuLedger[lane]} CPU=${ledger[lane]}`);
      }
      assert.deepEqual(differences, [], JSON.stringify({ receipt, gpuLedger, ledger }));
    } finally {
      solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
