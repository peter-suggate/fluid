import assert from "node:assert/strict";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("coarse-first mini32 supports the diagonal dam front at eight default steps",
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "mini32-corner");
    let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const errors: string[] = [];
      device!.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        selectorMode: "coarse-first", timeStep: "paper",
      });
      solver = await adaptiveMassMethod.createSolverAsync!(device!, scene, "balanced", values,
        undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      const mass = (density: Float32Array) => density.reduce((sum, value) => sum + value, 0);
      const initialMass = mass((await solver.readDiagnosticFields(true)).density);
      const initialActivity = await solver.readGPUActivityPolicy();
      assert.equal(initialActivity.bricks.find(b => b.coordinate.join("/") === "3/0/3")?.active, false,
        "refinement backing must not activate the dry corner at initialization");
      for (let step = 1; step <= 8; step++) {
        while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
        if (step === 1) {
          const firstStep = await solver.readGPUActivityPolicy();
          // Regression: almost vertical shared motion used to count as a B8
          // impact through a touching lateral face. Its 2:1 closure then
          // refined the back-top corner B2 -> B4 at exactly the default dt.
          const top = firstStep.bricks.filter(b => b.active && b.coordinate[1] === 3);
          for (const brick of top) {
            const initial = initialActivity.bricks.find(b => b.leafId === brick.leafId)!;
            assert.equal(brick.acceptedResolution, initial.acceptedResolution,
              `first-step top refinement at ${brick.coordinate}: ${brick.planReasons}`);
          }
          assert.equal(top.find(b => b.coordinate.join("/") === "0/3/0")?.acceptedResolution, 2);
          assert.ok(top.some(b => b.acceptedResolution === 8), "retain the sharp dam-edge geometry");
          assert.equal(firstStep.faultFlags, 0);
        }
      }
      const [activity, fields, stats] = await Promise.all([
        solver.readGPUActivityPolicy(), solver.readDiagnosticFields(true), solver.readStats(),
      ]);
      assert.ok(Math.abs((stats.simulatedTime_s ?? NaN) - 8 * CM12_PAPER_DT_S) < 1e-9, "capture the simulated clock, not the requested target");
      assert.equal(activity.commitFailed, false);
      assert.equal(activity.faultFlags, 0);
      assert.equal(stats.topologyGenerationCount ?? 0, 0, "the front must not wait for background generation replacement");
      const active = activity.bricks.filter(b => b.active);
      assert.ok(active.some(b => b.coordinate[0] === 3 && b.coordinate[1] === 0 && b.coordinate[2] === 3),
        "the floor corner must be resident before the dam front reaches it");
      let cornerMass = 0, symmetryError = 0;
      for (let z = 0; z < 32; z++) for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
        const rho = fields.density[x + 32 * (y + 32 * z)]!;
        assert.ok(Number.isFinite(rho));
        if (x >= 24 && z >= 24 && y < 8) cornerMass += rho;
        symmetryError = Math.max(symmetryError, Math.abs(rho - fields.density[z + 32 * (y + 32 * x)]!));
      }
      console.log(JSON.stringify({ cornerMass, symmetryError, relativeMassError: Math.abs(mass(fields.density) / initialMass - 1), activeCells: stats.activeSampleCount }));
      assert.ok(cornerMass > 1, "allocated corner must actually accept liquid");
      const byCoordinate = new Map(active.map(b => [b.coordinate.join("/"), b]));
      for (const brick of active) {
        const [x, y, z] = brick.coordinate;
        assert.equal(byCoordinate.get([z, y, x].join("/"))?.acceptedResolution,
          brick.acceptedResolution, "corner growth must preserve x/z topology symmetry");
      }
      assert.ok(Math.abs(mass(fields.density) / initialMass - 1) < 0.005);
      assert.ok(active.some(b => b.acceptedResolution < 8), "retain adaptive coarse support");
      assert.deepEqual(errors, []);
    } finally {
      solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
