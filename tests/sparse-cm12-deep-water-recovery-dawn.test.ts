import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

const maximumSpeed = (velocity: Float32Array): number => {
  let maximum = 0;
  for (let at = 0; at < velocity.length; at += 4) {
    maximum = Math.max(maximum, Math.hypot(
      velocity[at]!, velocity[at + 1]!, velocity[at + 2]!,
    ));
  }
  return maximum;
};

const assertExactField = (actual: Float32Array, expected: Float32Array, label: string): void => {
  assert.equal(actual.length, expected.length);
  for (let at = 0; at < actual.length; at += 1) {
    assert.equal(actual[at], expected[at], `${label} changed at finest cell ${at}`);
  }
};

dawnTest("large hydrostatic stays still, refines on impact, and restores deep coarseness",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-deep-water-recovery-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      const scene = sceneDocument(getSceneDefinition("water-box-tank-fill"));
      scene.rigidBodies = [];
      scene.container.width_m = 1.6;
      scene.container.height_m = 2.4;
      scene.container.depth_m = 1.6;
      scene.container.fillFraction = 0.7;
      scene.voxelDomain.finestCellSize_m = 0.05;
      scene.fluid.surfaceTension_N_m = 0;
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined,
        {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          timeStep: "paper",
          activityPolicy: SPARSE_CM12_ACTIVITY_POLICY,
          pressureIterations: 128,
          pressureRelativeTolerance: 0,
        },
        () => {},
      );
      assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [32, 48, 32]);

      const initialDensity = (await solver.readDiagnosticFields()).density;
      let oneSecondTopology = "";
      for (let step = 1; step <= 60; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step % 15 !== 0) continue;
        await device.queue.onSubmittedWorkDone();
        const fields = await solver.readDiagnosticFields();
        assertExactField(fields.density, initialDensity,
          `hydrostatic density at ${step * CM12_PAPER_DT_S}s`);
        assert.ok(maximumSpeed(fields.velocity) <= 5e-6,
          `hydrostatic maximum speed at ${step * CM12_PAPER_DT_S}s was ${
            maximumSpeed(fields.velocity)} m/s`);
        if (step === 30) {
          const activity = await solver.readGPUActivityPolicy();
          oneSecondTopology = activity.bricks.filter((brick) => brick.active)
            .map((brick) => `${brick.key}:${brick.acceptedResolution}`).join("|");
        }
      }

      const calm = await solver.readGPUActivityPolicy();
      const calmTopology = calm.bricks.filter((brick) => brick.active)
        .map((brick) => `${brick.key}:${brick.acceptedResolution}`).join("|");
      assert.equal(calmTopology, oneSecondTopology,
        "the calm topology must stop churning before the two-second benchmark endpoint");
      const deepBaseline = new Map(calm.bricks.filter((brick) => brick.active
        && (brick.reasons & 64) !== 0 && (brick.reasons & 1) === 0)
        .map((brick) => [brick.key, brick.acceptedResolution] as const));
      assert.ok(deepBaseline.size >= 32, "the benchmark must contain a substantial deep bulk");
      assert.ok([...deepBaseline.values()].some((resolution) => resolution <= 2),
        "calm deep water must reach an aggressive coarse rung");

      solver.injectLiquidBall({ centre_m: { x: 0, y: 2.05, z: 0 }, radius_m: 0.16 });
      let impactFine = false;
      let impactRefinedDeep = false;
      for (let step = 61; step <= 90; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step % 5 !== 0) continue;
        await device.queue.onSubmittedWorkDone();
        const sample = await solver.readGPUActivityPolicy();
        impactFine = impactFine || sample.bricks.some((brick) => brick.active
          && brick.acceptedResolution === 8);
        impactRefinedDeep = impactRefinedDeep || sample.bricks.some((brick) => {
          const baseline = deepBaseline.get(brick.key);
          return baseline !== undefined && brick.acceptedResolution > baseline;
        });
      }
      await device.queue.onSubmittedWorkDone();
      assert.ok(impactFine,
      "the falling ball must create accepted fine-grained regions");
      assert.ok(impactRefinedDeep,
        "impact motion must refine at least one previously calm deep brick");

      let recoveredAt_s: number | undefined;
      let quietestRecoveredSpeed = Number.POSITIVE_INFINITY;
      for (let step = 91; step <= 450 && recoveredAt_s === undefined; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step % 15 !== 0) continue;
        await device.queue.onSubmittedWorkDone();
        const recovered = await solver.readGPUActivityPolicy();
        const recoveredByKey = new Map(recovered.bricks.filter((brick) => brick.active)
          .map((brick) => [brick.key, brick.acceptedResolution] as const));
        const exactDeepTopology = [...deepBaseline].every(([key, baseline]) =>
          recoveredByKey.get(key) === baseline);
        if (!exactDeepTopology) continue;
        const speed = maximumSpeed((await solver.readDiagnosticFields()).velocity);
        quietestRecoveredSpeed = Math.min(quietestRecoveredSpeed, speed);
        if (speed <= 0.1) recoveredAt_s = (step - 60) * CM12_PAPER_DT_S;
      }
      assert.ok(recoveredAt_s !== undefined,
        `deep topology never reached an exact quiet recovery; minimum speed was ${
          quietestRecoveredSpeed} m/s`);
      assert.deepEqual(uncaptured, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
