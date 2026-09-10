import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createAnalyticMotionScene } from "../lib/core/analytic-motion-scenes";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("adaptive mass follows all gravity axes and live reversals without losing mass", { timeout: 240_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "gravity-direction");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    const definition = getSceneDefinition("coarse-surface-free-fall");
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", definition.methodProfile!.overrides);
    // A suspended cube has room to fall in all six directions. Mixed rungs
    // exercise the actual pressure, transport and presentation pipeline.
    for (const axis of [0, 1, 2]) for (const sign of [-1, 1]) {
      const scene = createAnalyticMotionScene("free-fall");
      scene.container.depth_m = 1.6;
      scene.fluid.initialDamBreakDimensions_m = { x: .4, y: .4, z: .4 };
      scene.fluid.initialDamBreakOrigin_m = { x: .6, y: .6, z: .6 };
      scene.fluid.refinementRegions = scene.fluid.refinementRegions!.map(region => ({ ...region,
        min_m: { ...region.min_m, z: -.8 }, max_m: { ...region.max_m, z: .8 } }));
      const vector = [0, 0, 0]; vector[axis] = sign * 9.81;
      scene.fluid.gravity_m_s2 = { x: vector[0]!, y: vector[1]!, z: vector[2]! };
      const solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
      try {
        await solver.waitForSimulationReady();
        const measure = async () => {
          const fields = await solver.readDiagnosticFields(true);
          assert.ok(fields.density.every(Number.isFinite) && fields.velocity.every(Number.isFinite));
          let mass = 0; const momentum = [0, 0, 0], moment = [0, 0, 0];
          for (let i = 0; i < fields.density.length; i++) {
            const rho = fields.density[i]!; mass += rho;
            const coordinate = [i % 32, Math.floor(i / 32) % 32, Math.floor(i / 1024)];
            for (let a = 0; a < 3; a++) {
              momentum[a]! += rho * fields.velocity[4 * i + a]!;
              moment[a]! += rho * (coordinate[a]! + .5) * .05;
            }
          }
          return { mass, velocity: momentum.map(value => value / mass), center: moment.map(value => value / mass) };
        };
        const initial = await measure();
        const dt = scene.numerics.fixedDt_s;
        for (let step = 1; step <= 6; step++) {
          while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
          await solver.waitForTopologyReady(); await solver.assertSimulationHealthy();
        }
        const fallen = await measure();
        console.log(JSON.stringify({ axis, sign, initial, fallen }));
        assert.ok(Math.abs(fallen.mass / initial.mass - 1) < .005, "mass drift under directional gravity");
        assert.ok(Math.abs(fallen.velocity[axis]! - vector[axis]! * 6 * dt) < .02,
          `axis ${axis}, sign ${sign}: must match g × t`);
        // Transport precedes acceleration in CM12, so use its discrete trajectory.
        const expectedTravel = vector[axis]! * dt * dt * 6 * 5 / 2;
        assert.ok(Math.abs(fallen.center[axis]! - initial.center[axis]! - expectedTravel) < .012,
          `axis ${axis}, sign ${sign}: liquid must move along the discrete ballistic trajectory`);
        for (let a = 0; a < 3; a++) if (a !== axis)
          assert.ok(Math.abs(fallen.velocity[a]!) < .05, `spurious transverse acceleration on ${a}`);
        solver.applySceneUniforms({ ...scene, fluid: { ...scene.fluid,
          gravity_m_s2: { x: -vector[0]!, y: -vector[1]!, z: -vector[2]! } } });
        for (let step = 7; step <= 12; step++) {
          while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
          await solver.waitForTopologyReady(); await solver.assertSimulationHealthy();
        }
        const reversed = await measure();
        assert.ok(sign * (fallen.velocity[axis]! - reversed.velocity[axis]!) > .5, "live gravity reversal must decelerate existing liquid");
        assert.ok(Math.abs(reversed.mass / initial.mass - 1) < .005, "mass drift after reversal");
      } finally { solver.destroy(); }
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
