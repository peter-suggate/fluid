import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("mini32 settled bottom keeps stable coarse support",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-settled-dam-bottom-dawn.test.ts");
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
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      // Exercise the exact registered/UI scene. The document finalizer rebuilds
      // its SolidWorld shell for the final 32^3 lattice.
      const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        timeStep: "paper",
        resolutionMode: "adaptive",
        // This is the default sparse-CM12 policy used by the UI. Activity mode
        // is a stricter policy and is not a substitute for this regression.
        selectorMode: "surface",
        ...(process.env.FLUID_MINI32_SETTLED_SHARPENING === "off"
          ? { surfaceSharpening: "off" } : {}),
      });
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();

      const steps = Number(process.env.FLUID_MINI32_SETTLED_STEPS ?? 600);
      const stableWindowSteps = Number(
        process.env.FLUID_MINI32_SETTLED_WINDOW_STEPS ?? 48,
      );
      const stableWindowStart = steps - stableWindowSteps + 1;
      const histories = new Map<string, Array<Readonly<{
        step: number;
        accepted: number;
        planned: number;
        candidate: number;
        reasons: number;
        planReasons: number;
        meanDensity: number;
        velocityTravel: number;
        supportMask: number;
        minimumDensity?: number;
        maximumDensity?: number;
      }>>>();
      for (let step = 1; step <= steps; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step < stableWindowStart) {
          if (step % 20 === 0) await device.queue.onSubmittedWorkDone();
          continue;
        }
        await device.queue.onSubmittedWorkDone();
        const [snapshot, sampledFields] = await Promise.all([
          solver.readGPUActivityPolicy(),
          step >= steps - 1 ? solver.readDiagnosticFields() : undefined,
        ]);
        if (process.env.FLUID_MINI32_SETTLED_TRACE === "1") {
          process.stderr.write(`[mini32-settled ${step}] ${JSON.stringify(
            snapshot.bricks.filter((brick) => brick.active
              && brick.coordinate[1] === 0 && brick.coordinate[0] >= 2)
              .map((brick) => ({ coordinate: brick.coordinate,
                accepted: brick.acceptedResolution, reasons: brick.reasons,
                supportMask: brick.supportMask, meanDensity: brick.meanDensity,
                planReasons: brick.planReasons })),
          )}\n`);
        }
        for (const brick of snapshot.bricks) {
          if (!brick.active || brick.coordinate[1] !== 0) continue;
          const key = brick.coordinate.join(",");
          const history = histories.get(key) ?? [];
          let minimumDensity: number | undefined;
          let maximumDensity: number | undefined;
          if (sampledFields) {
            minimumDensity = Number.POSITIVE_INFINITY;
            maximumDensity = Number.NEGATIVE_INFINITY;
            const [nx, ny] = [solver.info.nx, solver.info.ny];
            for (let z = 8 * brick.coordinate[2]; z < 8 * brick.coordinate[2] + 8;
              z += 1) for (let y = 0; y < 8; y += 1) {
              for (let x = 8 * brick.coordinate[0]; x < 8 * brick.coordinate[0] + 8;
                x += 1) {
                const at = x + nx * (y + ny * z);
                if (sampledFields.solidOpenFraction[at]! <= 1e-6) continue;
                const density = sampledFields.density[at]!;
                minimumDensity = Math.min(minimumDensity, density);
                maximumDensity = Math.max(maximumDensity, density);
              }
            }
          }
          history.push({ step, accepted: brick.acceptedResolution,
            planned: brick.plannedResolution,
            candidate: brick.candidateResolution, reasons: brick.reasons,
            planReasons: brick.planReasons, meanDensity: brick.meanDensity,
            velocityTravel: brick.maximumVelocityTravelFineCells,
            supportMask: brick.supportMask,
            ...(minimumDensity === undefined ? {} : {
              minimumDensity, maximumDensity,
            }) });
          histories.set(key, history);
        }
      }
      await device.queue.onSubmittedWorkDone();

      const policy = await solver.readGPUActivityPolicy();
      const bottom = policy.bricks.filter((brick) =>
        brick.active && brick.coordinate[1] === 0);
      assert.equal(bottom.length, 16);
      const unstable = [...histories].flatMap(([coordinate, history]) => {
        const transitions = history.slice(1).filter((sample, index) =>
          sample.accepted !== history[index]!.accepted);
        if (transitions.length === 0) return [];
        return [{ coordinate, transitionCount: transitions.length,
          accepted: history.map((sample) => sample.accepted).join(""),
          surface: history.map((sample) => (sample.reasons & 1) !== 0 ? "S" : ".")
            .join(""),
          lastTwo: history.slice(-2) }];
      });
      assert.deepEqual(unstable, [],
        `settled bottom changed resolution during the final ${stableWindowSteps} steps: ${
          JSON.stringify(unstable)}`);
      for (const brick of bottom) {
        assert.ok(brick.meanDensity > 0.5,
          `bottom brick ${brick.coordinate.join(",")} is not flooded: ${
            brick.meanDensity}`);
        assert.equal(brick.reasons & 1, 0,
          `submerged bottom brick ${brick.coordinate.join(",")} was classified as surface`);
        assert.ok(brick.acceptedResolution <= 4,
          `submerged bottom brick ${brick.coordinate.join(",")} stayed ${
            brick.acceptedResolution}^3`);
      }
      assert.ok(policy.bricks.some((brick) => brick.active
        && brick.coordinate[1] > 0 && (brick.reasons & 1) !== 0
        && brick.acceptedResolution === 8),
      "the actual free surface above the bottom must remain fine");
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
