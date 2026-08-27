import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { getScenePreset } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
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

interface BottomBrickSample {
  readonly coordinate: readonly [number, number, number];
  readonly resolution: number;
  readonly reasons: number;
  readonly planReasons: number;
  readonly meanFineDensity: number;
  readonly minimumFineDensity: number;
}

interface StepSample {
  readonly step: number;
  readonly topologyGeneration: number;
  readonly bricks: readonly BottomBrickSample[];
  readonly verticalLadder: readonly {
    readonly resolution: number;
    readonly reasons: number;
    readonly planReasons: number;
  }[];
}

dawnTest("Sparse CM12 publishes the full planar-wall ladder without a fine bottom", {
  timeout: 180_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test",
    "tests/sparse-cm12-deep-bottom-coarsening-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([
      `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
      "enable-dawn-features=disable_blob_cache",
    ]);
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: { gpu },
    });
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter, "Dawn must expose a WebGPU adapter");
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });

    const sourceScene = getScenePreset("water-box-dam-break").create();
    const scene = sceneAtContainerExtents(sourceScene, {
      width_m: sourceScene.container.width_m,
      height_m: 2,
      depth_m: sourceScene.container.depth_m,
    });
    // Preserve the production corner-dam shape in a taller tank and union in a
    // 26-cell-deep pool. Four vertical B8 pages then expose the complete
    // surface-to-floor 8/4/2/1 ladder without violating strong 2:1 grading.
    scene.fluid.initialDamBreakDimensions_m = { x: 0.6, y: 1.8, z: 0.5 };
    scene.fluid.initialLiquidVolumes = [{
      shape: "box",
      min_m: { x: -0.5 * scene.container.width_m, y: 0,
        z: -0.5 * scene.container.depth_m },
      max_m: { x: 0.5 * scene.container.width_m, y: 1.3,
        z: 0.5 * scene.container.depth_m },
    }];
    const values = {
      ...adaptiveMassMethod.presetFor("balanced"),
      resolutionMode: "adaptive",
      brickFineResolution: "8",
      presentationPageResolution: "8",
      surfaceFineRings: 1,
      timeStep: "paper",
      secondaryParticles: "off",
    };
    solver = await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {},
    ) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [24, 40, 16]);

    const samples: StepSample[] = [];
    const sample = async (step: number) => {
      await device!.queue.onSubmittedWorkDone();
      const [activity, fields] = await Promise.all([
        solver!.readGPUActivityPolicy(),
        solver!.readDiagnosticFields(),
      ]);
      const bottom = activity.bricks.filter((brick) => brick.active
        && brick.coordinate[1] === 0
        && brick.coordinate[0] >= 0 && brick.coordinate[0] < 3
        && brick.coordinate[2] >= 0 && brick.coordinate[2] < 2)
        .map<BottomBrickSample>((brick) => {
          let sum = 0, minimum = Infinity, count = 0;
          const x0 = brick.coordinate[0] * 8;
          const z0 = brick.coordinate[2] * 8;
          for (let z = z0; z < z0 + 8; z += 1) {
            for (let y = 0; y < 8; y += 1) {
              for (let x = x0; x < x0 + 8; x += 1) {
                const rho = fields.density[x + 24 * (y + 40 * z)]!;
                sum += rho;minimum = Math.min(minimum, rho);count += 1;
              }
            }
          }
          return {
            coordinate: brick.coordinate,
            resolution: brick.acceptedResolution,
            reasons: brick.reasons,
            planReasons: brick.planReasons,
            meanFineDensity: sum / count,
            minimumFineDensity: minimum,
          };
        }).sort((left, right) => left.coordinate[2] - right.coordinate[2]
          || left.coordinate[0] - right.coordinate[0]);
      assert.equal(bottom.length, 6,
        `the full-floor pool must keep all six bottom bricks resident on step ${step}`);
      const verticalLadder = Array.from({ length: 4 }, (_, y) => {
        const brick = activity.bricks.find((candidate) => candidate.active
          && candidate.coordinate[0] === 2 && candidate.coordinate[1] === y
          && candidate.coordinate[2] === 1);
        return { resolution: brick?.acceptedResolution ?? 0,
          reasons: brick?.reasons ?? 0, planReasons: brick?.planReasons ?? 0 };
      });
      samples.push({ step, topologyGeneration: activity.acceptedTopologyGeneration,
        bricks: bottom, verticalLadder });
    };

    await sample(0);
    assert.deepEqual(samples[0]!.verticalLadder.map((entry) => entry.resolution),
      [1, 2, 4, 8],
      "initial planar-wall restriction must publish the full 8/4/2/1 ladder");
    // Two simulated seconds cover fifteen topology epochs and the dam impact,
    // long enough for the former B8/B4 ping-pong to complete several cycles.
    for (let step = 1; step <= 60; step += 1) {
      assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true,
        `advance ${step}`);
      await sample(step);
    }

    const failures: string[] = [];
    for (let brick = 0; brick < 6; brick += 1) {
      const history = samples.map((entry) => ({
        step: entry.step,
        generation: entry.topologyGeneration,
        ...entry.bricks[brick]!,
      }));
      const independentlyDeep = history.filter((entry) =>
        entry.meanFineDensity >= 0.95 && entry.minimumFineDensity >= 0.75);
      const label = history[0]!.coordinate.join(",");
      const profile = history.map((entry) =>
        `${entry.step}:${entry.resolution}/${entry.reasons}@${entry.generation}`).join(" ");
      // A genuinely fine interface two pages above may move its strong-2:1
      // support cone down by one rung. The floor itself must still remain B2
      // or coarser; B4/B8 here is the original blanket-boundary regression.
      const fine = independentlyDeep.filter((entry) => entry.resolution > 2);
      if (fine.length > 0) failures.push(
        `${label} fine while deep at [${fine.map((entry) => entry.step).join(",")}]; ${profile}`,
      );
      const falseSurface = independentlyDeep.filter((entry) => (entry.reasons & 1) !== 0);
      if (falseSurface.length > 0) failures.push(
        `${label} false surface at [${falseSurface.map((entry) => entry.step).join(",")}]; ${profile}`,
      );
    }
    assert.deepEqual(validationErrors, []);
    const ladderProfile = samples.map((entry) => entry.step + ":"
      + entry.verticalLadder.map((brick) => brick.resolution + "/"
        + brick.reasons + "/p" + brick.planReasons).join(",")).join(" ");
    assert.deepEqual(failures, [],
      `deep bottom coarsening was not stable:\n${failures.join("\n")}`
        + `\nvertical ladder ${ladderProfile}`);
  } finally {
    solver?.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
});
