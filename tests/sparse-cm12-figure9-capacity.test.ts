import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  adaptiveMassPresentationDimensionsForScene,
  adaptiveMassReceiverScaleForScene,
  dormantReceiverDomain,
  residentSupportAtlas,
  WebGPUAdaptiveMassSolver,
} from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickAtlasStats,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { sparseCM12HostTemplateVariantsEnabled } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

test("Figure 9 receiver capacity reaches the authored far wall without host variants", () => {
  const scene = getScenePreset("mass-conserving-figure-9-dam-break").create();
  let atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
    surfaceFineRings: 1,
  });
  atlas = residentSupportAtlas(atlas, "adaptive");
  const initiallyActiveBrickCount = atlas.bricks.length;
  const receiverScale = adaptiveMassReceiverScaleForScene(scene, 9);
  atlas = dormantReceiverDomain(
    atlas, "adaptive", receiverScale.supportRings, "auto",
    receiverScale.minimumCapacityScale,
  );

  assert.deepEqual(atlas.brickDimensions, [16, 16, 8]);
  assert.equal(atlas.bricks.length, 16 * 16 * 8,
    "a tank that fits the receiver budget must not contain a hidden apron wall");
  assert.ok(atlas.bricks.some((brick) => brick.coordinate[0] === 15),
    "the advancing dam needs physical receiver topology at the far wall");
  assert.equal(initiallyActiveBrickCount, 616);
  assert.equal(sparseBrickAtlasStats(atlas).leafCount, 234_896);
  assert.equal(sparseCM12HostTemplateVariantsEnabled(
    sparseBrickAtlasStats(atlas).leafCount, 0, initiallyActiveBrickCount,
  ), false, "the 616-brick live frontier must use accepted-only startup packing");
});

dawnTest("Figure 9 dam reaches the far wall through activated receivers",
  { timeout: 60_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-figure9-capacity.test.ts");
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
      const scene = getScenePreset("mass-conserving-figure-9-dam-break").create();
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, adaptiveMassSolverOptions({}), () => {},
      );
      for (let step = 1; step <= 120; step += 1) {
        assert.equal(solver.advanceTo(step / 30, []), true);
      }
      await device.queue.onSubmittedWorkDone();
      const density = (await solver.readDiagnosticFields()).density;
      let front = -1, farWallMass = 0;
      for (let z = 0; z < 64; z += 1) for (let y = 0; y < 128; y += 1) {
        for (let x = 0; x < 128; x += 1) {
          const rho = Math.max(0, density[x + 128 * (y + 128 * z)]!);
          if (rho > 0.05) front = Math.max(front, x);
          if (x >= 120) farWallMass += rho;
        }
      }
      assert.equal(front, 127, "the dam front stopped before the physical far wall");
      assert.ok(farWallMass > 100,
        `the far-wall receiver layer retained only ${farWallMass} fine-cell mass`);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
