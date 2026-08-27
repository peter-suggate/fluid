import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("MiniDam64 deep pool keeps the full wall/corner coarsening ladder", {
  timeout: 180_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test",
    "tests/sparse-cm12-mini64-boundary-coarsening-dawn.test.ts");
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
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });

    const scene = createMinimalPowerDamBreak64Scene();
    // The reported regression is easiest to see with deep water beneath the
    // released column. This is fixture geometry only: the production selector
    // receives the same density/SolidWorld fields as every other scene.
    scene.fluid.initialLiquidVolumes = [{
      shape: "box",
      min_m: { x: -0.4, y: 0, z: -0.4 },
      max_m: { x: 0.4, y: 0.4, z: 0.4 },
    }];
    solver = await WebGPUAdaptiveMassSolver.createAsync(
      device, scene, "balanced", undefined, {
        resolutionMode: "adaptive",
        brickFineResolution: 8,
        surfaceFineRings: 1,
        timeStep: "scene",
      }, () => {},
    );
    await solver.waitForSimulationReady();
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [64, 64, 64]);

    const initial = await solver.readGPUActivityPolicy();
    const acceptedAt = (coordinate: readonly [number, number, number]) =>
      initial.bricks.find((brick) => brick.active
        && brick.coordinate.every((value, axis) => value === coordinate[axis]))
        ?.acceptedResolution;
    const profiles = [[7, 0], [0, 7], [7, 7]].map(([x, z]) =>
      Array.from({ length: 4 }, (_, y) => acceptedAt([x!, y, z!])));
    for (const profile of profiles) {
      assert.deepEqual(profile, [1, 2, 4, 8],
        `MiniDam64 wall/corner profile stopped before B1: ${profile.join("/")}`);
    }
    const occupiedRungs = new Set(initial.bricks.filter((brick) => brick.active)
      .map((brick) => brick.acceptedResolution));
    assert.deepEqual([...occupiedRungs].sort((left, right) => left - right),
      [1, 2, 4, 8]);
    const initialBottom = initial.bricks.filter((brick) => brick.active
      && brick.coordinate[1] === 0);
    assert.equal(initialBottom.length, 64);
    assert.ok(initialBottom.every((brick) => brick.acceptedResolution === 1),
      "the complete deep bottom must construct at B1");

    const cornerHistory: string[] = [];
    const fineBottomHistory: string[] = [];
    for (let step = 1; step <= 72; step += 1) {
      assert.equal(solver.advanceTo(step * scene.numerics.maxDt_s, []), true,
        `MiniDam64 advance ${step}`);
      if (step % 4 !== 0) continue;
      const activity = await solver.readGPUActivityPolicy();
      const corner = activity.bricks.find((brick) => brick.active
        && brick.coordinate[0] === 7 && brick.coordinate[1] === 0
        && brick.coordinate[2] === 7);
      assert.ok(corner, `deep tank corner disappeared on step ${step}`);
      const vertical = Array.from({ length: 5 }, (_, y) => activity.bricks.find(
        (brick) => brick.active && brick.coordinate[0] === 7
          && brick.coordinate[1] === y && brick.coordinate[2] === 7,
      )?.acceptedResolution ?? 0);
      cornerHistory.push(`${step}:${corner.acceptedResolution}`
        + `>${corner.plannedResolution}/${corner.reasons}/p${corner.planReasons}`
        + `/v${vertical.join(",")}`);
      const fineBottom = activity.bricks.filter((brick) => brick.active
        && brick.coordinate[1] === 0 && brick.acceptedResolution > 2);
      if (fineBottom.length > 0) fineBottomHistory.push(`${step}:`
        + fineBottom.map((brick) => `${brick.coordinate[0]},${brick.coordinate[2]}`
          + `=${brick.acceptedResolution}/${brick.reasons}/p${brick.planReasons}`)
          .join(";"));
    }
    const fineCorner = cornerHistory.filter((entry) => {
      const accepted = Number(entry.slice(entry.indexOf(":") + 1, entry.indexOf(">")));
      return accepted > 2;
    });
    assert.deepEqual(fineCorner, [],
      `the deep planar-wall corner must not churn into B4/B8:\n${cornerHistory.join(" ")}`);
    assert.deepEqual(fineBottomHistory, [],
      `the complete deep bottom must remain B2 or coarser:\n${fineBottomHistory.join("\n")}`);
    assert.deepEqual(validationErrors, []);
  } finally {
    solver?.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
});
