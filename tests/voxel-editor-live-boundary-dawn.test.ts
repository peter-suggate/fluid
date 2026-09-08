import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("live voxel insertion and removal change fluid capacity without replacing the world or clock", { timeout: 240000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/voxel-editor-live-boundary-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", (event) => { event.preventDefault(); errors.push(event.error.message); });
    const scene = createMinimalPowerDamBreak32Scene();
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
      adaptiveMassMethod.presetFor("balanced"), undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const dt = 1 / 30;
    const advance = async (time: number) => {
      while (!solver!.advanceTo(time, [])) await new Promise(setImmediate);
      await device!.queue.onSubmittedWorkDone();
    };
    await advance(dt);
    const world = solver.sparseWorld;
    const before = await solver.readDiagnosticFields();
    const activity = await solver.readGPUActivityPolicy();
    const wetEdit = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [4, 0, 4], maximumExclusive: [8, 4, 8], materialId: 2 }]);
    const wetGeneration = world.status().acceptedGeneration;
    const wetTime = solver.info.submittedTime_s;
    await assert.rejects(solver.prepareLiveSolidEdit(wetEdit), /overlaps moving water/i);
    assert.deepEqual((await solver.readDiagnosticFields()).density, before.density);
    assert.equal(world.status().acceptedGeneration, wetGeneration);
    assert.equal(solver.info.submittedTime_s, wetTime);
    // Choose an actually resident dry native cell, so the exact aperture change
    // can be inspected without manufacturing fluid or allocating a new solver.
    let dry: { minimum: [number, number, number]; width: number; index: number } | undefined;
    for (const owner of activity.bricks.filter(brick => brick.active)) {
      const width = 8 * owner.spanBricks / owner.acceptedResolution;
      const extent = 8 * owner.spanBricks;
      for (let z = 0; z < extent && !dry; z += width)
        for (let y = 0; y < extent && !dry; y += width)
          for (let x = 0; x < extent && !dry; x += width) {
            const minimum = [x, y, z].map((value, axis) => value + 8 * owner.coordinate[axis]!) as [number, number, number];
            if (minimum.some((value, axis) => value < 1 || value + width >= [solver!.info.nx, solver!.info.ny, solver!.info.nz][axis]!)
              || minimum[1] < solver.info.ny / 2) continue;
            const index = minimum[0] + solver.info.nx * (minimum[1] + solver.info.ny * minimum[2]);
            if (before.density[index] === 0 && before.solidOpenFraction[index] === 1) dry = { minimum, width, index };
          }
    }
    assert.ok(dry, "fixture must have an open resident dry cell");
    const index = dry.index, expectedOpen = 0;
    const edits = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: dry.minimum,
      maximumExclusive: dry.minimum.map(value => value + dry.width) as [number, number, number], materialId: 2 }]);
    const timeBefore = solver.info.submittedTime_s;
    const started = performance.now();
    solver.validateLiveSolidEdit(edits);
    await solver.prepareLiveSolidEdit(edits);
    solver.applySceneUniforms(edits);
    const editHostMs = performance.now() - started;
    assert.equal(solver.sparseWorld, world);
    assert.equal(solver.info.submittedTime_s, timeBefore);
    const filled = await solver.readDiagnosticFields();
    assert.ok(expectedOpen < 1);
    assert.ok(Math.abs(filled.solidOpenFraction[index]! - expectedOpen) < 1e-6,
      `solid insertion must set exact finite-volume capacity immediately: expected ${expectedOpen}, got ${filled.solidOpenFraction[index]}`);
    await advance(2 * dt);
    assert.ok(solver.info.submittedTime_s! > timeBefore!);
    solver.validateLiveSolidEdit(scene);
    await solver.prepareLiveSolidEdit(scene);
    solver.applySceneUniforms(scene);
    const cleared = await solver.readDiagnosticFields();
    assert.equal(cleared.solidOpenFraction[index], 1,
      "removing the obstacle must restore capacity before another simulation step");
    await advance(3 * dt);
    const final = await solver.readDiagnosticFields();
    assert.ok(final.density.every(Number.isFinite));
    assert.ok(final.pressure.every(Number.isFinite));
    assert.equal(solver.sparseWorld, world);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ editHostMs, finalTime_s: solver.info.submittedTime_s,
      openBefore: before.solidOpenFraction[index], openFilled: filled.solidOpenFraction[index], openCleared: cleared.solidOpenFraction[index] }));
  } finally {
    solver?.destroy();
    if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});

(modulePath ? test : test.skip)("voxel scene presentation accepts repeated solid edits without rebuilding its SVO", { timeout: 240000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "voxel-editor-live-presentation");
  let device: GPUDevice | undefined;
  let display: import("../lib/svo/features/scene-publication/webgpu-live-svo-scene").WebGPULiveSvoScene | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    device.pushErrorScope("validation");
    const { createEmptyScene } = await import("../lib/core/empty-scene");
    const { WebGPULiveSvoScene } = await import("../lib/svo/features/scene-publication/webgpu-live-svo-scene");
    const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .05 });
    display = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {});
    const source = display.sparseVoxelSceneSource;
    for (const operation of ["fill", "clear", "fill"] as const) {
      const edited = sceneWithSolidStroke(scene, [{ operation, minimum: [4, 0, 4], maximumExclusive: [8, 4, 8], materialId: 2 }]);
      display.validateLiveSolidEdit(edited);
      display.stageSceneUpdate(edited);
      const encoder = device.createCommandEncoder();
      display.encodeSceneMaintenance(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      assert.equal(display.sparseVoxelSceneSource, source);
    }
    assert.equal(await device.popErrorScope(), null);
  } finally {
    display?.destroy();
    if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});
