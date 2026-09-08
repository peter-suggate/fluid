import { readFileSync } from "node:fs";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { parseScene } from "../lib/core/model";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const wetOverlapRejection = "Solid insertion overlaps moving water. Remove water from this space or place the solid in an empty area.";
(modulePath ? test : test.skip)("retained density stays nonnegative through the recorded live solid edit sequence", { timeout: 240000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/retained-density-live-editor-sequence-dawn.test.ts");
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
    const authored = parseScene(readFileSync(new URL("./fixtures/retained-density-live-editor-sequence.json", import.meta.url), "utf8"));
    let scene = { ...authored, solidVoxels: authored.solidVoxels.slice(0, 99) };
    // Reproduce the recorded stroke append ranges [99,106), [106,107), and
    // [107,149) at frame clocks 116, 176, and 432 respectively. The current
    // public contract rejects insertion into moving water: replay each exact
    // batch as an attempt, without moving it or reappending rejected strokes.
    const edits = new Map<number, readonly [number, number]>([
      [116, [99, 106]], [176, [106, 107]], [432, [107, 149]],
    ]);
    const outcomes: { frame: number; patchRange: readonly [number, number]; accepted: boolean }[] = [];
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
      { ...adaptiveMassMethod.presetFor("balanced"), selectorMode: "coarse-first", timeStep: "paper", brickFineResolution: "8" }, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const dt = 1 / 30;
    const advance = async (time: number) => {
      while (!solver!.advanceTo(time, [])) await new Promise(setImmediate);
      await device!.queue.onSubmittedWorkDone();
    };
    await advance(dt);
    const world = solver.sparseWorld;
    for (let frame = 2; frame <= 450; frame++) {
      await advance(frame * dt);
      // Include runtime leaves and select the GPU-accepted scalar bank. The
      // world-growth receipt's max(A,B) envelope is not a conserved mass oracle.
      const before = await solver.readDiagnosticFields(true);
      const patchRange = edits.get(frame);
      if (patchRange) {
        const proposal = sceneWithSolidStroke(scene,
          authored.solidVoxels.slice(patchRange[0], patchRange[1]));
        const timeBefore = solver.info.submittedTime_s;
        const generationBefore = world.status().acceptedGeneration;
        solver.validateLiveSolidEdit(proposal);
        let accepted = true;
        let preflightCommitted: boolean | undefined;
        try {
          preflightCommitted = await solver.prepareLiveSolidEdit(proposal);
        } catch (error) {
          if (!(error instanceof Error) || error.message !== wetOverlapRejection) throw error;
          accepted = false;
        }
        if (accepted) {
          // A false result is also admitted: no new closure required an atomic
          // GPU commit. Both admitted paths still adopt scene uniforms.
          solver.applySceneUniforms(proposal);
          scene = proposal;
        }
        const after = await solver.readDiagnosticFields(true);
        assert.deepEqual(after.density, before.density,
          `frame ${frame}: a dry edit or rejected wet edit must preserve accepted liquid`);
        assert.equal(solver.info.submittedTime_s, timeBefore,
          `frame ${frame}: edit acceptance must preserve the simulation clock`);
        assert.equal(solver.sparseWorld, world);
        if (!accepted) {
          assert.deepEqual(after.solidOpenFraction, before.solidOpenFraction,
            `frame ${frame}: wet rejection must preserve solid geometry`);
          assert.equal(world.status().acceptedGeneration, generationBefore,
            `frame ${frame}: wet rejection must preserve the accepted generation`);
        }
        outcomes.push({ frame, patchRange, accepted });
        console.log(JSON.stringify({ editAtFrame: frame, patchRange,
          outcome: accepted ? "accepted" : "rejected-wet-overlap", preflightCommitted,
          patchCount: scene.solidVoxels.length, time: solver.info.submittedTime_s }));
      }
      if (frame % 30 === 0 || patchRange) console.log(JSON.stringify({ frame, time: solver.info.submittedTime_s }));
    }
    assert.deepEqual(outcomes.map(({ frame, patchRange }) => [frame, patchRange]), [...edits],
      "every recorded batch must be attempted at its original frame clock");
    const final = await solver.readDiagnosticFields(true);
    assert.ok(final.density.every(Number.isFinite));
    assert.ok(final.pressure.every(Number.isFinite));
    assert.equal(solver.sparseWorld, world);
    assert.deepEqual(errors, []);
    assert.ok(solver.info.submittedTime_s! >= 15 - 1e-8);
  } finally {
    solver?.destroy();
    if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});
