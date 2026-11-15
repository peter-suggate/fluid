/**
 * The brick engine through the app's path (lane E): createBrickSolver on the dam with the app's resolved values and
 * the scene's authored pools, driven by the renderer's own advance ledger (submitNextPreparedGPUAdvance,
 * canQueuePreparedGPUAdvance), one presentation completion a frame and the health receipt after it.
 *
 * Regression: the solver set no `info.submittedTime_s`, so the ledger never counted an advance. Every frame then
 * queued another ~100 ms step with no bound, the app froze, and t stayed 0.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { canQueuePreparedGPUAdvance, submitNextPreparedGPUAdvance, BROWSER_GPU_THROUGHPUT_DEPTH } from "../../core/webgpu-renderer";
import { H } from "./brick-abi";
import { createBrickSolver } from "./brick-method";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };

test("brick solver (GPU): the app's ledger advances the dam one step a frame, bounded, and t retires", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick solver app", async device => {
    const METHOD = "uniform-narrow-band-flip";
    const method = getMethod(METHOD), definition = getSceneDefinition("high-resolution-dam-break"), scene = sceneDocument(definition);
    const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
    const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
    const solver = await createBrickSolver(device, scene, values as Record<string, unknown>, quality);
    try {
      const dt = 1 / 60, frames = 30;
      let pending = 0, accounted = 0, maxPending = 0;
      const retire: Promise<void>[] = [];
      const t0 = performance.now();
      for (let frame = 1; frame <= frames; frame++) {
        // The renderer's prepared clock runs ahead in real time; the ledger alone bounds the queue.
        const clock = frame * 4 * dt;
        if (canQueuePreparedGPUAdvance(pending, BROWSER_GPU_THROUGHPUT_DEPTH)) {
          const { submittedTime } = submitNextPreparedGPUAdvance(solver, clock, []);
          if (submittedTime > accounted) { accounted = submittedTime; pending++; }
        }
        maxPending = Math.max(maxPending, pending);
        // The presentation's completion retires the admitted advances, then the health receipt (no second fence).
        const done = device.queue.onSubmittedWorkDone();
        const at = accounted;
        retire.push(done.then(async () => { await solver.assertSimulationHealthy(done); solver.info.completedTime_s = Math.max(solver.info.completedTime_s ?? 0, at); pending = 0; }));
        if (retire.length >= BROWSER_GPU_THROUGHPUT_DEPTH) await retire.shift();
      }
      await Promise.all(retire);
      const ms = (performance.now() - t0) / frames;
      await solver.awaitFrameCompletion();
      const stats = await solver.readStats();
      const step = stats.brickEngine?.step ?? 0;
      console.log(`brick app: ${frames} frames, ${(stats.submittedTime_s! / dt).toFixed(0)} steps submitted, engine step ${step}, `
        + `completed t ${stats.completedTime_s?.toFixed(4)}, max pending ${maxPending}, ${ms.toFixed(1)} ms/frame (check slot), liveF ${stats.brickEngine?.liveF}`);
      assert.ok(maxPending <= BROWSER_GPU_THROUGHPUT_DEPTH, `the ledger bounds the queue (max pending ${maxPending})`);
      assert.ok((stats.submittedTime_s ?? 0) > 0, "submittedTime_s advances");
      assert.ok((stats.completedTime_s ?? 0) > 0, "completedTime_s retires (t leaves 0)");
      assert.equal(Math.round(stats.submittedTime_s! / solver.brickEngine.options.dt_s), step, "one engine step per accounted advance");
      assert.equal(solver.info.simulationFailure, undefined, "no latched failure");
      const header = await solver.brickEngine.read("set", 0, 4);
      assert.equal(header[H.FAIL], 0, `FAIL ${header[H.FAIL]} info 0x${header[H.FAIL_INFO]!.toString(16)}`);
    } finally { solver.destroy(); }
  });
});
