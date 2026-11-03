import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { NBFLIP_LETTERS_SCENE_ID } from "../lib/core/nbflip-paper-scenes";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
// The falling letters keep their volume through the fall and the splash. Their
// corners and strokes are the surface shape 4h loses (Sato et al. 2018, Eq. 4:
// geometric heat "acts to retain sharp corners during the liquid's free-fall").
// While the shape criterion refines and heats them the scene ends 120 steps
// (5 s) at -1.07% to -1.12% (three runs). Without the shape criterion
// (detailShape "off") it ends at -1.51%, and with shape refining h but heating
// no particles at -1.51% as well. (The -4.21% of the pre-2026-10-28 body gate,
// one-pass plan §9, predates the level-set curvature term and is not
// reproduced on this code.) A change to the shape measure or to the samples'
// heat must keep the letters above -1.3%, between the two.
const LETTERS_DRIFT_FLOOR = -0.013;
test("the falling letters keep their volume through 120 steps", gpu, async t => {
  await withUniformDevice("falling letters volume", async device => {
    const definition = getSceneDefinition(NBFLIP_LETTERS_SCENE_ID), scene = structuredClone(sceneDocument(definition)), steps = 120;
    scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
    const solver = await createUniformSolver(device, scene, { ...definition.methodProfile?.overrides });
    const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
    try {
      for (let step = 1; step <= steps; step++) {
        if (step % 30 === 0) frame.requestNarrowBandVolumeProbe();
        assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, []));
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        if (step % 30 !== 0) continue;
        const volumeDrift = (await solver.readStats()).volumeDrift ?? Number.NaN;
        assert.ok(Number.isFinite(volumeDrift), `step ${step}: drift ${volumeDrift}`);
        t.diagnostic(`step ${step}: drift ${(100 * volumeDrift).toFixed(3)}%, h tiles ${solver.info.uniformMixedFineTiles}`);
        if (step === steps) assert.ok(volumeDrift > LETTERS_DRIFT_FLOOR, `step ${step}: the letters lost ${(-100 * volumeDrift).toFixed(2)}% (the shape criterion keeps about 1.1%, none 1.5%)`);
      }
    } finally { solver.destroy(); }
  });
});
