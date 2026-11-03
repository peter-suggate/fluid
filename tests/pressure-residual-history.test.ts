import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { NBFLIP_LETTERS_SCENE_ID } from "../lib/core/nbflip-paper-scenes";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";
import { mooredMotorBodies } from "../tools/moored-motor-bodies";

const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
// The pressure solve's convergence over frames 31-90: the mean V-cycles and
// the geometric mean of each frame's accepted (final) residual. Paddleboats
// (its moored boats' wheels driven) mostly converges in 3 cycles; the falling
// letters run to the 4-cycle limit, so their final residual is the solve's
// accuracy. Measured here, level 1's residual on the box view gave
// paddleboats 3.15 cycles, geometric mean 0.0286, and letters 0.1035; on the
// neighbour bricks' cores (pDownR) 3.05, 0.0219 and 0.0840. The bounds sit
// between, so a change that loosens either scene's solve fails here. Single
// runs vary about 5% in the geometric mean.
const scenes = [
  { id: "paddleboats", bodies: true, maxCycles: 3.1, maxGeo: 0.025 },
  { id: NBFLIP_LETTERS_SCENE_ID, bodies: false, maxCycles: 4, maxGeo: 0.093 },
];
for (const s of scenes) test(`${s.id}: pressure cycles and final residual over frames 31-90`, gpu, async t => {
  await withUniformDevice(`${s.id} pressure history`, async device => {
    const definition = getSceneDefinition(s.id), scene = structuredClone(sceneDocument(definition)), frames = 90, from = 31;
    scene.duration_s = Math.max(scene.duration_s, frames * scene.numerics.fixedDt_s);
    const solver = await createUniformSolver(device, scene, { ...definition.methodProfile?.overrides });
    const bodies = s.bodies ? mooredMotorBodies(scene, scene.numerics.fixedDt_s) : () => [];
    let cycles = 0, logSum = 0, n = 0;
    try {
      for (let frame = 1; frame <= frames; frame++) {
        const time = frame * scene.numerics.fixedDt_s;
        assert.ok(solver.advanceTo(time, bodies(time)));
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        if (frame < from) continue;
        const residual = solver.info.uniformPressureAcceptedResidual ?? Number.NaN, c = solver.info.uniformPressureCyclesExecuted ?? Number.NaN;
        assert.ok(Number.isFinite(residual) && residual > 0 && Number.isFinite(c), `frame ${frame}: residual ${residual}, cycles ${c}`);
        cycles += c; logSum += Math.log10(residual); n++;
      }
    } finally { solver.destroy(); }
    const meanCycles = cycles / n, geo = 10 ** (logSum / n);
    t.diagnostic(`${s.id}: mean cycles ${meanCycles.toFixed(3)}, geometric mean final ${geo.toPrecision(4)}`);
    assert.ok(meanCycles <= s.maxCycles, `${s.id}: ${meanCycles.toFixed(3)} mean cycles (at most ${s.maxCycles})`);
    assert.ok(geo <= s.maxGeo, `${s.id}: geometric mean final ${geo.toPrecision(4)} (at most ${s.maxGeo})`);
  });
});
