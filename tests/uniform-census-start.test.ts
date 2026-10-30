import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { NBFLIP_STAIR_CUBE_ID } from "../lib/core/nbflip-stair-cube-scene";
import { getScenePreset } from "../lib/core/scenes";
import { createUniformTroughScene } from "../lib/core/uniform-trough-scenes";
import { interactionScene } from "../lib/interaction/adapters/scene";
import { SceneInteractionHost } from "../lib/interaction/runtime/host";
import type { SceneDescription } from "../lib/core/model";
import { advanceUniform, createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// The census start (fluid.tiers.start "census", docs/plans/bulk16-plan-2026-10-28.md, V2a):
// t=0 holds h where frame 1's census would, on a 4h background, so a scene keeps the h start's
// liquid. The gate is the volume after two steps within 1% of the h start's. The scenes are the
// two that lost liquid under the 4h start: the sealed rigid vessel (5.3% short) and the trough
// dam-break's thin bath walls (+5.2% at step 1).
type Start = "h" | "census";
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };

type Result = { drift: number; fine: number };
async function stairCube(start: Start): Promise<Result> {
  let result: Result | undefined;
  await withUniformDevice(`census start vessel ${start}`, async device => {
    const scene = getScenePreset(NBFLIP_STAIR_CUBE_ID).create();
    scene.fluid.tiers = { ...scene.fluid.tiers, start };
    const solver = await createUniformSolver(device, interactionScene(scene), { rigidCoupling: "on" });
    const host = new SceneInteractionHost(scene, { fluid: { step: async request => solver.advanceExternalRigidStep(request) } });
    try {
      await host.ready;
      const fine = solver.info.uniformMixedFineTiles!;
      for (let step = 1; step <= 2; step++) await host.advance(1 / 60);
      assert.equal(solver.info.simulationPipelineError, undefined);
      const stats = await solver.readStats();
      assert.equal(stats.volumeTelemetrySource, "narrow-band-phi");
      result = { drift: stats.volumeDrift!, fine };
    } finally { solver.destroy(); }
  });
  return result!;
}

async function troughDam(start: Start): Promise<Result> {
  let result: Result | undefined;
  await withUniformDevice(`census start dam ${start}`, async device => {
    const scene: SceneDescription = createUniformTroughScene("dam-break");
    scene.fluid.tiers = { ...scene.fluid.tiers, start };
    const solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    try {
      const fine = solver.info.uniformMixedFineTiles!;
      for (let step = 1; step <= 2; step++) await advanceUniform(solver, step / 60);
      assert.equal(solver.info.simulationPipelineError, undefined);
      result = { drift: (await solver.readStats()).volumeDrift!, fine };
    } finally { solver.destroy(); }
  });
  return result!;
}

for (const [name, run] of [["the sealed rigid vessel (stair cube)", stairCube], ["the trough dam-break", troughDam]] as const) {
  test(`a census start keeps the h start's liquid after two steps: ${name}`, gpu, async () => {
    const h = await run("h"), census = await run("census");
    console.log(JSON.stringify({ scene: name, h, census }));
    assert.ok(Number.isFinite(h.drift) && Math.abs(h.drift) < 0.02, `h start drift ${h.drift}`);
    assert.ok(Math.abs(census.drift - h.drift) < 0.01, `drift after two steps: census ${census.drift} against h ${h.drift}`);
    assert.ok(census.fine < h.fine, `the census start's t=0 h tiles (${census.fine}) are fewer than the h start's (${h.fine})`);
  });
}
