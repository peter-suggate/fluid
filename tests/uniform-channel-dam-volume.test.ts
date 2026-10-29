import assert from "node:assert/strict";
import test from "node:test";
import { createUniformTroughScene } from "../lib/core/uniform-trough-scenes";
import type { SceneDescription } from "../lib/core/model";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// The dam's volume gain (docs/plans/bulk16-plan-2026-10-28.md, M4b): the
// trough dam-break gained +8.9 to +10.9% by step 120 under the h start, all
// of it in the surface reconstruct (Ferstl et al. 2016, Eq. 4: the advected
// level set united with the samples' surface), which adds back more than the
// advection loses. No solid is needed: a 0.6 x 0.8 m column released down a
// 2.5 m channel 0.55 m wide, bounded by the domain's own walls, gains +4.35%
// in 60 steps (+1.2% at 40), where the same liquid spreading in the open
// 3.2 x 1.2 m domain holds +0.3%. Inviscid, no tension, no volume control,
// one step a frame at 1/60 s.
function channelDam(): SceneDescription {
  const scene = createUniformTroughScene("dam-break");
  scene.container = { ...scene.container, width_m: 2.5, depth_m: 0.55 };
  scene.solidVoxels = [];
  scene.solidSurface = undefined;
  scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -1.25, y: 0, z: -0.275 }, max_m: { x: -0.65, y: 0.8, z: 0.275 } }];
  scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 };
  scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks: 64 };
  return scene;
}

// Expected to fail until the M4b merge item closes: with separation's normal
// part removed at the free surface the channel still gains +2.5 to +3.1% by
// step 60 (+4.1% before), a source not yet found (docs/plans/bulk16-plan-2026-10-28.md).
test("a dam released down a channel keeps its volume through the run-up (60 steps)", { skip: !process.env.WEBGPU_NODE_MODULE, todo: "M4b merge item: +2.5 to +3.1% at step 60 after the surface shift" }, async () => {
  await withUniformDevice("channel dam volume", async device => {
    const scene = channelDam(), solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    try {
      const drift: number[] = [];
      for (let step = 1; step <= 60; step++) {
        if (step % 20 === 0) (solver as unknown as { mixedFrame?: { requestNarrowBandVolumeProbe(): void } }).mixedFrame?.requestNarrowBandVolumeProbe();
        assert.ok(solver.advanceTo(step / 60, []));
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        if (step % 20 === 0) drift.push((await solver.readStats()).volumeDrift!);
      }
      assert.ok(drift.every(d => Number.isFinite(d) && Math.abs(d) < 0.01), `drift at steps 20, 40, 60: ${drift.map(d => `${(100 * d).toFixed(2)}%`).join(", ")}`);
    } finally { solver.destroy(); }
  });
});
