import assert from "node:assert/strict";
import test from "node:test";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import type { SceneDescription } from "../lib/core/model";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// The coarse start's volume (docs/plans/bulk16-plan-2026-10-28.md, M4b): the
// impact scene (a solid-walled basin, a ball at rest above it) lost 56-82% of
// its liquid in its first step under {start: "4h"} and none under the h start.
// The surface advection kept 4h tile corners (far air, deep air, buried in a
// solid) by a plain load, which beside an h texture reads the h texture: at a
// block claimed this step that is the slot init's empty value (the domain's
// far phi, 96 h), not the corner's stored value in the base block. One step
// under each start: the same liquid.
function impactScene(start: "h" | "4h"): SceneDescription {
  const scene = createBackyardPoolPartyScene();
  scene.interaction = undefined;
  scene.solidBoundary = undefined;
  scene.container = { ...scene.container, width_m: 3.2, height_m: 4.8, depth_m: 3.2 };
  scene.solidVoxels = ([[[0, 0, 0], [64, 4, 64]], [[0, 0, 0], [8, 40, 64]], [[56, 0, 0], [64, 40, 64]], [[8, 0, 0], [56, 40, 8]], [[8, 0, 56], [56, 40, 64]]] as const)
    .map(([minimum, maximumExclusive]) => ({ operation: "fill", minimum: [...minimum], maximumExclusive: [...maximumExclusive], materialId: 1 })) as unknown as SceneDescription["solidVoxels"];
  scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -1.2, y: 0.2, z: -1.2 }, max_m: { x: 1.2, y: 1.8, z: 1.2 } }];
  Object.assign(scene.rigidBodies[0]!, { position_m: { x: -0.8, y: 2.2, z: -0.8 }, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
  scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 };
  scene.fluid.tiers = { ...scene.fluid.tiers, start };
  scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks: 64 };
  return scene;
}

test("a coarse start keeps the h start's liquid through the first step (impact, ball at rest)", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const drift: Record<string, number> = {};
  for (const start of ["h", "4h"] as const) {
    await withUniformDevice(`coarse start volume ${start}`, async device => {
      const scene = impactScene(start), solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
      try {
        const bodies = initializeRigidBodies(scene.rigidBodies);
        // The narrow band's volume is measured on a step that asks for it.
        (solver as unknown as { mixedFrame?: { requestNarrowBandVolumeProbe(): void } }).mixedFrame?.requestNarrowBandVolumeProbe();
        assert.ok(solver.advanceTo(1 / 60, bodies));
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        drift[start] = (await solver.readStats()).volumeDrift!;
      } finally { solver.destroy(); }
    });
  }
  assert.ok(Number.isFinite(drift.h) && Math.abs(drift.h!) < 0.05, `h start drift ${drift.h}`);
  assert.ok(Math.abs(drift["4h"]! - drift.h!) < 0.01, `step-1 drift: 4h ${drift["4h"]} against h ${drift.h}`);
});
