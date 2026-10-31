import assert from "node:assert/strict";
import test from "node:test";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { sceneFluidBoxPage, sceneFluidLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { UNIFORM_PAGE_POOL_EDGE, uniformPagePoolResidency } from "../lib/methods/uniform/uniform-page-pool";
import { uniformLatticePagedVolume } from "../lib/methods/uniform/uniform-volume-initial";
import { uniformOpenTile, uniformPageSources } from "../lib/methods/uniform/webgpu-uniform-reference";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* The pool rule counts what commit 1 asks for (docs/plans/bulk16-plan-2026-10-28.md,
 * "the pool rule and the stage footprints"). The backyard's basin wall at z = 8 m
 * lies on a page face (cell 160), so the water's last tile row is its page's last.
 * The page request (lrAsk) asks for every page a wet tile's stage footprints reach,
 * solid or not: the six wholly solid deck pages (2..7, 0, 5). Commit 1 claims them
 * before it retires anything, so the rule must store them, or a pool sized by the
 * rule (no slack here: v = 0, no reserve, no BULK16) is exhausted at commit 1. */
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };

test("the rule's stored set holds the wall pages the wet tiles' stage footprints reach", () => {
  const scene = createBackyardPoolPartyScene(), cells = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  const { volume } = uniformLatticePagedVolume(scene, UNIFORM_PAGE_POOL_EDGE);
  const { air } = uniformPagePoolResidency(cells, scene.fluid.pagePool, uniformPageSources(scene, undefined, 1 / 60), true, () => volume, sceneFluidBoxPage(scene), uniformOpenTile(scene));
  const absent = new Set(air.map(p => p.join(",")));
  for (let x = 2; x <= 7; x++) assert.ok(!absent.has(`${x},0,5`), `deck page (${x}, 0, 5) beside the water's last tile row is stored`);
  // The deck pages no wet tile's footprint reaches stay AIR (one page past the wall, and under the decks).
  for (const page of ["2,0,6", "0,0,3", "9,0,3", "2,0,1"]) assert.ok(absent.has(page), `page (${page}) stays AIR`);
});

test("the backyard pool sized by the rule runs its first commits", gpu, async () => {
  await withUniformDevice("pool by rule", async device => {
    const scene = createBackyardPoolPartyScene();
    const bodies = initializeRigidBodies(scene.rigidBodies), solver = await createUniformSolver(device, scene);
    try {
      for (let step = 1; step <= 2; step++) {
        assert.ok(solver.advanceTo(step / 60, bodies)); await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined, `frame ${step}`);
      }
    } finally { solver.destroy(); }
  });
});
