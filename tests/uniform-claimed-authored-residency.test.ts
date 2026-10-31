import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { advanceUniform, createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* Item 2 (docs/plans/bulk16-plan-2026-10-28.md): the seep scene's rest
 * variant (the throat filled, the top open) failed its first frame at every
 * odd page count (57, 59, 61, 63 pages at 1 cm) and held at every even one.
 * At an odd count the middle wall's low face lies on a page boundary, so the
 * wall and the throat pin's far side (minimum-cell-size 1 to high + 4 cells)
 * share a page the liquid does not reach at build. The first commit claims
 * that page (the wall's region is within the liquid's reach), the layout
 * gives the pin's tiles their authored width 1, and an h tile is required to
 * be resident, but pageMark marks only from census records, which a claimed
 * block does not have yet: residency audit 1, absent near. Both widths run
 * a few steps; the even one is the control. */
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
const STEPS = 4;

function restScene(widthScale: number) {
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale }) }, { cellSize_m: 0.01 });
  const s = scene as any;
  // The rest gate (tools/probe-reservoir-bulk.ts --closed): the throat filled, the passage's wetted start dropped.
  s.solidVoxels = s.solidVoxels.filter((v: { operation: string }) => v.operation === "fill");
  s.fluid.initialLiquidVolumes = s.fluid.initialLiquidVolumes.slice(0, 1);
  delete s.fluid.pagePool;
  s.fluid.dynamicViscosity_Pa_s = 0; s.fluid.surfaceTension_N_m = 0;
  return { scene, values: { ...base.methodProfile!.overrides, bulkTier: "16h", volumeControlSeconds: 0 } as never };
}

for (const [pages, widthScale] of [[8, 1], [9, 1.125]] as const) {
  test(`a rest box ${pages} pages long: the throat pin's tiles in a page the first commit claims are resident when the layout makes them h`, gpu, async () => {
    const { scene, values } = restScene(widthScale);
    const nx = Math.round(scene.container.width_m / 0.01), high = 16 * Math.floor(nx / 32);
    assert.equal(nx, 32 * pages);
    assert.equal((high - 16) % 32 === 0, pages % 2 === 1, "an odd count puts the wall's low face on a page boundary");
    await withUniformDevice(`rest ${pages} pages`, async device => {
      const solver: any = await createUniformSolver(device, scene, values);
      try {
        for (let step = 1; step <= STEPS; step++) await advanceUniform(solver, step / 60);
        const stats = await solver.readStats();
        console.log(`# ${pages} pages: drift ${stats.volumeDrift}, max speed ${stats.maxSpeed_m_s}`);
        assert.ok(Math.abs(stats.volumeDrift!) < 1e-3, `drift ${stats.volumeDrift}`);
      } finally { solver.destroy(); }
    });
  });
}
