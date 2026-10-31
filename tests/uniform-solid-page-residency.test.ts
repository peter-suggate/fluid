import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { uniformBlockStored, uniformPageWord } from "../lib/methods/uniform/uniform-lattice-address";
import { advanceUniform, createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* V2d (docs/plans/bulk16-plan-2026-10-28.md): the closure margin through
 * open cells. A page the liquid cannot reach is not stored at build, but a
 * page wholly solid beside the liquid is its wall: the question is whether
 * the wall's faces need it stored. A 4x2x1-page closed tank at 5 mm: liquid
 * in page column x = 0, 40 cells deep; page column x = 1 wholly solid; x = 2
 * and 3 open and dry, past the wall. Two authored stored sets: the wall's
 * page stored, and not. At rest both must hold the liquid still and whole. */
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
const STEPS = 120;

function sceneWith(air: readonly (readonly [number, number, number])[]) {
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale: 4 / 16, heightScale: 2 / 8, depthScale: 1 / 2 }) }, { cellSize_m: 0.005 });
  const s = scene as any, w = s.container.width_m, d = s.container.depth_m;
  s.solidVoxels = [{ operation: "fill", minimum: [32, 0, 0], maximumExclusive: [64, 64, 32], materialId: 2 }];
  s.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -w / 2, y: 0, z: -d / 2 }, max_m: { x: -w / 2 + 32 * 0.005, y: 40 * 0.005, z: d / 2 } }];
  s.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  s.fluid.pagePool = { blocks: 10, air };
  s.fluid.refinementRegions = [];
  s.fluid.dynamicViscosity_Pa_s = 0; s.fluid.surfaceTension_N_m = 0;
  return { scene, values: { ...base.methodProfile!.overrides, volumeControlSeconds: 0 } as never };
}

async function rest(label: string, air: readonly (readonly [number, number, number])[]) {
  const { scene, values } = sceneWith(air);
  let out: { drift: number; speed: number; wallStored: boolean; dryStored: boolean } | undefined;
  await withUniformDevice(label, async device => {
    const solver: any = await createUniformSolver(device, scene, values);
    try {
      const a = solver.address, stored = (x: number, y: number) => uniformBlockStored(a, uniformPageWord(a, x, y, 0));
      let speed = 0;
      for (let step = 1; step <= STEPS; step++) {
        await advanceUniform(solver, step / 60);
        if (step % 30 === 0) speed = Math.max(speed, (await solver.readStats()).maxSpeed_m_s);
      }
      const stats = await solver.readStats();
      out = { drift: stats.volumeDrift!, speed, wallStored: stored(1, 0), dryStored: stored(2, 0) || stored(3, 0) };
      console.log(`# ${label}: ${JSON.stringify(out)}`);
    } finally { solver.destroy(); }
  });
  return out!;
}

const dry = [[2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0]] as const;
test("a wholly solid page beside the liquid: the rest holds with the wall's page stored, and without it", gpu, async () => {
  const withWall = await rest("wall page stored", dry);
  const without = await rest("wall page AIR", [[1, 0, 0], [1, 1, 0], ...dry]);
  assert.ok(withWall.wallStored && !withWall.dryStored && !without.wallStored && !without.dryStored, "the authored stored sets");
  assert.ok(Math.abs(withWall.drift) < 1e-3 && withWall.speed < 1e-3, `wall stored: ${JSON.stringify(withWall)}`);
  assert.ok(Math.abs(without.drift) < 1e-3 && without.speed < 1e-3, `wall page AIR: ${JSON.stringify(without)}`);
});
