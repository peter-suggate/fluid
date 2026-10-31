import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { uniformBlockStored, uniformPageWord } from "../lib/methods/uniform/uniform-lattice-address";
import { advanceUniform, createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* V2d (docs/plans/bulk16-plan-2026-10-28.md): the closure margin through
 * open cells. A page the liquid cannot reach is not stored at build. A page
 * wholly solid beside the liquid is its wall: the page request (lrAsk) asks
 * for every page a wet tile's stage footprints reach, solid or not, and
 * commit 1 claims before it retires, so the pool rule stores the wall's page
 * (uniformPagePoolReachablePages) and holds no block for the dry pages past
 * it. A 4x2x1-page closed tank at 5 mm: liquid in page column x = 0, 40
 * cells deep; page column x = 1 wholly solid; x = 2 and 3 open and dry, past
 * the wall. The pool is the rule's, with no slack (a closed top caps the
 * reserve, no BULK16), and the GPU table is read, not the build's address. */
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
const STEPS = 120;

function sceneWith() {
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale: 4 / 16, heightScale: 2 / 8, depthScale: 1 / 2 }) }, { cellSize_m: 0.005 });
  const s = scene as any, w = s.container.width_m, d = s.container.depth_m;
  s.solidVoxels = [{ operation: "fill", minimum: [32, 0, 0], maximumExclusive: [64, 64, 32], materialId: 2 }];
  s.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -w / 2, y: 0, z: -d / 2 }, max_m: { x: -w / 2 + 32 * 0.005, y: 40 * 0.005, z: d / 2 } }];
  s.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  // The rule's pool, with no slack: a closed top caps the ballistic reserve at the box's top page.
  delete s.fluid.pagePool; s.container = { ...s.container, top: "closed" };
  s.fluid.refinementRegions = [];
  s.fluid.dynamicViscosity_Pa_s = 0; s.fluid.surfaceTension_N_m = 0;
  return { scene, values: { ...base.methodProfile!.overrides, volumeControlSeconds: 0 } as never };
}

/** The pool's resident pages, as "x,y,z" in the box: its GPU block records. */
async function residentPages(device: GPUDevice, solver: any): Promise<Set<string>> {
  const pool = solver.mixedFrame.pagePool, l = pool.layout;
  const read = device.createBuffer({ size: 4 * l.state.words, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(pool.state, 0, read, 0, 4 * l.state.words); device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ); const state = new Int32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy();
  const pages = new Set<string>();
  for (let b = 1; b < l.blocks - 1; b++) {
    if (state[l.state.blocks + b] === -1) continue;
    const r = l.state.records + 8 * b;
    pages.add([0, 1, 2].map(a => state[r + a]! - l.box.origin[a]!).join(","));
  }
  return pages;
}

test("a wholly solid page beside the liquid: the rule stores it for the request, the dry pages past it stay AIR, and the rest holds", gpu, async () => {
  const { scene, values } = sceneWith();
  await withUniformDevice("wall page by rule", async device => {
    const solver: any = await createUniformSolver(device, scene, values);
    try {
      const a = solver.address, stored = (x: number, y: number) => uniformBlockStored(a, uniformPageWord(a, x, y, 0));
      assert.ok(stored(1, 0) && stored(1, 1) && ![2, 3].some(x => stored(x, 0) || stored(x, 1)), "the rule stores the wall's pages and not the dry ones");
      let held = 0; for (let y = 0; y < 2; y++) for (let x = 0; x < 4; x++) if (stored(x, y)) held++;
      assert.equal(a.blocks - 2, held, "the rule's pool holds no slot past the pages stored at build");
      let speed = 0;
      for (let step = 1; step <= STEPS; step++) {
        await advanceUniform(solver, step / 60);
        if (step % 30 === 0) speed = Math.max(speed, (await solver.readStats()).maxSpeed_m_s);
      }
      const stats = await solver.readStats(), resident = await residentPages(device, solver);
      console.log(`# by rule: drift ${stats.volumeDrift}, speed ${speed}, resident ${[...resident].sort().join(" ")}`);
      assert.ok(resident.has("1,0,0") && resident.has("1,1,0"), "the request keeps the wall's pages");
      assert.ok(![...resident].some(p => p.startsWith("2,") || p.startsWith("3,")), "no dry page past the wall is claimed");
      assert.ok(Math.abs(stats.volumeDrift!) < 1e-3 && speed < 1e-3, `rest: drift ${stats.volumeDrift}, speed ${speed}`);
    } finally { solver.destroy(); }
  });
});
