import assert from "node:assert/strict";
import test from "node:test";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import type { InitialLiquidVolume, SceneDescription } from "../lib/core/model";
import { createUniformSolver, readUniformFields, readUniformLiveAddress, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedTexture } from "./helpers/uniform-mixed-native-fields";

// 4h liquid in free fall (docs/plans/bulk16-plan-2026-10-28.md, M4b): a drop
// with no solid near it, every tile at 4h, must fall at g. The tiles are held
// at 4h by a Coarse region over the box (Dynamic holds a Coarse region at 4h
// but for solid contact), on the default (h) start: the census can't refine
// the drop, so the test keeps testing 4h whatever the start's seed becomes. Under {start: "4h"} pool-rest's drop (r 0.2 m)
// held a max speed of g dt, 0.163 m/s, from step 5 to step 10: each step's
// velocity reached the next step as zero. The liquid's speed against g t, a
// few percent; volume is not asserted (a 4h owner resolves a
// drop one tile across only coarsely, a separate resolution loss).
// A 3.2 m box (x, z centred, floor at y 0; h 0.05 m) with no solids, every
// tile held at 4h, the given liquid.
function all4hScene(liquid: InitialLiquidVolume[]): SceneDescription {
  const scene = createBackyardPoolPartyScene();
  scene.container = { ...scene.container, width_m: 3.2, height_m: 3.2, depth_m: 3.2 };
  scene.solidBoundary = undefined; scene.rigidBodies = []; scene.solidVoxels = [];
  scene.fluid.initialLiquidVolumes = liquid;
  scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 };
  scene.fluid.refinementRegions = [{ id: "all-4h", min_m: { x: -10, y: -10, z: -10 }, max_m: { x: 10, y: 10, z: 10 }, rule: "minimum-cell-size", minimumCellSize_cells: 4, maximumCellSize_cells: 4 }];
  return scene;
}
async function advance(solver: Awaited<ReturnType<typeof createUniformSolver>>, step: number) {
  assert.ok(solver.advanceTo(step / 60, []));
  await solver.awaitFrameCompletion();
  assert.equal(solver.info.simulationPipelineError, undefined);
}

// Every speed here is a 4h owner's vertical velocity, the mean of its two y
// faces (a face's value is packed at the texel 3 above its owner's origin),
// weighted by the owner's V: the mean over owners at or above tile row
// `rows`. The solver's speeds lead g t by g dt/2 from the first step (a
// resolved r 0.3 m drop reads 1.5, 2.5, 3.5 g dt), so the free fall is
// asserted on the increments: each step adds g dt.
async function meanVy(device: GPUDevice, solver: Awaited<ReturnType<typeof createUniformSolver>>, rows = 0) {
  const fields = await readUniformFields(device, solver), v = await readMixedTexture(device, solver.velocityTexture, await readUniformLiveAddress(device, solver));
  const { nx, ny, nz } = solver.info, face = (x: number, y: number, z: number) => y < 0 ? 0 : v[4 * (x + nx * (y + ny * z)) + 1]!;
  let mass = 0, momentum = 0;
  for (let z = 0; z < nz; z += 4) for (let y = 4 * rows; y < ny; y += 4) for (let x = 0; x < nx; x += 4) {
    const volume = fields.density[x + nx * (y + ny * z)]!;
    if (!(volume > 0)) continue;
    assert.equal(fields.widthAt(x, y, z), 4);
    mass += volume; momentum += volume * 0.5 * (face(x, y + 3, z) + face(x, y - 1, z));
  }
  return { mass, velocity: momentum / mass };
}

test("4h liquid in free fall accelerates at g", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("coarse free fall", async device => {
    const scene = all4hScene([{ shape: "sphere", center_m: { x: 0, y: 2.2, z: 0 }, radius_m: 0.2 }]);
    const solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    try {
      const g = scene.fluid.gravity_m_s2.y, velocity: number[] = [];
      for (let step = 1; step <= 8; step++) { await advance(solver, step); velocity.push((await meanVy(device, solver)).velocity); }
      assert.equal(solver.info.uniformMixedFineTiles ?? 0, 0, "the drop stays at 4h (the case under test)");
      for (let k = 1; k < velocity.length; k++) assert.ok(Math.abs(velocity[k]! - velocity[k - 1]! - g / 60) < 0.05 * Math.abs(g) / 60, `step ${k + 1}: ${velocity[k]} m/s after ${velocity[k - 1]} (g dt ${g / 60})`);
    } finally { solver.destroy(); }
  });
});

// Free flight ends where the drop meets a row: from then on the row's
// pressure and the extension carry it. A drop falls onto a 4h pool (y 0 to
// 0.6) at g; from contact the pool's pressure decelerates it, but no step
// loses more than half its momentum (a stall loses all of it, to g dt) or
// gains more than g dt (gravity twice). The volume does not jump at contact:
// no step's change exceeds the largest change of the fall.
async function dropMerge(radius: number, centre: number) {
  await withUniformDevice("coarse drop merge", async device => {
    const scene = all4hScene([{ shape: "box", min_m: { x: -1.6, y: 0, z: -1.6 }, max_m: { x: 1.6, y: 0.6, z: 1.6 } },
      { shape: "sphere", center_m: { x: 0, y: centre, z: 0 }, radius_m: radius }]);
    const solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    try {
      const g = Math.abs(scene.fluid.gravity_m_s2.y), dv = g / 60, contact = Math.floor(Math.sqrt(2 * (centre - radius - 0.6) / g) * 60);
      const drop: number[] = [], volume: number[] = [(await solver.readStats()).volumeCellSum!];
      for (let step = 1; step <= contact + 6; step++) {
        await advance(solver, step);
        // The drop: every owner above the pool's top row.
        drop.push((await meanVy(device, solver, 3)).velocity); volume.push((await solver.readStats()).volumeCellSum!);
      }
      assert.equal(solver.info.uniformMixedFineTiles ?? 0, 0, "every tile stays at 4h");
      for (let k = 1; k < contact; k++) assert.ok(Math.abs(drop[k]! - drop[k - 1]! + dv) < 0.05 * dv, `step ${k + 1}: ${drop[k]} m/s after ${drop[k - 1]} (free fall)`);
      for (let k = contact; k < drop.length; k++)
        assert.ok(drop[k]! <= 0.5 * drop[k - 1]! && drop[k]! >= drop[k - 1]! - 1.05 * dv, `step ${k + 1}: ${drop[k]} m/s after ${drop[k - 1]} (contact)`);
      const change = volume.slice(1).map((v, i) => Math.abs(v - volume[i]!)), fall = Math.max(...change.slice(0, contact - 1));
      for (let k = contact; k < change.length; k++) assert.ok(change[k]! <= Math.max(fall, 1), `step ${k + 1}: volume changed ${change[k]} cells against the fall's ${fall}`);
    } finally { solver.destroy(); }
  });
}

// A resolved drop (r 0.5 m, 2.5 owners across), 0.5 m above the pool.
test("a resolved 4h drop hands over to the liquid it falls onto", { skip: !process.env.WEBGPU_NODE_MODULE,
  todo: "O6: the drop erodes in free flight, 3847 to 2621 cells by contact (step 19); its mean's increments miss g dt by up to 33% (step 11) and the volume jumps +316 cells at contact" },
  () => dropMerge(0.5, 1.6));

// A sub-resolution drop (r 0.2 m, one owner across), 0.6 m above the pool.
test("a sub-resolution 4h drop hands over to the liquid it falls onto", { skip: !process.env.WEBGPU_NODE_MODULE,
  todo: "O6: the drop erodes in free flight, 132 to 3 cells in 10 steps, before it reaches the pool" },
  () => dropMerge(0.2, 1.4));

// A film thinner than half a 4h owner rests on the floor: no owner centre is
// liquid, so the film is all detached mass, but it rests on the floor and so
// is not in free flight (the support exclusion). Over 60 steps it stays at
// rest: no step's speed above g dt's noise, and its volume does not drift.
async function filmRest(check: (initial: number, stats: Awaited<ReturnType<Awaited<ReturnType<typeof createUniformSolver>>["readStats"]>>) => void) {
  await withUniformDevice("coarse film rest", async device => {
    const scene = all4hScene([{ shape: "box", min_m: { x: -0.6, y: 0, z: -0.6 }, max_m: { x: 0.6, y: 0.08, z: 0.6 } }]);
    const solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    try {
      // Stats are a step behind their request: read every step, so the last read is current.
      const initial = (await solver.readStats()).volumeCellSum!;
      await solver.readStats();
      for (let step = 1; step <= 60; step++) {
        await advance(solver, step);
        const stats = await solver.readStats();
        assert.equal(stats.uniformMixedFineTiles ?? 0, 0, "the film stays at 4h");
        check(initial, stats);
      }
    } finally { solver.destroy(); }
  });
}

test("a 4h film on the floor stays at rest", { skip: !process.env.WEBGPU_NODE_MODULE },
  () => filmRest((_, stats) => assert.ok(stats.maxSpeed_m_s! < 0.01, `speed ${stats.maxSpeed_m_s} m/s`)));

test("a 4h film on the floor keeps its volume", { skip: !process.env.WEBGPU_NODE_MODULE,
  todo: "O6: the film at rest loses 3.9% in 60 steps (685 to 659 cells), the same before the free-flight rule" },
  () => filmRest((initial, stats) => assert.ok(Math.abs(stats.volumeCellSum! - initial) < 1e-3 * initial, `volume ${initial} to ${stats.volumeCellSum} cells`)));
