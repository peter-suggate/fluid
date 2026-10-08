import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { parseScene, serializeScene, validateScene, type SceneDescription } from "../lib/core/model";
import {
  NBFLIP_DAM_CYLINDERS, NBFLIP_DAM_GRID, NBFLIP_DAM_SCENE_ID, NBFLIP_METHOD_PROFILE,
  NBFLIP_POUR_GRID, NBFLIP_POUR_SCENE_ID, NBFLIP_SIMPLE_DAM_GRID, NBFLIP_SIMPLE_DAM_SCENE_ID,
  NBFLIP_TEASER_CYLINDER_COUNT, NBFLIP_TEASER_GRID, NBFLIP_TEASER_LATTICE, NBFLIP_TEASER_SCENE_ID,
  NBFLIP_WAVES_GRID, NBFLIP_WAVES_SCENE_ID, nbflipTeaserCylinderCentre,
} from "../lib/core/nbflip-paper-scenes";

type Cells = readonly [number, number, number];

/** A scene as its registry entry builds it, checked the way every paper scene is. */
function paperScene(id: string): SceneDescription {
  const definition = getSceneDefinition(id), scene = sceneDocument(definition);
  assert.deepEqual(validateScene(scene), [], id);
  assert.deepEqual(parseScene(serializeScene(scene)).fluid, scene.fluid, id);
  assert.deepEqual(parseScene(serializeScene(scene)).solidVoxels, scene.solidVoxels, id);
  assert.deepEqual(definition.methodProfile, NBFLIP_METHOD_PROFILE, id);
  assert.ok(lattice(scene).every((cells) => cells % 4 === 0), `${id}: the method's tiles are four cells wide`);
  assert.equal(scene.container.top, "closed");
  assert.equal(scene.container.fluidWallMode, "free-slip");
  return scene;
}
const lattice = (scene: SceneDescription): Cells => {
  const h = scene.voxelDomain.finestCellSize_m, c = scene.container;
  return [c.width_m / h, c.height_m / h, c.depth_m / h].map(Math.round) as unknown as Cells;
};
/** The fall, in cells per solver frame squared. */
const fall = (scene: SceneDescription) => -scene.fluid.gravity_m_s2.y * scene.numerics.fixedDt_s! ** 2 / scene.voxelDomain.finestCellSize_m;
const solid = (scene: SceneDescription, cell: Cells) => scene.solidVoxels!.some((patch) =>
  patch.operation === "fill" && cell.every((value, axis) => value >= patch.minimum[axis]! && value < patch.maximumExclusive[axis]!));
/** A point given in cells from the lattice's minimum corner, in metres. */
const cells = (scene: SceneDescription, point_m: { x: number; y: number; z: number }): Cells => {
  const h = scene.voxelDomain.finestCellSize_m, [nx, , nz] = lattice(scene);
  return [point_m.x / h + nx / 2, point_m.y / h, point_m.z / h + nz / 2];
};

test("NB-FLIP Figure 9 stands the published 200^3 box behind solid, with seven cylinders", () => {
  const scene = paperScene(NBFLIP_TEASER_SCENE_ID), margin = (NBFLIP_TEASER_LATTICE[0] - NBFLIP_TEASER_GRID[0]) / 2;
  assert.deepEqual(NBFLIP_TEASER_GRID, [200, 200, 200]);
  assert.deepEqual(lattice(scene), [...NBFLIP_TEASER_LATTICE]);
  assert.ok(Math.abs(fall(scene) - 0.6) < 1e-12, "mantaflow: 0.003 x 200");
  assert.equal(scene.numerics.fixedDt_s, 1 / 24);
  // The liquid's box is the published grid: solid begins exactly outside it.
  for (const y of [0, 100, 199]) {
    assert.ok(solid(scene, [margin - 1, y, 112]) && !solid(scene, [margin, y, margin]));
    assert.ok(solid(scene, [margin + 200, y, 112]) && !solid(scene, [margin + 199, y, margin + 199]));
    assert.ok(solid(scene, [112, y, margin - 1]) && solid(scene, [112, y, margin + 200]));
  }
  assert.ok(solid(scene, [112, 200, 112]) && !solid(scene, [112, 199, 112]), "the ceiling is at the grid's height");
  // The pool and the dam, in cells of the published grid.
  assert.equal(scene.container.fillFraction * NBFLIP_TEASER_LATTICE[1], 40);
  const [dam] = scene.fluid.initialLiquidVolumes!;
  assert.ok(dam && dam.shape === "box");
  const low = cells(scene, dam.min_m), high = cells(scene, dam.max_m);
  for (const axis of [0, 1, 2]) {
    assert.ok(Math.abs(low[axis]! - [margin, 40, margin][axis]!) < 1e-9 && Math.abs(high[axis]! - [margin + 60, 120, margin + 60][axis]!) < 1e-9);
  }
  // Each cylinder is solid on its axis to one cell above the dam and open beside it.
  for (let index = 0; index < NBFLIP_TEASER_CYLINDER_COUNT; index++) {
    const [x, z] = nbflipTeaserCylinderCentre(index).map((value) => Math.floor(value + margin)) as [number, number];
    assert.ok(solid(scene, [x, 0, z]) && solid(scene, [x, 120, z]) && !solid(scene, [x, 121, z]), `cylinder ${index}`);
    assert.ok(solid(scene, [x + 4, 60, z]) && !solid(scene, [x + 6, 60, z]) && !solid(scene, [x, 60, z + 6]), `cylinder ${index} has radius 5`);
  }
});

test("NB-FLIP Figure 10 is the published channel with eight thin cylinders", () => {
  const scene = paperScene(NBFLIP_DAM_SCENE_ID), [nx, ny, nz] = NBFLIP_DAM_GRID;
  assert.deepEqual(NBFLIP_DAM_GRID, [256, 128, 64]);
  assert.deepEqual(lattice(scene), [nx, ny, nz]);
  assert.equal(scene.numerics.fixedDt_s, 1 / 120);
  // A metre of channel under 9.8 m/s^2 at 120 Hz.
  assert.ok(Math.abs(scene.container.width_m - 1) < 1e-12 && Math.abs(fall(scene) - 9.8 * 256 / 120 ** 2) < 1e-12);
  assert.ok(Math.abs(scene.container.fillFraction * ny - 0.06 * nx) < 1e-9, "the sheet");
  const [block] = scene.fluid.initialLiquidVolumes!;
  assert.ok(block && block.shape === "box");
  const high = cells(scene, block.max_m);
  assert.ok(Math.abs(high[0] - 0.2 * nx) < 1e-9 && Math.abs(high[1] - 0.3 * nx) < 1e-9 && Math.abs(high[2] - nz) < 1e-9);
  assert.equal(NBFLIP_DAM_CYLINDERS.length, 8);
  for (const [x, z] of NBFLIP_DAM_CYLINDERS) {
    const column: Cells = [Math.floor(x * nx), 0, Math.floor(z * nz)];
    assert.ok(solid(scene, column) && solid(scene, [column[0], 76, column[2]]) && !solid(scene, [column[0], 77, column[2]]));
    assert.ok(!solid(scene, [column[0] + 4, 0, column[2]]) && !solid(scene, [column[0], 0, column[2] + 4]), "radius 2.5");
  }
});

test("the two coupling scenes carry mantaflow's wall layer and play two frames per video frame", () => {
  for (const [id, grid, basin] of [[NBFLIP_WAVES_SCENE_ID, NBFLIP_WAVES_GRID, 0.3], [NBFLIP_SIMPLE_DAM_SCENE_ID, NBFLIP_SIMPLE_DAM_GRID, 0.1]] as const) {
    const scene = paperScene(id), [nx, ny, nz] = grid;
    assert.deepEqual(lattice(scene), [...grid]);
    assert.equal(scene.numerics.fixedDt_s, 1 / 48);
    assert.ok(Math.abs(fall(scene) - 0.003 * nx) < 1e-12, id);
    assert.ok(Math.abs(scene.duration_s - 500 / 48) < 1e-12);
    // The outermost layer of cells is wall on all six sides, and nothing else is.
    for (let b = 0; b < nx; b += 5) for (let a = 0; a < nx; a += 5) {
      for (const cell of [[0, a, b], [nx - 1, a, b], [a, 0, b], [a, ny - 1, b], [a, b, 0], [a, b, nz - 1]] as const) assert.ok(solid(scene, cell), `${id} ${cell}`);
    }
    for (const cell of [[1, 1, 1], [nx - 2, ny - 2, nz - 2], [1, ny - 2, 1], [nx / 2, ny / 2, nz / 2]] as const) assert.ok(!solid(scene, cell), `${id} ${cell}`);
    assert.ok(Math.abs(scene.container.fillFraction - basin) < 1e-12);
  }
  // The dam's block reaches the camera-side wall and stops short of the far one.
  const dam = paperScene(NBFLIP_SIMPLE_DAM_SCENE_ID), [block] = dam.fluid.initialLiquidVolumes!;
  assert.ok(block && block.shape === "box");
  const low = cells(dam, block.min_m), high = cells(dam, block.max_m);
  assert.ok(Math.abs(low[2] - 0.3 * 64) < 1e-9 && Math.abs(high[2] - 64) < 1e-9 && Math.abs(high[0] - 0.6 * 64) < 1e-9 && Math.abs(high[1] - 32) < 1e-9);
});

test("NB-FLIP Figure 7 pours the measured stream into a watertight glass", () => {
  const scene = paperScene(NBFLIP_POUR_SCENE_ID), [nx, ny, nz] = NBFLIP_POUR_GRID, h = scene.voxelDomain.finestCellSize_m;
  assert.deepEqual(NBFLIP_POUR_GRID, [128, 256, 128]);
  assert.deepEqual(lattice(scene), [nx, ny, nz]);
  assert.ok(Math.abs(fall(scene) - 0.768) < 1e-12, "mantaflow: 0.003 x 256");
  assert.equal(scene.container.fillFraction, 0);
  assert.equal(scene.fluid.initialLiquidVolumes, undefined);
  // The glass: open inside four tenths of the grid, solid just outside it, to three quarters of the height.
  for (const y of [0, 100, 191]) {
    assert.ok(!solid(scene, [64 + 50, y, 64]) && solid(scene, [64 + 52, y, 64]) && !solid(scene, [64 + 55, y, 64]), `rim row ${y}`);
    assert.ok(!solid(scene, [64, y, 64 - 51]) && solid(scene, [64, y, 64 - 53]));
  }
  assert.ok(!solid(scene, [64 + 52, 192, 64]), "nothing above the rim");
  // Watertight: a walk through face neighbours from the axis never leaves the glass.
  const seen = new Set<number>([64 + nx * 64]), queue = [[64, 64]];
  while (queue.length) {
    const [x, z] = queue.pop()!;
    assert.ok(Math.hypot(x! + 0.5 - 64, z! + 0.5 - 64) < 51.2, `the glass leaks at ${x}, ${z}`);
    for (const [u, w] of [[x! + 1, z!], [x! - 1, z!], [x!, z! + 1], [x!, z! - 1]] as const) {
      if (!seen.has(u + nx * w) && !solid(scene, [u, 0, w])) { seen.add(u + nx * w); queue.push([u, w]); }
    }
  }
  assert.ok(Math.abs(seen.size - Math.PI * 51.2 ** 2) < 60, `${seen.size} cells of floor`);
  // The stream: a 19.2-cell bore, 3 down in 5 across, for 126 frames from the first.
  const inflow = scene.fluid.inflow!, dt = scene.numerics.fixedDt_s!;
  const [u, v, w] = [inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z].map((speed) => speed * dt / h) as [number, number, number];
  assert.ok(Math.abs(u - 3.5) < 1e-9 && Math.abs(v + 2.1) < 1e-9 && w === 0);
  assert.ok(Math.abs(inflow.radius_m / h - 19.2) < 1e-9);
  assert.deepEqual([inflow.start_s, Math.round(inflow.end_s / dt), inflow.ramp_s], [0, 126, 0]);
  const centre = cells(scene, inflow.center_m);
  assert.ok(Math.abs(centre[2] - 64) < 1e-9 && Math.abs((centre[1] - 214.5) / (centre[0] - 31) + 0.6) < 1e-9, "on the pipe's axis");
  assert.ok(Math.abs(Math.hypot(centre[0] - 31, centre[1] - 214.5) - 5.5) < 1e-9, "5.5 cells beyond the mouth");
  // The tilted plug each frame adds stays under the ceiling and within the glass's bore.
  const across = 19.2 * 3 / Math.sqrt(34), up = 19.2 * 5 / Math.sqrt(34);
  assert.ok(centre[1] + up < ny && centre[0] - across > 64 - 51.2 && centre[0] + 3.5 + across < 64 + 51.2);
});
