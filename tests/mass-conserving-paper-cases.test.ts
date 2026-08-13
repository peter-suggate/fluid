import assert from "node:assert/strict";
import test from "node:test";
import { createMassConservingFigure9DamBreak } from "../lib/paper-scenarios";
import { sceneLatticeDimensions } from "../lib/scene-lattice";
import { validateScene } from "../lib/model";
import { scenePresets } from "../lib/scenes";
import {
  UNIFORM_CM11A_COARSEST_LANES, planUniformCM11aHierarchy,
} from "../lib/webgpu-uniform-pressure-multigrid";
import {
  CM12_CELL_SIZE_M, CM12_FIGURES, CM12_GRAVITY_M_S2, CM12_SLAB_DEPTH_CELLS,
  CM12_SOLID_EXCESS_STRENGTH, CM12_TIME_STEP_S, CM12_TRACE_DISTANCE_CELLS,
  cm12CharacteristicSpeed_m_s, cm12Figure, cm12Grid, cm12MethodProfile, cm12Scene,
} from "../lib/cm12-paper-scenes";

test("CM12 Figure 9 harness uses every published numerical parameter", () => {
  const scene = createMassConservingFigure9DamBreak();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(sceneLatticeDimensions(scene), [128, 128, 64]);
  assert.equal(scene.voxelDomain.finestCellSize_m, 0.05);
  assert.equal(scene.numerics.fixedDt_s, 1 / 30);
  assert.equal(scene.numerics.maxDt_s, 1 / 30);
  assert.deepEqual(scene.fluid.gravity_m_s2, { x: 0, y: -10, z: 0 });
  assert.equal(scene.fluid.surfaceTension_N_m, 0);
});

/**
 * Every figure of the paper, checked against what the paper actually prints.
 *
 * The point of this test is that a CM12 scene is a *citation*: the grid, the
 * cell size, the step, gravity, D and S are quoted numbers, so a default that
 * moves underneath the catalog has to break here rather than quietly change
 * what the figure means.
 */
test("every CM12 figure reproduces the published grid and parameter tuple", () => {
  assert.equal(CM12_FIGURES.length, 10, "the paper's ten simulated figures");
  for (const figure of CM12_FIGURES) {
    const where = `figure ${figure.figure}`;
    const scene = cm12Scene(figure.id);
    assert.deepEqual(validateScene(scene), [], where);

    // Sec. 4 and Table 2: the grid is published, and the domain is exactly
    // that many cells of the published size — no rounding, no padding.
    const grid = cm12Grid(figure);
    assert.deepEqual(sceneLatticeDimensions(scene), [...grid], `${where} grid`);
    assert.equal(scene.voxelDomain.finestCellSize_m, CM12_CELL_SIZE_M, `${where} dx`);
    assert.equal(scene.container.width_m, grid[0] * CM12_CELL_SIZE_M, `${where} width`);
    assert.equal(scene.container.height_m, grid[1] * CM12_CELL_SIZE_M, `${where} height`);
    assert.equal(scene.container.depth_m, grid[2] * CM12_CELL_SIZE_M, `${where} depth`);

    assert.equal(scene.numerics.fixedDt_s, CM12_TIME_STEP_S, `${where} dt`);
    assert.equal(scene.numerics.maxDt_s, CM12_TIME_STEP_S, `${where} max dt`);
    assert.deepEqual(scene.fluid.gravity_m_s2, { x: 0, y: -CM12_GRAVITY_M_S2, z: 0 }, `${where} g`);
    assert.equal(scene.fluid.surfaceTension_N_m, 0, `${where} has no surface tension`);

    // The method *is* the paper, and it carries D, S and Sec. 3.8's switch.
    const profile = cm12MethodProfile(figure);
    assert.equal(profile.methodId, "uniform", `${where} runs the paper's method`);
    assert.equal(profile.overrides?.timeStep, "paper", `${where} pins the paper step`);
    assert.equal(profile.overrides?.sharpeningDistance, CM12_TRACE_DISTANCE_CELLS, `${where} D`);
    assert.equal(profile.overrides?.sharpeningStrength, CM12_SOLID_EXCESS_STRENGTH, `${where} S`);
    // Sec. 3.8 is off "unless otherwise stated"; only Figure 6 states otherwise.
    assert.equal(profile.overrides?.densityPostProcessing,
      figure.figure === 6 ? "on" : "off", `${where} density post-processing`);
  }
});

test("the published 2D figures are the only ones extended into depth", () => {
  for (const figure of CM12_FIGURES) {
    const flat = figure.grid[2] === undefined;
    assert.equal(flat, figure.figure === 2 || figure.figure === 3,
      `figure ${figure.figure} 2D-ness`);
    if (!flat) continue;
    // A 2D case keeps its published x/y and gains the slab, and nothing in the
    // third dimension may do work.
    assert.deepEqual(cm12Grid(figure), [figure.grid[0], figure.grid[1], CM12_SLAB_DEPTH_CELLS]);
    assert.equal(cm12Scene(figure.id).container.fluidWallMode, "free-slip",
      `figure ${figure.figure} slab walls`);
  }
});

test("Table 2's CFL column is what sets the speed of the two driven scenes", () => {
  // CFL = v dt/dx with both fixed, so 25 and 32 are speeds: 37.5 and 48 m/s.
  assert.equal(cm12CharacteristicSpeed_m_s(25), 37.5);
  assert.equal(cm12CharacteristicSpeed_m_s(32), 48);

  const jet = cm12Scene("cm12-figure-1").fluid.inflow;
  assert.ok(jet, "Figure 1 is driven by a jet");
  const jetSpeed = Math.hypot(jet.velocity_m_s.x, jet.velocity_m_s.y, jet.velocity_m_s.z);
  assert.equal(jetSpeed, cm12CharacteristicSpeed_m_s(cm12Figure("cm12-figure-1").cfl!));

  const bodies = cm12Scene("cm12-figure-11").rigidBodies ?? [];
  assert.ok(bodies.length > 0, "Figure 11 is driven by moving obstacles");
  const fastest = Math.max(...bodies.map((body) => Math.hypot(
    body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z)));
  assert.equal(fastest, cm12CharacteristicSpeed_m_s(cm12Figure("cm12-figure-11").cfl!));
});

test("every CM12 figure is in the catalog under the paper's own shelf", () => {
  const catalog = new Map(scenePresets.map((preset) => [preset.id, preset]));
  for (const figure of CM12_FIGURES) {
    const preset = catalog.get(figure.id);
    assert.ok(preset, `${figure.id} is a shipped preset`);
    assert.deepEqual(validateScene(preset.create()), [], figure.id);
  }
});

test("Figures 8 and 12 share the true closed spherical boundary", () => {
  for (const id of ["cm12-figure-8", "cm12-figure-12"] as const) {
    const scene = cm12Scene(id);
    assert.equal(scene.container.shape, "sphere", id);
    assert.equal(scene.container.top, "closed", id);
    assert.equal(scene.container.vessel, "glass", id);
    assert.equal(scene.terrain, undefined, `${id} must not degrade the upper hemisphere to a heightfield wall`);
    assert.equal(scene.surfaceStyle, "smooth", id);
  }
  const dam = cm12Scene("cm12-figure-8");
  assert.deepEqual(dam.fluid.initialLiquidVolumes, [{
    shape: "hemisphere",
    center_m: { x: 0, y: 3.2, z: 0 },
    radius_m: 3.2,
    outwardNormal: { x: 1, y: 0, z: 0 },
  }]);
  const drop = cm12Scene("cm12-figure-12");
  assert.equal(drop.fluid.initialLiquidVolumes?.length, 1);
  assert.equal(drop.fluid.initialLiquidVolumes?.[0]?.shape, "sphere");
});

/**
 * The precondition that is invisible to `validateScene`.
 *
 * A CM12 scene declares the `uniform` method, and that solver builds a dense
 * CM11a pressure hierarchy in its *constructor*. A lattice the hierarchy cannot
 * carry throws at load: the scene validates, builds, renders its dry world, and
 * then never starts. Nothing else in the CPU suite can see that, so it is
 * asserted here against the same arithmetic the constructor uses.
 */
test("every scene declaring the uniform method can carry a CM11a hierarchy", () => {
  const declaring = scenePresets.filter((preset) => preset.methodProfile?.methodId === "uniform");
  assert.ok(declaring.length >= CM12_FIGURES.length, "the paper's figures declare the method");
  for (const preset of declaring) {
    const [nx, ny, nz] = sceneLatticeDimensions(preset.create());
    const plan = planUniformCM11aHierarchy([nx, ny, nz]);
    assert.equal(plan.rejection, undefined, `${preset.id} at ${nx}x${ny}x${nz}`);
    assert.ok(plan.coarsestCells <= UNIFORM_CM11A_COARSEST_LANES, preset.id);
  }
});

test("a thin 2D slab is carried by a semi-coarsened hierarchy", () => {
  const cross = 128;
  const slab = planUniformCM11aHierarchy([cross, cross, CM12_SLAB_DEPTH_CELLS]);
  assert.equal(slab.rejection, undefined, "the published 2D cases must load");
  // The lockstep rule cannot reach this lattice, so the plan is the per-axis
  // one, and it coarsens *further* than lockstep manages on a cube: the 256
  // lanes were never what stopped a slab.
  assert.equal(slab.semiCoarsened, true);
  assert.deepEqual(slab.levelDimensions[slab.levelCount - 1], [2, 2, 2]);
  assert.ok(slab.coarsestCells <= UNIFORM_CM11A_COARSEST_LANES);
  // Every axis reaches its floor by halving, never by jumping.
  for (let index = 1; index < slab.levelCount; index += 1) {
    const finer = slab.levelDimensions[index - 1]!, coarser = slab.levelDimensions[index]!;
    assert.ok(coarser.some((value, axis) => value === finer[axis]! / 2), "a level must coarsen something");
    for (const [axis, value] of coarser.entries()) {
      assert.ok(value === finer[axis] || value === finer[axis]! / 2, `axis ${axis} halves or holds`);
    }
  }
});

/**
 * The rule that keeps semi-coarsening from disturbing anything that works.
 *
 * Both rules can describe some lattices, and they disagree: on 128x128x64 the
 * lockstep plan stops at 4x4x2 where the per-axis one reaches 2x2x2. Since the
 * D4 folds downstream make a different hierarchy a different *rounding*, the
 * lockstep plan has to win wherever it exists, or every scene that loads today
 * quietly changes its numbers.
 */
test("semi-coarsening is reached only by lattices the lockstep rule refuses", () => {
  for (const dimensions of [[128, 128, 128], [256, 128, 128], [128, 128, 64], [32, 32, 32]] as const) {
    const plan = planUniformCM11aHierarchy(dimensions);
    assert.equal(plan.rejection, undefined, String(dimensions));
    assert.equal(plan.semiCoarsened, false, `${dimensions} must keep its lockstep hierarchy`);
    // Lockstep: every axis divided by the same power of two at every level.
    for (const [index, level] of plan.levelDimensions.entries()) {
      assert.deepEqual([...level], dimensions.map((value) => value / 2 ** index), `level ${index}`);
    }
  }
  for (const dimensions of [[128, 128, 8], [320, 96, 80], [24, 18, 16]] as const) {
    const plan = planUniformCM11aHierarchy(dimensions);
    assert.equal(plan.rejection, undefined, String(dimensions));
    assert.equal(plan.semiCoarsened, true, `${dimensions} has no lockstep hierarchy`);
  }
});
