import assert from "node:assert/strict";
import test from "node:test";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickAtlasStats,
  materializeSparseBrickAtlasDensity,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  initializeSparseAtlasDynamics,
  injectSparseAtlasLiquid,
  stepSparseAtlasDynamics,
} from "../lib/methods/adaptive-mass/sparse-atlas-dynamics";

const DIMENSIONS = [32, 32, 32] as const;

function state() {
  const scene = createMinimalPowerDamBreak32Scene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: DIMENSIONS,
    maximumFinestCells: 1_048_576,
  });
  return initializeSparseAtlasDynamics(atlas, { time_s: 4 });
}

const mass = (atlas: Parameters<typeof sparseBrickAtlasStats>[0]) =>
  sparseBrickAtlasStats(atlas).integratedMassFineCells;

test("an injected ball adds its own volume of liquid to a running state", () => {
  const before = state();
  const radiusFine = 3;
  const after = injectSparseAtlasLiquid(before, {
    centerFine: [26, 26, 16],
    radiusFine: [radiusFine, radiusFine, radiusFine],
  });

  const added = mass(after.atlas) - mass(before.atlas);
  const exact = (4 / 3) * Math.PI * radiusFine ** 3;
  assert.ok(Math.abs(added - exact) / exact < 0.08,
    `added ${added.toFixed(2)} fine cells for a ball of ${exact.toFixed(2)}`);

  // The point of injecting rather than authoring: the run continues from where
  // it was. A restart would put the clock back to zero.
  assert.equal(after.time_s, before.time_s);
  // A closed SolidWorld shell may preseed the target as an empty structural
  // brick. Injection must still add liquid authority; it need not duplicate
  // an already represented page merely to make the brick count increase.
  const wetBricks = (state: typeof before) => state.atlas.bricks.filter((brick) =>
    brick.density.some((density) => density > 0)).length;
  assert.ok(after.atlas.bricks.length > before.atlas.bricks.length
    || wetBricks(after) > wetBricks(before),
  "a ball dropped into air must add or wet represented bricks");
  assert.equal(after.cellVelocity.length, 3 * after.grid.cells.length);
  assert.equal(after.faceNormalVelocity.length, after.grid.gradientRows.length);
  assert.equal(after.cellPressure.length, after.grid.cells.length);
});

test("an injected ball never erases the liquid already there", () => {
  const before = state();
  const beforeDensity = materializeSparseBrickAtlasDensity(before.atlas);
  // Centred inside the dam, so every sample lands on cells that are already full.
  const after = injectSparseAtlasLiquid(before, {
    centerFine: [4, 4, 16],
    radiusFine: [3, 3, 3],
  });
  const afterDensity = materializeSparseBrickAtlasDensity(after.atlas);
  for (let index = 0; index < beforeDensity.length; index += 1) {
    assert.ok(afterDensity[index] >= beforeDensity[index] - 1e-12,
      `cell ${index} lost density: ${beforeDensity[index]} -> ${afterDensity[index]}`);
  }
});

test("a step after an injection conserves the mass the ball brought", () => {
  const scene = createMinimalPowerDamBreak32Scene();
  const injected = injectSparseAtlasLiquid(state(), {
    centerFine: [26, 26, 16],
    radiusFine: [3, 3, 3],
  });
  const before = mass(injected.atlas);
  const step = stepSparseAtlasDynamics(injected, {
    dt_s: 1 / 30,
    accelerationFinePerSecond2: [0, scene.fluid.gravity_m_s2.y / 0.025, 0],
    projection: { relativeTolerance: 1e-10, absoluteTolerance: 1e-12 },
  });
  assert.ok(step.stats.massAbsoluteErrorFineCells / before < 1e-9,
    `mass error ${step.stats.massAbsoluteErrorFineCells} on ${before} fine cells`);
  assert.ok(Number.isFinite(step.stats.maximumOutgoingCfl));
  for (const value of step.state.cellVelocity) assert.ok(Number.isFinite(value));
});
