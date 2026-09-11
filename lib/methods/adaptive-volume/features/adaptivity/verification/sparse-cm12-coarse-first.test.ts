import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../../../../../core/scene-definition";
import { getSceneDefinition } from "../../../../../core/scenes";
import { initializeSparseBrickAtlasFromScene } from "../../../sparse-brick-atlas";
import { adaptiveMassMethod, adaptiveMassSolverOptions, ADAPTIVE_MASS_RUNTIME_PARAM_KEYS } from "../../../method";
import { SPARSE_CM12_STAGES } from "../../../sparse-cm12-stages";
import { resolveMethodValues } from "../../../../../core/method-contract";

test("coarse-first starts the pool at B1 and curved liquid fine, conserving authored mass", () => {
  const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact"));
  // Keep the original high-curvature fixture independent of the demo ball size.
  scene.fluid.initialLiquidVolumes = [
    { shape: "sphere", center_m: { x: 0, y: 3.65, z: 0 }, radius_m: 0.5 },
  ];
  const options = { finestDimensions: [128, 96, 128] as const, brickFineResolution: 8 as const };
  const adaptive = initializeSparseBrickAtlasFromScene(scene, {
    ...options, coarseFirstCurvatureTolerance: 0.25,
  });
  const reference = initializeSparseBrickAtlasFromScene(scene, {
    ...options, resolutionForBrick: () => 8,
  });
  const pool = adaptive.bricks.filter(b => b.coordinate[1] === 3 && b.density.some(v => v > 0));
  assert.ok(adaptive.bricks.some(b => (b.spanBricks ?? 1) >= 2), "deep bulk needs a coarser macro rung");
  const ball = adaptive.bricks.filter(b => b.coordinate[1] >= 8 && b.density.some(v => v > 0));
  assert.ok(pool.length >= 64);
  assert.ok(pool.filter(b => b.resolution === 1).length >= pool.length * 0.8,
    JSON.stringify(pool.map(b => [b.coordinate, b.resolution])));
  assert.ok(ball.length > 0 && ball.every(b => b.resolution === 8));
  const mass = (atlas: typeof adaptive) => atlas.bricks.reduce((sum, b) => sum
    + b.density.reduce((a, v) => a + v, 0) * (8 * (b.spanBricks ?? 1) / b.resolution) ** 3, 0);
  assert.ok(Math.abs(mass(adaptive) - mass(reference)) < 1e-8);
  const relaxed = initializeSparseBrickAtlasFromScene(scene, {
    ...options, coarseFirstCurvatureTolerance: 2,
  });
  const cells = (atlas: typeof adaptive) => atlas.bricks.reduce((n, b) => n + b.resolution ** 3, 0);
  assert.ok(cells(relaxed) < cells(adaptive), "curvature control must change physical topology");
  assert.ok(Math.abs(mass(relaxed) - mass(reference)) < 1e-8);
});

test("coarse-first controls survive method normalization and runtime routing", () => {
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    selectorMode: "coarse-first", energyThreshold: 9, curvatureTolerance: 0.4,
    anticipationSeconds: 0.8, anticipationRadiusBricks: 4, surfaceQuietEpochs: 5,
  });
  const policy = adaptiveMassSolverOptions(values).activityPolicy!;
  assert.equal(policy.coarseFirst, true);
  assert.equal(policy.activitySignals, true);
  for (const key of ["energyThreshold", "curvatureTolerance", "anticipationSeconds",
    "anticipationRadiusBricks", "surfaceQuietEpochs"] as const) {
    assert.equal(policy[key], values[key]);
    assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes(key));
    assert.ok(SPARSE_CM12_STAGES["resolution-planning"].controls.some(control =>
      control.kind === "param-range" && control.param === key),
    `${key} needs a slider in the resolution toolstrip`);
  }
});
