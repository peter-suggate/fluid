import assert from "node:assert/strict";
import test from "node:test";

import { sceneDocument } from "../lib/core/scene-definition";
import {
  createSparseCM12LongDamBreakScene,
  getSceneDefinition,
} from "../lib/core/scenes";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  initializeSparseBrickAtlasFromScene,
  sparseCM12StaticSolidRestrictionError,
  sparseCM12StaticSolidResolutionFloor,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";

test("planar SolidWorld walls and their intersections restrict exactly", () => {
  const scene = createSparseCM12LongDamBreakScene();
  const dimensions = sceneLatticeDimensions(scene);
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions,
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  const at = (coordinate: readonly [number, number, number]) =>
    atlas.bricks.find((brick) => brick.coordinate.every((value, axis) =>
      value === coordinate[axis]));

  const planar = at([1, 0, 1]);
  assert.ok(planar, "the deep planar-floor fixture brick must be resident");
  assert.equal(sparseCM12StaticSolidRestrictionError(
    scene, dimensions, planar.coordinate, 8, 4,
  ), 0, "one flat wall must be affine-exact geometry evidence");
  assert.ok(planar.resolution < 8,
    `the planar floor was blanket-promoted to ${planar.resolution}^3`);

  const corner = at([0, 0, 0]);
  assert.ok(corner, "the tank-corner fixture brick must be resident");
  assert.equal(sparseCM12StaticSolidRestrictionError(
    scene, dimensions, corner.coordinate, 8, 4,
  ), 0, "orthogonal wall apertures must not be averaged across axes");
  assert.equal(sparseCM12StaticSolidResolutionFloor(
    scene, dimensions, corner.coordinate, 8,
  ), 1, "an aligned tank corner must add no solid-resolution floor");
  assert.ok(corner.resolution < 8,
    `the aligned tank corner was blanket-promoted to ${corner.resolution}^3`);
});

test("activity bootstrap shifts the exact offset hydrostatic surface from B8 to B4", () => {
  const scene = sceneDocument(getSceneDefinition("hydrostatic-power-large-offset"));
  const finestDimensions = sceneLatticeDimensions(scene);
  const surfaceDistance = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions,
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  const activity = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions,
    brickFineResolution: 8,
    surfaceFineRings: 1,
    initialSurfaceCoarseningBiasRings: 1,
  });
  const surfaceRungs = (atlas: typeof activity) => atlas.bricks
    .filter((brick) => brick.coordinate[1] === 1)
    .map((brick) => brick.resolution);

  assert.deepEqual(surfaceRungs(surfaceDistance), Array(8).fill(8),
    "Surface distance must retain its explicitly authored finest ring");
  assert.deepEqual(surfaceRungs(activity), Array(8).fill(4),
    "activity mode must present the broad calm reset surface one rung coarse");
});
