import assert from "node:assert/strict";
import test from "node:test";

import { createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
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
