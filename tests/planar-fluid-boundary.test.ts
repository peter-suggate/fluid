import assert from "node:assert/strict";
import test from "node:test";
import { compileE0PlanarFluidBoundaries } from
  "../lib/core/planar-fluid-boundary";
import {
  e0PlanarFluidBoundaryFaceMask,
} from "../lib/core/planar-fluid-boundary";
import { getScenePreset } from "../lib/core/scenes";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";

test("water-box shell compiles all six exact E0 planar faces", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const compiled = compileE0PlanarFluidBoundaries(scene);
  assert.deepEqual(compiled.rejectedFaces, []);
  assert.deepEqual(compiled.admitted.map(({ face }) => face),
    ["yLow", "xLow", "xHigh", "zLow", "zHigh", "yHigh"]);
  assert.ok(compiled.admitted.every(({ patch }) =>
    Math.abs(patch.halfThickness_m - 0.025) < 1e-12));
  assert.equal(e0PlanarFluidBoundaryFaceMask(compiled), 0b11_1111);
});

test("open tops and wall cuts fail closed in the E0 compiler", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  scene.container.top = "open";
  scene.solidVoxels = scene.solidVoxels.filter((patch) =>
    !(patch.operation === "fill" && patch.minimum[1] === 16));
  let compiled = compileE0PlanarFluidBoundaries(scene);
  assert.deepEqual(compiled.admitted.map(({ face }) => face),
    ["yLow", "xLow", "xHigh", "zLow", "zHigh"]);

  scene.solidVoxels.push({
    operation: "clear", minimum: [24, 4, 6], maximumExclusive: [25, 8, 10],
  });
  compiled = compileE0PlanarFluidBoundaries(scene);
  assert.equal(compiled.admitted.some(({ face }) => face === "xHigh"), false);
  assert.ok(compiled.rejectedFaces.includes("xHigh"));
});

test("Sparse CM12 keeps featureless wet water-box wall bricks coarse", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [24, 16, 16],
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  const wetWallBricks = atlas.bricks.filter((brick) =>
    brick.density.some((density) => density > 0)
      && (brick.coordinate[0] === 0 || brick.coordinate[1] === 0
        || brick.coordinate[2] === 0));
  assert.ok(wetWallBricks.length > 0);
  assert.ok(wetWallBricks.some((brick) => brick.resolution < atlas.brickFineResolution),
    "an admitted planar enclosure must not pin every adjacent wet brick to B8");
  assert.equal(atlas.directory.get(0)?.resolution, 4,
  "deep floor contact should retain the ordinary one-rung surface grading level");
});
