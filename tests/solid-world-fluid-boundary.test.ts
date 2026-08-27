import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { getScenePreset } from "../lib/core/scenes";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";

const residentSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");
const shaderSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

test("Sparse CM12 has no inferred tank-plane boundary authority", () => {
  for (const source of [residentSource, shaderSource]) {
    assert.doesNotMatch(source, /planarFluidBoundary|hasPlanarFluidBoundaries/);
  }
  assert.match(residentSource, /const staticSolidWorld = Boolean\(this\.solidOccupancyLayout\)/);
  assert.match(shaderSource,
    /return solid\*solidVoxelRowOpenFraction\(id\);/,
    "row openness must come directly from SolidWorld voxels");
});

test("SolidWorld boundary transitions keep touching wet bricks on the finest rung", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [24, 16, 16],
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  const wetBoundaryBricks = atlas.bricks.filter((brick) =>
    brick.density.some((density) => density > 0)
      && (brick.coordinate[0] === 0 || brick.coordinate[1] === 0
        || brick.coordinate[2] === 0));
  assert.ok(wetBoundaryBricks.length > 0);
  assert.ok(wetBoundaryBricks.every((brick) =>
    brick.resolution === atlas.brickFineResolution));
});
