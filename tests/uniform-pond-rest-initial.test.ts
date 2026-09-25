import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sampleSolidWorld, solidWorldForScene } from "../lib/core/solid-world";
import { uniformInitialVolume } from "../lib/methods/uniform/uniform-volume-initial";

test("filled hero pond seeds no mass in cells closed by the static solid mask", () => {
  const scene = sceneDocument(getSceneDefinition("hero-garden-hose-x10"));
  const h = scene.voxelDomain.finestCellSize_m;
  const dims = [scene.container.width_m, scene.container.height_m, scene.container.depth_m].map(v => Math.round(v / h)) as [number, number, number];
  const [nx, ny, nz] = dims;
  const world = solidWorldForScene(scene);
  const seed = uniformInitialVolume(scene, dims);
  let fractionalCells = 0, liquid = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const fraction = sampleSolidWorld(world, [x, y, z]).solidFraction;
    const volume = seed.volume[x + nx * (y + ny * z)]!;
    if (fraction > 0) assert.equal(volume, 0, `closed cell ${x},${y},${z}`);
    if (fraction > 0 && fraction < 1) fractionalCells++;
    liquid += volume;
  }
  assert.ok(fractionalCells > 100, "exercise the curved fractional terrain boundary");
  assert.ok(liquid > 1000, "retain the filled pond");
  assert.ok(Math.abs(liquid - seed.initial) < 1e-3, "reference mass counts the same accessible water");
});
