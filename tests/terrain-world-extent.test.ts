import assert from "node:assert/strict";
import test from "node:test";
import { terrainWorldContainer, type TerrainDescription } from "../lib/core/terrain";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { defaultScene } from "../lib/core/model";

test("sampled scenery may surround a smaller fluid domain without enlarging its solver lattice", () => {
  const container = { ...defaultScene.container, width_m: 32, height_m: 32, depth_m: 32 };
  const terrain: TerrainDescription = { baseHeight_m: 3, features: [], grid: {
    kind: "grid", origin_m: { x: -40, z: -48 }, spacing_m: 8,
    size: { nx: 11, nz: 11 }, heights_m: Array(121).fill(3),
  } };
  const scene = { container, terrain, voxelDomain: { ...defaultScene.voxelDomain, finestCellSize_m: .25 } };
  assert.deepEqual(terrainWorldContainer(scene), { width_m: 80, height_m: 32, depth_m: 96 });
  assert.deepEqual(sceneLatticeDimensions(scene), [128, 128, 128]);
  assert.equal(container.width_m, 32);
});

test("small sculpted grids retain the container footprint and elevated grids keep their headroom", () => {
  const container = { ...defaultScene.container, width_m: 8, height_m: 4, depth_m: 8 };
  const terrain: TerrainDescription = { baseHeight_m: 1, features: [], grid: {
    kind: "grid", origin_m: { x: -1, z: -1 }, spacing_m: 1,
    size: { nx: 3, nz: 3 }, heights_m: Array(9).fill(1),
  } };
  assert.deepEqual(terrainWorldContainer({ container, terrain }), { width_m: 8, height_m: 4, depth_m: 8 });
  terrain.grid!.heights_m[4] = 7;
  assert.equal(terrainWorldContainer({ container, terrain }).height_m, 7);
});
