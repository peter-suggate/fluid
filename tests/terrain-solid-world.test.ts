import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import {
  sampleSolidWorld,
  solidWorldForScene,
  solidWorldPageAddress,
  SOLID_WORLD_TERRAIN_MATERIAL_ID,
} from "../lib/core/solid-world";
import {
  createSparseCM12SolidOccupancyLayout,
  packSparseCM12SolidOccupancy,
  SPARSE_CM12_SOLID_FRACTION_PAGE_WORDS,
  SPARSE_CM12_SOLID_OCCUPANCY_PAGE_WORDS,
} from "../lib/methods/adaptive-mass/sparse-cm12-solid-occupancy";
import {
  createWebgpuSolidWorldPageLayout,
  packWebgpuSolidWorldPages,
  WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS,
} from "../lib/core/webgpu-solid-world-pages";

test("voxel terrain compiles page-native into the uniform SolidWorld", () => {
  const scene = createEmptyScene({ extents_m: { x: 0.8, y: 0.8, z: 0.8 },
    finestCellSize_m: 0.1 });
  scene.terrain = {
    baseHeight_m: 0.15,
    features: [],
  };
  scene.solidVoxels.push({ operation: "clear", minimum: [3, 0, 4],
    maximumExclusive: [4, 2, 5] });

  const world = solidWorldForScene(scene);
  assert.deepEqual(sampleSolidWorld(world, [2, 0, 4]), {
    solidFraction: 1,
    signedDistance_cells: -1,
    materialId: SOLID_WORLD_TERRAIN_MATERIAL_ID,
  });
  assert.ok(Math.abs(sampleSolidWorld(world, [2, 1, 4]).solidFraction - 0.5)
    <= 1 / 255, "the top terrain voxel retains quantized cut-cell coverage");
  assert.equal(sampleSolidWorld(world, [2, 2, 4]).solidFraction, 0);
  assert.equal(sampleSolidWorld(world, [3, 0, 4]).solidFraction, 0,
    "a generic clear edit removes terrain with no separate opening concept");
  assert.equal(world.patches.length, scene.solidVoxels.length,
    "terrain columns must not expand into one host descriptor per voxel");

  const layout = createSparseCM12SolidOccupancyLayout({ baseWords: 0,
    authoredPageCount: world.pages.length });
  const packed = packSparseCM12SolidOccupancy(layout, world, [0, 0, 0]);
  const address = solidWorldPageAddress([2, 1, 4]);
  const page = world.pages.findIndex((candidate) => candidate.coordinate.every(
    (value, axis) => value === address.page[axis]));
  assert.ok(page >= 0);
  const payload = layout.pageBaseWords + page * SPARSE_CM12_SOLID_OCCUPANCY_PAGE_WORDS;
  const fractionWord = packed[payload + (address.localIndex >>> 2)]!;
  assert.equal((fractionWord >>> (8 * (address.localIndex & 3))) & 0xff,
    Math.round(0.5 * 255), "the GPU projection retains fractional occupancy");
  const sdfWord = packed[payload + SPARSE_CM12_SOLID_FRACTION_PAGE_WORDS
    + (address.localIndex >>> 1)]!;
  const sdfQ8 = (sdfWord >>> (16 * (address.localIndex & 1))) & 0xffff;
  assert.equal(sdfQ8, 0, "the GPU projection retains the terrain SDF sample");

  const renderLayout = createWebgpuSolidWorldPageLayout({ baseWords: 0,
    authoredPageCount: world.pages.length, includesMaterial: true });
  const renderPacked = packWebgpuSolidWorldPages(renderLayout, world, [0, 0, 0], {
    origin_m: [-0.4, 0, -0.4], cellSize_m: [0.1, 0.1, 0.1],
  });
  const renderPayload = renderLayout.pageBaseWords
    + page * renderLayout.pageWords;
  const materialWord = renderPacked[renderPayload
    + WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS + (address.localIndex >>> 1)]!;
  assert.equal((materialWord >>> (16 * (address.localIndex & 1))) & 0xffff,
    SOLID_WORLD_TERRAIN_MATERIAL_ID,
    "the direct render page stream retains the canonical material ID");
});
