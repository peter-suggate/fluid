import assert from "node:assert/strict";
import test from "node:test";
import {
  boxSolidVoxelShell,
  createSolidWorld,
  planSolidWorldMemory,
  sampleSolidWorld,
  solidWorldForScene,
  solidWorldContentStamp,
  solidWorldPageAddress,
  SolidWorldDirectory,
} from "../lib/core/solid-world";
import { sparseCM12FinePresentationPlan } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { FINE_LEVELSET_SIGNED_SPARSE_ADDRESS_FLAG } from
  "../lib/core/fine-levelset-brick-abi";
import { getScenePreset } from "../lib/core/scenes";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  createSparseCM12SolidOccupancyLayout,
  packSparseCM12SolidOccupancy,
  SPARSE_CM12_SOLID_OCCUPANCY_MAGIC,
} from "../lib/methods/adaptive-mass/sparse-cm12-solid-occupancy";

test("SolidWorld content stamp ignores unrelated scene edits", () => {
  const scene = getScenePreset("bounded-pool-transfer").create();
  const original = solidWorldContentStamp(scene);
  scene.fluid.gravity_m_s2.y *= 0.5;
  assert.equal(solidWorldContentStamp(scene), original);
  scene.solidVoxels.push({ operation: "clear", minimum: [0, 0, 0],
    maximumExclusive: [1, 1, 1] });
  assert.notEqual(solidWorldContentStamp(scene), original);
});

test("SolidWorld uses exact signed pages and a floor-reaching voxel cut", () => {
  const world = createSolidWorld([
    ...boxSolidVoxelShell([16, 8, 16]),
    { operation: "clear", minimum: [16, 0, 5], maximumExclusive: [17, 4, 11] },
  ]);

  assert.equal(sampleSolidWorld(world, [16, 0, 6]).solidFraction, 0,
    "the authored right-wall hole is empty voxel space");
  assert.equal(sampleSolidWorld(world, [16, 4, 6]).solidFraction, 1,
    "uncut right-wall voxels remain solid");
  assert.equal(sampleSolidWorld(world, [6, -1, 6]).solidFraction, 1,
    "a floor-reaching side cut does not erase the independent floor voxel");
  assert.equal(sampleSolidWorld(world, [40, 0, 0]).solidFraction, 0,
    "an absent signed page is empty space, not an implicit boundary");
  assert.deepEqual(solidWorldPageAddress([-1, -1, -9]), {
    page: [-1, -1, -2], local: [7, 7, 7], localIndex: 511,
  });
  const directory = new SolidWorldDirectory(2);
  directory.insert([-2, 0, 3], 0);
  directory.insert([2, 0, -3], 1);
  assert.equal(directory.lookup([-2, 0, 3]), 0);
  assert.equal(directory.lookup([2, 0, -3]), 1);
  assert.equal(directory.lookup([-2, 0, -3]), undefined);

  const memory = planSolidWorldMemory({ staticPageCapacity: 8,
    dynamicPageCapacity: 4, maximumBufferBytes: 64 * 1024 });
  assert.equal(memory.directoryCapacity, 32, "the exact directory stays <=50% loaded");
  assert.ok(memory.totalBytes < 64 * 1024);
  assert.throws(() => planSolidWorldMemory({ staticPageCapacity: 8,
    dynamicPageCapacity: 4, maximumBufferBytes: memory.totalBytes - 1 }));

  const scene = getScenePreset("bounded-pool-transfer").create();
  const sceneWorld = solidWorldForScene(scene);
  assert.equal(sampleSolidWorld(sceneWorld, [8, 6, 7]).solidFraction, 1,
    "the transfer dam is an ordinary occupied voxel");
  assert.equal(sampleSolidWorld(sceneWorld, [8, 0, 7]).solidFraction, 0,
    "the dam opening reaches the pool floor");
  assert.equal(sampleSolidWorld(sceneWorld, [16, -1, 7]).solidFraction, 1,
    "the wider receiving pool retains its independent voxel floor");
  const solidLayout = createSparseCM12SolidOccupancyLayout({
    baseWords: 64, authoredPageCount: sceneWorld.pages.length,
  });
  const solidWords = packSparseCM12SolidOccupancy(solidLayout, sceneWorld, [-8, 0, 0]);
  assert.equal(solidWords[0], SPARSE_CM12_SOLID_OCCUPANCY_MAGIC);
  assert.ok(4 * (solidLayout.totalWords - solidLayout.baseWords) <= 8 * 1024 * 1024,
    "the exact solid cache has an independent hard memory bound");
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: sceneLatticeDimensions(scene), brickFineResolution: 8,
    maximumMacroSpanBricks: 1,
  });
  const presentation = sparseCM12FinePresentationPlan(atlas, 8, {
    signedSparseAddressing: true,
  });
  assert.deepEqual(presentation.plan.sampleDimensions, atlas.dimensions,
    "signed presentation must not advertise page-capacity headroom as extent");
  assert.notEqual(presentation.worklist[3]!
    & FINE_LEVELSET_SIGNED_SPARSE_ADDRESS_FLAG, 0);
});
