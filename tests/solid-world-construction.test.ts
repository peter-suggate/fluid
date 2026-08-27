import assert from "node:assert/strict";
import test from "node:test";
import {
  boxSolidVoxelShell,
  createSolidWorld,
  fluidColliderVoxelPatchesForScene,
  fluidSolidWorldForScene,
  planSolidWorldMemory,
  planarBoundaryForSolidWorldVoxelPatch,
  sampleSolidWorld,
  solidWorldForScene,
  solidWorldContentStamp,
  solidWorldPageAddress,
  solidWorldVoxelPatchBounds_m,
  SolidWorldDirectory,
} from "../lib/core/solid-world";
import { sparseCM12FinePresentationPlan } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { FINE_LEVELSET_SIGNED_SPARSE_ADDRESS_FLAG } from
  "../lib/core/fine-levelset-brick-abi";
import { findSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  createSparseCM12SolidOccupancyLayout,
  packSparseCM12SolidOccupancy,
  SPARSE_CM12_SOLID_OCCUPANCY_MAGIC,
} from "../lib/methods/adaptive-mass/sparse-cm12-solid-occupancy";

test("thin authored tank voxels compile to exact planar slabs", () => {
  const scene = getScenePreset("bounded-pool-transfer").create();
  scene.container.width_m = 16;
  scene.container.height_m = 8.4;
  scene.container.depth_m = 16.4;
  scene.voxelDomain.finestCellSize_m = 1;
  const shell = boxSolidVoxelShell([16, 8, 16]);

  const floor = planarBoundaryForSolidWorldVoxelPatch(scene, shell[0]!);
  assert.ok(floor);
  assert.deepEqual(solidWorldVoxelPatchBounds_m(scene, shell[0]!), {
    minimum: [-8, -1.05, -8.2],
    maximum: [8, 0, 8.2],
  });
  assert.deepEqual(floor.center_m, [0, -0.525, 0]);
  assert.deepEqual(floor.normal, [0, 1, 0]);
  assert.deepEqual(floor.tangentU, [1, 0, 0]);
  assert.deepEqual(floor.tangentV, [0, 0, 1]);
  assert.equal(floor.halfExtentU_m, 8);
  assert.equal(floor.halfExtentV_m, 8.2);
  assert.equal(floor.halfThickness_m, 0.525,
    "the slab retains one exact anisotropic voxel of thickness");

  const accepted = shell.map((patch) =>
    planarBoundaryForSolidWorldVoxelPatch(scene, patch));
  assert.equal(accepted.filter(Boolean).length, 5,
    "all five open tank faces meet the canonical thin-slab criterion");
  assert.equal(accepted[1]!.halfThickness_m, 0.5,
    "the x wall retains its realized x-cell thickness");
  assert.equal(accepted[1]!.halfExtentU_m, 4.2,
    "the x wall uses the anisotropic y extent");
  assert.equal(accepted[1]!.halfExtentV_m, 8.2);

  assert.equal(planarBoundaryForSolidWorldVoxelPatch(scene, {
    operation: "clear", minimum: [0, 0, 0], maximumExclusive: [16, 1, 16],
  }), null, "subtractive edits remain residual geometry");
  assert.equal(planarBoundaryForSolidWorldVoxelPatch(scene, {
    operation: "fill", minimum: [0, 0, 0], maximumExclusive: [7, 1, 16],
  }), null, "a box below the exact 8:1 second-axis threshold remains residual");
});

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
  assert.ok(presentation.plan.domainOrigin.every((value) => value === 0),
    "signed WDR page coordinates are already global fine-lattice addresses");
  assert.notEqual(presentation.worklist[3]!
    & FINE_LEVELSET_SIGNED_SPARSE_ADDRESS_FLAG, 0);
  assert.throws(() => sparseCM12FinePresentationPlan(atlas, 8, {
    signedSparseAddressing: true,
    coordinateOffsetPages: [1, 0, 0],
  }), /cannot advertise a padded address lattice/,
  "signed consumers must never add a second domain-origin page offset");
});

test("the finite studio slab, not y=0, supports outside-tank fluid", () => {
  const definition = findSceneDefinition("hydrostatic-power-large-offset");
  assert.ok(definition);
  const scene = sceneDocument(definition);
  const patches = fluidColliderVoxelPatchesForScene(scene);
  assert.equal(patches.length, 1);
  assert.ok(patches[0]!.minimum[1] < 0);
  assert.equal(patches[0]!.maximumExclusive[1], 0,
    "the collider ends at its authored top face without filling y=0 cells");
  const world = fluidSolidWorldForScene(scene);
  assert.equal(world.pages.length, solidWorldForScene(scene).pages.length,
    "a planar collider must remain a compact region, not expand into voxel pages");
  assert.deepEqual(world.regions, patches);
  const outsideX = sceneLatticeDimensions(scene)[0] + 1;
  assert.equal(sampleSolidWorld(world, [outsideX, -1, 0]).solidFraction, 1);
  assert.equal(sampleSolidWorld(world, [outsideX, 0, 0]).solidFraction, 0,
    "empty space above the slab remains open");

  const longDam = sceneDocument(findSceneDefinition(
    "sparse-cm12-ladder-long-dam",
  )!);
  const longDamWorld = fluidSolidWorldForScene(longDam);
  const layout = createSparseCM12SolidOccupancyLayout({ baseWords: 64,
    authoredPageCount: longDamWorld.pages.length,
    authoredRegionCount: longDamWorld.regions?.length ?? 0 });
  assert.ok(layout.totalBytes < 8 * 1024 * 1024,
    "the long dam's planar stage floor must stay inside the SolidWorld budget");
  assert.doesNotThrow(() => packSparseCM12SolidOccupancy(
    layout, longDamWorld, [0, 0, 0]));
});
