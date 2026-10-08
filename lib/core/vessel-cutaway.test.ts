import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, type SceneDescription } from "./model";
import { createNbflipPour } from "./nbflip-paper-scenes";
import { createSolidWorld, sampleSolidWorld, solidWorldForScene, type SolidWorldVoxelPatch } from "./solid-world";
import { vesselCutawayPatches } from "./vessel-cutaway";
import { buildSvoSolidWorldPlanarBoundaryCatalog, svoPlanarResidualSolidWorld } from "../svo/features/scene-publication/svo-planar-boundary";

/** A 32-cell lattice holding `solidVoxels`, viewed from in front (+z) and above. */
function sceneWith(solidVoxels: SolidWorldVoxelPatch[]): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.container = { ...scene.container, width_m: 3.2, height_m: 3.2, depth_m: 3.2 };
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: 0.1 };
  scene.solidVoxels = solidVoxels;
  scene.cutaway = { eye_m: { x: 0, y: 4, z: 12 } };
  return scene;
}
const fill = (minimum: [number, number, number], maximumExclusive: [number, number, number]): SolidWorldVoxelPatch =>
  ({ operation: "fill", minimum, maximumExclusive, materialId: 17 });
/** An open-topped box on the ground: walls two thick around x, z in [8, 24), 16 tall. */
const openBox = () => [
  fill([8, 0, 8], [24, 16, 10]), fill([8, 0, 22], [24, 16, 24]),
  fill([8, 0, 10], [10, 16, 22]), fill([22, 0, 10], [24, 16, 22]),
];

test("the pour's glass is drawn without the half that faces the footage camera", () => {
  const scene = createNbflipPour();
  const world = solidWorldForScene(scene);
  const residual = svoPlanarResidualSolidWorld(world,
    buildSvoSolidWorldPlanarBoundaryCatalog(scene, world.patches, 0, { promoteEditablePatches: false }), scene);
  const near = [64, 100, 116] as const, far = [64, 100, 11] as const;
  assert.notEqual(sampleSolidWorld(world, near).materialId, 0, "the solver keeps the near wall");
  assert.equal(sampleSolidWorld(residual, near).materialId, 0, "the near wall is not drawn");
  assert.notEqual(sampleSolidWorld(residual, far).materialId, 0, "the far wall is drawn");
  delete scene.cutaway;
  assert.equal(svoPlanarResidualSolidWorld(world,
    buildSvoSolidWorldPlanarBoundaryCatalog(scene, world.patches, 0, { promoteEditablePatches: false }), scene), world);
});

test("only the wall between the eye and the cavity is cut", () => {
  const scene = sceneWith([...openBox(), fill([14, 0, 14], [18, 24, 18])]);
  const drawn = createSolidWorld([...scene.solidVoxels, ...vesselCutawayPatches(scene, scene.solidVoxels)]);
  const solid = (x: number, y: number, z: number) => sampleSolidWorld(drawn, [x, y, z]).materialId !== 0;
  assert.equal(solid(16, 8, 22), false, "near wall, inner layer");
  assert.equal(solid(16, 8, 23), false, "near wall, outer layer");
  assert.equal(solid(16, 8, 8), true, "far wall");
  assert.equal(solid(8, 8, 12), true, "side wall, far end");
  assert.equal(solid(23, 8, 12), true, "other side wall, far end");
  for (const z of [14, 17]) assert.equal(solid(16, 8, z), true, "a pillar standing in the cavity is not wall");
});

test("solids that hold nothing are left whole", () => {
  // A slab, and a box with a doorway down to the ground on its far side.
  for (const solids of [[fill([8, 0, 8], [24, 16, 24])], [...openBox(), { operation: "clear", minimum: [14, 0, 8], maximumExclusive: [18, 16, 10] } as const]]) {
    const scene = sceneWith(solids);
    assert.deepEqual(vesselCutawayPatches(scene, scene.solidVoxels), []);
  }
});

test("a thick wall and a lid are cut through, and the walls beside the eye's line are kept whole", () => {
  // A closed box: cavity x in [6, 26), z in [2, 18), 16 tall, behind a wall twelve thick.
  const scene = sceneWith([
    fill([4, 0, 0], [28, 16, 2]), fill([4, 0, 18], [28, 16, 30]),
    fill([4, 0, 2], [6, 16, 18]), fill([26, 0, 2], [28, 16, 18]),
    fill([4, 16, 0], [28, 19, 30]),
  ]);
  const drawn = createSolidWorld([...scene.solidVoxels, ...vesselCutawayPatches(scene, scene.solidVoxels)]);
  const solid = (x: number, y: number, z: number) => sampleSolidWorld(drawn, [x, y, z]).materialId !== 0;
  for (const z of [18, 23, 29]) assert.equal(solid(16, 8, z), false, `near wall at depth ${z - 18}`);
  for (const y of [16, 18]) assert.equal(solid(16, y, 10), false, "lid");
  assert.equal(solid(16, 8, 1), true, "far wall");
  for (const y of [0, 8, 15]) for (const z of [3, 8, 12]) {
    assert.equal(solid(5, y, z), true, `side wall at ${y}, ${z}`);
    assert.equal(solid(26, y, z), true, `other side wall at ${y}, ${z}`);
  }
});
