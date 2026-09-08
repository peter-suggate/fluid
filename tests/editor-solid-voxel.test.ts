import assert from "node:assert/strict";
import test from "node:test";
import { pickSolidVoxel, containerShellContains, solidVoxelClearPreview,
  withSolidVoxelClearRegion } from "../lib/core/editor-solid-voxel";
import { cloneScene, defaultScene, markSceneRevision } from "../lib/core/model";
import { sceneCellSizes_m, sceneLatticeDimensions, solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { sampleSolidWorld, solidWorldForScene, sceneWithSolidStroke, reuseSolidWorld } from "../lib/core/solid-world";

test("generic clear selection uses exact occupied voxels and preserves prior edits", () => {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [
    { operation: "fill", minimum: [3, 4, -2], maximumExclusive: [5, 6, 0],
      materialId: 2 },
    { operation: "clear", minimum: [4, 4, -2], maximumExclusive: [5, 5, -1] },
    { operation: "fill", minimum: [1_000_000, 1, 20],
      maximumExclusive: [1_000_001, 2, 21],
      materialId: 3 },
  ];
  const region = { minimum: [3, 4, -2], maximumExclusive: [5, 6, 0] } as const;
  const preview = solidVoxelClearPreview(scene, region);
  assert.equal(preview.affectedCount, 7);
  assert.equal(preview.coordinates.length, 7);

  const prior = structuredClone(scene.solidVoxels);
  scene.solidVoxels = withSolidVoxelClearRegion(scene.solidVoxels, region);
  assert.deepEqual(scene.solidVoxels.slice(0, prior.length), prior);
  const world = solidWorldForScene(scene);
  assert.equal(sampleSolidWorld(world, [3, 4, -2]).solidFraction, 0);
  assert.equal(sampleSolidWorld(world, [1_000_000, 1, 20]).materialId, 3);

  scene.solidVoxels = prior;
  const [hx, hy, hz] = sceneCellSizes_m(scene);
  const originX = -0.5 * scene.container.width_m;
  const originZ = -0.5 * scene.container.depth_m;
  const picked = pickSolidVoxel(scene, {
    origin: { x: originX + 2 * hx, y: 4.5 * hy, z: originZ - 1.5 * hz },
    direction: { x: 1, y: 0, z: 0 },
  });
  assert.deepEqual(picked?.coordinate, [3, 4, -2]);
  assert.equal(picked?.faceAxis, 0);
  assert.equal(picked?.faceSign, -1);
});

test("preview detail truncation never limits a large generic clear edit", () => {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [{ operation: "fill", minimum: [-700, 9, 31],
    maximumExclusive: [-187, 11, 32], materialId: 2 }];
  const region = { minimum: [-700, 9, 31],
    maximumExclusive: [-187, 11, 32] } as const;
  const preview = solidVoxelClearPreview(scene, region);
  assert.equal(preview.affectedCount, 1_026);
  assert.equal(preview.coordinates.length, 512);
  assert.equal(preview.truncated, true);

  scene.solidVoxels = withSolidVoxelClearRegion(scene.solidVoxels, region);
  assert.equal(scene.solidVoxels.at(-1)?.operation, "clear");
  assert.equal(sampleSolidWorld(solidWorldForScene(scene), [-188, 10, 31])
    .solidFraction, 0);
});


test("lazy terrain ray picking agrees with materialized ordered fill/clear authority", () => {
  const scene = cloneScene(defaultScene);
  scene.terrain = { baseHeight_m: .15, features: [] };
  scene.solidVoxels = [
    { operation: "clear", minimum: [4, 0, 4], maximumExclusive: [5, 8, 5] },
    { operation: "fill", minimum: [5, 6, 4], maximumExclusive: [6, 7, 5], materialId: 2 },
  ];
  const h = sceneCellSizes_m(scene), world = solidWorldForScene(scene);
  for (const x of [3, 4, 5, 6]) {
    const ray = { origin: { x: -scene.container.width_m / 2 + (x + .5) * h[0], y: 1, z: -scene.container.depth_m / 2 + 4.5 * h[2] }, direction: { x: 0, y: -1, z: 0 } };
    assert.deepEqual(pickSolidVoxel(scene, ray), pickSolidVoxel(scene, ray, world));
  }
});

test("cold terrain input only samples the ray and authoring does not materialize the domain", () => {
  const scene = cloneScene(defaultScene);
  scene.container.width_m = 10000; scene.container.depth_m = 10000;
  let samples = 0;
  scene.terrain = { get baseHeight_m() { samples++; return .2; }, features: [] };
  scene.solidVoxels = [];
  const hit = pickSolidVoxel(scene, { origin: { x: 0, y: .3, z: 0 }, direction: { x: 0, y: -1, z: 0 } });
  assert.ok(hit);
  assert.ok(samples < 64, `one ray sampled ${samples} terrain columns`);
  const before = samples;
  const next = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [1, 2, 3], maximumExclusive: [2, 3, 4], materialId: 2 }]);
  assert.equal(samples, before, "a cold document stamp must not bake terrain or evaluate its heightfield");
  assert.equal(next.solidVoxels.length, 1);
});

test("bounded hover misses safely while tool picks reject, and near faces match the page oracle", () => {
  const scene = cloneScene(defaultScene);
  scene.terrain = undefined;
  scene.solidVoxels = [{ operation: "fill", minimum: [2, 2, 2], maximumExclusive: [3, 3, 3], materialId: 2 }];
  const h = sceneCellSizes_m(scene), world = solidWorldForScene(scene);
  for (const epsilon of [-1e-7, 1e-7]) {
    const ray = { origin: { x: -scene.container.width_m / 2 + (2 + epsilon) * h[0], y: 2.5 * h[1], z: -scene.container.depth_m / 2 + 2.5 * h[2] }, direction: { x: 1, y: 0, z: 0 } };
    assert.deepEqual(pickSolidVoxel(scene, ray), pickSolidVoxel(scene, ray, world));
  }
  scene.solidVoxels = [{ operation: "fill", minimum: [1000000, 2, 2], maximumExclusive: [1000001, 3, 3] }];
  const ray = { origin: { x: -scene.container.width_m / 2, y: 2.5 * h[1], z: -scene.container.depth_m / 2 + 2.5 * h[2] }, direction: { x: 1, y: 0, z: 0 } };
  assert.equal(pickSolidVoxel(scene, ray), undefined);
  assert.throws(() => pickSolidVoxel(scene, ray, undefined, { rejectBudgetExhaustion: true }), /bounded 4096-cell/);
});


test("worker document clones reuse the constructed terrain and first stroke only replaces changed pages", () => {
  const source = cloneScene(defaultScene);
  source.terrain = { baseHeight_m: .15, features: [] };
  source.solidVoxels = [];
  assert.equal(markSceneRevision(source), source, "worker stamping preserves renderer construction identity");
  const residentWorld = solidWorldForScene(source);
  const renderClone = structuredClone(source);
  reuseSolidWorld(source, renderClone);
  assert.equal(solidWorldForScene(renderClone), residentWorld, "uniform-only render publication cannot trigger a terrain rebake");
  const strokeBase = structuredClone(renderClone);
  reuseSolidWorld(renderClone, strokeBase);
  const prepared = sceneWithSolidStroke(strokeBase, [{ operation: "fill", minimum: [4, 20, 4], maximumExclusive: [5, 21, 5], materialId: 2 }]);
  const messageScene = structuredClone(prepared);
  reuseSolidWorld(prepared, messageScene);
  const proposal = solidWorldForScene(messageScene);
  assert.equal(solidWorldForScene(strokeBase), residentWorld);
  assert.ok(residentWorld.pages.length > 0);
  assert.ok(residentWorld.pages.every(page => proposal.pages.includes(page)), "the entire original terrain remains the same immutable page objects");
});


test("constant-space shell picking matches every authored box and sphere shell cell", () => {
  for (const shape of ["box", "sphere"] as const) for (const top of ["open", "closed"] as const) {
    const scene = cloneScene(defaultScene);
    scene.container = { ...scene.container, width_m: .4, height_m: .4, depth_m: .4, shape, top };
    scene.voxelDomain.finestCellSize_m = .05;
    const dimensions = sceneLatticeDimensions(scene), patches = solidVoxelShellForScene(scene);
    for (let z = -2; z <= dimensions[2] + 1; z++) for (let y = -2; y <= dimensions[1] + 1; y++)
      for (let x = -2; x <= dimensions[0] + 1; x++) {
        const q = [x, y, z] as const;
        const expected = patches.some(patch => q.every((value, axis) => value >= patch.minimum[axis]! && value < patch.maximumExclusive[axis]!));
        assert.equal(containerShellContains(scene, q), expected, `${shape}/${top} ${q}`);
      }
  }
});
