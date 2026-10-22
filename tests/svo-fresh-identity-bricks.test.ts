import assert from "node:assert/strict";
import test from "node:test";
import { OctreeSparseBrickWorld } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";

test("new identity-transition leaves rebuild geometry and resample siblings of split terrain terminals", () => {
  const activated: unknown[] = [];
  const sibling = { minimum: [10, 0, 20], maximum: [26, 8, 36] };
  let sampledGeometry = false;
  const world = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    liveScenePrimitiveList: [], sceneWorldOrigin: [10, 0, 20], cellSize: [.5, .5, .5], brickSize: 8,
    terrainResamplingBounds(bounds: { identityOnly?: boolean }[]) {
      sampledGeometry = bounds.some(bound => !bound.identityOnly);
      return sampledGeometry ? [sibling] : [];
    }, raiseBackdropContentCeiling() {}, releaseUnclaimedTopologyBricks() {},
    growLiveDerivedAddressPlan() {}, topologyMirror: { activate(coordinate: unknown) { activated.push(coordinate); },
      releasable: () => false, hasPendingRecords: false },
    device: { queue: { writeBuffer() {} } }, topologyMutationWorklist: {},
  });
  const region = { minimum: [0, 0, 0], maximum: [100, 100, 100], identityOnly: true };
  const fresh = { x: 2, y: 0, z: 1 };
  world.stagePrimitivePublication(2, [region], [region], false, [fresh]);
  assert.deepEqual(activated, [fresh]);
  assert.ok(world.pendingScenePublication.dirtyRegions.some((bounds: typeof region) => !bounds.identityOnly
    && bounds.minimum.every((value, axis) => value <= [18, 0, 24][axis]!)
    && bounds.maximum.every((value, axis) => value >= [22, 4, 28][axis]!)),
  "every newly activated cell must get real occupancy, material and DMC data");
  assert.equal(sampledGeometry, true, "new leaf geometry must reach sampled-terrain split planning");
  assert.ok(world.pendingScenePublication.dirtyRegions.includes(sibling),
    "the split terrain terminal's siblings must be resampled in the same publication");
  world.stagePrimitivePublication(3, [region], [region], false, []);
  assert.deepEqual(world.pendingScenePublication.dirtyRegions, [region], "existing leaves retain the palette-only optimization");
});

test("promoted fresh geometry is refused during topology preflight before publication mutation", () => {
  const region = { minimum: [0, 0, 0], maximum: [100, 100, 100], identityOnly: true };
  const priorPublication = { revision: 7 }, priorSolid = {};
  let calls = 0;
  const world = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    sceneWorldOrigin: [10, 0, 20], cellSize: [.5, .5, .5], brickSize: 8,
    pendingScenePublication: priorPublication, solidWorld: priorSolid, renderSolidWorld: priorSolid,
    liveBricksToActivate: () => [{ x: 2, y: 0, z: 1 }],
    terrainResamplingBounds(bounds: { identityOnly?: boolean }[]) {
      calls++;
      if (bounds.some(bound => !bound.identityOnly)) throw new RangeError("Terrain edit exceeds resampling budget");
      return [];
    },
    topologyMirror: { demand() { assert.fail("terrain capacity must be validated before reserving topology"); } },
  });
  assert.throws(() => world.assertLiveTopologyCapacity([region]), /resampling budget/);
  assert.equal(calls, 2, "preflight checks the same promoted geometry that staging would publish");
  assert.equal(world.pendingScenePublication, priorPublication);
  assert.equal(world.solidWorld, priorSolid);
});
