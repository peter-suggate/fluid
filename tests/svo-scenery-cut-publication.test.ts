import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createEmptyScene } from "../lib/core/empty-scene";
import { cloneScene, parseScene, serializeScene } from "../lib/core/model";
import { terrainFieldStamp } from "../lib/core/live-terrain-overlay";
import { validateSolidPatch } from "../lib/core/solid-patch/compose";
import { sameSolidPatchList } from "../lib/core/solid-patch-list";
import { sceneWithSolidStroke, solidVoxelPatchesKey, solidWorldContentStamp, solidWorldForScene, solidWorldVoxelPatchBounds_m,
  type SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { environmentProxyPrimitives, type EnvironmentProxyPrimitive } from "../lib/core/voxel-environments";
import { successorEnvironmentCatalog } from "../lib/svo/features/scene-publication/scene-environment-catalog";
import { OctreeSparseBrickWorld } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";
import { svoRenderResidual } from "../lib/svo/features/scene-publication/svo-planar-boundary";
import { sceneSculptCuts, sceneryPrimitiveAffectedBySculptCuts, sculptPrimitiveBounds, svoSculptChangeBounds } from "../lib/svo/features/scene-publication/svo-sculpt-change-bounds";

const legacy: SolidWorldVoxelPatch = { operation: "clear", minimum: [1, 2, 3], maximumExclusive: [4, 5, 6] };
const cut: SolidWorldVoxelPatch = { ...legacy, affectsScenery: true };
const scene = () => ({ ...createEmptyScene({ extents_m: { x: 1, y: 1, z: 1 }, finestCellSize_m: .05 }), solidVoxels: [] as SolidWorldVoxelPatch[] });

test("scenery clear marker persists, changes content identity, and preserves legacy stamps", () => {
  assert.equal(solidVoxelPatchesKey([legacy]), "1:b4f08319:bf91d6b1");
  assert.equal(solidVoxelPatchesKey([{ ...legacy, kind: "ellipsoid" }]), "1:49a12f71:8eaad1af");
  assert.equal(solidVoxelPatchesKey([{ ...legacy, affectsScenery: false }]), solidVoxelPatchesKey([legacy]));
  assert.equal(sameSolidPatchList([legacy], [cut]), false);
  const base = scene(), marked = sceneWithSolidStroke(base, [cut]);
  assert.notEqual(solidWorldContentStamp(marked), solidWorldContentStamp(sceneWithSolidStroke(base, [legacy])));
  for (const restored of [cloneScene(marked), structuredClone(marked), parseScene(serializeScene(marked))]) {
    assert.equal(restored.solidVoxels[0]!.affectsScenery, true);
    assert.equal(solidWorldContentStamp(restored), solidWorldContentStamp(marked));
  }
  assert.throws(() => validateSolidPatch({ ...cut, operation: "fill" }), /Only clear/);
  assert.throws(() => validateSolidPatch({ ...cut, affectsScenery: "yes" } as unknown as SolidWorldVoxelPatch), /boolean/);
});

test("scenery clears survive render residual, undo, and redo without occupied static pages", () => {
  const base = scene(), marked = sceneWithSolidStroke(base, [cut]);
  const world = solidWorldForScene(base), next = solidWorldForScene(marked);
  assert.equal(world.pages.length, 0); assert.equal(next.pages.length, 0);
  const before = svoRenderResidual(base, world), after = svoRenderResidual(marked, next, before);
  assert.deepEqual(after.residual.patches, [cut]);
  const undo = svoRenderResidual(base, world, after);
  assert.equal(undo.residual.patches.length, 0);
  assert.deepEqual(svoRenderResidual(marked, next, undo).residual.patches, [cut]);
});

const imported = {
  kind: "field-program", key: "import", ownerIndex: 0, group: "test", tags: [],
  center_m: { x: 0, y: .5, z: 0 }, halfExtent_m: { x: .1, y: .1, z: .1 },
  aabb_m: { min: { x: -.1, y: .4, z: -.1 }, max: { x: .1, y: .6, z: .1 } },
  program: { sampledVolume: {} }, material: { colorLinear: [1, 1, 1], roughness: 1, emission: 0 },
} as unknown as EnvironmentProxyPrimitive;
const stamp: SolidWorldVoxelPatch = { operation: "clear", kind: "ellipsoid", affectsScenery: true,
  minimum: [12, 10, 10], maximumExclusive: [13, 11, 11] };

test("first and last imported cuts dirty the full owner using shared conservative ownership bounds", () => {
  const base = scene();
  // x=.10..15 touches the padded native box boundary.
  assert.equal(sceneryPrimitiveAffectedBySculptCuts(sceneSculptCuts(base, [stamp]), imported), true);
  const first = svoSculptChangeBounds(base, [], [stamp], [imported]);
  assert.deepEqual(first.dirtyBounds, [solidWorldVoxelPatchBounds_m(base, stamp), { ...sculptPrimitiveBounds(imported), identityOnly: true }]);
  assert.deepEqual(first.addedPrimitiveBounds, [sculptPrimitiveBounds(imported)]);
  const second = { ...stamp, minimum: [10, 10, 10] as const, maximumExclusive: [11, 11, 11] as const };
  assert.deepEqual(svoSculptChangeBounds(base, [stamp], [stamp, second], [imported]).dirtyBounds,
    [solidWorldVoxelPatchBounds_m(base, second)], "continuing a stroke updates only its new volume");
  assert.deepEqual(svoSculptChangeBounds(base, [stamp], [], [imported]).dirtyBounds, first.dirtyBounds);
  assert.deepEqual(svoSculptChangeBounds(base, [stamp], structuredClone([stamp]), [imported]).dirtyBounds, []);
  assert.equal(svoSculptChangeBounds(base, [], [legacy], [imported]).changed, false);
});

test("scenery-only cut staging uploads originals and publishes dirty regions across undo/redo", () => {
  const base = scene(), marked = sceneWithSolidStroke(base, [cut]);
  const solid = solidWorldForScene(base), residual = svoRenderResidual(base, solid);
  const catalog = successorEnvironmentCatalog(undefined, base), authored = environmentProxyPrimitives(catalog, true);
  const uploads: (readonly SolidWorldVoxelPatch[])[] = [];
  const regions: unknown[] = [];
  const world = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    destroyed: false, sceneRevision: 1, solidWorld: solid, renderSolidWorld: residual.residual,
    renderResidual: residual, solidWorldStamp: solidWorldContentStamp(base), terrainFieldStamp: terrainFieldStamp(base),
    stagedScene: base, stagedDocument: base, stagedCatalog: catalog, stagedAuthored: authored, stagedStaticBodies: "",
    liveScenePrimitiveStates: new Map(), liveScenePrimitives: new Map(), liveScenePrimitiveList: [],
    environmentDetailCellSize_m: 0, planarTerminalsBaked: false,
    proxyVoxelizer: { setSolidWorld() {}, setSculptCuts(patches: SolidWorldVoxelPatch[]) { uploads.push(patches); } },
    assertLiveTopologyCapacity() { return []; }, stageAppearanceTables() { return false; },
    stagePrimitivePublication(_revision: number, dirty: unknown) { regions.push(dirty); },
    advanceSceneRevision() { return ++this.sceneRevision; },
  }) as OctreeSparseBrickWorld;
  assert.equal(world.stageSceneUpdate(marked), true);
  assert.equal(world.stageSceneUpdate(marked), false);
  assert.equal(world.stageSceneUpdate(base), true);
  assert.equal(world.stageSceneUpdate(marked), true);
  assert.deepEqual(uploads, [marked.solidVoxels, base.solidVoxels, marked.solidVoxels]);
  assert.equal(regions.length, 3);
  for (const dirty of regions) assert.ok((dirty as unknown[]).length > 0);
});

test("a cut temporarily restores an affected planar proxy to the live voxel list", () => {
  const base = scene(), carved = sceneWithSolidStroke(base, [stamp]);
  const plate = { kind: "box", key: "plate", ownerIndex: 0, group: "test", tags: [],
    center_m: { x: 0, y: .5, z: 0 }, halfSize_m: { x: .4, y: .01, z: .4 },
    material: { colorLinear: [1, 1, 1], roughness: 1, emission: 0 },
    aabb_m: { min: { x: -.4, y: .49, z: -.4 }, max: { x: .4, y: .51, z: .4 } },
  } as EnvironmentProxyPrimitive;
  const world = Object.assign(Object.create(OctreeSparseBrickWorld.prototype), {
    liveScenePrimitiveStates: new Map(), liveScenePrimitives: new Map(),
  });
  const before = world.listLivePrimitives(base, [plate], [], []);
  assert.equal(before.primitives.size, 0, "the intact plate uses its exact planar representation");
  const cuts = svoSculptChangeBounds(base, [], [stamp], [plate]);
  assert.equal(cuts.relistPlanar, true);
  assert.deepEqual(cuts.addedPrimitiveBounds, [sculptPrimitiveBounds(plate)]);
  const during = world.listLivePrimitives(carved, [plate], [], []);
  assert.equal(during.primitives.has("plate"), true, "the voxel proxy supplies newly cut faces");
  world.liveScenePrimitiveStates = during.states; world.liveScenePrimitives = during.primitives;
  const dirty: unknown[] = [];
  const undo = world.listLivePrimitives(base, [plate], dirty, []);
  assert.equal(undo.primitives.size, 0);
  assert.equal(dirty.length, 1, "undo invalidates the removed voxel copy");
});
