import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sampleSolidWorld, sceneWithSolidStroke, solidWorldForScene } from "../lib/core/solid-world";
import { createSolidEditSceneCache } from "../lib/core/voxel-editor/solid-edit-scene-cache";

const patch = (x: number) => ({ operation: "fill" as const, minimum: [x, 2, 2] as const,
  maximumExclusive: [x + 1, 3, 3] as const, materialId: 2 });

test("worker history reuses exact accepted images for Undo, Redo and branch replacements", () => {
  const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .05 });
  const baseWorld = solidWorldForScene(scene);
  const first = sceneWithSolidStroke(scene, [patch(2)]);
  const firstWorld = solidWorldForScene(first);
  const second = sceneWithSolidStroke(first, [patch(3)]);
  const secondWorld = solidWorldForScene(second);
  const cache = createSolidEditSceneCache();
  for (const endpoint of [scene, first, second]) cache.remember(endpoint);
  for (const [target, previous, world] of [[first, second, firstWorld], [second, first, secondWorld], [scene, second, baseWorld]] as const) {
    const clone = structuredClone(target);
    cache.prepare(clone, previous);
    assert.equal(solidWorldForScene(clone), world, "history must retain its compiled immutable image");
  }
  const branch = sceneWithSolidStroke(scene, [patch(5)]);
  const clone = structuredClone(branch);
  cache.prepare(clone, second);
  const world = solidWorldForScene(clone);
  assert.equal(sampleSolidWorld(world, [5, 2, 2]).solidFraction, 1);
  assert.equal(sampleSolidWorld(world, [2, 2, 2]).solidFraction, 0);
});

test("append-only worker samples retain untouched page identity", () => {
  const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .05 });
  const world = solidWorldForScene(scene);
  const target = structuredClone(sceneWithSolidStroke(scene, [patch(2)]));
  createSolidEditSceneCache().prepare(target, scene);
  const next = solidWorldForScene(target);
  const untouched = world.pages.find(page => page.coordinate[0] < 0)!;
  assert.ok(untouched);
  assert.ok(next.pages.includes(untouched));
});

test("worker coordinator publishes structured-clone history only after real acceptance receipt", async () => {
  const { createWorkerSolidEditAcceptance } = await import("../lib/core/voxel-editor/worker-solid-edit-acceptance");
  const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .05 });
  const baseWorld = solidWorldForScene(scene);
  let current = { revision: 7, document: scene };
  let release: (() => void) | undefined;
  const coordinator = createWorkerSolidEditAcceptance({
    readScene: () => current,
    writeScene: value => { current = value; },
    accept: async (_scene, stillCurrent) => {
      await new Promise<void>(resolve => { release = resolve; });
      assert.equal(stillCurrent(), true);
    },
  });
  const target = sceneWithSolidStroke(scene, [patch(2)]);
  const request = structuredClone({ scene: target, base: scene });
  const pending = coordinator.accept(request);
  assert.equal(current.document, scene);
  release!(); await pending;
  assert.equal(current.document, request.scene);
  assert.equal(current.revision, 7, "queued worker draws retain the same revision");
  const undo = structuredClone({ scene, base: target });
  const undoing = coordinator.accept(undo);
  release!(); await undoing;
  assert.equal(solidWorldForScene(current.document), baseWorld);
});

test("worker acceptance rejection and replacement preserve the current document", async () => {
  const { createWorkerSolidEditAcceptance } = await import("../lib/core/voxel-editor/worker-solid-edit-acceptance");
  const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .05 });
  const target = sceneWithSolidStroke(scene, [patch(2)]);
  let current = { revision: 7, document: scene };
  const rejected = createWorkerSolidEditAcceptance({
    readScene: () => current, writeScene: value => { current = value; },
    accept: async () => { throw new Error("Solid insertion overlaps moving water"); },
  });
  await assert.rejects(rejected.accept(structuredClone({ scene: target, base: scene })), /overlaps/);
  assert.equal(current.document, scene);
  assert.equal(rejected.preparedScene, undefined);
  const replacement = { revision: 8, document: structuredClone(scene) };
  const stale = createWorkerSolidEditAcceptance({
    readScene: () => current, writeScene: value => { current = value; },
    accept: async (_scene, stillCurrent) => { current = replacement; assert.equal(stillCurrent(), false); },
  });
  await assert.rejects(stale.accept(structuredClone({ scene: target, base: scene })), /newer scene remains active/);
  assert.equal(current, replacement);
  assert.equal(stale.preparedScene, undefined);
});
