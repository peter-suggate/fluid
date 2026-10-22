import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createEmptyScene } from "../lib/core/empty-scene";
import type { SceneDescription } from "../lib/core/model";
import { createPaneSession } from "../lib/core/session/session";
import type { SolidWorldVoxelPatch } from "../lib/core/solid-world";
import type { VoxelToolPlugin } from "../lib/core/voxel-editor/plugin";
import { sessionToolHost } from "../lib/core/voxel-editor/session-host";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";

const ray = (x: number) => ({ origin: { x, y: 4, z: 0 }, direction: { x: 0, y: -1, z: 0 } });
const plugin: VoxelToolPlugin = {
  id: "live-fixture", version: 1, execution: "live",
  ui: { label: "Live stroke", hint: "Fixture", group: "Fixture", order: 0, icon: "M0 0", controls: [] },
  unavailable: () => undefined,
  begin: () => {
    let patches: readonly SolidWorldVoxelPatch[] = [];
    return { update: input => {
      patches = [...patches, { operation: "fill", minimum: [input.origin.x, 0, 0], maximumExclusive: [input.origin.x + 1, 1, 1] }];
      return { patches, caption: "Fixture", highlight: { kind: "point", position_m: { x: 0, y: 0, z: 0 }, radius_m: 1 } };
    } };
  },
};
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(done => { resolve = done; });
  return { promise, resolve };
}
function fixture(validate?: (next: SceneDescription, base: SceneDescription) => Promise<void>) {
  const session = createPaneSession("a");
  const base = createEmptyScene({ extents_m: { x: 1, y: 1, z: 1 }, finestCellSize_m: .05 });
  session.scene.getState().setScene(base);
  const publications: { next: SceneDescription; base: SceneDescription }[] = [];
  let runtime = base;
  const host = sessionToolHost(session, async (next, before) => {
    assert.equal(runtime, before, "validation is serialized against runtime authority");
    publications.push({ next, base: before });
    await validate?.(next, before);
    runtime = next;
  });
  return { session, base, publications, runtime: () => runtime,
    transaction: beginToolTransaction(plugin, host, ray(0))! };
}

test("live samples publish during the drag with one history snapshot on release", async () => {
  const f = fixture();
  await f.transaction.update(ray(1));
  const first = f.session.scene.getState().scene;
  assert.notEqual(first, f.base);
  assert.equal(f.runtime(), first);
  assert.equal(f.session.history.getState().past.length, 0);
  assert.equal(f.session.ui.getState().voxelStrokePending, true);
  await f.transaction.update(ray(2));
  assert.equal(f.publications.length, 2);
  assert.equal(f.publications[1]!.base, first);
  assert.equal(f.runtime().solidVoxels.length, f.base.solidVoxels.length + 2);
  await f.transaction.finish();
  assert.equal(f.session.history.getState().past.length, 1);
  assert.equal(f.session.history.getState().past[0]!.scene.solidVoxels, f.base.solidVoxels);
  assert.equal(f.session.ui.getState().voxelStrokePending, false);
});

test("slow validation coalesces queued samples into one suffix and release drains it", async () => {
  const gate = deferred();
  let first = true;
  const f = fixture(async () => { if (first) { first = false; await gate.promise; } });
  const updates = [f.transaction.update(ray(1))];
  await tick();
  for (let x = 2; x <= 20; x++) updates.push(f.transaction.update(ray(x)));
  assert.equal(f.publications.length, 1, "only one validation is in flight");
  const completion = f.transaction.finish();
  gate.resolve();
  await Promise.all(updates); await completion;
  assert.equal(f.publications.length, 2, "the latest proposal carries every skipped sample once");
  assert.equal(f.runtime().solidVoxels.length, f.base.solidVoxels.length + 20);
  assert.deepEqual(f.runtime().solidVoxels.slice(f.base.solidVoxels.length).map(patch => patch.minimum[0]),
    Array.from({ length: 20 }, (_, index) => index + 1));
  assert.equal(f.session.history.getState().past.length, 1);
});

test("cancel during validation drops queued work, rolls back accepted samples, and preserves redo", async () => {
  const gate = deferred();
  let first = true;
  const f = fixture(async () => { if (first) { first = false; await gate.promise; } });
  const future = [{ label: "Redo", scene: f.base, presetId: "fixture" }];
  f.session.history.setState({ future });
  const firstUpdate = f.transaction.update(ray(1));
  await tick();
  const queuedUpdate = f.transaction.update(ray(2));
  const completion = f.transaction.finish(true);
  assert.equal(f.transaction.finish(), completion);
  gate.resolve();
  await Promise.all([firstUpdate, queuedUpdate, completion]);
  assert.equal(f.publications.length, 2, "one accepted sample, then the rollback; queued sample is discarded");
  assert.equal(f.publications[1]!.next, f.base);
  assert.equal(f.runtime(), f.base);
  assert.equal(f.session.scene.getState().scene, f.base);
  assert.equal(f.session.history.getState().past.length, 0);
  assert.equal(f.session.history.getState().future, future);
  assert.equal(f.session.ui.getState().voxelStrokePending, false);
});

test("refused later live sample rolls back the accepted prefix and releases the gesture", async () => {
  let count = 0;
  const f = fixture(async () => { if (++count === 2) throw new Error("capacity"); });
  await f.transaction.update(ray(1));
  await assert.rejects(f.transaction.update(ray(2)), /capacity/);
  await assert.rejects(f.transaction.finish(), /capacity/);
  assert.equal(f.runtime(), f.base);
  assert.equal(f.session.scene.getState().scene, f.base);
  assert.equal(f.session.history.getState().past.length, 0);
  assert.equal(f.session.ui.getState().voxelStrokePending, false);
});

test("cancelling a live stroke cannot overwrite an external replacement", async () => {
  const f = fixture();
  await f.transaction.update(ray(1));
  const replacement = { ...f.base, name: "External" };
  f.session.scene.getState().setScene(replacement);
  await f.transaction.finish(true);
  assert.equal(f.session.scene.getState().scene, replacement);
  assert.equal(f.publications.length, 1);
  assert.equal(f.session.history.getState().past.length, 0);
  assert.equal(f.session.ui.getState().voxelStrokePending, false);
});

test("a runtime that refuses rollback leaves the accepted prefix undoable and reports the failure", async () => {
  let count = 0;
  const f = fixture(async () => { if (++count === 2) throw new Error("runtime lost"); });
  await f.transaction.update(ray(1));
  const accepted = f.runtime();
  await assert.rejects(f.transaction.finish(true), /runtime lost/);
  assert.equal(f.session.scene.getState().scene, accepted);
  assert.equal(f.session.history.getState().past.length, 1);
  assert.equal(f.session.history.getState().past[0]!.scene.solidVoxels, f.base.solidVoxels);
  assert.equal(f.session.ui.getState().voxelStrokePending, false);
});

test("a live gesture cancelled before its first accepted sample authors nothing", async () => {
  const f = fixture();
  const update = f.transaction.update(ray(1));
  await f.transaction.finish(true); await update;
  assert.equal(f.publications.length, 0);
  assert.equal(f.runtime(), f.base);
  assert.equal(f.session.history.getState().past.length, 0);
  assert.equal(f.session.ui.getState().voxelStrokePending, false);
});
