import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import type { SceneDescription } from "../lib/core/model";
import { beginToolTransaction, type ToolHost } from "../lib/core/voxel-editor/transaction";
import type { VoxelToolPlugin } from "../lib/core/voxel-editor/plugin";

const ray = (x = 0) => ({ origin: { x, y: 0, z: 0 }, direction: { x: 0, y: -1, z: 0 } });
const plugin: VoxelToolPlugin = {
  id: "lifecycle-fixture", version: 1,
  ui: { label: "Fixture", hint: "Fixture", group: "Fixture", order: 0, icon: "M0 0", controls: [] },
  unavailable: () => undefined,
  begin: () => ({ update: (input) => ({
    patches: [{ operation: "fill", minimum: [input.origin.x, 0, 0], maximumExclusive: [input.origin.x + 1, 1, 1] }],
    highlight: { kind: "point", position_m: { x: 0, y: 0, z: 0 }, radius_m: 1 },
    caption: "Fixture",
  }) }),
};
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}
function fixture(publish?: (next: SceneDescription, before: SceneDescription) => Promise<void>) {
  const base = cloneScene(defaultScene);
  let scene = base;
  const events: string[] = [];
  const host: ToolHost = {
    scene: () => scene,
    async publish(next) {
      const before = scene;
      events.push("preflight");
      await publish?.(next, before);
      if (scene !== before) throw new Error("Scene changed during preflight");
      scene = next;
      events.push(next === base ? "rollback" : "publish");
    },
    begin: () => { events.push("begin"); },
    finish: () => { events.push("finish"); },
    cancel: () => { events.push("cancel"); },
  };
  return { base, events, scene: () => scene, replace: (next: SceneDescription) => { scene = next; },
    transaction: beginToolTransaction(plugin, host, ray())! };
}

test("cancel waits for an outstanding preflight and rolls back exactly once", async () => {
  const gate = deferred();
  const f = fixture(() => gate.promise);
  const update = f.transaction.update(ray());
  await Promise.resolve();
  const finish = f.transaction.finish(true);
  assert.equal(f.transaction.finish(), finish);
  assert.equal(await f.transaction.update(ray(3)), undefined);
  assert.equal(f.scene(), f.base);
  gate.resolve();
  await Promise.all([update, finish]);
  assert.equal(f.scene(), f.base);
  assert.deepEqual(f.events, ["begin", "preflight", "publish", "preflight", "rollback", "cancel"]);
});

test("queued updates and release publish in order before one history entry", async () => {
  const gate = deferred();
  const f = fixture(() => gate.promise);
  const first = f.transaction.update(ray());
  const second = f.transaction.update(ray(3));
  const finish = f.transaction.finish();
  await Promise.resolve();
  assert.deepEqual(f.events, ["begin", "preflight"]);
  gate.resolve();
  await Promise.all([first, second, finish]);
  assert.deepEqual(f.events, ["begin", "preflight", "publish", "preflight", "publish", "finish"]);
  assert.equal(f.scene().solidVoxels?.at(-1)?.minimum[0], 3);
});

test("external undo or import during preflight cannot be overwritten or recorded as this stroke", async () => {
  const gate = deferred();
  const f = fixture(() => gate.promise);
  const update = f.transaction.update(ray());
  await Promise.resolve();
  const external = cloneScene(defaultScene);
  f.replace(external);
  const finish = f.transaction.finish(true);
  gate.resolve();
  await assert.rejects(update, /Scene changed/);
  await finish;
  assert.equal(f.scene(), external);
  assert.deepEqual(f.events, ["begin", "preflight", "cancel"]);
});

test("a rejected final sample retains the accepted stroke as one undo entry", async () => {
  let reject = false;
  const f = fixture(async () => { if (reject) throw new Error("capacity"); });
  await f.transaction.update(ray());
  const accepted = f.scene();
  reject = true;
  const update = f.transaction.update(ray(3));
  const finish = f.transaction.finish();
  await assert.rejects(update, /capacity/);
  await finish;
  assert.equal(f.scene(), accepted);
  assert.equal(f.events.filter((event) => event === "finish").length, 1);
});

test("failed rollback keeps accepted geometry undoable, without taking ownership of an external scene", async () => {
  for (const replace of [false, true]) {
    let reject = false;
    const external = cloneScene(defaultScene);
    const f = fixture(async () => {
      if (reject) { if (replace) f.replace(external); throw new Error("rollback rejected"); }
    });
    await f.transaction.update(ray());
    const accepted = f.scene();
    reject = true;
    await assert.rejects(f.transaction.finish(true), /rollback rejected/);
    assert.equal(f.scene(), replace ? external : accepted);
    assert.equal(f.events.at(-1), replace ? "cancel" : "finish");
  }
});
