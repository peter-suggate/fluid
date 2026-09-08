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
  let pending = false;
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
    begin: () => { pending = true; events.push("begin"); },
    finish: () => { pending = false; events.push("finish"); },
    cancel: () => { pending = false; events.push("cancel"); },
  };
  return { base, events, pending: () => pending, scene: () => scene, replace: (next: SceneDescription) => { scene = next; },
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
  assert.equal(f.pending(), true);
  gate.resolve();
  await Promise.all([update, finish]);
  assert.equal(f.pending(), false);
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


test("abandoning a gesture before its first sample clears the pending state", async () => {
  const f = fixture();
  assert.equal(f.pending(), true);
  await f.transaction.finish(true);
  assert.equal(f.pending(), false);
  assert.deepEqual(f.events, ["begin", "cancel"]);
});

test("controller save and history commands cannot bypass an outstanding voxel stroke", async () => {
  await import("../lib/methods");
  const { simulation } = await import("../lib/core/simulation/controller");
  const session = simulation.session();
  const entry = { scene: session.scene.getState().scene, presetId: session.scene.getState().presetId, label: "Prior edit" };
  session.history.setState({ past: [entry], future: [entry] });
  const before = session.history.getState();
  session.ui.setState({ voxelStrokePending: true });
  try {
    assert.equal(simulation.undo(), false);
    assert.equal(simulation.redo(), false);
    simulation.saveNamedScene("Should wait");
    assert.equal(session.history.getState(), before);
    assert.equal(session.scene.getState().scene, entry.scene);
    assert.match(JSON.stringify(session.runtime.getState()), /Finish the voxel stroke before saving/);
  } finally {
    session.ui.setState({ voxelStrokePending: false });
    session.history.getState().clear();
  }
});

test("invalid JSON imports preserve the scene and explain the rejected file", async () => {
  await import("../lib/methods");
  const { simulation } = await import("../lib/core/simulation/controller");
  const session = simulation.session();
  const scene = session.scene.getState().scene;
  const history = session.history.getState();
  for (const contents of ["{", "{}"] ) {
    simulation.importScene("invalid.json", contents);
    assert.equal(session.scene.getState().scene, scene);
    assert.equal(session.history.getState(), history);
    const status = JSON.stringify(session.runtime.getState());
    assert.match(status, /Scene import failed:/);
    assert.doesNotMatch(status, /Cannot read properties/);
  }
});

test("an asynchronous import finishing during a stroke preserves document, tool and history", async () => {
  await import("../lib/methods");
  const { simulation } = await import("../lib/core/simulation/controller");
  const { serializeScene } = await import("../lib/core/model");
  const session = simulation.session();
  const scene = session.scene.getState().scene, history = session.history.getState();
  const gate = deferred();
  const contents = serializeScene(scene);
  const read = gate.promise.then(() => simulation.importScene("delayed.json", contents));
  const previousTool = session.ui.getState().voxelToolId;
  session.ui.setState({ voxelStrokePending: true });
  try {
    gate.resolve(); await read;
    assert.equal(session.scene.getState().scene, scene);
    assert.equal(session.history.getState(), history);
    assert.equal(session.ui.getState().voxelToolId, previousTool);
    assert.equal(session.ui.getState().voxelStrokePending, true);
    assert.match(JSON.stringify(session.runtime.getState()), /Finish the voxel stroke before importing/);
  } finally { session.ui.setState({ voxelStrokePending: false }); }
});
