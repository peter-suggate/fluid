import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import type { SceneDescription } from "../lib/core/model";
import { beginToolTransaction, type ToolHost } from "../lib/core/voxel-editor/transaction";
import type { ToolAction, VoxelToolPlugin } from "../lib/core/voxel-editor/plugin";

const ray = (x = 0) => ({ origin: { x, y: 1, z: 0 }, direction: { x: 0, y: -1, z: 0 } });
const releasePlugin: VoxelToolPlugin = {
  id: "fluid-transaction-fixture", version: 1, execution: "release",
  ui: { label: "Fluid fixture", hint: "Preview then release", group: "Fluid", order: 0, icon: "M0 0", controls: [] },
  unavailable: () => undefined,
  begin: () => ({ update(input) {
    if (input.origin.x === -1) return undefined;
    if (input.origin.x === -2) throw new Error("Invalid preview");
    return {
      patches: [],
      action: { kind: "fluid", edit: { operation: "add", shape: "ball", center_m: input.origin, radius_m: .1 } },
      highlight: { kind: "point", position_m: input.origin, radius_m: .1 },
      caption: "Preview",
    };
  } }),
};
function fixture(execute?: (action: ToolAction) => Promise<void>) {
  const initial = cloneScene(defaultScene);
  let scene = initial;
  const events: string[] = [];
  const actions: ToolAction[] = [];
  let pending = false;
  const host: ToolHost = {
    scene: () => scene,
    publish: async () => { events.push("publish"); throw new Error("Fluid preview must not author the scene"); },
    execute: async action => { actions.push(action); events.push("execute"); await execute?.(action); },
    begin: () => { pending = true; events.push("begin"); },
    finish: () => { pending = false; events.push("history"); },
    cancel: () => { pending = false; events.push("clear"); },
  };
  return { initial, scene: () => scene, replace: (next: SceneDescription) => { scene = next; },
    events, actions, pending: () => pending, transaction: beginToolTransaction(releasePlugin, host, ray())! };
}

test("fluid preview is read-only and release executes only the latest shape once", async () => {
  const f = fixture();
  await f.transaction.update(ray(1));
  await f.transaction.update(ray(2));
  assert.equal(f.scene(), f.initial);
  assert.deepEqual(f.events, ["begin"]);
  assert.equal(f.actions.length, 0);
  const released = f.transaction.finish();
  assert.equal(f.transaction.finish(), released);
  await released;
  assert.equal(f.actions.length, 1);
  assert.equal(f.actions[0].edit.center_m.x, 2);
  assert.deepEqual(f.events, ["begin", "execute", "clear"]);
  assert.equal(f.scene(), f.initial);
  assert.equal(f.pending(), false);
});

test("canceling a queued preview publishes no fluid or authored history", async () => {
  const f = fixture();
  const preview = f.transaction.update(ray(3));
  const canceled = f.transaction.finish(true);
  await Promise.all([preview, canceled]);
  assert.deepEqual(f.events, ["begin", "clear"]);
  assert.equal(f.actions.length, 0);
  assert.equal(f.scene(), f.initial);
  assert.equal(f.pending(), false);
});

test("cancel can upgrade a queued release before its action executes", async () => {
  const f = fixture();
  const preview = f.transaction.update(ray());
  const release = f.transaction.finish();
  assert.equal(f.transaction.finish(true), release);
  await Promise.all([preview, release]);
  assert.equal(f.actions.length, 0);
  assert.deepEqual(f.events, ["begin", "clear"]);
});

test("releasing after a missed or rejected preview cannot execute a stale shape", async () => {
  for (const x of [-1, -2]) {
    const f = fixture();
    await f.transaction.update(ray(4));
    if (x === -2) await assert.rejects(f.transaction.update(ray(x)), /Invalid preview/);
    else assert.equal(await f.transaction.update(ray(x)), undefined);
    await f.transaction.finish();
    assert.equal(f.actions.length, 0);
    assert.equal(f.scene(), f.initial);
    assert.deepEqual(f.events, ["begin", "clear"]);
  }
});

test("external scene replacement invalidates an uncommitted fluid preview", async () => {
  const f = fixture();
  await f.transaction.update(ray());
  const replacement = cloneScene(defaultScene);
  f.replace(replacement);
  await f.transaction.finish();
  assert.equal(f.actions.length, 0);
  assert.equal(f.scene(), replacement);
  assert.deepEqual(f.events, ["begin", "clear"]);
});

test("capacity rejection clears pending without creating authored history or retrying", async () => {
  const f = fixture(async () => { throw new Error("Fluid edit exceeds capacity"); });
  await f.transaction.update(ray());
  const release = f.transaction.finish();
  await assert.rejects(release, /exceeds capacity/);
  assert.equal(f.transaction.finish(), release);
  assert.equal(f.actions.length, 1);
  assert.equal(f.scene(), f.initial);
  assert.equal(f.pending(), false);
  assert.deepEqual(f.events, ["begin", "execute", "clear"]);
});

test("pending remains owned until accepted action completes", async () => {
  let resolve!: () => void;
  const gate = new Promise<void>(done => { resolve = done; });
  const f = fixture(() => gate);
  await f.transaction.update(ray());
  const release = f.transaction.finish();
  await Promise.resolve();
  assert.equal(f.pending(), true);
  assert.deepEqual(f.events, ["begin", "execute"]);
  resolve();
  await release;
  assert.equal(f.pending(), false);
  assert.equal(f.events.filter(event => event === "clear").length, 1);
});
