import assert from "node:assert/strict";
import test from "node:test";
import { ownedBy, RunScope } from "../lib/core/run-scope";
import { createLabStore, labRunToReload } from "./lab-store";

/** A creation the test resolves by hand, so the race is the test's to order. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}

function controllerLedger() {
  const destroyed: string[] = [];
  return { destroyed, make: (name: string) => ({ name, destroy: async () => { destroyed.push(name); } }) };
}

test("a controller that arrives after its boot was unmounted is destroyed, never adopted", async () => {
  const runs = new RunScope(), ledger = controllerLedger();
  const created = deferred<ReturnType<typeof ledger.make>>();
  const boot = ownedBy(runs.start(), created.promise, controller => controller.destroy());
  runs.cancel("unmounted");
  created.resolve(ledger.make("boot"));
  assert.equal(await boot, undefined);
  assert.deepEqual(ledger.destroyed, ["boot"]);
});

test("a reload during the boot keeps only the newer load's controller", async () => {
  const runs = new RunScope(), ledger = controllerLedger();
  const first = deferred<ReturnType<typeof ledger.make>>(), second = deferred<ReturnType<typeof ledger.make>>();
  const stale = ownedBy(runs.start(), first.promise, controller => controller.destroy());
  const fresh = ownedBy(runs.start("reloaded"), second.promise, controller => controller.destroy());
  // Out of order: the newer controller lands first.
  second.resolve(ledger.make("second"));
  first.resolve(ledger.make("first"));
  assert.equal((await fresh)?.name, "second");
  assert.equal(await stale, undefined);
  assert.deepEqual(ledger.destroyed, ["first"]);
});

test("a back/forward during the boot is read by the boot, not raced by a reload", () => {
  const lab = createLabStore();
  const opened = { sceneId: lab.getState().sceneId, transport: lab.getState().transport };
  lab.getState().setSceneId("elsewhere");
  const address = { sceneId: lab.getState().sceneId, transport: lab.getState().transport };
  // Nothing loaded yet: the address change asks for no second load.
  assert.equal(labRunToReload(null, address), undefined);
  // The boot reads the address when its controller arrives, so it loads what was named.
  assert.notDeepEqual(address, opened);
  assert.equal(lab.getState().sceneId, "elsewhere");
});

test("once a run is loaded, an address naming another scene or transport reloads it", () => {
  const loaded = { sceneId: "dam", transport: "cellwise-remap" } as const;
  assert.equal(labRunToReload(loaded, { ...loaded }), undefined);
  assert.deepEqual(labRunToReload(loaded, { ...loaded, sceneId: "seep" }), { ...loaded, sceneId: "seep" });
  const other = { ...loaded, transport: "level-set-volume" } as const;
  assert.deepEqual(labRunToReload(loaded, other), other);
});
