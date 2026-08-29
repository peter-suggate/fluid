import assert from "node:assert/strict";
import test from "node:test";
import {
  acquireBrowserGPULease,
  manualGPUControlTargetsPane,
  type BrowserGPULeaseResult,
  type PaneId,
} from "../lib/core/gpu-startup";
import { createPaneLeaseBroker, MAXIMUM_PANE_LEASES } from "../lib/core/session/pane-lease";

type LockManager = Parameters<typeof acquireBrowserGPULease>[0];

/**
 * A `navigator.locks` stand-in with the one behaviour the safety lock relies
 * on: `ifAvailable` hands the callback `null` instead of queueing when the
 * name is already held. `heldElsewhere` plays the second browser tab.
 */
function fakeLockManager(options: { heldElsewhere?: boolean } = {}) {
  let held = options.heldElsewhere === true;
  let requests = 0;
  const manager: NonNullable<LockManager> = {
    async request(name, _options, callback) {
      requests += 1;
      if (held) {
        await callback(null);
        return;
      }
      held = true;
      await callback({ name });
      held = false;
    },
  };
  return {
    manager,
    requests: () => requests,
    held: () => held,
  };
}

function brokerOver(locks: LockManager, maximumLeases = MAXIMUM_PANE_LEASES) {
  return createPaneLeaseBroker(() => acquireBrowserGPULease(locks), maximumLeases);
}

/** A count cap is about counting; `PaneId` only names the two panes that exist. */
const thirdPane = "c" as PaneId;

/**
 * `release()` resolves the promise the lock callback is parked on; the browser
 * only takes the name back once that callback has actually returned. The
 * broker's own ledger is synchronous, the hand-back is not.
 */
const lockHandback = () => new Promise((resolve) => setTimeout(resolve, 0));

/** Read a refusal's reason without a narrowing dance at every call site. */
function refusalMessage(lease: BrowserGPULeaseResult): string {
  return lease.status === "acquired" ? "<acquired>" : lease.message;
}

test("two panes share one page-exclusive lock", async () => {
  const locks = fakeLockManager();
  const broker = brokerOver(locks.manager);

  const a = await broker.acquire("a");
  const b = await broker.acquire("b");

  assert.equal(a.status, "acquired");
  assert.equal(b.status, "acquired");
  assert.deepEqual([...broker.heldPanes()], ["a", "b"]);
  assert.equal(broker.lockHeld(), true);
  assert.equal(locks.requests(), 1, "the second pane rides the first pane's lock");
});

test("a third lease is refused while both are out", async () => {
  const locks = fakeLockManager();
  const broker = brokerOver(locks.manager);
  await broker.acquire("a");
  await broker.acquire("b");

  const third = await broker.acquire(thirdPane);

  assert.equal(third.status, "exhausted");
  assert.match(refusalMessage(third), /All 2 in-page WebGPU pane leases are held/);
  assert.deepEqual([...broker.heldPanes()], ["a", "b"], "a refusal takes no slot");
  assert.equal(locks.requests(), 1);
});

test("a pane cannot take a second lease for itself", async () => {
  const broker = brokerOver(fakeLockManager().manager);
  await broker.acquire("a");

  const again = await broker.acquire("a");

  assert.equal(again.status, "exhausted");
  assert.deepEqual([...broker.heldPanes()], ["a"]);
});

test("the lock is released only when the last lease is released", async () => {
  const locks = fakeLockManager();
  const broker = brokerOver(locks.manager);
  const a = await broker.acquire("a");
  const b = await broker.acquire("b");
  assert.equal(a.status, "acquired");
  assert.equal(b.status, "acquired");

  if (a.status === "acquired") a.release();
  await lockHandback();
  assert.equal(broker.lockHeld(), true, "pane B still holds the device");
  assert.equal(locks.held(), true, "one pane leaving must not free the page's lock");
  assert.deepEqual([...broker.heldPanes()], ["b"]);

  if (b.status === "acquired") b.release();
  assert.equal(broker.lockHeld(), false);
  assert.deepEqual([...broker.heldPanes()], []);
  await lockHandback();
  assert.equal(locks.held(), false, "the Web Lock is handed back to the browser");

  const reacquired = await broker.acquire("a");
  assert.equal(reacquired.status, "acquired");
  assert.equal(locks.requests(), 2, "the next session asks for the lock again");
});

test("releasing twice, or releasing a stale handle, leaves the live lease alone", async () => {
  const broker = brokerOver(fakeLockManager().manager);
  const first = await broker.acquire("a");
  assert.equal(first.status, "acquired");
  if (first.status === "acquired") {
    first.release();
    first.release();
  }
  await lockHandback();
  const second = await broker.acquire("a");
  assert.equal(second.status, "acquired");

  if (first.status === "acquired") first.release();

  assert.deepEqual([...broker.heldPanes()], ["a"], "the stale handle does not drop the new lease");
  assert.equal(broker.lockHeld(), true);
});

test("panes starting in the same tick make one lock request, and overshoot is refused", async () => {
  const locks = fakeLockManager();
  const broker = brokerOver(locks.manager);

  const [a, b, c] = await Promise.all([
    broker.acquire("a"),
    broker.acquire("b"),
    broker.acquire(thirdPane),
  ]);

  assert.equal(a.status, "acquired");
  assert.equal(b.status, "acquired");
  assert.equal(c.status, "exhausted", "the cap is claimed before the lock is awaited");
  assert.equal(locks.requests(), 1);
});

test("a second tab still owns the lock, and says so unchanged", async () => {
  const locks = fakeLockManager({ heldElsewhere: true });
  const broker = brokerOver(locks.manager);

  const a = await broker.acquire("a");
  const b = await broker.acquire("b");

  for (const lease of [a, b] as readonly BrowserGPULeaseResult[]) {
    assert.equal(lease.status, "held");
    assert.equal(refusalMessage(lease), "Another Fluid Lab tab owns the WebGPU safety lock");
  }
  assert.deepEqual([...broker.heldPanes()], [], "a refused pane holds nothing to release");
  assert.equal(broker.lockHeld(), false);
});

test("a browser without Web Locks refuses every pane the same way it refused one", async () => {
  const broker = brokerOver(undefined);

  const a = await broker.acquire("a");
  const b = await broker.acquire("b");

  assert.equal(a.status, "unsupported");
  assert.equal(b.status, "unsupported");
  assert.deepEqual([...broker.heldPanes()], []);
});

test("a manual start/stop acts on the pane it names, or on every pane", () => {
  const everyPane = new CustomEvent("fluid-lab:start-gpu", { detail: {} });
  assert.equal(manualGPUControlTargetsPane(everyPane, "a"), true);
  assert.equal(manualGPUControlTargetsPane(everyPane, "b"), true);

  const paneB = new CustomEvent("fluid-lab:stop-gpu", { detail: { paneId: "b" } });
  assert.equal(manualGPUControlTargetsPane(paneB, "a"), false);
  assert.equal(manualGPUControlTargetsPane(paneB, "b"), true);

  // The page's START/STOP buttons dispatch with no detail at all.
  assert.equal(manualGPUControlTargetsPane(new Event("fluid-lab:start-gpu"), "a"), true);
});
