import test from "node:test";
import assert from "node:assert/strict";

import type { RunState } from "../lib/core/model";
import {
  applyHostRunState,
  hostResetPlan,
  paneResetPlan,
  type TransportPane,
} from "../lib/core/simulation/pane-transport";

/**
 * The transport is the host's.
 *
 * These are the rules the controller applies when a transport gesture arrives:
 * which panes hear it, and what each of them keeps. They are tested apart from
 * the controller because the controller is the singleton that imports the
 * method registry, the renderer and every store — nothing in it is reachable
 * from a CPU test — and apart from `session.ts` because a pane's realm is a
 * zustand store set that has no business in a node test either. `TransportPane`
 * is satisfied structurally by `Pick<PaneSession, "id" | "runtime">`.
 */

/** One pane's realm, narrowed to the half a transport gesture touches. */
function fakePane(id: string, runState: RunState = "paused"): TransportPane & { state: RunState; writes: number } {
  const pane = {
    id,
    state: runState,
    writes: 0,
    runtime: {
      getState: () => ({
        runState: pane.state,
        setRunState: (next: RunState) => { pane.state = next; pane.writes += 1; },
      }),
    },
  };
  return pane;
}

test("play and pause reach every attached pane", () => {
  const a = fakePane("a", "paused");
  const b = fakePane("b", "paused");
  applyHostRunState([a, b], "running");
  assert.equal(a.state, "running");
  assert.equal(b.state, "running", "pane B never heard the transport");
  applyHostRunState([a, b], "paused");
  assert.equal(a.state, "paused");
  assert.equal(b.state, "paused");
});

test("a pane already in the asked-for state is not written again", () => {
  const a = fakePane("a", "running");
  const b = fakePane("b", "paused");
  applyHostRunState([a, b], "running");
  assert.equal(a.writes, 0, "pane A was told to do what it was already doing");
  assert.equal(b.writes, 1);
  applyHostRunState([a, b], "running");
  assert.equal(b.writes, 1, "the fan-out is not idempotent");
});

test("single-pane mode is the one-pane case of the same fan-out", () => {
  const a = fakePane("a", "paused");
  applyHostRunState([a], "running");
  assert.equal(a.state, "running");
  assert.equal(a.writes, 1);
});

test("a reset of one pane re-seeds every other pane's clock", () => {
  // `PaneClockHost.reset` zeroes every pane, so a pane left un-re-seeded is
  // stranded above a target that rewound past it — and its solver will never
  // re-encode a time it already submitted.
  const plan = paneResetPlan(["a", "b"], "b");
  assert.deepEqual(plan.map((step) => step.id), ["b", "a"], "the pane that was asked resets first");
  assert.deepEqual(plan, [
    { id: "b", resynchronizing: false },
    { id: "a", resynchronizing: true },
  ]);
});

test("a pane re-seeded by someone else's reset keeps the run state it had", () => {
  const plan = paneResetPlan(["a", "b"], "a");
  const bystander = plan.find((step) => step.id === "b");
  assert.ok(bystander, "pane B was left out of pane A's reset");
  assert.equal(bystander.resynchronizing, true,
    "choosing a scene in one pane stopped the other");
});

test("a single-pane reset is exactly the reset that shipped", () => {
  assert.deepEqual(paneResetPlan(["a"], "a"), [{ id: "a", resynchronizing: false }]);
  assert.deepEqual(hostResetPlan(["a"]), [{ id: "a", resynchronizing: false }]);
});

test("RESET puts every pane at t = 0 and stops all of them", () => {
  const plan = hostResetPlan(["a", "b"]);
  assert.deepEqual(plan.map((step) => step.id).sort(), ["a", "b"]);
  assert.deepEqual(plan.filter((step) => step.resynchronizing), [],
    "RESET left a pane running");
});

test("the reset plans cover every attached pane exactly once", () => {
  for (const plan of [paneResetPlan(["a", "b"], "a"), hostResetPlan(["a", "b"])]) {
    assert.equal(new Set(plan.map((step) => step.id)).size, plan.length);
    assert.deepEqual([...plan.map((step) => step.id)].sort(), ["a", "b"]);
  }
});
