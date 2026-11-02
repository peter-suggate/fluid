import assert from "node:assert/strict";
import test from "node:test";
import { pickLanding } from "../components/viewport-pick";
import { RunScope } from "../lib/core/run-scope";

test("a pick's answer goes to the press that asked for it", () => {
  const picks = new RunScope();
  const first = picks.start();
  const held = { action: "pick", pick: first };
  assert.equal(pickLanding(held, first, true), "throw");
  assert.equal(pickLanding({ ...held, released: true }, first, true), "select-body", "a fast click selects without opening a throw");
  assert.equal(pickLanding(held, first, false), "orbit");
  assert.equal(pickLanding({ ...held, released: true }, first, false), "select");
});

test("a pick superseded, cancelled or outgrown by its press is stale", () => {
  const picks = new RunScope();
  const first = picks.start();
  const second = picks.start();
  // Same pointer id, newer press: the slow first readback must not start a throw on it.
  assert.equal(pickLanding({ action: "pick", pick: second }, first, true), "stale");
  assert.equal(pickLanding({ action: "pick", pick: second }, second, true), "throw");
  assert.equal(pickLanding(null, second, true), "stale", "pointercancel cleared the press");
  assert.equal(pickLanding({ action: "orbit" }, second, true), "stale", "the press already became an orbit");
});
