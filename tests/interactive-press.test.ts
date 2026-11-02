import assert from "node:assert/strict";
import test from "node:test";
import { RunScope } from "../lib/core/run-scope";
import { latestRefusal, pressRelease } from "../lib/features/interactive/press";
import type { CommandResult } from "../lib/interaction/contracts";

test("a released press places a clicked draft and lands a dragged base; a cancelled one does neither", () => {
  assert.equal(pressRelease({ kind: "placement", dragged: false }, false), "place");
  assert.equal(pressRelease({ kind: "placement", dragged: true }, false), "nothing", "a drag on a draft orbited");
  assert.equal(pressRelease({ kind: "move", dragged: true }, false), "move");
  assert.equal(pressRelease({ kind: "move", dragged: false }, false), "nothing", "a click on a base only selected it");
  assert.equal(pressRelease({ kind: "look", dragged: true }, false), "nothing");
  assert.equal(pressRelease({ kind: "select", dragged: false }, false), "nothing");
  for (const kind of ["placement", "move", "look", "select"] as const) {
    for (const dragged of [false, true]) assert.equal(pressRelease({ kind, dragged }, true), "nothing");
  }
});

function answer() {
  let settle!: (result: CommandResult) => void;
  const promise = new Promise<CommandResult>(done => { settle = done; });
  return { promise, refuse: (reason: string) => settle({ accepted: false, reason }),
    accept: () => settle({ accepted: true, documentRevision: 1, impact: "live" }) };
}

test("a draft's refusal is shown only while its request is the latest", async () => {
  const requests = new RunScope();
  const first = answer(), second = answer();
  const firstRefusal = latestRefusal(requests, () => first.promise);
  const secondRefusal = latestRefusal(requests, () => second.promise);
  second.accept();
  first.refuse("overlaps the crate");
  assert.equal(await secondRefusal, undefined);
  assert.equal(await firstRefusal, undefined, "a superseded refusal must not cover the accepted value");

  const third = answer();
  const thirdRefusal = latestRefusal(requests, () => third.promise);
  third.refuse("out of bounds");
  assert.equal(await thirdRefusal, "out of bounds");
});

test("a refusal that answers after the record changed or the row closed is dropped", async () => {
  const requests = new RunScope();
  const pending = answer();
  const refusal = latestRefusal(requests, () => pending.promise);
  requests.cancel();
  pending.refuse("stale");
  assert.equal(await refusal, undefined);
});
