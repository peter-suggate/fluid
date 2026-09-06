import assert from "node:assert/strict";
import test from "node:test";
import { driveCooperativeBuild } from "../lib/core/cooperative-build";

test("GPU stages fence before the generator consumes their results", async () => {
  let result = 0;
  function* build() {
    yield new Promise<void>(resolve => setImmediate(() => { result = 42; resolve(); }));
    return result;
  }
  assert.equal(await driveCooperativeBuild(build()), 42);
});

test("failed GPU stages unwind temporary build resources", async () => {
  let disposed = false;
  function* build() {
    try { yield Promise.reject(new Error("device lost")); }
    finally { disposed = true; }
  }
  await assert.rejects(driveCooperativeBuild(build()), /device lost/);
  assert.equal(disposed, true);
});

test("supersession during a GPU fence never resumes stale publication", async () => {
  const abort = new AbortController();
  let resumed = false, disposed = false;
  function* build() {
    try {
      yield new Promise<void>(resolve => setImmediate(() => { abort.abort(); resolve(); }));
      resumed = true;
    } finally { disposed = true; }
  }
  await assert.rejects(driveCooperativeBuild(build(), { signal: abort.signal }), { name: "AbortError" });
  assert.equal(resumed, false);
  assert.equal(disposed, true);
});
