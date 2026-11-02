import assert from "node:assert/strict";
import test from "node:test";
import type { EditorRay } from "../lib/core/editor-entity";
import { VoxelStroke, type StrokeTransaction } from "../lib/core/voxel-editor/stroke";
import type { ToolUpdate } from "../lib/core/voxel-editor/plugin";

const at = (x: number): EditorRay => ({ origin: { x, y: 1, z: 0 }, direction: { x: 0, y: -1, z: 0 } });
const flush = () => new Promise(resolve => setTimeout(resolve, 0));

/** A transaction that records what it was asked, and a host whose frames the test fires by hand. */
function harness(options: { releaseOnly?: boolean; continues?: () => Promise<boolean> } = {}) {
  const log: string[] = [];
  const frames: (() => void)[] = [];
  const transaction: StrokeTransaction = {
    async update(sample) { log.push(`update ${sample.origin.x}`); return { patches: [] } as unknown as ToolUpdate; },
    advance: options.continues ?? (async () => { log.push("advance"); return false; }),
    async finish(cancelled) { log.push(`finish ${cancelled}`); },
  };
  const stroke = new VoxelStroke(7, transaction, options.releaseOnly ?? false, {
    outline: update => log.push(update ? "outline" : "outline down"),
    notice: error => log.push(`notice ${String(error)}`),
    frame: run => { frames.push(run); return () => { frames.splice(frames.indexOf(run), 1); }; },
    released: () => log.push("released"),
    settled: () => log.push("settled"),
  });
  const tick = async () => { for (const run of frames.splice(0)) run(); await flush(); };
  return { stroke, log, frames, tick };
}

test("a drag samples once a frame, latest winning, and a release that does not continue concludes it", async () => {
  const { stroke, log, tick } = harness();
  stroke.begin(at(0));
  await flush();
  stroke.sample(at(1), 7);
  stroke.sample(at(2), 7);
  stroke.sample(at(3), 99); // another pointer's move is not this stroke's
  await tick();
  stroke.release(at(4), false);
  await flush();
  assert.equal(stroke.phase, "concluded");
  assert.deepEqual(log, ["update 0", "outline", "update 2", "outline", "update 4", "outline", "advance",
    "released", "finish false", "settled"]);
});

test("a gesture that continues hovers on the bare pointer until the next press commits it", async () => {
  const { stroke, log, tick } = harness({ continues: async () => true });
  stroke.begin(at(0));
  stroke.release(at(1), false);
  await flush();
  assert.equal(stroke.phase, "hovering");
  stroke.sample(at(2), 12); // any pointer drives a hovering phase
  await tick();
  stroke.commit(at(3));
  await flush();
  assert.equal(stroke.phase, "concluded");
  assert.deepEqual(log, ["update 0", "outline", "update 1", "outline", "update 2", "outline", "update 3", "outline",
    "released", "finish false", "settled"]);
});

test("an end while the release is still being answered is not taken back to hovering", async () => {
  let answer!: (continues: boolean) => void;
  const { stroke, log } = harness({ continues: () => new Promise(done => { answer = done; }) });
  stroke.begin(at(0));
  stroke.release(at(1), false);
  await flush();
  assert.equal(stroke.phase, "releasing");
  stroke.end(true); // Escape
  answer(true);
  await flush();
  await flush();
  assert.equal(stroke.phase, "concluded");
  assert.ok(log.includes("finish true"), "an Escape rolls the stroke back");
});

test("a cancelled pointer ends the stroke without authoring where it was taken away", async () => {
  const { stroke, log } = harness();
  stroke.begin(at(0));
  await flush();
  stroke.release(at(9), true);
  await flush();
  assert.equal(stroke.phase, "concluded");
  assert.ok(!log.includes("update 9"));
  assert.ok(log.includes("finish true"));
});

test("a runtime that can no longer validate drops the waiting sample but keeps what was accepted", async () => {
  const { stroke, log, frames } = harness();
  stroke.begin(at(0));
  await flush();
  stroke.sample(at(5), 7);
  assert.equal(frames.length, 1);
  stroke.end(false, true);
  await flush();
  assert.equal(frames.length, 0, "the waiting frame is cancelled");
  assert.ok(!log.includes("update 5"));
  assert.ok(log.includes("finish false"));
});

test("a viewport unmounting mid-drag finishes the transaction once and samples nothing more", async () => {
  const { stroke, log, frames, tick } = harness({ releaseOnly: true });
  stroke.begin(at(0));
  await flush();
  stroke.sample(at(1), 7);
  stroke.abandon();
  stroke.abandon();
  stroke.end(true);
  await tick();
  assert.equal(frames.length, 0);
  assert.equal(stroke.phase, "concluded");
  assert.deepEqual(log.filter(line => line.startsWith("finish")), ["finish true"], "a release-only proposal is discarded");
  assert.ok(!log.includes("update 1"));
  assert.ok(!log.includes("released"), "the host clears its own reference when it unmounts");
});
