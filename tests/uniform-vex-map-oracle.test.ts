import assert from "node:assert/strict";
import test from "node:test";
import { assertUniformVexCaptureUsages, decodeUniformVexMapReceipt, uniformVexSnapshotLayout,
  type UniformVexSnapshotSource } from "../tools/implicit-density/uniform-vex-map-gpu";
import { inspectUniformVexFixture, uniformVexCorruptions, uniformVexFixture } from "./helpers/uniform-vex-map-fixture";

test("native padded/reversed fixture has an independently determined uniform physical map", () => {
  const q = uniformVexFixture(), oracle = inspectUniformVexFixture(q);
  assert.equal(oracle.anchor, 4); assert.equal(oracle.valid, 23); assert.equal(oracle.accepted, 24);
  assert.equal(oracle.coverage[0], 0);
  assert.deepEqual(oracle.coverage.slice(1), Uint32Array.from({ length: 23 }, (_, i) => i + 5));
  for (let axis = 0; axis < 3; axis++) assert.equal(oracle.translation[axis], -oracle.physicalVelocity[axis]! * oracle.dt);
});

for (const [name, fault, corrupt] of uniformVexCorruptions) test(name, () => {
  const q = uniformVexFixture(); corrupt(q); assert.throws(() => inspectUniformVexFixture(q), new RegExp(fault));
});

test("an unexecuted or failed map receipt cannot become a host affine map", () => {
  assert.throws(() => decodeUniformVexMapReceipt(new Uint32Array(40)), /not admitted/);
  const failed = new Uint32Array(40); failed.set([0x55564d31, 1, 2, 32]);
  assert.throws(() => decodeUniformVexMapReceipt(failed), /fault=32/);
  assert.throws(() => decodeUniformVexMapReceipt(new Uint32Array(39)), /not admitted/);
});

test("snapshot allocation is bounded and keeps every copied arena disjoint", () => {
  const l = uniformVexSnapshotLayout(32, 28);
  assert.equal(l.scmt, l.parameters + 64); assert.equal(l.frame, l.topology + 32 + 64);
  assert.equal(l.velocity, l.depth + 32); assert.equal(l.cells, l.velocity + 128);
  assert.equal(l.open, l.cells + 224); assert.equal(l.words, l.voxelOpen + 32);
  assert.throws(() => uniformVexSnapshotLayout(0, 0), /capacity/);
  assert.throws(() => uniformVexSnapshotLayout(1_048_577, 1), /capacity/);
  assert.throws(() => uniformVexSnapshotLayout(4, 5), /capacity/);
});

test("native capture rejects unreadable copies and non-uniform parameter buffers before encoding", () => {
  const copied = { usage: 4 }, parameters = { usage: 64 };
  const source = { topologyArena: copied, activity: copied, effectiveTransportVelocity: copied,
    state: copied, parameters, solidCellOpenBaseWords: 1, solidVoxelCellOpenBaseWords: 0 } as unknown as UniformVexSnapshotSource;
  assert.doesNotThrow(() => assertUniformVexCaptureUsages(source));
  assert.throws(() => assertUniformVexCaptureUsages({ ...source, activity: { usage: 8 } as GPUBuffer }), /COPY_SRC/);
  assert.throws(() => assertUniformVexCaptureUsages({ ...source, state: { usage: 8 } as GPUBuffer }), /COPY_SRC/);
  assert.throws(() => assertUniformVexCaptureUsages({ ...source, parameters: { usage: 4 } as GPUBuffer }), /UNIFORM/);
  const legacy = { ...source, topologyArena: {} as GPUBuffer, activity: {} as GPUBuffer,
    effectiveTransportVelocity: {} as GPUBuffer, state: {} as GPUBuffer, parameters: {} as GPUBuffer };
  assert.doesNotThrow(() => assertUniformVexCaptureUsages(legacy));
});
