import assert from "node:assert/strict";
import test from "node:test";

import { WebGPURigidBodySystem } from "../lib/core/webgpu-rigid-body";

type Internals = Pick<WebGPURigidBodySystem, "encode" | "finishNativeStep" | "readPoses" | "pick"> & { nativeStep?: { dt_s: number } };

test("concurrent finishers share the one native step encode owed them", async () => {
  const ran: number[] = [];
  let finish = (_: boolean) => {};
  const rigid = Object.assign(Object.create(WebGPURigidBodySystem.prototype), {
    nativeStep: { dt_s: 0.02 }, disposed: false,
    runNativeStep: (dt: number) => { ran.push(dt); return new Promise<boolean>(resolve => { finish = resolve; }); },
  }) as unknown as Internals;
  const first = rigid.finishNativeStep(), second = rigid.finishNativeStep();
  assert.equal(first, second);
  assert.throws(() => rigid.encode({} as GPUCommandEncoder, 0.02, 1), /Finish the native rigid step/);
  finish(true);
  assert.deepEqual(await Promise.all([first, second]), [true, true]);
  assert.deepEqual(ran, [0.02], "the contact solve runs once per encoded step");
  assert.equal(rigid.nativeStep, undefined);
  assert.equal(await rigid.finishNativeStep(), false, "nothing is owed after it completes");
});

test("a destroyed rigid world answers pose reads and picks without touching its buffers", async () => {
  const rigid = Object.assign(Object.create(WebGPURigidBodySystem.prototype), {
    disposed: true, bodyCount: 2,
    device: new Proxy({}, { get: () => assert.fail("a destroyed world must not encode a readback") }),
  }) as unknown as Internals;
  assert.equal(await rigid.readPoses(), undefined);
  assert.equal(await rigid.pick({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }), undefined);
});
