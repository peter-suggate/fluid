import assert from "node:assert/strict";
import test from "node:test";

import { WebGPURigidBodySystem } from "../lib/core/webgpu-rigid-body";

type Internals = Pick<WebGPURigidBodySystem, "readPoses" | "pick">;

test("a destroyed rigid world answers pose reads and picks without touching its buffers", async () => {
  const rigid = Object.assign(Object.create(WebGPURigidBodySystem.prototype), {
    disposed: true, bodyCount: 2,
    device: new Proxy({}, { get: () => assert.fail("a destroyed world must not encode a readback") }),
  }) as unknown as Internals;
  assert.equal(await rigid.readPoses(), undefined);
  assert.equal(await rigid.pick({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }), undefined);
});
