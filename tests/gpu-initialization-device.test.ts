import assert from "node:assert/strict";
import test from "node:test";
import { gpuInitializationDevice } from "../lib/core/gpu-initialization-device";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";

function fixture() {
  let allocated = 0, destroyed = 0;
  const raw = { createBuffer() { allocated++; return { destroy() { destroyed++; } }; } } as unknown as GPUDevice;
  const device = managedGPUDevice(raw, { requireWorkerRealm: false });
  const controller = new AbortController();
  const scope = gpuInitializationDevice(device, controller.signal);
  return { device, controller, scope, counts: () => ({ allocated, destroyed }) };
}

test("a constructor failure releases allocations made before its owner exists", () => {
  const f = fixture();
  assert.throws(() => {
    try {
      f.scope.device.createBuffer({ size: 64, usage: 0 });
      throw new Error("constructor failed");
    } catch (error) { f.scope.rollback(); throw error; }
  }, /constructor failed/);
  assert.deepEqual(f.counts(), { allocated: 1, destroyed: 1 });
  assert.throws(() => f.scope.device.createBuffer({ size: 64, usage: 0 }), { name: "AbortError" });
  f.scope.rollback();
  assert.equal(f.counts().destroyed, 1);
});

test("an aborted build cannot allocate from a late continuation or commit", () => {
  const f = fixture();
  f.controller.abort();
  assert.throws(() => f.scope.device.createBuffer({ size: 64, usage: 0 }), { name: "AbortError" });
  assert.throws(() => f.scope.commit(), { name: "AbortError" });
  assert.deepEqual(f.counts(), { allocated: 0, destroyed: 0 });
});

test("a committed solver survives cancellation of the next scene and shares the scheduler", () => {
  const f = fixture();
  assert.equal(gpuCompilationManagerFor(f.scope.device), gpuCompilationManagerFor(f.device));
  f.scope.device.createBuffer({ size: 64, usage: 0 });
  f.scope.commit();
  f.controller.abort();
  f.scope.device.createBuffer({ size: 64, usage: 0 });
  assert.deepEqual(f.counts(), { allocated: 2, destroyed: 0 });
});
