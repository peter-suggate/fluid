import assert from "node:assert/strict";
import test from "node:test";
import { detectPhysicsWasmCapabilities } from "./capabilities";

test("non-isolated hosts choose the real single-thread Wasm artifact", () => {
  const scope = {
    WebAssembly,
    SharedArrayBuffer,
    crossOriginIsolated: false,
    navigator: { hardwareConcurrency: 10 },
  } as unknown as typeof globalThis;
  const capabilities = detectPhysicsWasmCapabilities(scope);
  assert.equal(capabilities.wasm, true);
  assert.equal(capabilities.simd, true);
  assert.equal(capabilities.sharedMemory, false);
  assert.equal(capabilities.recommendedArtifact, "simd");
  assert.equal(capabilities.recommendedThreadCount, 1);
});

test("isolated shared-memory hosts reserve one core and cap the pool", () => {
  const scope = {
    WebAssembly,
    SharedArrayBuffer,
    crossOriginIsolated: true,
    navigator: { hardwareConcurrency: 12 },
  } as unknown as typeof globalThis;
  const capabilities = detectPhysicsWasmCapabilities(scope);
  assert.equal(capabilities.sharedMemory, true);
  assert.equal(capabilities.recommendedArtifact, "threaded");
  assert.equal(capabilities.recommendedThreadCount, 8);
});
