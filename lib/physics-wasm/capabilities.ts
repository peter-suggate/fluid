import type { PhysicsWasmArtifact } from "./protocol";

const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 5, 1, 96, 0, 1, 123,
  3, 2, 1, 0,
  10, 22, 1, 20, 0, 253, 12,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
]);

export interface PhysicsWasmCapabilities {
  readonly wasm: boolean;
  readonly simd: boolean;
  readonly sharedMemory: boolean;
  readonly crossOriginIsolated: boolean;
  readonly hardwareConcurrency: number;
  readonly recommendedArtifact: PhysicsWasmArtifact;
  readonly recommendedThreadCount: number;
}

export function detectPhysicsWasmCapabilities(scope: typeof globalThis = globalThis): PhysicsWasmCapabilities {
  const wasm = typeof scope.WebAssembly === "object";
  const simd = wasm && scope.WebAssembly.validate(SIMD_PROBE);
  const isolated = scope.crossOriginIsolated === true;
  let sharedMemory = false;
  if (wasm && isolated && typeof scope.SharedArrayBuffer === "function") {
    try {
      sharedMemory = new scope.WebAssembly.Memory({ initial: 1, maximum: 1, shared: true }).buffer
        instanceof scope.SharedArrayBuffer;
    } catch { sharedMemory = false; }
  }
  const concurrency = Math.max(1, Math.floor(scope.navigator?.hardwareConcurrency ?? 1));
  const threadCount = Math.max(1, Math.min(8, concurrency - 1));
  return {
    wasm,
    simd,
    sharedMemory,
    crossOriginIsolated: isolated,
    hardwareConcurrency: concurrency,
    recommendedArtifact: sharedMemory ? "threaded" : simd ? "simd" : "scalar",
    recommendedThreadCount: sharedMemory ? threadCount : 1,
  };
}
