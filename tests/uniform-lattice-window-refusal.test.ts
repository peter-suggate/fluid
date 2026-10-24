import assert from "node:assert/strict";
import test from "node:test";
import { createReservoirSeepScene } from "../lib/core/reservoir-seep-scene";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

test("a lattice past the page window is refused before the residency's host passes", () => {
  // The seep at 1.25 mm is 2048 cells wide: 64 pages, past the 62 a page-table axis holds.
  const scene = createReservoirSeepScene(0.00125);
  // No GPU: the refusal must come before anything but the timestamp probe touches the device.
  const device = { features: new Set<string>() } as unknown as GPUDevice;
  const Solver = WebGPUUniformReferenceSolver as unknown as new (...args: unknown[]) => unknown;
  const started = performance.now();
  assert.throws(() => new Solver(device, scene, "high", undefined, {}),
    (error: unknown) => error instanceof RangeError && /exceeds 62 pages/.test(error.message));
  // The dense residency pass over this lattice takes minutes; the refusal takes none of it.
  assert.ok(performance.now() - started < 5000, `refusal took ${(performance.now() - started).toFixed(0)} ms`);
});
