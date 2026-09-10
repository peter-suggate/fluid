import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SimulationFailureDetails } from "../components/SimulationFailureDetails";
import { SimulationFailureError, type SimulationFailure } from "../lib/core/simulation-failure";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
import type { GPUStatus } from "../lib/core/gpu-status";

const failure: SimulationFailure = {
  method: "adaptive-mass", code: "INCIDENCE_RANGE", message: "Corrupt incidence range",
  kernel: "forceFaces", frame: 42, generation: 17, ownerId: 123,
  operands: [900, 2, 384], rawWords: [1, 1, 0, 42, 17, 123, 900, 2, 384],
};
test("renderer preserves a structured first failure and refuses restart", async () => {
  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, status => statuses.push(status));
  renderer.stopAfterSimulationFailure(new SimulationFailureError(failure));
  renderer.stopAfterSimulationFailure(new SimulationFailureError({ ...failure, message: "late error" }));
  renderer.setSimulationRunning(true);
  await assert.rejects(renderer.initialize(), /HALTED.*INCIDENCE_RANGE/);
  assert.equal(statuses.length, 1);
  assert.equal(statuses[0].state, "unavailable");
  if (statuses[0].state === "unavailable") assert.deepEqual(statuses[0].failure, failure);
});
test("failure details expose provenance and downloadable scene inputs", () => {
  const html = renderToStaticMarkup(createElement(SimulationFailureDetails,
    { failure, configuration: { scene: { sceneId: "example", numerics: { fixedDt_s: 1 / 30 } } } }));
  for (const text of ["SIMULATION HALTED", "INCIDENCE_RANGE", "forceFaces", "42", "123",
    "Copy failure report", "Download failure report", "fixedDt_s", "example"]) assert.ok(html.includes(text), text);
});

test("a failure check retired during receipt readback never submits against destroyed buffers", async () => {
  let release!: () => void;
  const receipt = new Promise<void>(resolve => { release = resolve; });
  const solver = Object.assign(Object.create(WebGPUAdaptiveMassSolver.prototype), {
    disposed: false,
    failureReceipts: new Set([receipt]),
    sparseRuntime: { assertSimulationHealthy: () => assert.fail("checkpoint submitted after disposal") },
  });
  const check = solver.assertSimulationHealthy();
  solver.disposed = true;
  release();
  await check;
  await solver.assertSimulationHealthy();
});


test("presentation health maps its submitted receipt once without another GPU checkpoint", async () => {
  const encoder = {} as GPUCommandEncoder;
  let captures = 0, reads = 0;
  const solver = Object.assign(Object.create(WebGPUAdaptiveMassSolver.prototype), {
    disposed: false, lastTime_s: 1 / 30, failureReceipts: new Set(),
    sparseRuntime: {
      captureSimulationFailure: (actual: GPUCommandEncoder) => {
        assert.equal(actual, encoder); captures++;
        return async () => { reads++; return undefined; };
      },
      assertSimulationHealthy: () => assert.fail("redundant GPU checkpoint"),
    },
  });
  const read = solver.captureSimulationHealth(encoder);
  assert.equal(captures, 1); assert.equal(reads, 0);
  await Promise.all([read(), read()]);
  assert.equal(reads, 1); assert.equal(solver.failureReceipts.size, 0);
});

test("presentation health preserves the first GPU failure and retires stale receipts", async () => {
  let reads = 0, cancellations = 0;
  const solver = Object.assign(Object.create(WebGPUAdaptiveMassSolver.prototype), {
    disposed: false, lastTime_s: 1 / 30, failureReceipts: new Set(),
    scene: { sceneId: "fixture" }, info: {},
    sparseRuntime: {
      captureSimulationFailure: () => async () => { reads++; return failure; },
      cancelTopologyPreparation: () => { cancellations++; },
    },
  });
  await assert.rejects(solver.captureSimulationHealth({} as GPUCommandEncoder)(), /INCIDENCE_RANGE/);
  assert.equal(cancellations, 1);
  assert.equal(solver.info.simulationFailure.scene, "fixture");
  const stale = solver.captureSimulationHealth({} as GPUCommandEncoder);
  solver.disposed = true;
  await stale();
  assert.equal(reads, 2); assert.equal(solver.failureReceipts.size, 0);
});
