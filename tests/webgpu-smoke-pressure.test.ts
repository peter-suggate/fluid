import assert from "node:assert/strict";
import test from "node:test";
import {
  decodeOctreeMGPCGDiagnostics,
  octreeMGPCGDiagnosticsAreAcceptable,
} from "../tools/webgpu-smoke-pressure";

function control(input: {
  flags?: number; converged?: boolean; iterations?: number; rows?: number;
  residualSquared?: number; rhsSquared?: number;
} = {}) {
  const words = new Uint32Array(16);
  const floats = new Float32Array(words.buffer);
  words[0] = input.flags ?? 0;
  words[1] = (input.converged ?? true) ? 1 : 0;
  words[2] = input.iterations ?? 7;
  words[3] = input.rows ?? 42;
  floats[4] = input.residualSquared ?? 1e-10;
  floats[5] = input.rhsSquared ?? 1;
  return words;
}

test("MGPCG smoke diagnostics decode the GPU control ABI", () => {
  const decoded = decodeOctreeMGPCGDiagnostics(control());
  assert.equal(decoded.flags, 0);
  assert.equal(decoded.converged, true);
  assert.equal(decoded.iterations, 7);
  assert.equal(decoded.rows, 42);
  assert.ok(Math.abs(decoded.relativeResidualSquared - 1e-10) < 1e-16);
  assert.ok(Math.abs(decoded.relativeResidual - 1e-5) < 1e-10);
  assert.equal(octreeMGPCGDiagnosticsAreAcceptable(decoded), true);
});

test("MGPCG smoke acceptance fails closed on errors, rejection, empty rows, and residual misses", () => {
  for (const words of [
    control({ flags: 1 }),
    control({ converged: false }),
    control({ rows: 0 }),
    control({ residualSquared: 1.01e-8, rhsSquared: 1 }),
    control({ residualSquared: Number.NaN }),
    control({ rhsSquared: Number.NaN }),
  ]) assert.equal(octreeMGPCGDiagnosticsAreAcceptable(decodeOctreeMGPCGDiagnostics(words)), false);
  assert.equal(octreeMGPCGDiagnosticsAreAcceptable(undefined), false);
});
