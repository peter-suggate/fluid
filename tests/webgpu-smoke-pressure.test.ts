import assert from "node:assert/strict";
import test from "node:test";
import {
  decodeOctreeMGPCGDiagnostics,
  octreeMGPCGDiagnosticsAreAcceptable,
  octreePowerPressureDiagnosticsAreAcceptable,
  octreePowerPressureEnvelopeIsAcceptable,
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

test("Galerkin smoke acceptance honors its absolute RMS floor without weakening MGPCG", () => {
  const rows = 1_248;
  const acceptedAbsolute = decodeOctreeMGPCGDiagnostics(control({
    rows,
    residualSquared: 3.72e-12,
    rhsSquared: 9.62e-7,
  }));
  assert.equal(octreeMGPCGDiagnosticsAreAcceptable(acceptedAbsolute), false);
  assert.equal(octreePowerPressureDiagnosticsAreAcceptable(
    "Octree power fixed native-L2 Galerkin · 4 levels",
    acceptedAbsolute,
  ), true);
  assert.equal(octreePowerPressureDiagnosticsAreAcceptable(
    "Octree power persistent PCG · Section 4.3 hybrid",
    acceptedAbsolute,
  ), false);
  assert.equal(octreePowerPressureDiagnosticsAreAcceptable(
    "Octree power fixed native-L2 Galerkin · 4 levels",
    decodeOctreeMGPCGDiagnostics(control({
      rows,
      residualSquared: rows * 1.01e-14,
      rhsSquared: 9.62e-7,
    })),
  ), false);
});

test("Galerkin stability envelopes use the same absolute RMS floor as production", () => {
  const galerkin = "Octree power fixed native-L2 Galerkin · 4 levels";
  const mgpcg = "Octree power persistent PCG · Section 4.3 hybrid";
  assert.equal(octreePowerPressureEnvelopeIsAcceptable(galerkin, 3e-3, 9.9e-8), true);
  assert.equal(octreePowerPressureEnvelopeIsAcceptable(mgpcg, 3e-3, 9.9e-8), false);
  assert.equal(octreePowerPressureEnvelopeIsAcceptable(galerkin, 9.9e-5, 2e-7), true);
  assert.equal(octreePowerPressureEnvelopeIsAcceptable(galerkin, 3e-3, 1.01e-7), false);
  assert.equal(octreePowerPressureEnvelopeIsAcceptable(galerkin, Number.NaN, Number.NaN), false);
});
