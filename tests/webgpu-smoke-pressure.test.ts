import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  decodeOctreeMGPCGDiagnostics,
  octreeMGPCGDiagnosticsAreAcceptable,
  octreePowerPressureDiagnosticsAreAcceptable,
  octreePowerPressureEnvelopeIsAcceptable,
  octreeProjectedVariationalResidualRms,
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
  words[4] = input.rows ?? 42;
  floats[10] = input.residualSquared ?? 1e-10;
  floats[11] = 1e-18;
  floats[8] = input.rhsSquared ?? 1;
  floats[9] = 1e-9;
  return words;
}

test("MGPCG smoke diagnostics decode the GPU control ABI", () => {
  const decoded = decodeOctreeMGPCGDiagnostics(control());
  assert.equal(decoded.flags, 0);
  assert.equal(decoded.converged, true);
  assert.equal(decoded.iterations, 7);
  assert.equal(decoded.rows, 42);
  assert.ok(Math.abs(decoded.relativeResidualSquared - 1.00000001e-10) < 1e-16);
  assert.ok(Math.abs(decoded.relativeResidual - Math.sqrt(1.00000001e-10)) < 1e-10);
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

test("production pressure acceptance uses only the MGPCG relative residual", () => {
  const rows = 1_248;
  const acceptedAbsolute = decodeOctreeMGPCGDiagnostics(control({
    rows,
    residualSquared: 3.72e-12,
    rhsSquared: 9.62e-7,
  }));
  assert.equal(octreeMGPCGDiagnosticsAreAcceptable(acceptedAbsolute), false);
  assert.equal(octreePowerPressureDiagnosticsAreAcceptable(
    "Octree power persistent PCG · Section 4.3 hybrid",
    acceptedAbsolute,
  ), false);
});

test("stability envelopes have no solver-specific fallback", () => {
  const mgpcg = "Octree power persistent PCG · Section 4.3 hybrid";
  assert.equal(octreePowerPressureEnvelopeIsAcceptable(mgpcg, 3e-3, 9.9e-8), false);
  assert.equal(octreePowerPressureEnvelopeIsAcceptable(mgpcg, 9.9e-5, 2e-7), true);
  assert.equal(octreePowerPressureEnvelopeIsAcceptable(mgpcg, Number.NaN, Number.NaN), false);
});

test("projected variational diagnostics convert algebraic pressure residual to Eq. (4) flux", () => {
  assert.equal(octreeProjectedVariationalResidualRms(0.011144211939353173, 0.004, 1000),
    4.457684775741269e-8);
  assert.equal(octreeProjectedVariationalResidualRms(2, 0, 1000), 0);
  for (const value of [
    octreeProjectedVariationalResidualRms(undefined, 0.004, 1000),
    octreeProjectedVariationalResidualRms(Number.NaN, 0.004, 1000),
    octreeProjectedVariationalResidualRms(-1, 0.004, 1000),
    octreeProjectedVariationalResidualRms(1, -0.004, 1000),
    octreeProjectedVariationalResidualRms(1, 0.004, 0),
  ]) assert.equal(value, undefined);

  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(smoke,
    /projectedVariationalResidual = octreeProjectedVariationalResidualRms\([\s\S]*sample\.pressureResidual, stepDt, scene\.fluid\.density_kg_m3\)/,
    "the per-step envelope must consume the Eq. (4) flux residual derived from the current solve and timestep");
  assert.doesNotMatch(smoke,
    /maximumProjectedVariationalResidual[^\n]*sample\.pressureResidual/,
    "raw algebraic pressure RMS must not be mislabeled as projected variational flux");
});
