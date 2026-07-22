import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  compactLiquidVelocityDiagnostic,
  compactMechanicalEnergyDiagnostic,
  compactPowerFaceIntegratedFlux,
  compactPowerFaceMetricKineticEnergy,
} from "../tools/webgpu-smoke-power-diagnostics";

test("compact projected flux uses the already aperture-weighted face velocity", () => {
  assert.equal(compactPowerFaceIntegratedFlux(3, 2), 6);

  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(smoke, /compactPowerFaceIntegratedFlux\(area, normalVelocity\)/);
  assert.doesNotMatch(smoke, /const flux = area \* openFraction \* normalVelocity/,
    "the compact residual must not apply the solid aperture twice");
});

test("compact projection-energy proxy uses the pressure operator's dual metric", () => {
  // H = area / (openFraction * inverseDistance), and E = 1/2 u^T H u.
  assert.equal(compactPowerFaceMetricKineticEnergy(4, 0.5, 0.25, 3), 144);
  assert.equal(compactPowerFaceMetricKineticEnergy(4, 0.5, 0, 3), 0,
    "fully constrained faces are outside the projected velocity space");
  assert.ok(Number.isNaN(compactPowerFaceMetricKineticEnergy(4, 0, 1, 3)));
  assert.ok(Number.isNaN(compactPowerFaceMetricKineticEnergy(4, 0.5, 1.1, 3)));
});

test("compact mechanical-energy diagnostic measures potential-to-kinetic conversion and loss", () => {
  assert.deepEqual(compactMechanicalEnergyDiagnostic(10, 7, 2), {
    gravitationalPotentialEnergyProxy: 7,
    reconstructedKineticEnergyProxy: 2,
    mechanicalEnergyProxy: 9,
    potentialEnergyReleasedProxy: 3,
    mechanicalEnergyLossProxy: 1,
    mechanicalEnergyRetentionRatio: 0.9,
    releasedPotentialToKineticRatio: 2 / 3,
  });
  assert.equal(compactMechanicalEnergyDiagnostic(10, 10, 0).releasedPotentialToKineticRatio, null);
  assert.throws(() => compactMechanicalEnergyDiagnostic(0, 0, 0), RangeError);
  assert.throws(() => compactMechanicalEnergyDiagnostic(10, 9, -1), RangeError);

  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(smoke, /compactMechanicalEnergyCheckpoints/);
  assert.match(smoke, /readCompactOctreeVelocityField3D[\s\S]*?compactMechanicalEnergyDiagnostic/,
    "checkpoint QA must derive energy from the authoritative compact velocity publication");
});

test("compact velocity energy ignores deliberately uncovered air but fails visible liquid gaps", () => {
  const clean = compactLiquidVelocityDiagnostic([3, 4, 0, NaN, NaN, NaN], [0.5, 0], 2);
  assert.deepEqual(clean, { kineticEnergyProxy: 12.5, liquidCellCount: 1, finiteLiquidCellCount: 1,
    liquidVolumeCellSum: 0.5, finiteLiquidVolumeCellSum: 0.5,
    nonFiniteLiquidComponentCount: 0, maximumLiquidComponentSpeed_m_s: 4 });
  const gap = compactLiquidVelocityDiagnostic([3, 4, 0, NaN, NaN, NaN], [0.5, 0.25], 2);
  assert.equal(gap.liquidCellCount, 2);
  assert.equal(gap.finiteLiquidCellCount, 1);
  assert.equal(gap.nonFiniteLiquidComponentCount, 3);
});

test("2017 pressure comments do not attribute ICCG or the QA tolerance to the paper", () => {
  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  const pressure = readFileSync(new URL("../tools/webgpu-smoke-pressure.ts", import.meta.url), "utf8");
  assert.doesNotMatch(smoke, /paper example uses ICCG|paper.*relative residual\s+1e-4/i);
  assert.match(smoke, /float32 QA relative-residual limit 1e-4/);
  assert.match(smoke, /Eq\. \(3\)-form projected residual[\s\S]*1e-6 QA gate/);
  assert.doesNotMatch(pressure, /Paper-result acceptance|ICCG\/PCG solves use a 1e-4/);
  assert.match(pressure, /2017 paper reports iteration counts, not this tolerance/);
});
