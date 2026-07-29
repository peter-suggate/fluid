import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import {
  compactLiquidVelocityDiagnostic,
  compactMechanicalEnergyDiagnostic,
} from "../tools/webgpu-smoke-power-diagnostics";

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

  const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  assert.match(smoke, /compactMechanicalEnergyCheckpoints/);
  assert.match(smoke, /readCompactOctreeVelocityField3D[\s\S]*?compactMechanicalEnergyDiagnostic/,
    "checkpoint QA must derive energy from the authoritative compact velocity publication");
});

test("compact velocity energy ignores unrepresented cells but fails partial row corruption", () => {
  const clean = compactLiquidVelocityDiagnostic(
    [3, 4, 0, NaN, NaN, NaN], [0.5, 0], 2, [2, 4, 8], 0.5,
  );
  assert.deepEqual(clean, { kineticEnergyProxy: 12.5, liquidCellCount: 1, finiteLiquidCellCount: 1,
    liquidVolumeCellSum: 0.5, finiteLiquidVolumeCellSum: 0.5,
    nonFiniteLiquidComponentCount: 0, maximumLiquidComponentSpeed_m_s: 4,
    maximumLiquidComponentCfl: 0.75 });
  const uncovered = compactLiquidVelocityDiagnostic(
    [3, 4, 0, NaN, NaN, NaN], [0.5, 0.25], 2, [2, 4, 8], 0.5,
  );
  assert.equal(uncovered.liquidCellCount, 1);
  assert.equal(uncovered.finiteLiquidCellCount, 1);
  assert.equal(uncovered.nonFiniteLiquidComponentCount, 0);
  const corrupt = compactLiquidVelocityDiagnostic(
    [3, 4, 0, 2, NaN, 0], [0.5, 0.25], 2, [2, 4, 8], 0.5,
  );
  assert.equal(corrupt.liquidCellCount, 2);
  assert.equal(corrupt.finiteLiquidCellCount, 1);
  assert.equal(corrupt.nonFiniteLiquidComponentCount, 1);
  assert.throws(() => compactLiquidVelocityDiagnostic([0, 0, 0], [1], 1, [1, 0, 1], 0.1),
    /dimensions are inconsistent/);
});

test("exact octree QA reuses compact GPU velocity evidence for speed and CFL", () => {
  const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  assert.match(smoke,
    /compactLiquidVelocityDiagnostic\(compact\.field, cubic\.field,[\s\S]*?\[spacing\.x, spacing\.y, spacing\.z\], stepDt\)/,
    "component CFL must retain axis-specific fine spacing");
  assert.match(smoke,
    /checkpoints\.findLast[\s\S]*?info\.maxSpeed_m_s = compactVelocity\.maximumLiquidComponentSpeed_m_s;[\s\S]*?info\.maxComponentCfl = compactVelocity\.maximumLiquidComponentCfl;/,
    "the exact gate must consume already-read compact GPU QA evidence");
});

test("2017 pressure comments do not attribute ICCG or the QA tolerance to the paper", () => {
  const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  const pressure = readFileSync(new URL("../tools/webgpu-smoke-pressure.ts", import.meta.url), "utf8");
  assert.doesNotMatch(smoke, /paper example uses ICCG|paper.*relative residual\s+1e-4/i);
  assert.doesNotMatch(smoke, /1e-4 relative-residual limit is this regression's float32 QA/,
    "the executor must not own a scenario residual threshold");
  assert.equal(getSceneWebGPUSmokeLane("minimal-power-dam-break").acceptance
    .find(({ id }) => id === "minimal-power-variational-residual")?.expected, 3.5e-6,
  "the scene lane, not a paper attribution or runner literal, owns the projected-residual tolerance");
  assert.doesNotMatch(pressure, /Paper-result acceptance|ICCG\/PCG solves use a 1e-4/);
  assert.match(pressure, /2017 paper reports iteration counts, not this tolerance/);
});

test("structured-stage audit exposes only accepted velocity, boundary, and fine generations", () => {
  const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  assert.match(smoke, /unpackStructuredVelocityControl/);
  assert.match(smoke, /unpackStructuredBoundaryControl/);
  assert.match(smoke, /exactStructuredGenerationAuditFailures/);
  assert.doesNotMatch(smoke, /powerFaceControl|globalFinePowerVelocityControl|globalFineFaceBandControl/);
});
