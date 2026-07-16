import assert from "node:assert/strict";
import test from "node:test";
import { classifyTallCellStability, planGPUAdvance, summarizeDriftOscillation } from "../lib/tall-cell-diagnostics";

test("GPU clock advances by maxDt without discarding requested time", () => {
  const first = planGPUAdvance(0.1, 0, 1 / 30);
  assert.ok(first);
  assert.equal(first.dt_s, 1 / 30);
  assert.equal(first.nextTime_s, 1 / 30);
  assert.ok(Math.abs(first.lag_s - 2 / 30) < 1e-12);

  const second = planGPUAdvance(0.1, first.nextTime_s, 1 / 30);
  assert.ok(second);
  assert.ok(Math.abs(second.nextTime_s - 2 / 30) < 1e-12);
  assert.ok(Math.abs(second.lag_s - 1 / 30) < 1e-12);
  assert.equal(planGPUAdvance(0.01, 0.02, 1 / 30), undefined);
});

test("stability classifier distinguishes expected collocated divergence from eruption", () => {
  assert.deepEqual(classifyTallCellStability({
    nonFiniteCount: 0,
    pressureRelativeResidual: 0.02,
    maxComponentCfl: 0.21,
    highCflCellCount: 0,
    maxDivergenceBefore_s: 17,
    maxDivergenceAfter_s: 4.8,
    dt_s: 1 / 30
  }), []);

  assert.deepEqual(classifyTallCellStability({
    nonFiniteCount: 0,
    pressureRelativeResidual: 0.35,
    maxComponentCfl: 24,
    highCflCellCount: 80,
    maxDivergenceBefore_s: 115,
    maxDivergenceAfter_s: 360,
    dt_s: 1 / 30
  }), ["pressure-residual", "advective-cfl", "post-projection-divergence", "projection-amplified-divergence"]);
});

test("stability classifier permits isolated paper-regime CFL excursions", () => {
  assert.deepEqual(classifyTallCellStability({
    nonFiniteCount: 0,
    pressureRelativeResidual: 0.02,
    maxComponentCfl: 3.5,
    highCflCellCount: 1,
    maxDivergenceBefore_s: 17,
    maxDivergenceAfter_s: 4.8,
    dt_s: 1 / 30
  }), []);
});

test("drift oscillation summary median-smooths and counts late reversals", () => {
  assert.deepEqual(summarizeDriftOscillation([]), { driftSignChanges: 0, latePeakToPeakDrift: 0 });
  const monotone = summarizeDriftOscillation([0, 0.01, 0.02, 0.03, 0.04, 0.05]);
  assert.equal(monotone.driftSignChanges, 0);
  assert.ok(Math.abs(monotone.latePeakToPeakDrift - 0.02) < 1e-12);
  const ringing = summarizeDriftOscillation([0, 0, 0, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]);
  assert.ok(ringing.driftSignChanges >= 2);
  assert.ok(ringing.latePeakToPeakDrift >= 0.02 - 1e-12);
});
