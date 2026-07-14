import assert from "node:assert/strict";
import test from "node:test";
import { validateScene } from "../lib/model";
import {
  compareScalarFields,
  createSmokeScenario,
  smokeScenarioIds,
  summarizeScalarField,
  summarizeTallCellActivity
} from "../tools/webgpu-smoke-scenarios";

test("native WebGPU matrix covers equilibrium, moving boundaries, rigid geometry, and deep compression", () => {
  assert.deepEqual(smokeScenarioIds, ["settled-tank", "dam-break-boxes", "hose-tank", "sphere-jet", "deep-water"]);
  for (const id of smokeScenarioIds) {
    const scenario = createSmokeScenario(id);
    assert.deepEqual(validateScene(scenario.scene), []);
    assert.ok(scenario.oracleSteps > 0);
    assert.ok(scenario.target_s >= scenario.oracleSteps * scenario.scene.numerics.maxDt_s);
  }
  assert.ok(createSmokeScenario("deep-water").scene.container.height_m >= 20);
  assert.ok(createSmokeScenario("hose-tank").scene.fluid.inflow);
  assert.ok(createSmokeScenario("sphere-jet").scene.rigidBodies.length > 0);
});

test("tall-cell activity explains ordinary-grid lock-in and mixed layouts", () => {
  assert.deepEqual(summarizeTallCellActivity(new Float32Array([0, 0, 0]), 46, 46), {
    totalColumns: 3,
    tallColumns: 0,
    ordinaryColumns: 3,
    tallFraction: 0,
    minimumTallHeight: 0,
    maximumTallHeight: 0,
    meanTallHeight: 0,
    maximumPermittedHeight: 0,
    canRemeshToTall: false,
    classification: "none"
  });
  const mixed = summarizeTallCellActivity(new Float32Array([0, 2, 8, 1]), 20, 12);
  assert.equal(mixed.classification, "mixed");
  assert.equal(mixed.tallColumns, 2);
  assert.equal(mixed.tallFraction, 0.5);
  assert.equal(mixed.minimumTallHeight, 2);
  assert.equal(mixed.maximumTallHeight, 8);
  assert.equal(mixed.meanTallHeight, 5);
  assert.equal(mixed.maximumPermittedHeight, 8);
  assert.equal(mixed.canRemeshToTall, true);
});

test("scalar discrepancy metrics are symmetric and distinguish shape from volume", () => {
  const left = new Float32Array([1, 1, 0, 0, 0, 0, 0, 0]);
  const right = new Float32Array([1, 0, 1, 0, 0, 0, 0, 0]);
  const leftSummary = summarizeScalarField(left, 2, 2, 2);
  assert.equal(leftSummary.cellSum, 2);
  assert.equal(leftSummary.componentCount, 1);
  const difference = compareScalarFields(left, right, 2, 2, 2);
  const reverse = compareScalarFields(right, left, 2, 2, 2);
  assert.equal(difference.volumeRelativeDifference, 0);
  assert.equal(difference.wetIntersectionOverUnion, 1 / 3);
  assert.equal(difference.meanAbsoluteError, 0.25);
  assert.deepEqual(difference, reverse);
  assert.deepEqual(compareScalarFields(left, left, 2, 2, 2), {
    meanAbsoluteError: 0,
    rootMeanSquareError: 0,
    volumeRelativeDifference: 0,
    wetIntersectionOverUnion: 1,
    centroidDistanceCells: 0
  });
});
