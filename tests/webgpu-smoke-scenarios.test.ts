import assert from "node:assert/strict";
import test from "node:test";
import { validateScene } from "../lib/model";
import {
  compareScalarFields,
  compareSingleTallCellNeighborhood,
  createSmokeScenario,
  smokeScenarioIds,
  summarizeScalarField,
  summarizeTallCellActivity
} from "../tools/webgpu-smoke-scenarios";

test("native WebGPU matrix covers the UI dam break, equilibrium, moving boundaries, rigid geometry, and deep compression", () => {
  assert.deepEqual(smokeScenarioIds, ["dam-break-ui", "settled-tank", "dam-break-boxes", "hose-tank", "sphere-jet", "deep-water", "garden-pond", "garden-dam-break"]);
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

test("single-tall-cell differential separates the probe stencil from the far field", () => {
  const left = new Float32Array(3 * 2 * 3), right = left.slice();
  const index = (x: number, y: number, z: number) => x + 3 * (y + 2 * z);
  right[index(1, 0, 1)] = 1;
  right[index(0, 1, 1)] = 0.5;
  right[index(0, 0, 0)] = 0.25;
  const difference = compareSingleTallCellNeighborhood(left, right, 3, 2, 3, 1, 1);
  assert.equal(difference.probeColumn.maximumAbsoluteError, 1);
  assert.deepEqual(difference.probeColumn.maximumLocation, { x: 1, y: 0, z: 1 });
  assert.equal(difference.neighborColumns.maximumAbsoluteError, 0.5);
  assert.equal(difference.farField.maximumAbsoluteError, 0.25);
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
  assert.equal(leftSummary.interfaceFaceCount, 4);
  assert.equal(leftSummary.enclosedAirComponentCount, 0);
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

test("scalar topology metrics expose enclosed air that volume and liquid connectivity miss", () => {
  const field = new Float32Array(27).fill(1);
  field[1 + 3 * (1 + 3)] = 0;
  const summary = summarizeScalarField(field, 3, 3, 3);
  assert.equal(summary.componentCount, 1);
  assert.equal(summary.largestComponent, 26);
  assert.equal(summary.interfaceFaceCount, 6);
  assert.equal(summary.enclosedAirComponentCount, 1);
  assert.equal(summary.enclosedAirCells, 1);
});
