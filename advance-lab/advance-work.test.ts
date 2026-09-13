import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES,
  SPARSE_CM12_RESIDENT_STAGES,
} from "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";
import {
  ADVANCE_STAGE_ORDER, ADVANCE_WORK, ADVANCE_WORK_SCENES,
  advanceCosts, advanceStageCost, advanceWorkModel,
} from "./advance-work";

const inputs = (over: Partial<Parameters<typeof advanceWorkModel>[0]> = {}) =>
  advanceWorkModel({
    scene: ADVANCE_WORK_SCENES.mini32, pressureIterations: 64, cfl: 1,
    limiterPasses: 8, churn: 0.06, markers: 65_536, ...over,
  });

test("the work table covers the resident stage ABI, in encode order", () => {
  assert.deepEqual([...ADVANCE_STAGE_ORDER], [...SPARSE_CM12_RESIDENT_STAGES]);
  for (const stage of ADVANCE_STAGE_ORDER) {
    const declared = SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES[stage] as readonly string[];
    const modelled = ADVANCE_WORK[stage].seams
      .flatMap(seam => (seam.id === null ? [] : [seam.id as string]));
    assert.deepEqual(modelled, [...declared]);
  }
});

test("every kernel the work table prices is one the encoder dispatches", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts", import.meta.url), "utf8");
  const missing: string[] = [];
  for (const stage of ADVANCE_STAGE_ORDER) for (const seam of ADVANCE_WORK[stage].seams) {
    for (const kernel of seam.kernels) {
      if (kernel.isCopy) continue;
      const found = kernel.host ? source.includes(kernel.name) : source.includes(`"${kernel.name}"`);
      if (!found) missing.push(`${stage}/${kernel.name}`);
    }
  }
  assert.deepEqual(missing, []);
});

test("a capability a scene lacks encodes no dispatches for it", () => {
  const withInflow = advanceStageCost(
    ADVANCE_WORK["body-forces"], inputs({ scene: ADVANCE_WORK_SCENES.dam }));
  const without = advanceStageCost(ADVANCE_WORK["body-forces"], inputs());
  assert.ok(withInflow.dispatches > without.dispatches);
  assert.equal(without.dispatches, 2);
});

test("the work roster follows full pressure publication and topology-lifetime transport", () => {
  const names = (stage: keyof typeof ADVANCE_WORK) => ADVANCE_WORK[stage].seams
    .flatMap(seam => seam.kernels.map(kernel => kernel.name));
  const pressure = names("pressure-topology");
  assert.deepEqual(pressure, [
    "refreshGeometricInterface", "extendGeometricInterface",
    "beginSparseCM12PressureTopologyRepair", "beginFullPressureImage",
    "classifyFullPressureCellWords", "scanFullPressureCellWords",
    "publishFullPressureCellIds", "classifyFullPressureRowWords",
    "scanFullPressureRowWords", "publishFullPressureCoefficients",
    "sealFullPressureImage", "pressure-image cell indirect → pressure arguments",
    "finalizeFullPressureTopologyJournal", "beginSparseCM12PressureTopologyRepair",
    "preparePressure",
  ]);
  assert.deepEqual(names("pressure-rhs"), [
    "beginPressureSolve", "initializePCG", "reduceInitialize",
    "initializePipelinedImage", "reducePipelinedInitialize",
    "publishPressureSolveDispatchGate", "dispatch gate → cell + solve indirect (2 copies)",
  ]);
  assert.deepEqual(names("velocity-projection"), [
    "beginIncrementalActivity", "projectSparseCM12AcceptedFaceRows",
    "collocateAndDiagnose", "reduceDivergenceDiagnostics",
    "publishSparseCM12FrameFaceOutput", "encodeTopologyEditTransaction",
    "beginSparseCM12VelocityExtensionSchedule",
    "compileSparseCM12VelocityExtensionSchedule",
    "sealSparseCM12VelocityExtensionSchedule",
    "projected packet schedule → transport indirect arguments",
    "initializeVelocityExtensionPackets", "advanceVelocityExtensionPackets",
  ]);
  const transport = names("conservative-transport");
  assert.ok(!transport.includes("compileGeometricVolumeSubfaces"));
  assert.ok(!transport.includes("compileGeometricVolumeCellFaces"));
  const movingInitialize = ADVANCE_WORK["conservative-transport"].seams
    .flatMap(seam => seam.kernels)
    .find(kernel => kernel.name === "initializeGeometricLowFluxLimits");
  assert.deepEqual({ gate: movingInitialize?.gate, microstep: movingInitialize?.microstep },
    { gate: "solids", microstep: true });
});

test("the pressure solve keeps its tail encoded past the residual guard", () => {
  const budget = 64;
  const cost = advanceStageCost(
    ADVANCE_WORK["pressure-solve"], inputs({ pressureIterations: budget }));
  assert.equal(cost.dispatches, budget * 3 + (budget / 8 - 1) * 8 + 3);
});

test("transport is the frame's largest stage at mini32 scale", () => {
  const costs = advanceCosts(inputs({ cfl: 1.5 }));
  const largest = costs.indexOf(costs.reduce((a, b) => b.workgroups > a.workgroups ? b : a));
  assert.equal(ADVANCE_STAGE_ORDER[largest], "conservative-transport");
});
