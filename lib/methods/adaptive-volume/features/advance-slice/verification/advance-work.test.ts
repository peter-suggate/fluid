import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  ADVANCE_STAGE_ORDER, ADVANCE_WORK, ADVANCE_WORK_SCENES,
  advanceCosts, advanceStageCost, advanceWorkModel,
} from "../advance-work";

const inputs = (over: Partial<Parameters<typeof advanceWorkModel>[0]> = {}) =>
  advanceWorkModel({
    scene: ADVANCE_WORK_SCENES.mini32, pressureIterations: 64, cfl: 1,
    limiterPasses: 8, churn: 0.06, markers: 65_536, ...over,
  });

test("every kernel the work table prices is one the encoder dispatches", () => {
  const source = readFileSync(new URL(
    "../../../webgpu-sparse-cm12-resident.ts", import.meta.url), "utf8");
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
  // Projection publishes its faces, commits the projected frontier, and
  // recompiles the accepted topology before transport can read it.
  const projection = names("velocity-projection");
  assert.deepEqual(projection.slice(0, 8), [
    "beginIncrementalActivity", "projectSparseCM12InteriorFaceTiles",
    "projectSparseCM12SeamFacePackets", "projectSparseCM12SparseAirFacePackets",
    "projectSparseCM12DynamicFaceRows", "collocateAndDiagnose",
    "reduceDivergenceDiagnostics", "publishSparseCM12FrameFaceOutput",
  ]);
  assert.ok(projection.indexOf("encodeTopologyEditTransaction")
    < projection.indexOf("sealCompiledTopologyGeneration"));
  assert.ok(projection.indexOf("sealCompiledTopologyGeneration")
    < projection.indexOf("initializeVelocityExtensionPackets"));
  // Topology compilation is a topology-lifetime cost, never a transport one.
  const transport = names("conservative-transport");
  assert.ok(!transport.includes("compileGeometricVolumeSubfaces"));
  assert.ok(!transport.includes("compileGeometricVolumeCellFaces"));
  // Transport is one fixed schedule: no packet loop, no limiter continuation.
  assert.equal(ADVANCE_WORK["conservative-transport"].loop, undefined);
  const movingRows = ADVANCE_WORK["conservative-transport"].seams
    .flatMap(seam => seam.kernels)
    .find(kernel => kernel.name === "reexpressGeometricSolidRows");
  assert.equal(movingRows?.gate, "solids");
});

test("the pressure solve keeps its tail encoded past the residual guard", () => {
  const budget = 64;
  const cost = advanceStageCost(
    ADVANCE_WORK["pressure-solve"], inputs({ pressureIterations: budget }));
  assert.equal(cost.dispatches, budget * 3 + (budget / 8 - 1) * 8 + 3);
});

test("transport is the frame's largest stage outside the pressure solve", () => {
  const costs = advanceCosts(inputs({ cfl: 1.5 }));
  const ranked = [...ADVANCE_STAGE_ORDER]
    .map((stage, index) => ({ stage, workgroups: costs[index]!.workgroups }))
    .filter(entry => entry.stage !== "pressure-solve")
    .sort((a, b) => b.workgroups - a.workgroups);
  assert.equal(ranked[0]!.stage, "conservative-transport");
});
