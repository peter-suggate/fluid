import assert from "node:assert/strict";
import test from "node:test";
import { normalizeWebGPUSmokeEvidence } from "../tools/webgpu-smoke-evidence";

test("smoke results normalize into stable diagnostic namespaces and explicit capabilities", () => {
  const field = new Float32Array([0, 1]);
  const evidence = normalizeWebGPUSmokeEvidence([{
    method: "octree",
    info: { simulatedTime_s: 0.2, surfaceField: "levelset", referenceLiquidVolume_cells: 1,
      representedVolumeCellSum: 1, nonFiniteCount: 0 },
    steps: 2,
    grid: [2, 1, 1],
    matchedField: field,
    matchedSummary: { minimum: 0, maximum: 1, cellSum: 1 },
    checkpoints: [{ time_s: 0.2, field, summary: { centroidCells: { y: 0.5 } },
      evidence: { "free-fall-contact-attribution": { velocityByContact: [] } } }],
    collectedEvidence: ["free-fall contact attribution"],
    powerGenerationAuditedSteps: 2,
    validationErrors: [],
    energyTrace: [],
  }]);
  const method = evidence.methods.octree;
  assert.ok(method.available.includes("run"));
  assert.ok(method.available.includes("field summary"));
  assert.ok(method.available.includes("free-fall contact attribution"));
  assert.equal((method.diagnostics.run as { steps: number }).steps, 2);
  assert.equal((method.diagnostics.info as { sourceAdjustedRepresentedVolumeRatio: number })
    .sourceAdjustedRepresentedVolumeRatio, 1);
  assert.equal(method.diagnostics.validationErrorCount, 0);
});

test("validation error arrays normalize to an explicit scalar count", () => {
  const evidence = normalizeWebGPUSmokeEvidence([{
    method: "octree", info: {}, steps: 1, grid: [1, 1, 1],
    matchedField: new Float32Array(1), matchedSummary: {}, checkpoints: [],
    validationErrors: ["first", "second"],
  }]);
  assert.equal(evidence.methods.octree.diagnostics.validationErrorCount, 2);
});

test("missing expensive evidence is not advertised", () => {
  const evidence = normalizeWebGPUSmokeEvidence([{
    method: "uniform", info: {}, steps: 1, grid: [1, 1, 1],
    matchedField: new Float32Array(1), matchedSummary: {}, checkpoints: [],
  }]);
  assert.equal(evidence.methods.uniform.available.includes("front/back raster"), false);
  assert.equal(evidence.methods.uniform.available.includes("performance authority"), false);
});

test("compact mechanical-energy checkpoints advertise mechanical-energy evidence", () => {
  const evidence = normalizeWebGPUSmokeEvidence([{
    method: "octree", info: {}, steps: 1, grid: [1, 1, 1],
    matchedField: new Float32Array(1), matchedSummary: {}, energyTrace: [],
    checkpoints: [{ time_s: 0.004, compactMechanicalEnergy: {
      maximumLiquidComponentSpeed_m_s: 0.1,
    } }],
  }]);
  assert.equal(evidence.methods.octree.available.includes("mechanical energy"), true);
});
