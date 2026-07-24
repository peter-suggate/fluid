import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type {
  OctreeFaceBandControlSnapshot,
  OctreeFaceBandPointFieldControlSnapshot,
  OctreeFaceBandPowerPublicationSnapshot,
  OctreeFaceBandTransientPowerControlSnapshot,
} from "../lib/webgpu-octree-face-closest-point";
import {
  exactSection5GenerationAuditFailures,
  finalPerformanceAuthorityFailures,
  type ExactSection5GenerationAudit,
} from "../tools/webgpu-smoke-section5-audit";
import { OCTREE_POWER_VELOCITY_VALID } from "../lib/webgpu-octree-power-velocity";

const faceBand = {
  flags: 0, firstError: 0xffff_ffff, rowCount: 320, faceCount: 840, incidenceCount: 1_680,
  generation: 6, valid: true, seedCount: 400, acceptedCount: 840, unresolvedCount: 0,
  sampleFailures: 0, coarsePhiSamples: 0, coarsePhiFailures: 0, bandPhiExtensions: 0,
  closestPointFaces: 440, closestPointFailures: 0, liquidInterpolationFailures: 0,
  cptNoOwnerFailures: 0, cptSupportOwnerFailures: 0, cptNoContainingSimplexFailures: 0,
  cptMissingLiquidVertexFailures: 0, capacityFailure: false,
  stageFirstFailures: {
    faceEmission: 0xffff_ffff, phi: 0xffff_ffff,
    closestPoint: 0xffff_ffff, vectorReconstruction: 0xffff_ffff,
  },
  firstPhiFailureSlot: 0xffff_ffff,
  firstClosestPointFailureSlotByCause: {
    noOwner: 0xffff_ffff, supportOwner: 0xffff_ffff,
    noContainingSimplex: 0xffff_ffff, missingLiquidVertex: 0xffff_ffff,
  },
  invalidDirectory: false,
  invalidSource: false, invalidRow: false, invalidFace: false, invalidPhi: false,
  unresolved: false, incompleteVector: false, outsideFineBand: false,
} satisfies OctreeFaceBandControlSnapshot;

const transientPower = {
  flags: 0, firstError: 0xffff_ffff, rowCount: 256, faceSlots: 7_680,
  emittedCount: 700, sampledCount: 700, validatedCount: 256, generation: 6, valid: true,
} satisfies OctreeFaceBandTransientPowerControlSnapshot;

const pointField = {
  flags: 0, firstError: 0xffff_ffff, rowCount: 256, generation: 6,
  solvedCount: 256, valid: true, wallContributions: 20, coreRowCount: 128,
} satisfies OctreeFaceBandPointFieldControlSnapshot;

const powerPublication = {
  flags: 0, firstError: 0xffff_ffff, faceCount: 512, targetCount: 180,
  interpolatedCount: 180, committedCount: 180, fineGeneration: 6,
  powerGeneration: 6, valid: true,
} satisfies OctreeFaceBandPowerPublicationSnapshot;

function coherentAudit(): ExactSection5GenerationAudit {
  return {
    publishedFineGeneration: 7,
    expectedPowerGeneration: 6,
    expectedPowerFaceCount: 512,
    previousAcceptedSection5FineGeneration: 5,
    previousPowerGeneration: 5,
    faceBand,
    transientPower,
    pointField,
    powerPublication,
  };
}

test("exact Section 5 audit accepts one coherent advancing transaction", () => {
  assert.deepEqual(exactSection5GenerationAuditFailures(coherentAudit()), []);
  const audit = coherentAudit();
  assert.deepEqual(exactSection5GenerationAuditFailures({
    ...audit,
    faceBand: { ...audit.faceBand, generation: audit.expectedPowerGeneration - 1 },
  }), [], "an aligned current point field may consume exactly one retained-band predecessor");
});

test("exact Section 5 audit rejects a retained old publication even when every header is valid", () => {
  const audit = coherentAudit();
  const retainedFineGeneration = 5;
  const failures = exactSection5GenerationAuditFailures({
    ...audit,
    faceBand: { ...audit.faceBand, generation: retainedFineGeneration },
    transientPower: { ...audit.transientPower, generation: retainedFineGeneration },
    pointField: { ...audit.pointField, generation: retainedFineGeneration },
    powerPublication: { ...audit.powerPublication, fineGeneration: retainedFineGeneration },
  });
  assert.deepEqual(failures, [
    "transient-power generation is not current",
    "point-field generation is not current",
    "power publication fine generation is not current",
  ]);
});

test("exact Section 5 audit rejects a face band older than the exact predecessor", () => {
  const audit = coherentAudit();
  assert.deepEqual(exactSection5GenerationAuditFailures({
    ...audit,
    faceBand: { ...audit.faceBand, generation: audit.expectedPowerGeneration - 2 },
  }), ["face-band generation is neither current nor the exact predecessor"]);
});

test("exact Section 5 audit rejects mixed clocks, stale audit cadence, and old power projection", () => {
  const audit = coherentAudit();
  const failures = exactSection5GenerationAuditFailures({
    ...audit,
    publishedFineGeneration: audit.expectedPowerGeneration,
    previousAcceptedSection5FineGeneration: audit.expectedPowerGeneration,
    previousPowerGeneration: audit.expectedPowerGeneration,
    transientPower: { ...audit.transientPower, generation: 9, rowCount: 255 },
    powerPublication: { ...audit.powerPublication, powerGeneration: 5, faceCount: 511 },
  });
  assert.deepEqual(failures, [
    "published fine generation is not the Section 5 successor",
    "Section 5 fine generation did not advance",
    "power generation did not advance",
    "transient-power generation is not current",
    "transient-power and point-field row counts differ",
    "power publication power generation is not current",
    "power publication face count is not current",
  ]);
});

test("exact Dawn power audit reads and gates every accepted Section 5 transaction", () => {
  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  for (const control of [
    "globalFineFaceBandControl",
    "globalFineFaceBandTransientPowerControl",
    "globalFineFaceBandPointFieldControl",
    "globalFineFaceBandPowerPublicationControl",
  ]) {
    assert.match(smoke, new RegExp(`readBufferBinding\\(device, \\{ buffer: audited\\.${control} \\}`),
      `${control} must be part of the fenced per-generation readback`);
  }
  assert.match(smoke,
    /section5GenerationFailures = exactSection5GenerationAuditFailures[\s\S]*\|\| section5GenerationFailures\.length !== 0/,
    "a Section 5 clock mismatch must directly reject the accepted advance");
  assert.match(smoke,
    /previousAuditedSection5FineGeneration = audit\.section5\.faceBand\.generation/,
    "the next audit must compare against the last accepted Section 5 transaction, not the successor fine source");
});

test("final performance authority accepts only the exact dam-break clock", () => {
  assert.deepEqual(finalPerformanceAuthorityFailures({
    expectedSteps: 62,
    observedSteps: 62,
    expectedTime_s: 0.496,
    targetTime_s: 0.496,
    submittedTime_s: 0.496,
    fineSourceGeneration: 64,
    fineWorklistHeader: [5_931, 64, 0, 1, 1],
    finePageCapacity: 16_384,
    powerFaces: {
      flags: 0, rowCount: 1_264, faceCount: 512, incidenceCount: 2_048,
      generation: 63, valid: true,
    },
    powerVelocity: {
      flags: OCTREE_POWER_VELOCITY_VALID, firstError: 0xffff_ffff,
      rowCount: 1_264, faceCount: 512, incidenceCount: 2_048,
      reconstructedCount: 1_264, illConditionedCount: 0, generation: 63,
    },
    faceBand: { ...faceBand, generation: 63 },
    transientPower: { ...transientPower, generation: 63 },
    pointField: { ...pointField, generation: 63 },
    powerPublication: {
      ...powerPublication, faceCount: 512, fineGeneration: 63, powerGeneration: 63,
    },
  }), []);
});

test("final performance authority rejects a retained coherent publication", () => {
  const failures = finalPerformanceAuthorityFailures({
    expectedSteps: 62,
    observedSteps: 62,
    expectedTime_s: 0.496,
    targetTime_s: 0.496,
    submittedTime_s: 0.496,
    fineSourceGeneration: 21,
    fineWorklistHeader: [5_931, 21, 0, 1, 1],
    finePageCapacity: 16_384,
    powerFaces: {
      flags: 0, rowCount: 1_264, faceCount: 512, incidenceCount: 2_048,
      generation: 20, valid: true,
    },
    powerVelocity: {
      flags: OCTREE_POWER_VELOCITY_VALID, firstError: 0xffff_ffff,
      rowCount: 1_264, faceCount: 512, incidenceCount: 2_048,
      reconstructedCount: 1_264, illConditionedCount: 0, generation: 20,
    },
    faceBand: { ...faceBand, generation: 20 },
    transientPower: { ...transientPower, generation: 20 },
    pointField: { ...pointField, generation: 20 },
    powerPublication: {
      ...powerPublication, faceCount: 512, fineGeneration: 20, powerGeneration: 20,
    },
  });
  assert.deepEqual(failures, [
    "fine source generation is not current for the exact step",
    "power-face generation is not current for the exact step",
    "power-velocity generation is not current for the exact step",
    "published fine generation is not the Section 5 successor",
    "face-band generation is neither current nor the exact predecessor",
    "transient-power generation is not current",
    "point-field generation is not current",
    "power publication fine generation is not current",
    "power publication power generation is not current",
  ]);
});

test("performance authority readback is packed, fenced, and outside measured simulation wall", () => {
  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(smoke,
    /const simulationWall_ms[\s\S]*await awaitAdvanceCompletion\(\);[\s\S]*readBufferBindingsPacked/,
    "the final authority fence and packed control readback must follow the measured simulation wall");
  assert.match(smoke, /phase: "final-performance-authority"/);
  assert.match(smoke, /if \(finalAuthorityFailures\.length !== 0\)[\s\S]*throw new Error/,
    "the final packed authority result must reject the performance run directly");
});
