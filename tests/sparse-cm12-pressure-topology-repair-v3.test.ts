import assert from "node:assert/strict";
import test from "node:test";

import {
  createSparseCM12PressureTopologyRepairLayout,
  sparseCM12PressureTopologyRepairEntryPoints,
} from "../lib/methods/adaptive-volume/sparse-cm12-pressure-topology-repair";
import { createSparseCM12PressureTopologyRepairWGSL } from
  "../lib/methods/adaptive-volume/sparse-cm12-pressure-topology-repair.wgsl";

test("PTR v3 stores only the compact changed-brick journal", () => {
  const layout = createSparseCM12PressureTopologyRepairLayout({
    brickCapacity: 1_152, brickFineResolution: 8, presentationPageResolution: 8,
  });
  assert.equal(layout.totalWords, 4_736);
  assert.equal(layout.totalBytes, 18_944);
  assert.deepEqual(Object.keys(layout.brick).sort(), [
    "candidateGenerationBaseWords", "capacity", "changedBrickListBaseWords",
    "dirtyLeafStampBaseWords", "leafCount",
  ]);
});

test("PTR v3 exposes no rank-tree repair or generation-capture shaders", () => {
  const layout = createSparseCM12PressureTopologyRepairLayout({
    brickCapacity: 1_152, brickFineResolution: 8, presentationPageResolution: 8,
  });
  assert.deepEqual(sparseCM12PressureTopologyRepairEntryPoints(layout), [
    "beginSparseCM12PressureTopologyRepair",
    "finalizeSparseCM12PressureTopologyBrickFrontier",
    "finalizeSparseCM12BoundedPressureTopologyRepair",
  ]);
  const wgsl = createSparseCM12PressureTopologyRepairWGSL({ layout });
  for (const legacy of ["seedPrevious", "repairSparse", "reduceSparse",
    "captureSparseCM12PressureTopologyConsumerGenerations", "ActiveLeaves",
    "TreeBase", "RepairIndirect", "WorkIndirect", "BrickCause"]) {
    assert.doesNotMatch(wgsl, new RegExp(legacy));
  }
  assert.match(wgsl,
    /ptrPressureCoefficientCandidateGeneration\(\)!=ptrPressureCoefficientAcceptedGeneration\(\)/);
});

test("production PTR commits only an accepted full pressure image", () => {
  const layout = createSparseCM12PressureTopologyRepairLayout({
    brickCapacity: 1_152, brickFineResolution: 8, presentationPageResolution: 8,
  });
  const wgsl = createSparseCM12PressureTopologyRepairWGSL({
    layout, fullImageAccepted: "peiFullImageAccepted()",
    publishFailure: (fault, owner) => `recordFailure(${fault},${owner});`,
  });
  assert.match(wgsl, /fn finalizeFullPressureTopologyJournal\(\)/);
  assert.match(wgsl, /if\(!\(peiFullImageAccepted\(\)\)\)/);
  assert.match(wgsl, /coefficientGenerationGap|17u/);
  assert.doesNotMatch(wgsl, /finalizeSparseCM12PressureTopologyBrickFrontier/);
  assert.doesNotMatch(wgsl, /finalizeSparseCM12BoundedPressureTopologyRepair/);
  assert.doesNotMatch(wgsl, /pcmCellQueueTopologyRetiredRange/);
  assert.match(wgsl, /recordFailure\(code,id\)/);
});
