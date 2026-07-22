import assert from "node:assert/strict";
import test from "node:test";
import { PAPER_VISUAL_PRESETS, paperPipelineHealthFlags, paperPipelineStages, paperSection5SpatialFailures, paperVisualAuthority } from "../lib/paper-pipeline-diagnostics";
import type { GPUEulerianInfo } from "../lib/webgpu-eulerian";

function info(patch: Partial<GPUEulerianInfo> = {}): GPUEulerianInfo {
  return {
    nx: 24, ny: 18, nz: 16, storedNy: 18, cellCount: 1, equivalentUniformCells: 1,
    compressionRatio: 1, regularLayers: 1, maximumNeighborDelta: 1, gridKind: "octree",
    cellSize_m: 0.1, pressureIterations: 32, allocatedBytes: 1, quality: "balanced",
    ...patch,
  };
}

test("paper pipeline inspector identifies one fully current authority chain", () => {
  const stages = paperPipelineStages(info({
    initialSparseAuthorityReady: true,
    encodedSteps: 1,
    powerDiagramProjection: "authoritative", powerDiagramAuthoritative: true, powerDiagramGeneration: 9,
    pressureRequiredRows: 120, pressureRequiredEntries: 840, pressureCapacityOverflow: false,
    globalFinePublished: true, globalFineRolledBack: false, globalFineGeneration: 17,
    globalFineSeedCount: 42, globalFineSeedError: 0, globalFineTopologyFlags: 0,
    globalFineDownstreamFinalizeReason: 0, globalFineRedistanceCommitted: true,
    globalFineRedistanceUnresolvedCells: 0, globalFineFaceBandValid: true,
    globalFineFaceBandTransitionValid: true, globalFineFaceBandTransientPowerValid: true,
    globalFineFaceBandPointFieldValid: true, globalFineFaceBandPowerPublicationValid: true,
    globalFineFaceBandPowerFineGeneration: 17, globalFineFaceBandPowerGeneration: 9,
    pressureSolver: "Section 4.3 hybrid MGPCG", pressureRelativeResidual: 8e-5,
  }), { surfaceGeometrySource: "global-fine-coarse", globalFineAttached: true, globalFineAttachedGeneration: 17, meshPublicationGeneration: 17, globalFineCrossingPublished: true, presentationFallbackActive: false });
  assert.ok(stages.length >= 8);
  assert.equal(stages.find((stage) => stage.id === "extrapolation")?.state, "PUBLISHED");
  assert.equal(stages.find((stage) => stage.id === "pressure")?.state, "CONVERGED");
  assert.equal(stages.find((stage) => stage.id === "raster")?.state, "CURRENT");
});

test("stale Section 5 and retained raster generations are visibly distinct from current authority", () => {
  const stages = paperPipelineStages(info({
    initialSparseAuthorityReady: true, powerDiagramGeneration: 11, globalFineGeneration: 20,
    globalFineFaceBandValid: true, globalFineFaceBandTransitionValid: true,
    globalFineFaceBandTransientPowerValid: true, globalFineFaceBandPointFieldValid: true,
    globalFineFaceBandPowerPublicationValid: true,
    globalFineFaceBandPowerFineGeneration: 19, globalFineFaceBandPowerGeneration: 10,
  }), { surfaceGeometrySource: "retained-previous", globalFineAttached: true, globalFineAttachedGeneration: 20, meshPublicationGeneration: 19, globalFineCrossingPublished: false, presentationFallbackActive: true });
  assert.equal(stages.find((stage) => stage.id === "extrapolation")?.tone, "stale");
  assert.equal(stages.find((stage) => stage.id === "raster")?.state, "STALE");
});

test("always-visible health flags expose rejected and stale 2017 authority", () => {
  const flags = paperPipelineHealthFlags(info({
    initialSparseAuthorityReady: true,
    powerDiagramProjection: "authoritative", powerDiagramAuthoritative: true,
    pressureRequiredRows: 10, pressureRequiredEntries: 30,
    globalFinePublished: true, globalFineRolledBack: true, globalFineGeneration: 20,
    globalFineTopologyFlags: 4, globalFineDownstreamFinalizeReason: 8,
    globalFineFaceBandValid: true, globalFineFaceBandTransitionValid: true,
    globalFineFaceBandTransientPowerValid: true, globalFineFaceBandPointFieldValid: true,
    globalFineFaceBandPowerPublicationValid: true,
    globalFineFaceBandPowerFineGeneration: 19, globalFineFaceBandPowerGeneration: 10,
    powerDiagramGeneration: 11,
  }));
  assert.ok(flags.includes("2017-fine-rejected"));
  assert.ok(flags.includes("2017-extrapolation-stale"));
});

test("no-causal-simplex coarse phi failure is attributed to grading coverage", () => {
  const stages = paperPipelineStages(info({
    globalFinePublished: true, globalFineRolledBack: false, globalFineGeneration: 17,
    globalFineSeedError: 0, globalFineTopologyFlags: 0, globalFineDownstreamFinalizeReason: 0,
    globalFineCoarseLevelSetFlags: 512,
    globalFineCoarseLevelSetFirstErrorRow: 37,
  }), undefined);
  const fine = stages.find((stage) => stage.id === "fine");
  assert.equal(fine?.state, "REJECTED");
  assert.match(fine?.detail ?? "", /no causal non-obtuse simplex at row 37/);
  assert.match(fine?.detail ?? "", /acute-simplex grading\/refinement coverage failed/);
});

test("one-click presets cover the requested paper structures without new diagnostic products", () => {
  assert.deepEqual(PAPER_VISUAL_PRESETS.map((preset) => preset.id), [
    "power-cells", "power-faces", "transitions", "fine-band", "fine-phi-values", "section5-march", "velocity", "pressure", "operator", "raster",
  ]);
  assert.equal(PAPER_VISUAL_PRESETS.find((preset) => preset.id === "raster")?.axis, "off");
  assert.equal(PAPER_VISUAL_PRESETS.find((preset) => preset.id === "fine-band")?.mode, "fine-band-lifecycle");
  assert.equal(PAPER_VISUAL_PRESETS.find((preset) => preset.id === "fine-phi-values")?.mode, "global-fine-phi");
});

test("raster CURRENT requires attached and mesh generations to match live fine authority", () => {
  const gpu = info({ globalFineGeneration: 21 });
  const stale = paperPipelineStages(gpu, {
    surfaceGeometrySource: "global-fine-coarse", globalFineAttached: true,
    globalFineAttachedGeneration: 21, meshPublicationGeneration: 20,
    globalFineCrossingPublished: true, presentationFallbackActive: false,
  }).find((stage) => stage.id === "raster");
  assert.equal(stale?.state, "STALE");
  assert.match(stale?.detail ?? "", /attached 21 · mesh 20 · live 21/);
});

test("active visual reports its exact authority and t=0/latest-step identity", () => {
  const gpu = info({
    initialSparseAuthorityReady: true, encodedSteps: 0, globalFineGeneration: 12,
    globalFinePublished: true, globalFineRolledBack: false, globalFineSeedError: 0,
    globalFineTopologyFlags: 0, globalFineDownstreamFinalizeReason: 0,
  });
  const stages = paperPipelineStages(gpu, undefined);
  const fine = paperVisualAuthority("fine-band-lifecycle", "volume", stages, gpu);
  assert.equal(fine.label, "Fine-band lifecycle");
  assert.equal(fine.stageId, "fine");
  assert.equal(fine.state, "PUBLISHED");
  assert.equal(fine.frame, "t=0 preflight");
  const stepped = paperVisualAuthority("power-faces", "volume", stages, { ...gpu, encodedSteps: 3 });
  assert.equal(stepped.stageId, "power");
  assert.equal(stepped.frame, "latest encoded substep 3");
});

test("Section 5 audit exposes bounded first failures and existing inspect overlays", () => {
  const audit = paperSection5SpatialFailures(info({
    globalFineGeneration: 7, globalFineFaceBandGeneration: 7,
    powerDiagramGeneration: 4,
    globalFineFaceBandPowerFineGeneration: 7, globalFineFaceBandPowerGeneration: 4,
    globalFineFaceBandFlags: 0x1f8,
    globalFineFaceBandValid: false,
    globalFineFaceBandFirstError: 69,
    globalFineFaceBandFaceCount: 15479,
    globalFineFaceBandAcceptedCount: 3113,
    globalFineFaceBandUnresolvedCount: 12366,
    globalFineFaceBandCoarsePhiFailures: 337,
    globalFineFaceBandPhiExtensions: 1800,
    globalFineFaceBandMarchHeapHighWater: 15479,
    globalFineFaceBandMarchPops: 13237,
    globalFineFaceBandMarchTrials: 15479,
    globalFineFaceBandMarchChunks: 16,
    globalFineFaceBandMarchChunkBound: 16,
    globalFineFaceBandMarchCapExhausted: 2242,
    globalFineFaceBandMarchUnresolvedWithPredecessor: 0,
    globalFineFaceBandMarchDisconnected: 0,
    globalFineFaceBandPhiFailureCounts: {
      missingRow: 2, exactCoarseMiss: 3, invalidMetric: 4, invalidSelector: 5,
    },
    globalFineFaceBandPhiFailure: {
      cause: 3, faceIndex: 19, globalFace: 4301, negativeRow: 71, positiveRow: 84,
      anchorRow: 71, centroid: [12.5, 7, 3.25], interpolantPath: 2,
      missingOrigin: [-4, 8, 10], missingSize: 2, selectorOrCorner: 17, detail: 0x205,
    },
    globalFineFaceBandTransitionFlags: 0,
    globalFineFaceBandTransitionValid: true,
    globalFineFaceBandTransitionRows: 448,
    globalFineFaceBandTransitionAdjacencyCount: 903,
    globalFineFaceBandTransientPowerFlags: 8,
    globalFineFaceBandTransientPowerValid: false,
    globalFineFaceBandTransientPowerFirstError: 69,
  }));
  assert.equal(audit.find((item) => item.id === "regular-band")?.state, "REJECTED");
  assert.equal(audit.find((item) => item.id === "regular-band")?.first,
    "first φ owner invalid selector: face 4,301 (slot 19, rows 71↔84) · anchor 71 · Delaunay · missing (-4,8,10) size 2 · selector/corner 17 · detail 0x205");
  assert.match(audit.find((item) => item.id === "regular-band")?.counts ?? "", /337 φ failures/);
  assert.match(audit.find((item) => item.id === "regular-band")?.counts ?? "",
    /owner causes row 2 \/ coarse 3 \/ metric 4 \/ selector 5/);
  assert.match(audit.find((item) => item.id === "regular-band")?.counts ?? "",
    /heap pop bound exhausted 2,242 \/ accepted-predecessor scheduler defect 0 \/ disconnected 0/);
  assert.match(audit.find((item) => item.id === "regular-band")?.counts ?? "",
    /heap 15,479 high-water · 13,237\/15,479 pops\/trials · 16\/16 chunks/);
  assert.equal(audit.find((item) => item.id === "transition")?.state, "CURRENT");
  assert.equal(audit.find((item) => item.id === "transient-power")?.inspectMode, "power-faces");
});

test("Section 5 audit preserves escaped acute mask attribution for Render Inspect", () => {
  const audit = paperSection5SpatialFailures(info({
    globalFineGeneration: 7, globalFineFaceBandGeneration: 7,
    globalFineFaceBandTransitionFlags: 16,
    globalFineFaceBandTransitionValid: false,
    globalFineFaceBandTransitionFirstError: 91,
    globalFineFaceBandAcuteGradingFailure: {
      band: 12, rowCell: 91, rowSize: 2, descriptor: 0xa580_0039, coarseMask: 57,
    },
  }));
  const transition = audit.find((item) => item.id === "transition");
  assert.equal(transition?.state, "REJECTED");
  assert.equal(transition?.first,
    "escaped acute grading: band 12 · row 91 size 2 · coarse mask 0x39 · raw descriptor 0xa5800039");
  assert.equal(transition?.inspectMode, "delaunay-tetrahedra");
});

test("Section 5 audit never labels valid but cross-generation controls current", () => {
  const audit = paperSection5SpatialFailures(info({
    globalFineGeneration: 8, globalFineFaceBandGeneration: 7,
    powerDiagramGeneration: 5,
    globalFineFaceBandFlags: 0, globalFineFaceBandValid: true,
    globalFineFaceBandTransitionFlags: 0, globalFineFaceBandTransitionValid: true,
    globalFineFaceBandTransientPowerFlags: 0, globalFineFaceBandTransientPowerValid: true,
    globalFineFaceBandPointFieldFlags: 0, globalFineFaceBandPointFieldValid: true,
    globalFineFaceBandPowerPublicationFlags: 0, globalFineFaceBandPowerPublicationValid: true,
    globalFineFaceBandPowerFineGeneration: 7, globalFineFaceBandPowerGeneration: 4,
  }));
  for (const item of audit) assert.equal(item.state, "STALE", item.id);
});
