import assert from "node:assert/strict";
import test from "node:test";
import { PAPER_VISUAL_PRESETS, paperPipelineHealthFlags, paperPipelineStages,
  paperVisualAuthority } from "../lib/paper-pipeline-diagnostics";
import type { GPUEulerianInfo } from "../lib/webgpu-eulerian";

function info(patch: Partial<GPUEulerianInfo> = {}): GPUEulerianInfo {
  return {
    nx: 24, ny: 18, nz: 16, storedNy: 18, cellCount: 1, equivalentUniformCells: 1,
    compressionRatio: 1, regularLayers: 1, maximumNeighborDelta: 1, gridKind: "octree",
    cellSize_m: 0.1, pressureIterations: 32, allocatedBytes: 1, quality: "balanced",
    ...patch,
  };
}

test("paper pipeline inspector reports the direct structured authority chain", () => {
  const stages = paperPipelineStages(info({
    initialSparseAuthorityReady: true, encodedSteps: 1,
    powerDiagramAuthoritative: true, powerDiagramGeneration: 9,
    pressureRequiredRows: 120, pressureCapacityOverflow: false,
    globalFinePublished: true, globalFineRolledBack: false, globalFineGeneration: 17,
    globalFineSeedCount: 42, globalFineSeedError: 0, globalFineTopologyFlags: 0,
    globalFineDownstreamFinalizeReason: 0, globalFineRedistanceCommitted: true,
    globalFineRedistanceUnresolvedCells: 0, globalFineTransportCommitted: true,
    pressureSolver: "Section 4.3 hybrid MGPCG", pressureRelativeResidual: 8e-5,
  }), { surfaceGeometrySource: "global-fine-coarse", globalFineAttached: true,
    globalFineAttachedGeneration: 17, meshPublicationGeneration: 17,
    globalFineCrossingPublished: true, presentationFallbackActive: false });
  assert.equal(stages.find((stage) => stage.id === "authority")?.state, "READY");
  assert.equal(stages.find((stage) => stage.id === "power")?.state, "AUTHORITATIVE");
  assert.equal(stages.find((stage) => stage.id === "fine")?.state, "PUBLISHED");
  assert.equal(stages.find((stage) => stage.id === "transport")?.state, "COMMITTED");
  assert.equal(stages.find((stage) => stage.id === "pressure")?.state, "CONVERGED");
  assert.equal(stages.find((stage) => stage.id === "raster")?.state, "CURRENT");
});

test("pipeline inspector rejects a non-M1 pressure authority", () => {
  const pressure = paperPipelineStages(info({
    initialSparseAuthorityReady: true, encodedSteps: 1,
    powerDiagramAuthoritative: true, pressureRequiredRows: 120,
    pressureCapacityOverflow: false,
    pressureSolver: "retired pressure authority", pressureRelativeResidual: 1.4e-5,
  }), undefined).find((stage) => stage.id === "pressure");
  assert.equal(pressure?.state, "REJECTED");
  assert.equal(pressure?.section, "pressure");
});

test("always-visible health flags expose rejected fine publication", () => {
  const flags = paperPipelineHealthFlags(info({
    initialSparseAuthorityReady: true, powerDiagramAuthoritative: true,
    pressureRequiredRows: 10,
    globalFinePublished: true, globalFineRolledBack: true, globalFineGeneration: 20,
    globalFineTopologyFlags: 4, globalFineDownstreamFinalizeReason: 8,
  }));
  assert.ok(flags.includes("2017-fine-rejected"));
});

test("structured telemetry is pending at t=0 but rejects after a dynamic attempt", () => {
  const base = info({
    initialSparseAuthorityReady: true,
    powerDiagramAuthoritative: true,
    pressureRequiredRows: 10,
    structuredVelocityValid: false,
    structuredBoundaryValid: false,
  });
  const preflight = paperPipelineStages({ ...base, encodedSteps: 0 }, undefined)
    .find((stage) => stage.id === "extrapolation");
  assert.equal(preflight?.state, "WAITING");
  assert.equal(preflight?.tone, "pending");
  assert.match(preflight?.detail ?? "", /t=0 preflight is fenced/);

  const dynamic = paperPipelineStages({ ...base, encodedSteps: 1 }, undefined)
    .find((stage) => stage.id === "extrapolation");
  assert.equal(dynamic?.state, "REJECTED");
  assert.equal(dynamic?.tone, "rejected");
});

test("fine and coarse-phi flags render high bits as unsigned hexadecimal", () => {
  const fine = paperPipelineStages(info({
    initialSparseAuthorityReady: true, globalFinePublished: false,
    globalFineTopologyFlags: 0x8000_0000,
    globalFineDownstreamFinalizeReason: 0x8000_0001,
    globalFineCoarseLevelSetFlags: 0x8000_0002,
  }), undefined).find((stage) => stage.id === "fine");
  assert.match(fine?.detail ?? "", /topology 0x80000000/);
  assert.match(fine?.detail ?? "", /downstream 0x80000001/);
  assert.match(fine?.detail ?? "", /coarse φ 0x80000002/);
  assert.doesNotMatch(fine?.detail ?? "", /0x-/);
});

test("coarse-phi causal failure is attributed to grading coverage", () => {
  const fine = paperPipelineStages(info({
    globalFinePublished: true, globalFineRolledBack: false, globalFineGeneration: 17,
    globalFineSeedError: 0, globalFineTopologyFlags: 0, globalFineDownstreamFinalizeReason: 0,
    globalFineCoarseLevelSetFlags: 512, globalFineCoarseLevelSetFirstErrorRow: 37,
  }), undefined).find((stage) => stage.id === "fine");
  assert.equal(fine?.state, "REJECTED");
  assert.match(fine?.detail ?? "", /no causal non-obtuse simplex at row 37/);
});

test("one-click presets reuse existing topology, fine, velocity, pressure, and raster views", () => {
  const ids = new Set(PAPER_VISUAL_PRESETS.map((preset) => preset.id));
  for (const id of ["power-cells", "transitions", "fine-band", "fine-phi-values",
    "velocity", "pressure", "operator", "raster"]) assert.ok(ids.has(id), id);
  assert.equal(PAPER_VISUAL_PRESETS.find((preset) => preset.id === "raster")?.axis, "off");
  assert.equal(PAPER_VISUAL_PRESETS.find((preset) => preset.id === "fine-band")?.mode,
    "fine-band-lifecycle");
});

test("raster CURRENT requires matching GPU-latched attachment and mesh", () => {
  const gpu = info({ globalFineGeneration: 21 });
  const stale = paperPipelineStages(gpu, {
    surfaceGeometrySource: "global-fine-coarse", globalFineAttached: true,
    globalFineAttachedGeneration: 21, meshPublicationGeneration: 20,
    globalFineCrossingPublished: true, presentationFallbackActive: false,
  }).find((stage) => stage.id === "raster");
  assert.equal(stale?.state, "STALE");
  const current = paperPipelineStages(gpu, {
    surfaceGeometrySource: "global-fine-coarse", globalFineAttached: true,
    globalFineAttachedGeneration: 22, meshPublicationGeneration: 22,
    globalFineCrossingPublished: true, presentationFallbackActive: false,
  }).find((stage) => stage.id === "raster");
  assert.equal(current?.state, "CURRENT");
});

test("active visual reports exact publication and t=0/latest-step identity", () => {
  const gpu = info({
    initialSparseAuthorityReady: true, encodedSteps: 0, globalFineGeneration: 12,
    globalFinePublished: true, globalFineRolledBack: false, globalFineSeedError: 0,
    globalFineTopologyFlags: 0, globalFineDownstreamFinalizeReason: 0,
  });
  const stages = paperPipelineStages(gpu, undefined);
  const fine = paperVisualAuthority("fine-band-lifecycle", "volume", stages, gpu);
  assert.equal(fine.stageId, "fine");
  assert.equal(fine.state, "PUBLISHED");
  assert.equal(fine.frame, "t=0 preflight");
  const stepped = paperVisualAuthority("power-cells", "volume", stages,
    { ...gpu, encodedSteps: 3 });
  assert.equal(stepped.stageId, "power");
  assert.equal(stepped.frame, "latest encoded substep 3");
});
