import assert from "node:assert/strict";
import test from "node:test";
import type { PerformanceTrace } from "../lib/performance-trace";
import {
  buildPowerDamPassBoundaryProfile,
  powerDamComputePassStage,
  powerDamPerformanceFailures,
  powerDamResultFromLine,
  powerDamResultWindow,
  summarizePowerDamPerformance,
} from "../tools/power-dam-performance-report";
import {
  buildPowerDamPressureKernelProfile,
  powerDamPressureKernelRegion,
} from "../tools/power-dam-pressure-kernel-profile";

const physicsTrace = (phases: PerformanceTrace["phases"]): PerformanceTrace => ({
  sampleId: 1,
  domain: "gpu",
  lane: "physics",
  context: "octree:test",
  capturedAt_ms: 1,
  total_ms: phases.reduce((sum, phase) => sum + phase.duration_ms, 0),
  phases,
});

test("power dam throughput summary normalizes command costs per advance", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "minimal-power-dam-break", method: "octree", phase: "result",
    steps: 4, simulationWall_ms: 240, validationErrors: [],
    gpuCommandAudit: {
      dispatches: 400, indirectDispatches: 240, computePasses: 80,
      clearBuffer: { calls: 8, bytes: 4_000_000 },
      copyBufferToBuffer: { calls: 12, bytes: 8_000_000 },
      computePassesByLabel: {
        "Octree owner pages": { calls: 12, bytes: 0 },
        "Octree MGPCG solve": { calls: 8, bytes: 0 },
        "Fine redistance": { calls: 60, bytes: 0 },
      },
      dispatchesByPassLabel: {
        "Octree MGPCG solve": { calls: 200, bytes: 0 },
        "SPGrid persistent small-domain MGPCG": { calls: 100, bytes: 0 },
      },
    },
  });
  assert.equal(summary.advanceWall_ms, 60);
  assert.deepEqual(summary.commands, {
    dispatchesPerAdvance: 100, indirectDispatchesPerAdvance: 60,
    computePassesPerAdvance: 20,
    computePassesByStage: {
      "Octree owner pages": 3, "MGPCG solve": 2, "Fine redistance / volume": 15,
    },
    computePassesByLabel: {
      "Octree owner pages": 3, "Octree MGPCG solve": 2, "Fine redistance": 15,
    },
    computePassAttributionComplete: true,
    unattributedComputePassesPerAdvance: 0,
    unownedComputePassLabels: [],
    mgpcgDispatchesPerAdvance: 75,
    mgpcgDispatchFraction: 0.75,
    clearBytesPerAdvance: 1_000_000,
    copyBytesPerAdvance: 2_000_000,
  });
});

test("fine GPU pass timestamps normalize by sampled advances, not the full smoke length", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "minimal-power-dam-break",
    method: "octree",
    phase: "result",
    steps: 62,
    simulationWall_ms: 620,
    gpuFineTimestamps: {
      measuredAdvances: 1,
      measuredPasses: 2,
      invalidPasses: 0,
      summedPass_ms: 12,
      byLabel: {
        "Fine topology": {
          samples: 2,
          total_ms: 12,
          mean_ms: 6,
          minimum_ms: 5,
          maximum_ms: 7,
        },
      },
    },
  });
  assert.equal(summary.fineTimestamps?.summedPassPerAdvance_ms, 12);
  assert.equal(summary.fineTimestamps?.byLabel["Fine topology"]?.totalPerAdvance_ms, 12);
});

/**
 * The pass budget is a BOUNDARY budget: a compute pass exists because something
 * fenced. `computePassesPerAdvance` counts the passes; this ranks the causes,
 * which is what a merge has to be aimed at.
 */
test("pass boundary causes rank by measured closures and separate batchable requests", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "minimal-power-dam-break", method: "octree", phase: "result",
    steps: 4, simulationWall_ms: 240,
    gpuPassBoundaryAudit: {
      schemaVersion: 1,
      brokers: 8, labelIsolatedBrokers: 0, labelIsolated: false, resets: 1,
      requests: 480, passClosures: 320, copyCommands: 80, clearCommands: 40,
      commandBytes: 4_000_000, absorbedLabelPairs: 12, compositeEncoderLabels: 3,
      byReason: {
        "stage indirect args": { requests: 160, passClosures: 160, copyCommands: 80,
          clearCommands: 0, commandBytes: 2_000_000 },
        "clear buffer": { requests: 40, passClosures: 40, copyCommands: 0,
          clearCommands: 40, commandBytes: 2_000_000 },
        "publication visibility": { requests: 240, passClosures: 80, copyCommands: 0,
          clearCommands: 0, commandBytes: 0 },
        "finish command encoder": { requests: 40, passClosures: 40, copyCommands: 0,
          clearCommands: 0, commandBytes: 0 },
      },
    },
  });
  const boundaries = summary.passBoundaries;
  assert.equal(boundaries?.exact, true);
  assert.equal(boundaries?.warmupExcluded, true);
  assert.equal(boundaries?.labelIsolated, false);
  assert.equal(boundaries?.passClosuresPerAdvance, 80);
  assert.equal(boundaries?.requestsPerAdvance, 120);
  // The actionable number: 40 fence calls per advance found no open pass, so
  // they are already batched behind an earlier boundary and cost no launch.
  assert.equal(boundaries?.idempotentRequestsPerAdvance, 40);
  assert.equal(boundaries?.brokersPerAdvance, 2);
  assert.equal(boundaries?.bytesPerAdvance, 1_000_000);
  assert.equal(boundaries?.absorbedLabelPairs, 12);
  assert.deepEqual(Object.keys(boundaries?.byReason ?? {}), [
    "stage indirect args", "publication visibility", "clear buffer", "finish command encoder",
  ], "causes rank by real closures, then command traffic, then bytes, then name");
  assert.deepEqual(boundaries?.byReason["publication visibility"], {
    requests: 240, passClosures: 80, copyCommands: 0, clearCommands: 0, commandBytes: 0,
    idempotentRequests: 160, passClosuresPerAdvance: 20, requestsPerAdvance: 60,
    idempotentRequestsPerAdvance: 40, bytesPerAdvance: 0, shareOfClosures: 0.25,
  });
  assert.deepEqual(boundaries?.warnings, []);
});

test("a label-isolated or unreset boundary census cannot report itself as exact", () => {
  const profile = buildPowerDamPassBoundaryProfile({
    schemaVersion: 1,
    brokers: 2, labelIsolatedBrokers: 2, labelIsolated: true,
    labelIsolationPrefixes: ["SPGrid V-cycle -"],
    // Never reset, so construction and the cold bootstrap encoders are still in
    // here and every per-advance number below is inflated by one-time setup.
    resets: 0,
    requests: 24, passClosures: 20, copyCommands: 0, clearCommands: 0, commandBytes: 0,
    absorbedLabelPairs: 0, compositeEncoderLabels: 0,
    byReason: {
      // The synthetic bucket label isolation manufactures. Reporting a table
      // that contains it without saying so would present a diagnostic command
      // stream as the production one.
      "pass label isolation": { requests: 16, passClosures: 16, copyCommands: 0,
        clearCommands: 0, commandBytes: 0 },
      "publication visibility": { requests: 8, passClosures: 4, copyCommands: 0,
        clearCommands: 0, commandBytes: 0 },
    },
  }, 2);
  assert.equal(profile?.exact, false);
  assert.equal(profile?.labelIsolated, true);
  assert.equal(profile?.labelIsolatedBrokers, 2);
  assert.deepEqual(profile?.labelIsolationPrefixes, ["SPGrid V-cycle -"]);
  assert.equal(profile?.warmupExcluded, false);
  assert.equal(profile?.byReason["pass label isolation"]?.passClosuresPerAdvance, 8);
  assert.ok(profile?.warnings.some((warning) =>
    /FLUID_GPU_ISOLATE_PASS_LABELS/.test(warning)), profile?.warnings.join("\n"));
  assert.ok(profile?.warnings.some((warning) =>
    /cold bootstrap/.test(warning)), profile?.warnings.join("\n"));
  assert.equal(buildPowerDamPassBoundaryProfile({
    schemaVersion: 1, brokers: 0, labelIsolatedBrokers: 0, labelIsolated: false, resets: 1,
    requests: 0, passClosures: 0, copyCommands: 0, clearCommands: 0, commandBytes: 0,
    absorbedLabelPairs: 0, compositeEncoderLabels: 0, byReason: {},
  }, 2), undefined, "an empty census is absent, not a table of zeroes");
});

test("pressure kernel timestamps roll up into optimization regions", () => {
  const bucket = (samples: number, total_ms: number) => ({
    samples, total_ms, mean_ms: total_ms / samples,
    minimum_ms: total_ms / samples, maximum_ms: total_ms / samples,
  });
  const profile = buildPowerDamPressureKernelProfile({
    capturedCommandBuffers: 2,
    measuredPasses: 10,
    invalidPasses: 0,
    capacityOverflows: 0,
    summedPass_ms: 15,
    encoderIsolated: false,
    labelIsolated: true,
    labelPrefixes: [
      "Pipelined MGPCG", "Section 4.3", "Factor-1 dense M1", "SPGrid V-cycle",
      "SPGrid accurate A2", "SPGrid Section 6.3",
    ],
    span_ms: 40,
    coverageRatio: 0.375,
    byLabel: {
      "SPGrid V-cycle - restrict level 0": bucket(4, 8),
      "SPGrid V-cycle - coarse V-cycle tail levels 2-bottom": bucket(2, 4),
      "Section 4.3 hybrid inner residual": bucket(2, 2),
      "Factor-1 dense M1 - pre-smooth level 0": bucket(2, 3),
      "Pipelined MGPCG merged reduction finish": bucket(2, 1),
      "Fine JFA - flood stride 4": bucket(2, 99),
    },
  });
  assert.equal(profile?.exact, true);
  assert.equal(profile?.completePressureScope, true);
  assert.equal(profile?.instrumentedPressurePerAdvance_ms, 9);
  assert.equal(profile?.byRegion["vcycle.transfer.restrict.level-0"]?.totalPerAdvance_ms, 4);
  assert.equal(profile?.byRegion["vcycle.coarse-tail"]?.totalPerAdvance_ms, 2);
  assert.equal(profile?.byRegion["preconditioner.inner-residual"]?.totalPerAdvance_ms, 1);
  assert.equal(profile?.byRegion["preconditioner.dense.smoothing.pre.level-0"]
    ?.totalPerAdvance_ms, 1.5);
  assert.equal(profile?.byRegion["outer.reductions"]?.totalPerAdvance_ms, 0.5);
  assert.equal(profile?.byLabel["Fine JFA - flood stride 4"], undefined,
    "unrelated timestamp labels must not contaminate pressure totals");
});

test("a micro-stage timestamp filter cannot claim a complete pressure profile", () => {
  const profile = buildPowerDamPressureKernelProfile({
    capturedCommandBuffers: 1,
    measuredPasses: 1,
    invalidPasses: 0,
    capacityOverflows: 0,
    summedPass_ms: 1,
    encoderIsolated: false,
    labelIsolated: true,
    labelPrefixes: ["SPGrid accurate A2 -", "SPGrid Section 6.3 -"],
    span_ms: 2,
    coverageRatio: 0.5,
    byLabel: {
      "SPGrid accurate A2 - parallel direct terms": {
        samples: 1, total_ms: 1, mean_ms: 1, minimum_ms: 1, maximum_ms: 1,
      },
    },
  });
  assert.equal(profile?.completePressureScope, false);
  assert.equal(profile?.exact, false);
  assert.ok(profile?.warnings.some((warning) => /partial/.test(warning)));
});

test("pressure kernel taxonomy exposes transfer direction and hierarchy level", () => {
  assert.equal(powerDamPressureKernelRegion("SPGrid V-cycle · pre-smooth level 1"),
    "vcycle.smoothing.pre.level-1");
  assert.equal(powerDamPressureKernelRegion("SPGrid V-cycle - prolong level 0"),
    "vcycle.transfer.prolong.level-0");
  assert.equal(powerDamPressureKernelRegion("Factor-1 dense M1 - restrict level 1"),
    "preconditioner.dense.transfer.restrict.level-1");
  assert.equal(powerDamPressureKernelRegion("Factor-1 dense M1 - tail levels 2-bottom"),
    "preconditioner.dense.coarse-tail");
  assert.equal(powerDamPressureKernelRegion(
    "SPGrid Section 6.3 - parallel merged-band adjoint children"),
  "operator.band.adjoint-children");
  assert.equal(powerDamPressureKernelRegion("unrelated stage"), undefined);
});

test("quiescent paired-prefix window subtracts cumulative work and retains terminal counters", () => {
  const prefix = {
    scenario: "minimal-power-dam-break",
    method: "octree",
    phase: "result" as const,
    steps: 500,
    simulatedTime_s: 2,
    simulationWall_ms: 27_500,
    validationErrors: [],
    gpuCommandAudit: {
      dispatches: 250_000,
      indirectDispatches: 100_000,
      computePasses: 25_000,
      clearBuffer: { calls: 1_000, bytes: 500_000_000 },
      copyBufferToBuffer: { calls: 2_000, bytes: 1_000_000_000 },
      computePassesByLabel: {
        "Octree MGPCG solve": { calls: 20_000, bytes: 0 },
        "Fine redistance": { calls: 5_000, bytes: 0 },
      },
      dispatchesByPassLabel: {
        "Octree MGPCG solve": { calls: 200_000, bytes: 0 },
      },
    },
    gpuPassBoundaryAudit: {
      schemaVersion: 1 as const,
      brokers: 1_000, labelIsolatedBrokers: 0, labelIsolated: false, resets: 1,
      requests: 37_500, passClosures: 25_000, copyCommands: 5_000, clearCommands: 2_500,
      commandBytes: 500_000_000, absorbedLabelPairs: 12, compositeEncoderLabels: 3,
      byReason: {
        "stage indirect args": { requests: 5_000, passClosures: 5_000, copyCommands: 5_000,
          clearCommands: 0, commandBytes: 250_000_000 },
        "publication visibility": { requests: 30_000, passClosures: 17_500, copyCommands: 0,
          clearCommands: 0, commandBytes: 0 },
        "clear buffer": { requests: 2_500, passClosures: 2_500, copyCommands: 0,
          clearCommands: 2_500, commandBytes: 250_000_000 },
      },
    },
  };
  const complete = {
    ...prefix,
    steps: 560,
    simulatedTime_s: 2.24,
    simulationWall_ms: 30_620,
    activeSampleCount: 60_000,
    globalFineActiveBricks: 118,
    globalFineDesiredBricks: 120,
    globalFineLevelSetResidentBrickCapacity: 200,
    globalFineLevelSetLogicalBrickCount: 1_000,
    globalFineTransportSegmentCount: 2,
    quadtreePressureIterationsUsed: 4,
    quadtreePressureIterationBudget: 20,
    quadtreePressureIterationHardBudget: 20,
    compactMechanicalEnergyCheckpoints: [
      { time_s: 2, mechanicalEnergyRetentionRatio: 0.96, publicationValid: true,
        rowCount: 20, reconstructedRows: 20, invalidRows: 0,
        liquidCellCount: 100, finiteLiquidCellCount: 100 },
      { time_s: 2.04, mechanicalEnergyRetentionRatio: 0.95, publicationValid: true,
        rowCount: 20, reconstructedRows: 20, invalidRows: 0,
        liquidCellCount: 100, finiteLiquidCellCount: 100 },
    ],
    gpuCommandAudit: {
      dispatches: 277_000,
      indirectDispatches: 110_800,
      computePasses: 27_880,
      clearBuffer: { calls: 1_120, bytes: 560_000_000 },
      copyBufferToBuffer: { calls: 2_240, bytes: 1_120_000_000 },
      computePassesByLabel: {
        "Octree MGPCG solve": { calls: 22_400, bytes: 0 },
        "Fine redistance": { calls: 5_480, bytes: 0 },
      },
      dispatchesByPassLabel: {
        "Octree MGPCG solve": { calls: 221_600, bytes: 0 },
      },
    },
    gpuPassBoundaryAudit: {
      schemaVersion: 1 as const,
      brokers: 1_120, labelIsolatedBrokers: 0, labelIsolated: false, resets: 1,
      requests: 44_700, passClosures: 29_800, copyCommands: 6_200, clearCommands: 3_100,
      commandBytes: 560_000_000, absorbedLabelPairs: 12, compositeEncoderLabels: 3,
      byReason: {
        "stage indirect args": { requests: 6_200, passClosures: 6_200, copyCommands: 6_200,
          clearCommands: 0, commandBytes: 280_000_000 },
        "publication visibility": { requests: 35_400, passClosures: 20_500, copyCommands: 0,
          clearCommands: 0, commandBytes: 0 },
        "clear buffer": { requests: 3_100, passClosures: 3_100, copyCommands: 0,
          clearCommands: 3_100, commandBytes: 280_000_000 },
      },
    },
  };
  const result = powerDamResultWindow(prefix, complete);
  const summary = summarizePowerDamPerformance(result);
  assert.equal(result.steps, 60);
  assert.equal(summary.advanceWall_ms, 52);
  assert.deepEqual(summary.measurementWindow, {
    kind: "paired-prefix-difference", startStep: 500, endStep: 560,
  });
  assert.equal(summary.commands?.dispatchesPerAdvance, 450);
  assert.equal(summary.commands?.indirectDispatchesPerAdvance, 180);
  assert.equal(summary.commands?.computePassesPerAdvance, 48);
  assert.equal(summary.commands?.mgpcgDispatchesPerAdvance, 360);
  assert.equal(summary.commands?.clearBytesPerAdvance, 1_000_000);
  assert.deepEqual(summary.terminalCounters, {
    activeSamples: 60_000,
    activeFineBricks: 118,
    desiredFineBricks: 120,
    fineBrickCapacity: 200,
    logicalFineBricks: 1_000,
    // Terminal snapshots survive the window difference, so occupancy is the
    // suffix's active count over the logical lattice, not a differenced rate.
    fineBandOccupancy: 0.118,
    transportSegments: 2,
    pressureIterationsExecuted: 4,
    pressureIterationsScheduled: 20,
    pressureIterationsHardLimit: 20,
  });
  // `powerDamResultWindow` rebuilds the record from an explicit field list, so
  // an unlisted field is silently dropped on the whole quiescent lane.
  assert.equal(result.gpuPassBoundaryAudit?.passClosures, 4_800);
  assert.equal(summary.passBoundaries?.passClosuresPerAdvance, 80);
  assert.equal(summary.passBoundaries?.requestsPerAdvance, 120);
  assert.equal(summary.passBoundaries?.idempotentRequestsPerAdvance, 40);
  assert.equal(summary.passBoundaries?.brokersPerAdvance, 2);
  assert.deepEqual(Object.keys(summary.passBoundaries?.byReason ?? {}), [
    "publication visibility", "stage indirect args", "clear buffer",
  ], "the differenced suffix is re-ranked, not left in the cumulative order");
  assert.equal(summary.passBoundaries?.byReason["stage indirect args"]?.bytesPerAdvance,
    500_000);
  assert.equal(result.gpuFineTimestamps, undefined,
    "cumulative timestamp distributions cannot be differenced exactly");
  assert.deepEqual(result.compactMechanicalEnergyCheckpoints?.map(({ time_s }) => time_s), [2.04],
    "quiescent dissipation evidence must belong to the measured suffix");
});

test("quiescent paired-prefix window fails closed when cumulative counters regress", () => {
  const common = {
    scenario: "minimal-power-dam-break",
    method: "octree",
    phase: "result" as const,
    simulationWall_ms: 100,
    gpuCommandAudit: { dispatches: 10 },
  };
  assert.throws(() => powerDamResultWindow(
    { ...common, steps: 5 },
    { ...common, steps: 6, simulationWall_ms: 110, gpuCommandAudit: { dispatches: 9 } },
  ), /cumulative counter decreased/);
  assert.throws(() => powerDamResultWindow(
    { ...common, steps: 5 },
    { ...common, scenario: "dam-break-ui", steps: 6, simulationWall_ms: 110 },
  ), /same scenario and method/);
});

test("compute-pass attribution aggregates indexed native labels into stable owning stages", () => {
  const computePassesByLabel: Record<string, { calls: number; bytes: number }> = {
    "Begin direct structured publication": { calls: 2, bytes: 0 },
    "Finalize direct structured publication": { calls: 2, bytes: 0 },
    "Publish structured boundary coefficients": { calls: 2, bytes: 0 },
  };
  computePassesByLabel["Prepare global fine trajectory chunk 1/4"] = { calls: 2, bytes: 0 };
  computePassesByLabel["Prepare global fine trajectory chunk 2/4"] = { calls: 2, bytes: 0 };
  computePassesByLabel["Publish complete global fine velocity cache 1/1"] =
    { calls: 2, bytes: 0 };
  computePassesByLabel["Summarize global fine departure chunk 1/1"] = { calls: 2, bytes: 0 };
  computePassesByLabel["Finalize global fine departure chunk 1/1"] = { calls: 2, bytes: 0 };
  computePassesByLabel["Rank fixed octree fine-seed candidate records"] = { calls: 2, bytes: 0 };
  computePassesByLabel["Finalize octree fine-seed candidate publication"] = { calls: 2, bytes: 0 };
  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 100, gpuCommandAudit: { computePasses: 60, computePassesByLabel },
  });
  assert.deepEqual(summary.commands?.computePassesByStage, {
    "Structured velocity publication": 2,
    "Structured boundary coefficients": 1,
    "Fine transport · prepare trajectory chunks": 2,
    "Fine transport · velocity cache": 1,
    "Fine transport · summarize departure chunks": 1,
    "Fine transport · finalize departure summaries": 1,
    "Fine seed adapter": 2,
  });
  assert.equal(summary.commands?.computePassesByLabel["Begin direct structured publication"], 1);
});

test("compute-pass attribution is a closed ownership table", () => {
  assert.equal(powerDamComputePassStage("Prepare exact owner-page delta dispatch"), "Octree owner pages");
  assert.equal(powerDamComputePassStage("Publish complete global fine velocity cache 1/1"),
    "Fine transport · velocity cache");
  assert.equal(powerDamComputePassStage("A newly introduced mystery pass"), undefined);

  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 1, gpuCommandAudit: {
      computePasses: 4,
      computePassesByLabel: { "A newly introduced mystery pass": { calls: 4, bytes: 0 } },
    },
  });
  assert.equal(summary.commands?.computePassAttributionComplete, false);
  assert.deepEqual(summary.commands?.unownedComputePassLabels, ["A newly introduced mystery pass"]);
  assert.deepEqual(summary.commands?.computePassesByStage, {});
  assert.deepEqual(powerDamPerformanceFailures(summary, {
    maximumComputePassesPerAdvance: 60,
  }), ["compute passes/stage attribution incomplete (2.0 unattributed/advance)"]);
});

test("generic physics attribution is exact and aggregates repeated semantic phases", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 62,
    simulationWall_ms: 4_200,
    physicsTrace: physicsTrace([
      { id: "coarse-grid", label: "Adaptive coarse-grid topology", duration_ms: 9 },
      { id: "pressure-system", label: "Power operator", duration_ms: 20 },
      { id: "pressure-system", label: "Pressure rows", duration_ms: 12 },
      { id: "pressure-solve", label: "MGPCG", duration_ms: 7.36 },
      { id: "velocity-extrapolation", label: "Closest point", duration_ms: 23 },
      { id: "adaptive-publication", label: "Publication", duration_ms: 13.64 },
    ]),
  });
  assert.equal(summary.physicsTrace?.total_ms, 85);
  assert.equal(summary.physicsTrace?.accounted_ms, 85);
  assert.equal(summary.physicsTrace?.exact, true);
  assert.equal(summary.physicsTrace?.phaseTotals_ms["pressure-system"], 32);
  assert.equal(summary.physicsTrace?.phaseTotals_ms["velocity-extrapolation"], 23);
  assert.equal(summary.advanceWall_ms, 4_200 / 62);
});

test("NDJSON parsing ignores logs and selects octree result records", () => {
  assert.equal(powerDamResultFromLine("SAFETY: close browser tabs"), undefined);
  assert.equal(powerDamResultFromLine(JSON.stringify({ phase: "result", method: "uniform" })), undefined);
  assert.equal(powerDamResultFromLine(JSON.stringify({
    phase: "result", method: "octree", scenario: "dam-break-ui", steps: 62, simulationWall_ms: 4200,
  }))?.scenario, "dam-break-ui");
});

test("performance limits gate throughput and generic pressure-system attribution independently", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 120,
    gpuCommandAudit: {
      dispatches: 300, computePasses: 50,
      computePassesByLabel: { "Structured velocity publication": { calls: 50, bytes: 0 } },
    },
    physicsTrace: physicsTrace([
      { id: "pressure-system", label: "Pressure operator", duration_ms: 22 },
      { id: "pressure-solve", label: "MGPCG", duration_ms: 8 },
      { id: "other", label: "Other measured work", duration_ms: 50 },
    ]),
  });
  assert.deepEqual(powerDamPerformanceFailures(summary, {
    maximumAdvanceWall_ms: 59,
    maximumDispatchesPerAdvance: 149,
    maximumComputePassesPerAdvance: 24,
    maximumPressureNonSolve_ms: 21,
  }), [
    "advance wall 60.00 ms exceeds 59.00 ms",
    "dispatches/advance 150.0 exceeds 149.0",
    "compute passes/advance 25.0 exceeds 24.0",
    "pressure-system 22.00 ms exceeds 21.00 ms",
  ]);
});

test("compute-pass budget rejects missing and unnamed stage ownership", () => {
  const missing = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 1, gpuCommandAudit: { computePasses: 4 },
  });
  assert.deepEqual(powerDamPerformanceFailures(missing, {
    maximumComputePassesPerAdvance: 60,
  }), ["compute passes/stage attribution incomplete (2.0 unattributed/advance)"]);

  // `<unlabeled compute pass>` now has an owner: the three post-fence
  // `broker.compute()` calls in the pipelined MGPCG open a pass with no
  // descriptor, and they are bucketed as their own stage rather than folded
  // into "MGPCG solve" so the count stays visible. Attribution is therefore
  // complete for it -- while still being reported separately, which is the
  // point of the separate bucket.
  const unlabeled = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 1, gpuCommandAudit: {
      computePasses: 4,
      computePassesByLabel: { "<unlabeled compute pass>": { calls: 4, bytes: 0 } },
    },
  });
  assert.equal(powerDamComputePassStage("<unlabeled compute pass>"),
    "MGPCG solve · unlabeled broker pass",
    "an unlabeled broker pass must land in its own bucket, never inflate the solve");
  assert.deepEqual(powerDamPerformanceFailures(unlabeled, {
    maximumComputePassesPerAdvance: 60,
  }), []);

  // A label with no owner at all is still rejected: that is the case this gate
  // exists for, and bucketing the known unlabeled pass must not have opened a
  // hole for genuinely unattributed work.
  const unowned = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 1, gpuCommandAudit: {
      computePasses: 4,
      computePassesByLabel: { "A newly introduced mystery pass": { calls: 4, bytes: 0 } },
    },
  });
  assert.deepEqual(powerDamPerformanceFailures(unowned, {
    maximumComputePassesPerAdvance: 60,
  }), ["compute passes/stage attribution incomplete (2.0 unattributed/advance)"]);
});

test("configured structural gates fail closed when their audit source is absent", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 1,
    simulationWall_ms: 1,
  });
  assert.deepEqual(powerDamPerformanceFailures(summary, {
    maximumDispatchesPerAdvance: 300,
    maximumComputePassesPerAdvance: 60,
    maximumPressureNonSolve_ms: 4,
  }), [
    "dispatches/advance unavailable for configured gate",
    "compute passes/advance unavailable for configured gate",
    "pressure-system phase unavailable for configured gate",
  ]);
});
