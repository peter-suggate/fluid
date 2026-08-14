import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { PerformanceTrace } from "../lib/core/performance-trace";
import {
  OCTREE_WORK_STAGES,
  type OctreeWorkSnapshot,
} from "../lib/methods/octree-shared/webgpu-octree-work-accounting";
import {
  buildOctreeRegressionArtifact,
  type OctreeRegressionArtifact,
  type OctreeRegressionResultRecord,
} from "../tools/octree-regression-artifact";
import {
  compareOctreeRegressionArtifacts,
  DEFAULT_OCTREE_REGRESSION_THRESHOLDS,
} from "../tools/compare-octree-regression";

const revisions = Object.freeze({
  sourceSha256: "1".repeat(64), shaderSha256: "2".repeat(64),
  sourceCount: 10, shaderCount: 20,
});

function workSnapshot(): OctreeWorkSnapshot {
  const stages = Object.fromEntries(OCTREE_WORK_STAGES.map((stage) => [stage, {
    scheduledLanes: 100, activeLanes: 50, activePages: 2, logicalPages: 4,
    worksets: 1, encodedIterations: 1, executedIterations: 1, reductionPasses: 0,
    estimatedBytesMoved: 400, estimatedBytesMovedPerActiveElement: 8,
  }])) as unknown as OctreeWorkSnapshot["stages"];
  return {
    topologyEpoch: 17, catalogStamps: 4, solverIterations: 8, reductionPasses: 0,
    scheduledLanes: 800, activeLanes: 400, activePages: 16, logicalPages: 32,
    worksetCount: 8, estimatedBytesMoved: 3_200, activeLaneRatio: 0.5,
    stagesComplete: true, missingStageMetrics: Object.fromEntries(
      OCTREE_WORK_STAGES.map((stage) => [stage, []]),
    ) as unknown as OctreeWorkSnapshot["missingStageMetrics"],
    stages,
    allocatedBytesByAuthority: { power: 1_000, "fine-level-set": 2_000 },
    allocatedScratchBytesByArena: { reductions: 500 },
    authoritativeBytes: 3_000, scratchBytes: 500, allocatedBytes: 3_500,
    allocationInventoryComplete: true,
  };
}

function trace(pressure_ms = 10): PerformanceTrace {
  return {
    sampleId: 1, domain: "gpu", lane: "physics", context: "regression-test",
    capturedAt_ms: 1, measurementSource: "gpu-hardware-timestamp",
    total_ms: 15 + pressure_ms,
    phases: [
      { id: "coarse-grid", label: "Coarse grid", duration_ms: 10 },
      { id: "pressure-system", label: "Pressure system", duration_ms: pressure_ms },
      { id: "velocity-projection", label: "Projection", duration_ms: 5 },
    ],
  };
}

function result(pressure_ms = 10): OctreeRegressionResultRecord {
  return {
    scenario: "minimal-power-dam-break", method: "power-liquids", phase: "result",
    steps: 500, simulatedTime_s: 2, simulationWall_ms: 1_000,
    physicsTrace: trace(pressure_ms),
    gpuCommandAudit: {
      dispatches: 20_000,
      dispatchesByPassLabel: {
        "Structured velocity publication": { calls: 5_000, bytes: 0 },
        "Octree MGPCG solve": { calls: 15_000, bytes: 0 },
      },
    },
    stabilityEnvelope: {
      maximumPressureRelativeResidual: 1e-5,
      maximumExactVolumeDrift: 1e-3,
      maximumProjectionEnergyRatio: 1.01,
      projectionEnergySampleCount: 10,
    },
    compactMechanicalEnergyCheckpoints: [{
      time_s: 2,
      mechanicalEnergyRetentionRatio: 0.94,
      publicationValid: true,
      rowCount: 32,
      reconstructedRows: 32,
      invalidRows: 0,
      liquidCellCount: 256,
      finiteLiquidCellCount: 256,
    }],
    octreeWorkAccounting: workSnapshot(),
  };
}

function artifact(pressure_ms = 10): OctreeRegressionArtifact {
  return buildOctreeRegressionArtifact({
    lane: "mini", result: result(pressure_ms), repositoryRoot: process.cwd(),
    adapter: "Apple M1 Max", capturedAt: "2026-07-26T00:00:00.000Z", revisions,
  });
}

test("artifact persists the complete attribution ledger without inventing counters", () => {
  const value = artifact();
  assert.equal(value.contract.complete, true);
  assert.equal(value.contract.actualSteps, 500);
  assert.equal(value.contract.actualSimulatedTime_s, 2);
  assert.equal(value.metrics.stageTime_ms?.["pressure-system"], 10);
  assert.equal(value.metrics.dispatchesPerAdvance?.total, 40);
  assert.deepEqual(value.metrics.dispatchesPerAdvance?.byStage, {
    "MGPCG solve": 30, "Structured velocity publication": 10,
  });
  assert.equal(value.metrics.activeScheduledRatio?.overall, 0.5);
  assert.equal(value.metrics.authorityBytes?.total, 3_000);
  assert.equal(value.metrics.residual, 1e-5);
  assert.equal(value.metrics.volumeDrift, 1e-3);
  assert.equal(value.metrics.energyRatio, 1.01);
  assert.ok(Math.abs(value.metrics.dissipationRatio! - 0.06) < 1e-12);
  assert.equal(value.metrics.topologyEpoch, 17);
  assert.deepEqual(value.blockers, []);
});

test("artifact records every unavailable runtime counter as a blocker", () => {
  const incomplete = buildOctreeRegressionArtifact({
    lane: "mini",
    result: {
      scenario: "minimal-power-dam-break", method: "power-liquids", phase: "result",
      steps: 500, simulatedTime_s: 2, simulationWall_ms: 1_000,
    },
    repositoryRoot: process.cwd(), revisions,
  });
  assert.deepEqual(incomplete.blockers.map(({ metric }) => metric), [
    "stageTime_ms", "dispatchesPerAdvance", "activeScheduledRatio", "authorityBytes",
    "residual", "volumeDrift", "energyRatio", "dissipationRatio", "topologyEpoch",
  ]);
  assert.equal(incomplete.metrics.activeScheduledRatio, null);
  assert.equal(incomplete.metrics.topologyEpoch, null);
});

test("comparator attributes the first regression to the earliest regressed stage", () => {
  const baseline = artifact(10);
  const candidate = artifact(11);
  const comparison = compareOctreeRegressionArtifacts(baseline, candidate);
  assert.equal(comparison.passed, false);
  assert.deepEqual(comparison.firstRegression, {
    kind: "regression", metric: "stageTime_ms", stage: "pressure-system",
    baseline: 10, candidate: 11, limit: 10.5,
    reason: "pressure-system stageTime_ms 11 exceeds 10.5",
  });
});

test("comparator fails closed on every null blocker metric", () => {
  const baseline = artifact();
  const metrics = ["stageTime_ms", "dispatchesPerAdvance", "activeScheduledRatio",
    "authorityBytes", "residual", "volumeDrift", "energyRatio", "dissipationRatio",
    "topologyEpoch"] as const;
  for (const metric of metrics) {
    const missing = {
      ...artifact(),
      blockers: [],
      metrics: { ...artifact().metrics, [metric]: null },
    } as OctreeRegressionArtifact;
    const comparison = compareOctreeRegressionArtifacts(baseline, missing);
    assert.equal(comparison.passed, false, metric);
    assert.ok(comparison.failures.some((failure) => failure.kind === "blocker"
      && failure.metric === metric && /fails closed/.test(failure.reason)), metric);
  }
});

test("mini artifact and comparator require exactly 500 steps and 2.0 simulated seconds", () => {
  const baseline = artifact();
  for (const [actualSteps, actualSimulatedTime_s] of [[499, 2], [500, 1.996], [501, 2.004]]) {
    const source = result();
    const built = buildOctreeRegressionArtifact({
      lane: "mini",
      result: { ...source, steps: actualSteps, simulatedTime_s: actualSimulatedTime_s },
      repositoryRoot: process.cwd(), revisions,
    });
    assert.equal(built.contract.complete, false);
    const comparison = compareOctreeRegressionArtifacts(baseline, built,
      DEFAULT_OCTREE_REGRESSION_THRESHOLDS);
    assert.equal(comparison.passed, false);
    assert.ok(comparison.failures.some(({ metric }) => metric === "candidate.mini-500-2s"));
  }
});

test("active/scheduled ratio is recomputed from a complete reconciled stage snapshot", () => {
  const source = result();
  const supplied = workSnapshot();
  const inconsistent = { ...supplied, activeLaneRatio: 0.75 } as OctreeWorkSnapshot;
  const built = buildOctreeRegressionArtifact({
    lane: "mini", result: { ...source, octreeWorkAccounting: inconsistent },
    repositoryRoot: process.cwd(), revisions,
  });
  assert.equal(built.metrics.activeScheduledRatio, null);
  assert.ok(built.blockers.some(({ metric }) => metric === "activeScheduledRatio"));
});

test("authority bytes require an exact inventory sum", () => {
  const source = result();
  const supplied = workSnapshot();
  const inconsistent = { ...supplied, authoritativeBytes: 3_001 } as OctreeWorkSnapshot;
  const built = buildOctreeRegressionArtifact({
    lane: "mini", result: { ...source, octreeWorkAccounting: inconsistent },
    repositoryRoot: process.cwd(), revisions,
  });
  assert.equal(built.metrics.authorityBytes, null);
  assert.ok(built.blockers.some(({ metric }) => metric === "authorityBytes"));
});

test("compact dissipation evidence rejects partial or non-finite reconstruction", () => {
  const source = result();
  for (const sample of [
    { ...source.compactMechanicalEnergyCheckpoints![0]!, reconstructedRows: 31 },
    { ...source.compactMechanicalEnergyCheckpoints![0]!, finiteLiquidCellCount: 255 },
    { ...source.compactMechanicalEnergyCheckpoints![0]!, mechanicalEnergyRetentionRatio: Number.NaN },
  ]) {
    const built = buildOctreeRegressionArtifact({
      lane: "mini", result: { ...source, compactMechanicalEnergyCheckpoints: [sample] },
      repositoryRoot: process.cwd(), revisions,
    });
    assert.equal(built.metrics.dissipationRatio, null);
    assert.ok(built.blockers.some(({ metric }) => metric === "dissipationRatio"));
  }
});

test("a numeric projection-energy value without a paired sample remains unavailable", () => {
  const source = result();
  const built = buildOctreeRegressionArtifact({
    lane: "mini",
    result: { ...source, stabilityEnvelope: {
      ...source.stabilityEnvelope!, projectionEnergySampleCount: 0,
    } },
    repositoryRoot: process.cwd(), revisions,
  });
  assert.equal(built.metrics.energyRatio, null);
  assert.ok(built.blockers.some(({ metric }) => metric === "energyRatio"));
});

test("package scripts expose all four capture lanes and the explicit comparator", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  assert.match(packageJson.scripts["capture:octree-regression-mini"], /--lane=mini .*--artifact=/);
  assert.match(packageJson.scripts["capture:octree-regression-ui"], /--lane=ui .*--artifact=/);
  assert.match(packageJson.scripts["capture:octree-regression-quiescent"], /--lane=quiescent .*--artifact=/);
  assert.match(packageJson.scripts["capture:octree-regression-moving-interface"],
    /--lane=moving-interface .*--artifact=/);
  assert.equal(packageJson.scripts["compare:octree-regression"],
    "node --import tsx tools/compare-octree-regression.ts");
});
