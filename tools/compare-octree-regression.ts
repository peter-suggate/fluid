#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { OctreeRegressionArtifact, OctreeRegressionMetric } from "./octree-regression-artifact";

export interface OctreeRegressionThresholds {
  readonly stageTimeFraction: number;
  readonly dispatchFraction: number;
  readonly activeRatioFraction: number;
  readonly authorityBytesFraction: number;
  readonly residualFraction: number;
  readonly residualAbsolute: number;
  readonly volumeDriftFraction: number;
  readonly volumeDriftAbsolute: number;
  readonly energyRatioFraction: number;
  readonly energyRatioAbsolute: number;
  readonly dissipationRatioFraction: number;
  readonly dissipationRatioAbsolute: number;
  readonly wallTimeFraction: number;
}

export const DEFAULT_OCTREE_REGRESSION_THRESHOLDS: OctreeRegressionThresholds = Object.freeze({
  stageTimeFraction: 0.05,
  dispatchFraction: 0,
  activeRatioFraction: 0.05,
  authorityBytesFraction: 0,
  residualFraction: 0.10,
  residualAbsolute: 1e-7,
  volumeDriftFraction: 0.10,
  volumeDriftAbsolute: 1e-5,
  energyRatioFraction: 0.05,
  energyRatioAbsolute: 1e-4,
  dissipationRatioFraction: 0.05,
  dissipationRatioAbsolute: 1e-4,
  wallTimeFraction: 0.05,
});

export interface OctreeRegressionFailure {
  readonly kind: "blocker" | "contract" | "regression";
  readonly metric: string;
  readonly stage?: string;
  readonly baseline?: number;
  readonly candidate?: number | null;
  readonly limit?: number;
  readonly reason: string;
}

export interface OctreeRegressionComparison {
  readonly passed: boolean;
  readonly lane: string;
  readonly baselineRevision: OctreeRegressionArtifact["revisions"];
  readonly candidateRevision: OctreeRegressionArtifact["revisions"];
  readonly firstFailure?: OctreeRegressionFailure;
  readonly firstRegression?: OctreeRegressionFailure;
  readonly failures: readonly OctreeRegressionFailure[];
}

const PHASE_ORDER = [
  "frame-control", "scene-upload", "command-encoding", "coarse-grid", "power-topology",
  "velocity-advection", "pressure-system", "pressure-solve", "velocity-projection",
  "velocity-extrapolation", "fine-sdf-advection", "fine-sdf-redistance",
  "adaptive-publication", "surface-extraction", "other",
] as const;
const WORK_STAGE_ORDER = [
  "topology-publication", "forces-and-solids", "fine-transport", "rhs",
  "pressure-solve", "projection", "fine-redistance", "candidate-build",
] as const;

function orderedKeys(values: Readonly<Record<string, unknown>>, preferred: readonly string[]): string[] {
  const keys = Object.keys(values);
  return [
    ...preferred.filter((key) => keys.includes(key)),
    ...keys.filter((key) => !preferred.includes(key)).sort(),
  ];
}

function regression(
  failures: OctreeRegressionFailure[],
  metric: string,
  baseline: number,
  candidate: number,
  limit: number,
  stage?: string,
): void {
  if (candidate <= limit) return;
  failures.push(Object.freeze({
    kind: "regression", metric, stage, baseline, candidate, limit,
    reason: `${stage ? `${stage} ` : ""}${metric} ${candidate} exceeds ${limit}`,
  }));
}

function decreaseRegression(
  failures: OctreeRegressionFailure[],
  metric: string,
  baseline: number,
  candidate: number,
  limit: number,
  stage?: string,
): void {
  if (candidate >= limit) return;
  failures.push(Object.freeze({
    kind: "regression", metric, stage, baseline, candidate, limit,
    reason: `${stage ? `${stage} ` : ""}${metric} ${candidate} is below ${limit}`,
  }));
}

function missingMetricFailure(
  artifactName: "baseline" | "candidate",
  metric: OctreeRegressionMetric,
): OctreeRegressionFailure {
  return Object.freeze({
    kind: "blocker", metric,
    reason: `${artifactName} ${metric} is unavailable; regression comparison fails closed`,
  });
}

export function compareOctreeRegressionArtifacts(
  baseline: OctreeRegressionArtifact,
  candidate: OctreeRegressionArtifact,
  thresholds: OctreeRegressionThresholds = DEFAULT_OCTREE_REGRESSION_THRESHOLDS,
): OctreeRegressionComparison {
  const failures: OctreeRegressionFailure[] = [];
  if (baseline.schemaVersion !== 1 || candidate.schemaVersion !== 1) {
    failures.push({ kind: "blocker", metric: "schemaVersion", reason: "unsupported artifact schema" });
  }
  if (baseline.lane !== candidate.lane) {
    failures.push({ kind: "contract", metric: "lane", reason: `lane mismatch ${baseline.lane} != ${candidate.lane}` });
  }
  if (baseline.scenario !== candidate.scenario || baseline.method !== candidate.method) {
    failures.push({ kind: "contract", metric: "scenario/method", reason: "scenario or method mismatch" });
  }
  for (const entry of baseline.blockers ?? []) failures.push({
    kind: "blocker", metric: entry.metric, reason: `baseline blocker: ${entry.reason}`,
  });
  for (const entry of candidate.blockers ?? []) failures.push({
    kind: "blocker", metric: entry.metric, reason: `candidate blocker: ${entry.reason}`,
  });
  const miniContractIsExact = (artifact: OctreeRegressionArtifact): boolean =>
    artifact.lane !== "mini"
      || (artifact.contract.expectedSteps === 500
        && artifact.contract.expectedSimulatedTime_s === 2
        && artifact.contract.actualSteps === 500
        && artifact.contract.actualSimulatedTime_s !== null
        && Math.abs(artifact.contract.actualSimulatedTime_s - 2) <= 1e-6);
  if (!baseline.contract.complete) {
    failures.push({ kind: "blocker", metric: "baseline.contract", reason: "baseline did not complete its lane contract" });
  }
  if (!candidate.contract.complete) {
    failures.push({
      kind: "contract", metric: "candidate.contract",
      candidate: candidate.contract.actualSimulatedTime_s,
      reason: `candidate completed ${candidate.contract.actualSteps}/${candidate.contract.expectedSteps} steps and ${candidate.contract.actualSimulatedTime_s ?? "unknown"}/${candidate.contract.expectedSimulatedTime_s} simulated seconds`,
    });
  }
  if (!miniContractIsExact(baseline)) failures.push({
    kind: "contract", metric: "baseline.mini-500-2s",
    reason: "baseline mini lane is not the exact 500-step/2-s minimal-dam contract",
  });
  if (!miniContractIsExact(candidate)) failures.push({
    kind: "contract", metric: "candidate.mini-500-2s",
    reason: "candidate mini lane is not the exact 500-step/2-s minimal-dam contract",
  });

  const metricPairs: Array<readonly [OctreeRegressionMetric, unknown, unknown]> = [
    ["stageTime_ms", baseline.metrics.stageTime_ms, candidate.metrics.stageTime_ms],
    ["dispatchesPerAdvance", baseline.metrics.dispatchesPerAdvance, candidate.metrics.dispatchesPerAdvance],
    ["activeScheduledRatio", baseline.metrics.activeScheduledRatio, candidate.metrics.activeScheduledRatio],
    ["authorityBytes", baseline.metrics.authorityBytes, candidate.metrics.authorityBytes],
    ["residual", baseline.metrics.residual, candidate.metrics.residual],
    ["volumeDrift", baseline.metrics.volumeDrift, candidate.metrics.volumeDrift],
    ["energyRatio", baseline.metrics.energyRatio, candidate.metrics.energyRatio],
    ["dissipationRatio", baseline.metrics.dissipationRatio, candidate.metrics.dissipationRatio],
    ["topologyEpoch", baseline.metrics.topologyEpoch, candidate.metrics.topologyEpoch],
  ];
  for (const [metric, baselineValue, candidateValue] of metricPairs) {
    if (baselineValue == null) failures.push(missingMetricFailure("baseline", metric));
    if (candidateValue == null) failures.push(missingMetricFailure("candidate", metric));
  }
  if (failures.some((failure) => failure.kind === "blocker")) {
    return Object.freeze({
      passed: false, lane: candidate.lane,
      baselineRevision: baseline.revisions, candidateRevision: candidate.revisions,
      firstFailure: failures[0], failures: Object.freeze(failures),
    });
  }

  const baseTimes = baseline.metrics.stageTime_ms!;
  const nextTimes = candidate.metrics.stageTime_ms!;
  for (const stage of orderedKeys({ ...baseTimes, ...nextTimes }, PHASE_ORDER)) {
    const before = baseTimes[stage as keyof typeof baseTimes];
    const after = nextTimes[stage as keyof typeof nextTimes];
    if (before === undefined || after === undefined) {
      failures.push({ kind: "blocker", metric: "stageTime_ms", stage,
        reason: `stage timing key ${stage} is absent from one artifact` });
    } else regression(failures, "stageTime_ms", before, after,
      before * (1 + thresholds.stageTimeFraction), stage);
  }

  const baseDispatch = baseline.metrics.dispatchesPerAdvance!;
  const nextDispatch = candidate.metrics.dispatchesPerAdvance!;
  for (const stage of orderedKeys({ ...baseDispatch.byStage, ...nextDispatch.byStage }, PHASE_ORDER)) {
    const before = baseDispatch.byStage[stage];
    const after = nextDispatch.byStage[stage];
    if (before === undefined || after === undefined) {
      failures.push({ kind: "blocker", metric: "dispatchesPerAdvance", stage,
        reason: `dispatch stage ${stage} is absent from one artifact` });
    } else regression(failures, "dispatchesPerAdvance", before, after,
      before * (1 + thresholds.dispatchFraction), stage);
  }
  regression(failures, "dispatchesPerAdvance.total", baseDispatch.total, nextDispatch.total,
    baseDispatch.total * (1 + thresholds.dispatchFraction));

  const baseRatio = baseline.metrics.activeScheduledRatio!;
  const nextRatio = candidate.metrics.activeScheduledRatio!;
  for (const stage of orderedKeys({ ...baseRatio.byStage, ...nextRatio.byStage }, WORK_STAGE_ORDER)) {
    const before = baseRatio.byStage[stage as keyof typeof baseRatio.byStage];
    const after = nextRatio.byStage[stage as keyof typeof nextRatio.byStage];
    if (before === undefined || after === undefined) {
      failures.push({ kind: "blocker", metric: "activeScheduledRatio", stage,
        reason: `active/scheduled stage ${stage} is absent from one artifact` });
    } else decreaseRegression(failures, "activeScheduledRatio", before, after,
      before * (1 - thresholds.activeRatioFraction), stage);
  }
  decreaseRegression(failures, "activeScheduledRatio.overall", baseRatio.overall,
    nextRatio.overall, baseRatio.overall * (1 - thresholds.activeRatioFraction));

  const baseBytes = baseline.metrics.authorityBytes!;
  const nextBytes = candidate.metrics.authorityBytes!;
  for (const authority of orderedKeys({ ...baseBytes.byAuthority, ...nextBytes.byAuthority }, [])) {
    const before = baseBytes.byAuthority[authority];
    const after = nextBytes.byAuthority[authority];
    if (before === undefined || after === undefined) {
      failures.push({ kind: "blocker", metric: "authorityBytes", stage: authority,
        reason: `authority ${authority} is absent from one artifact` });
    } else regression(failures, "authorityBytes", before, after,
      before * (1 + thresholds.authorityBytesFraction), authority);
  }
  regression(failures, "authorityBytes.total", baseBytes.total, nextBytes.total,
    baseBytes.total * (1 + thresholds.authorityBytesFraction));

  const scalar = (metric: "residual" | "volumeDrift" | "energyRatio" | "dissipationRatio", fraction: number,
    absolute: number): void => {
    const before = baseline.metrics[metric]!;
    const after = candidate.metrics[metric]!;
    regression(failures, metric, before, after, before * (1 + fraction) + absolute);
  };
  scalar("residual", thresholds.residualFraction, thresholds.residualAbsolute);
  scalar("volumeDrift", thresholds.volumeDriftFraction, thresholds.volumeDriftAbsolute);
  scalar("energyRatio", thresholds.energyRatioFraction, thresholds.energyRatioAbsolute);
  scalar("dissipationRatio", thresholds.dissipationRatioFraction,
    thresholds.dissipationRatioAbsolute);
  decreaseRegression(failures, "topologyEpoch", baseline.metrics.topologyEpoch!,
    candidate.metrics.topologyEpoch!, baseline.metrics.topologyEpoch!);
  regression(failures, "wallTime_ms", baseline.metrics.wallTime_ms,
    candidate.metrics.wallTime_ms, baseline.metrics.wallTime_ms * (1 + thresholds.wallTimeFraction));

  const firstRegression = failures.find((failure) => failure.kind === "regression");
  return Object.freeze({
    passed: failures.length === 0,
    lane: candidate.lane,
    baselineRevision: baseline.revisions,
    candidateRevision: candidate.revisions,
    firstFailure: failures[0], firstRegression,
    failures: Object.freeze(failures),
  });
}

function parseArgument(name: string): string | undefined {
  return process.argv.find((argument) => argument.startsWith(`--${name}=`))?.slice(name.length + 3);
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  const candidatePath = parseArgument("candidate");
  if (!candidatePath) throw new Error("--candidate=<artifact.json> is required");
  const candidate = JSON.parse(readFileSync(resolve(candidatePath), "utf8")) as OctreeRegressionArtifact;
  const baselinePath = parseArgument("baseline")
    ?? `docs/baselines/octree-regression/${candidate.lane}.json`;
  const baseline = JSON.parse(readFileSync(resolve(baselinePath), "utf8")) as OctreeRegressionArtifact;
  const comparison = compareOctreeRegressionArtifacts(baseline, candidate);
  if (process.argv.includes("--json")) console.log(JSON.stringify(comparison));
  else if (comparison.passed) console.log(`octree regression gate: ${candidate.lane} passed`);
  else {
    console.error(`octree regression gate: ${candidate.lane} failed (${comparison.failures.length})`);
    for (const failure of comparison.failures) console.error(`- ${failure.reason}`);
  }
  if (!comparison.passed) process.exitCode = 1;
}
