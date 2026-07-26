import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { extname, relative, resolve } from "node:path";
import { performanceTraceIsExact, type PaperPhaseId } from "../lib/performance-trace";
import {
  OCTREE_WORK_STAGES,
  type OctreeWorkSnapshot,
  type OctreeWorkStage,
} from "../lib/webgpu-octree-work-accounting";
import {
  discoverOctreeProductionSources,
  extractEmbeddedWgslSources,
} from "./audit-octree-production-source";
import {
  powerDamComputePassStage,
  type PowerDamResultRecord,
} from "./power-dam-performance-report";

export const OCTREE_REGRESSION_SCHEMA_VERSION = 1 as const;
export type OctreeRegressionLane = "mini" | "ui" | "quiescent" | "moving-interface";
export type OctreeRegressionMetric =
  | "stageTime_ms"
  | "dispatchesPerAdvance"
  | "activeScheduledRatio"
  | "authorityBytes"
  | "residual"
  | "volumeDrift"
  | "energyRatio"
  | "dissipationRatio"
  | "topologyEpoch";

export interface OctreeRegressionBlocker {
  readonly metric: OctreeRegressionMetric;
  readonly reason: string;
}

export interface OctreeRegressionRevision {
  readonly sourceSha256: string;
  readonly shaderSha256: string;
  readonly sourceCount: number;
  readonly shaderCount: number;
}

export interface OctreeRegressionArtifact {
  readonly schemaVersion: typeof OCTREE_REGRESSION_SCHEMA_VERSION;
  readonly lane: OctreeRegressionLane;
  readonly capturedAt: string;
  readonly scenario: string;
  readonly method: string;
  readonly adapter: string;
  readonly contract: {
    readonly expectedSteps: number;
    readonly expectedSimulatedTime_s: number;
    readonly actualSteps: number;
    readonly actualSimulatedTime_s: number | null;
    readonly complete: boolean;
  };
  readonly revisions: OctreeRegressionRevision;
  readonly metrics: {
    readonly wallTime_ms: number;
    readonly stageTime_ms: Readonly<Partial<Record<PaperPhaseId, number>>> | null;
    readonly dispatchesPerAdvance: {
      readonly total: number;
      readonly byStage: Readonly<Record<string, number>>;
    } | null;
    readonly activeScheduledRatio: {
      readonly overall: number;
      readonly byStage: Readonly<Partial<Record<OctreeWorkStage, number>>>;
    } | null;
    readonly authorityBytes: {
      readonly total: number;
      readonly byAuthority: Readonly<Record<string, number>>;
    } | null;
    readonly residual: number | null;
    readonly volumeDrift: number | null;
    /** Maximum projection-energy amplification reported by the stability envelope. */
    readonly energyRatio: number | null;
    /** Maximum sampled compact mechanical-energy loss relative to reset-time energy. */
    readonly dissipationRatio: number | null;
    readonly topologyEpoch: number | null;
  };
  readonly blockers: readonly OctreeRegressionBlocker[];
}

export interface OctreeRegressionResultRecord extends PowerDamResultRecord {
  readonly simulatedTime_s?: number;
  readonly pressureRelativeResidual?: number;
  readonly volumeDrift?: number;
  readonly stabilityEnvelope?: {
    readonly maximumPressureRelativeResidual?: number;
    readonly maximumExactVolumeDrift?: number;
    readonly maximumProjectionEnergyRatio?: number;
    readonly projectionEnergySampleCount?: number;
  };
  readonly octreeWorkAccounting?: OctreeWorkSnapshot;
}

export const OCTREE_REGRESSION_CONTRACTS: Readonly<Record<OctreeRegressionLane, {
  readonly scenario: string;
  readonly expectedSteps: number;
  readonly expectedSimulatedTime_s: number;
}>> = Object.freeze({
  mini: Object.freeze({
    scenario: "minimal-power-dam-break", expectedSteps: 500, expectedSimulatedTime_s: 2,
  }),
  ui: Object.freeze({
    scenario: "dam-break-ui", expectedSteps: 62, expectedSimulatedTime_s: 0.496,
  }),
  quiescent: Object.freeze({
    scenario: "minimal-power-dam-break", expectedSteps: 60, expectedSimulatedTime_s: 0.24,
  }),
  "moving-interface": Object.freeze({
    scenario: "minimal-power-dam-break", expectedSteps: 62, expectedSimulatedTime_s: 0.248,
  }),
});

const finiteNonNegative = (value: number | undefined): number | null =>
  value !== undefined && Number.isFinite(value) && value >= 0 ? value : null;

function digestEntries(entries: readonly (readonly [string, string])[]): string {
  const digest = createHash("sha256");
  for (const [name, contents] of entries) {
    digest.update(name);
    digest.update("\0");
    digest.update(contents);
    digest.update("\0");
  }
  return digest.digest("hex");
}

/** Content-address the production host sources and embedded/standalone WGSL.
 * A dirty worktree therefore has a distinct revision even before commit. */
export function octreeRegressionRevision(repositoryRoot: string): OctreeRegressionRevision {
  const root = resolve(repositoryRoot);
  const paths = [...discoverOctreeProductionSources(root)].sort();
  const sources = paths.map((path) => [relative(root, path), readFileSync(path, "utf8")] as const);
  const shaders: Array<readonly [string, string]> = [];
  for (const [name, source] of sources) {
    if (extname(name) === ".wgsl") shaders.push([name, source]);
    else for (const shader of extractEmbeddedWgslSources(name, source)) {
      shaders.push([shader.source, shader.body]);
    }
  }
  return Object.freeze({
    sourceSha256: digestEntries(sources),
    shaderSha256: digestEntries(shaders),
    sourceCount: sources.length,
    shaderCount: shaders.length,
  });
}

function dispatchAttribution(
  result: OctreeRegressionResultRecord,
): OctreeRegressionArtifact["metrics"]["dispatchesPerAdvance"] {
  const audit = result.gpuCommandAudit;
  if (!audit || !Number.isSafeInteger(audit.dispatches) || audit.dispatches! < 0
    || !Number.isSafeInteger(result.steps) || result.steps < 1
    || !audit.dispatchesByPassLabel) return null;
  const byStage: Record<string, number> = {};
  let attributed = 0;
  for (const [label, bucket] of Object.entries(audit.dispatchesByPassLabel)) {
    const stage = powerDamComputePassStage(label);
    if (!stage || !Number.isSafeInteger(bucket.calls) || bucket.calls < 0) return null;
    attributed += bucket.calls;
    byStage[stage] = (byStage[stage] ?? 0) + bucket.calls / result.steps;
  }
  if (attributed !== audit.dispatches) return null;
  return Object.freeze({
    total: audit.dispatches / result.steps,
    byStage: Object.freeze(Object.fromEntries(Object.entries(byStage).sort(([a], [b]) =>
      a.localeCompare(b)))),
  });
}

function activeRatios(
  work: OctreeWorkSnapshot | undefined,
): OctreeRegressionArtifact["metrics"]["activeScheduledRatio"] {
  if (!work || !work.stagesComplete || work.activeLaneRatio === null
    || work.activeLanes === null || work.scheduledLanes === null) return null;
  const byStage: Partial<Record<OctreeWorkStage, number>> = {};
  let active = 0;
  let scheduled = 0;
  for (const stage of OCTREE_WORK_STAGES) {
    const counters = work.stages[stage];
    if (!counters) return null;
    const stageActive = counters.activeLanes;
    const stageScheduled = counters.scheduledLanes;
    if (!Number.isSafeInteger(stageActive) || stageActive === null || stageActive < 0
      || !Number.isSafeInteger(stageScheduled) || stageScheduled === null || stageScheduled < 0
      || stageActive > stageScheduled) return null;
    active += stageActive;
    scheduled += stageScheduled;
    byStage[stage] = stageScheduled === 0 ? 1 : stageActive / stageScheduled;
  }
  const overall = scheduled === 0 ? 1 : active / scheduled;
  if (active !== work.activeLanes || scheduled !== work.scheduledLanes
    || !Number.isFinite(work.activeLaneRatio)
    || Math.abs(overall - work.activeLaneRatio) > 1e-12) return null;
  return Object.freeze({ overall, byStage: Object.freeze(byStage) });
}

/** Compact reconstruction is the surviving end-to-end dissipation authority.
 * Reject partial reconstruction rather than treating uncovered or non-finite
 * liquid cells as zero-energy samples. */
function compactDissipationRatio(result: OctreeRegressionResultRecord): number | null {
  const samples = result.compactMechanicalEnergyCheckpoints;
  if (!samples || samples.length === 0) return null;
  let maximumLossRatio = 0;
  for (const sample of samples) {
    const retention = sample.mechanicalEnergyRetentionRatio;
    if (!Number.isFinite(retention) || retention! < 0
      || sample.publicationValid !== true
      || !Number.isSafeInteger(sample.rowCount) || sample.rowCount! < 1
      || sample.reconstructedRows !== sample.rowCount
      || sample.invalidRows !== 0
      || !Number.isSafeInteger(sample.liquidCellCount) || sample.liquidCellCount! < 1
      || sample.finiteLiquidCellCount !== sample.liquidCellCount) return null;
    maximumLossRatio = Math.max(maximumLossRatio, 1 - retention!);
  }
  return maximumLossRatio;
}

function stageTimes(
  result: OctreeRegressionResultRecord,
): OctreeRegressionArtifact["metrics"]["stageTime_ms"] {
  const trace = result.physicsTrace;
  if (!trace || trace.phases.length === 0 || !performanceTraceIsExact(trace)) return null;
  const values: Partial<Record<PaperPhaseId, number>> = {};
  for (const phase of trace.phases) values[phase.id] = (values[phase.id] ?? 0) + phase.duration_ms;
  return Object.freeze(values);
}

function blocker(
  blockers: OctreeRegressionBlocker[],
  metric: OctreeRegressionMetric,
  value: unknown,
  reason: string,
): void {
  if (value === null) blockers.push(Object.freeze({ metric, reason }));
}

function authorityInventory(
  work: OctreeWorkSnapshot | undefined,
): OctreeRegressionArtifact["metrics"]["authorityBytes"] {
  if (!work?.allocationInventoryComplete
    || !Number.isSafeInteger(work.authoritativeBytes) || work.authoritativeBytes < 0) return null;
  let total = 0;
  for (const bytes of Object.values(work.allocatedBytesByAuthority)) {
    if (!Number.isSafeInteger(bytes) || bytes < 0) return null;
    total += bytes;
  }
  if (total !== work.authoritativeBytes) return null;
  return Object.freeze({
    total,
    byAuthority: Object.freeze({ ...work.allocatedBytesByAuthority }),
  });
}

export function buildOctreeRegressionArtifact(input: {
  readonly lane: OctreeRegressionLane;
  readonly result: OctreeRegressionResultRecord;
  readonly repositoryRoot: string;
  readonly adapter?: string;
  readonly capturedAt?: string;
  readonly revisions?: OctreeRegressionRevision;
}): OctreeRegressionArtifact {
  const contract = OCTREE_REGRESSION_CONTRACTS[input.lane];
  const result = input.result;
  const work = result.octreeWorkAccounting;
  const stages = stageTimes(result);
  const dispatches = dispatchAttribution(result);
  const ratios = activeRatios(work);
  const authorityBytes = authorityInventory(work);
  const residual = finiteNonNegative(
    result.stabilityEnvelope?.maximumPressureRelativeResidual
      ?? result.pressureRelativeResidual,
  );
  const volumeDrift = finiteNonNegative(result.stabilityEnvelope?.maximumExactVolumeDrift
    ?? (result.volumeDrift === undefined ? undefined : Math.abs(result.volumeDrift)));
  const energyRatio = Number.isSafeInteger(result.stabilityEnvelope?.projectionEnergySampleCount)
    && result.stabilityEnvelope!.projectionEnergySampleCount! > 0
    ? finiteNonNegative(result.stabilityEnvelope?.maximumProjectionEnergyRatio)
    : null;
  const dissipationRatio = compactDissipationRatio(result);
  const topologyEpoch = work && Number.isSafeInteger(work.topologyEpoch) && work.topologyEpoch >= 0
    ? work.topologyEpoch : null;
  const blockers: OctreeRegressionBlocker[] = [];
  blocker(blockers, "stageTime_ms", stages,
    "exact physics stage timing was not published; capture with --profile");
  blocker(blockers, "dispatchesPerAdvance", dispatches,
    "dispatch audit is absent, incomplete, or contains an unowned pass label");
  blocker(blockers, "activeScheduledRatio", ratios,
    "work-accounting active/scheduled counters are null for at least one stage");
  blocker(blockers, "authorityBytes", authorityBytes,
    "work-accounting authority allocation inventory is absent or incomplete");
  blocker(blockers, "residual", residual,
    "neither the stability-envelope maximum nor terminal pressure residual is available");
  blocker(blockers, "volumeDrift", volumeDrift,
    "neither the stability-envelope maximum nor terminal compact volume drift is available");
  blocker(blockers, "energyRatio", energyRatio,
    "the stability envelope has no paired pre/post projection energy sample");
  blocker(blockers, "dissipationRatio", dissipationRatio,
    "complete compact mechanical-energy checkpoints are unavailable");
  blocker(blockers, "topologyEpoch", topologyEpoch,
    "the runtime did not publish a valid work-accounting topology epoch");
  const actualSimulatedTime_s = finiteNonNegative(result.simulatedTime_s);
  const complete = result.scenario === contract.scenario
    && result.steps === contract.expectedSteps
    && actualSimulatedTime_s !== null
    && Math.abs(actualSimulatedTime_s - contract.expectedSimulatedTime_s) <= 1e-6;
  return Object.freeze({
    schemaVersion: OCTREE_REGRESSION_SCHEMA_VERSION,
    lane: input.lane,
    capturedAt: input.capturedAt ?? new Date().toISOString(),
    scenario: result.scenario,
    method: result.method,
    adapter: input.adapter ?? "unknown",
    contract: Object.freeze({
      expectedSteps: contract.expectedSteps,
      expectedSimulatedTime_s: contract.expectedSimulatedTime_s,
      actualSteps: result.steps,
      actualSimulatedTime_s,
      complete,
    }),
    revisions: input.revisions ?? octreeRegressionRevision(input.repositoryRoot),
    metrics: Object.freeze({
      wallTime_ms: result.simulationWall_ms,
      stageTime_ms: stages,
      dispatchesPerAdvance: dispatches,
      activeScheduledRatio: ratios,
      authorityBytes,
      residual,
      volumeDrift,
      energyRatio,
      dissipationRatio,
      topologyEpoch,
    }),
    blockers: Object.freeze(blockers),
  });
}
