import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { basename, extname, join, relative, resolve } from "node:path";
import { performanceTraceIsExact, type PaperPhaseId } from "../lib/core/performance-trace";
import type { PowerDamRunEnvironment } from "./power-dam-run-environment";
import {
  OCTREE_WORK_STAGES,
  type OctreeWorkSnapshot,
  type OctreeWorkStage,
} from "../lib/methods/octree-shared/webgpu-octree-work-accounting";
import {
  powerDamComputePassStage,
  type PowerDamResultRecord,
} from "./power-dam-performance-report";

/** Presentation and one-shot inspection paths are not part of the recurring
 * simulation graph. Everything else with an octree production prefix is
 * scanned; the list is intentionally an exclusion list so new files enter the
 * revision digest automatically. */
const NON_RECURRING_SOURCE_BASENAMES = new Set([
  "webgpu-octree-sparse-bricks.ts",
  "webgpu-octree-technique-overlay.ts",
  "webgpu-octree-voxel-inspection.ts",
  "webgpu-octree-work-accounting.ts",
]);

/** Recurring simulation modules whose names do not carry an octree prefix, so
 * the prefix filter above silently skipped them. Both are encoded every
 * advance from `webgpu-octree.ts` -- brick residency publishes the topology the
 * octree consumes, and the sparse mutation module rewrites it. Membership here
 * is an inclusion list only because the naming convention is not uniform;
 * anything that acquires an octree prefix is picked up automatically. */
const ADDITIONAL_RECURRING_SOURCE_BASENAMES = new Set([
  "webgpu-fluid-brick-residency.ts",
  "webgpu-sparse-brick-topology-mutation.ts",
]);

export interface EmbeddedWgslSource {
  readonly source: string;
  readonly body: string;
  readonly bodyOffset: number;
  readonly bodyLine: number;
  readonly templateStart: number;
  readonly templateEnd: number;
}

function lineAt(source: string, offset: number): number {
  let line = 1;
  for (let index = 0; index < offset; index += 1) {
    if (source.charCodeAt(index) === 10) line += 1;
  }
  return line;
}

/** Discover WGSL template literals by syntax, not exported constant names.
 * This makes every new embedded shader enter the revision digest automatically. */
export function extractEmbeddedWgslSources(
  sourceName: string,
  source: string,
): readonly EmbeddedWgslSource[] {
  const shaders: EmbeddedWgslSource[] = [];
  for (let cursor = 0; cursor < source.length;) {
    const templateStart = source.indexOf("`", cursor);
    if (templateStart < 0) break;
    let templateEnd = templateStart + 1;
    while (templateEnd < source.length) {
      if (source[templateEnd] === "\\") { templateEnd += 2; continue; }
      if (source[templateEnd] === "`") break;
      templateEnd += 1;
    }
    if (templateEnd >= source.length) break;
    const bodyOffset = templateStart + 1;
    const body = source.slice(bodyOffset, templateEnd);
    if (/(?:@(?:compute|vertex|fragment|group|binding)\b|\bstruct\s+[A-Za-z_]\w*\s*\{)/.test(body)) {
      shaders.push(Object.freeze({
        source: `${sourceName}#wgsl-${shaders.length + 1}`,
        body,
        bodyOffset,
        bodyLine: lineAt(source, bodyOffset),
        templateStart,
        templateEnd: templateEnd + 1,
      }));
    }
    cursor = templateEnd + 1;
  }
  return Object.freeze(shaders);
}

function walk(directory: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(directory).sort()) {
    const path = join(directory, entry);
    const stats = statSync(path);
    if (stats.isDirectory()) files.push(...walk(path));
    else files.push(path);
  }
  return files;
}

/** The recurring octree production surface, content-addressed by
 * `octreeRegressionRevision` so a dirty worktree gets its own revision. */
export function discoverOctreeProductionSources(repositoryRoot: string): readonly string[] {
  const libraryRoot = join(resolve(repositoryRoot), "lib");
  return walk(libraryRoot).filter((path) => {
    const name = basename(path);
    if (NON_RECURRING_SOURCE_BASENAMES.has(name)) return false;
    const extension = extname(name);
    if (extension !== ".ts" && extension !== ".wgsl") return false;
    return name === "webgpu-octree.ts"
      || name.startsWith("webgpu-octree-")
      || name.startsWith("octree-")
      || ADDITIONAL_RECURRING_SOURCE_BASENAMES.has(name);
  });
}

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
  /**
   * The environment this capture actually ran in. Optional only so artifacts
   * written before ledger C7 still parse; every new capture carries one, and a
   * comparison against an artifact without it is a comparison against an
   * unknown.
   */
  readonly environment?: PowerDamRunEnvironment;
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
  /** Exact cause when a work-accounting stage failed its validity gate. */
  readonly octreeWorkAccountingBlocker?: string;
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

/** A metric together with the precise reason it could not be published. The
 * reason is what the blocker records, so a capture names its own defect
 * instead of restating the generic failure class. */
interface OctreeRegressionAttempt<T> {
  readonly value: T | null;
  readonly reason: string;
}

const attempt = <T>(value: T): OctreeRegressionAttempt<T> => ({ value, reason: "" });
const rejected = <T>(reason: string): OctreeRegressionAttempt<T> => ({ value: null, reason });

/** Keep blocker reasons bounded when a whole family of labels/stages fails. */
function listed(names: readonly string[], limit = 6): string {
  return names.length <= limit
    ? names.join(", ")
    : `${names.slice(0, limit).join(", ")} (+${names.length - limit} more)`;
}

function dispatchAttribution(
  result: OctreeRegressionResultRecord,
): OctreeRegressionAttempt<OctreeRegressionArtifact["metrics"]["dispatchesPerAdvance"]> {
  const audit = result.gpuCommandAudit;
  if (!audit || !Number.isSafeInteger(audit.dispatches) || audit.dispatches! < 0
    || !Number.isSafeInteger(result.steps) || result.steps < 1
    || !audit.dispatchesByPassLabel) {
    return rejected("dispatch audit is absent or malformed; capture with the command audit enabled");
  }
  const byStage: Record<string, number> = {};
  const unowned: string[] = [];
  const malformed: string[] = [];
  let attributed = 0;
  for (const [label, bucket] of Object.entries(audit.dispatchesByPassLabel)) {
    if (!Number.isSafeInteger(bucket.calls) || bucket.calls < 0) { malformed.push(label); continue; }
    const stage = powerDamComputePassStage(label);
    if (!stage) { unowned.push(label); continue; }
    attributed += bucket.calls;
    byStage[stage] = (byStage[stage] ?? 0) + bucket.calls / result.steps;
  }
  if (malformed.length > 0) {
    return rejected(`dispatch audit bucket is not a counter for pass label(s): ${listed(malformed.sort())}`);
  }
  if (unowned.length > 0) {
    return rejected("dispatch audit contains pass label(s) with no owning stage in "
      + `POWER_DAM_COMPUTE_PASS_OWNERSHIP: ${listed(unowned.sort())}`);
  }
  if (attributed !== audit.dispatches) {
    return rejected(`dispatch label buckets sum to ${attributed} of ${audit.dispatches} audited dispatches`);
  }
  return attempt(Object.freeze({
    total: audit.dispatches / result.steps,
    byStage: Object.freeze(Object.fromEntries(Object.entries(byStage).sort(([a], [b]) =>
      a.localeCompare(b)))),
  }));
}

function activeRatios(
  work: OctreeWorkSnapshot | undefined,
): OctreeRegressionAttempt<OctreeRegressionArtifact["metrics"]["activeScheduledRatio"]> {
  if (!work) return rejected("work accounting was not published by the runtime");
  // Name the unobserved counters. Every stage record is authored by one
  // fail-closed decode in `recordOctreeRuntimeGPUWork`, so a null stage means
  // that stage's GPU validity gate rejected — not that it did no work.
  const missingLanes = OCTREE_WORK_STAGES.filter((stage) =>
    work.stages[stage]?.activeLanes === null || work.stages[stage]?.scheduledLanes === null);
  if (missingLanes.length > 0) {
    return rejected(`work-accounting active/scheduled lanes are null for stage(s): ${listed(missingLanes)}`);
  }
  if (!work.stagesComplete) {
    const incomplete = OCTREE_WORK_STAGES.flatMap((stage) =>
      (work.missingStageMetrics[stage] ?? []).map((metric) => `${stage}.${metric}`));
    return rejected(`work-accounting stage metrics are incomplete: ${listed(incomplete)}`);
  }
  if (work.activeLaneRatio === null || work.activeLanes === null || work.scheduledLanes === null) {
    return rejected("work-accounting lane totals are null while every stage published counters");
  }
  const byStage: Partial<Record<OctreeWorkStage, number>> = {};
  let active = 0;
  let scheduled = 0;
  for (const stage of OCTREE_WORK_STAGES) {
    const counters = work.stages[stage];
    const stageActive = counters.activeLanes;
    const stageScheduled = counters.scheduledLanes;
    if (!Number.isSafeInteger(stageActive) || stageActive === null || stageActive < 0
      || !Number.isSafeInteger(stageScheduled) || stageScheduled === null || stageScheduled < 0
      || stageActive > stageScheduled) {
      return rejected(`work-accounting lane counters for stage ${stage} are not an active<=scheduled pair`);
    }
    active += stageActive;
    scheduled += stageScheduled;
    byStage[stage] = stageScheduled === 0 ? 1 : stageActive / stageScheduled;
  }
  const overall = scheduled === 0 ? 1 : active / scheduled;
  if (active !== work.activeLanes || scheduled !== work.scheduledLanes
    || !Number.isFinite(work.activeLaneRatio)
    || Math.abs(overall - work.activeLaneRatio) > 1e-12) {
    return rejected("work-accounting stage lane sums do not reconcile with the published totals");
  }
  return attempt(Object.freeze({ overall, byStage: Object.freeze(byStage) }));
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
  readonly environment?: PowerDamRunEnvironment;
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
  blocker(blockers, "dispatchesPerAdvance", dispatches.value, dispatches.reason);
  blocker(blockers, "activeScheduledRatio", ratios.value, ratios.reason);
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
    ...(input.environment ? { environment: input.environment } : {}),
    metrics: Object.freeze({
      wallTime_ms: result.simulationWall_ms,
      stageTime_ms: stages,
      dispatchesPerAdvance: dispatches.value,
      activeScheduledRatio: ratios.value,
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
