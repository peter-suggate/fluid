import {
  performanceTraceAccounted_ms,
  performanceTraceIsExact,
  type PaperPhaseId,
  type PerformanceTrace,
} from "../lib/performance-trace";
import type { GPUDataFlowManifest } from "./webgpu-data-flow-manifest";

export interface PowerDamCommandBucket {
  readonly calls: number;
  readonly bytes: number;
}

export interface PowerDamCommandAudit {
  readonly clearBuffer?: PowerDamCommandBucket;
  readonly copyBufferToBuffer?: PowerDamCommandBucket;
  readonly computePasses?: number;
  readonly dispatches?: number;
  readonly indirectDispatches?: number;
  readonly computePassesByLabel?: Readonly<Record<string, PowerDamCommandBucket>>;
  readonly dispatchesByPassLabel?: Readonly<Record<string, PowerDamCommandBucket>>;
}

export interface PowerDamFineTimestampBucket {
  readonly samples: number;
  readonly total_ms: number;
  readonly mean_ms: number;
  readonly minimum_ms: number;
  readonly maximum_ms: number;
}

export interface PowerDamFineTimestampReport {
  readonly measuredAdvances: number;
  readonly measuredPasses: number;
  readonly invalidPasses: number;
  readonly summedPass_ms: number;
  readonly byLabel: Readonly<Record<string, PowerDamFineTimestampBucket>>;
}

export interface PowerDamResultRecord {
  readonly scenario: string;
  readonly method: string;
  readonly phase: "result";
  readonly steps: number;
  readonly simulationWall_ms: number;
  readonly validationErrors?: readonly string[];
  readonly gpuCommandAudit?: PowerDamCommandAudit;
  readonly gpuFineTimestamps?: PowerDamFineTimestampReport;
  readonly gpuDataFlowManifest?: GPUDataFlowManifest;
  readonly physicsTrace?: PerformanceTrace;
}

export interface PowerDamPerformanceSummary {
  readonly scenario: string;
  readonly method: string;
  readonly steps: number;
  readonly simulationWall_ms: number;
  readonly advanceWall_ms: number;
  readonly validationErrorCount: number;
  readonly commands?: {
    readonly dispatchesPerAdvance: number;
    readonly indirectDispatchesPerAdvance: number;
    readonly computePassesPerAdvance: number;
    /** Stable owning-stage buckets, normalized by accepted advances. */
    readonly computePassesByStage: Readonly<Record<string, number>>;
    /** Raw descriptor labels retained for contributor-level drill-down. */
    readonly computePassesByLabel: Readonly<Record<string, number>>;
    /** True only when every audited pass maps through the closed ownership
     * table and the label buckets exactly reconcile with the total counter. */
    readonly computePassAttributionComplete: boolean;
    readonly unattributedComputePassesPerAdvance: number;
    /** Non-empty raw labels which did not match a registered owning stage. */
    readonly unownedComputePassLabels: readonly string[];
    readonly mgpcgDispatchesPerAdvance: number;
    readonly mgpcgDispatchFraction: number;
    readonly clearBytesPerAdvance: number;
    readonly copyBytesPerAdvance: number;
  };
  readonly fineTimestamps?: {
    readonly measuredAdvances: number;
    readonly measuredPasses: number;
    readonly invalidPasses: number;
    readonly summedPass_ms: number;
    readonly summedPassPerAdvance_ms: number;
    readonly byLabel: Readonly<Record<string, PowerDamFineTimestampBucket & {
      readonly totalPerAdvance_ms: number;
    }>>;
  };
  /** Machine-readable lineage captured from actual pipeline/bind-group/dispatch
   * state during the same advances as fine GPU timestamps. */
  readonly dataFlow?: GPUDataFlowManifest;
  /** Exact, exclusive generic physics accounting from one sampled advance. */
  readonly physicsTrace?: {
    readonly sampleId: number;
    readonly measurementSource?: PerformanceTrace["measurementSource"];
    readonly total_ms: number;
    readonly accounted_ms: number;
    readonly exact: boolean;
    readonly phases: PerformanceTrace["phases"];
    readonly phaseTotals_ms: Partial<Record<PaperPhaseId, number>>;
  };
}

export interface PowerDamPerformanceLimits {
  readonly maximumAdvanceWall_ms?: number;
  readonly maximumDispatchesPerAdvance?: number;
  readonly maximumComputePassesPerAdvance?: number;
  readonly maximumPressureNonSolve_ms?: number;
}

const perAdvance = (value: number | undefined, steps: number): number => (value ?? 0) / steps;

interface PowerDamComputePassOwnershipRule {
  readonly stage: string;
  readonly label: RegExp;
}

/** Closed ownership table for compute-pass labels emitted by the octree power
 * lane. Rules may normalize bounded loop/chunk suffixes, but there is
 * deliberately no raw-label fallback: a newly introduced label must register
 * its owning module here or the pass-budget audit fails closed. */
const POWER_DAM_COMPUTE_PASS_OWNERSHIP: readonly PowerDamComputePassOwnershipRule[] = [
  { stage: "Octree owner pages", label: /owner[- ]page|owner pages|analytic octree owner generation/i },
  { stage: "Power energy ledger", label: /^Power energy ledger\b/i },
  { stage: "Fine seed adapter", label: /octree fine-seed|octree interface candidates|FineSeedLeaf to global fine seeds|Seed global fine bricks from (?:FineSeedLeaf candidates|every interface leaf)|fine-seed brick residency/i },
  { stage: "Section 5 face band · catalog adjacency", label: /^(?:Classify exact Section 5 catalog-adjacency delta|Resolve (?:exact affected )?Section 5 catalog adjacency|Validate and publish Section 5 catalog adjacency)$/i },
  { stage: "Section 5 face band · topology", label: /^(?:Build exact Section 5 support-row delta|Publish Section 5 regular-face topology)$/i },
  { stage: "Section 5 face band · extrapolation", label: /^(?:Extrapolate Section 5 regular-face velocities|Publish grouped face-band transport authority)$/i },
  { stage: "Section 5 face band · power publication", label: /^(?:Publish Section 5 regular velocities to power faces|Prepare mandatory Section 5 regular-face completion|Complete power-face advection from Section 5 regular band|Publish completed Section 5 power-face advection)$/i },
  { stage: "Section 5 old-mesh face advection", label: /(?:Section 5 old-mesh|old-mesh face advection|old power interpolation|live old power interpolation|2017 old-mesh)/i },
  { stage: "Fine transport · prepare trajectory chunks", label: /^Prepare global fine trajectory chunk \d+\/\d+$/i },
  { stage: "Fine transport · advance trajectories", label: /^Advance global fine trajectories \d+\/\d+$/i },
  { stage: "Fine transport · sample departure chunks", label: /^Sample global fine departure chunk \d+\/\d+$/i },
  { stage: "Fine transport · summarize departure chunks", label: /^Summarize global fine departure chunk \d+\/\d+$/i },
  { stage: "Fine transport · finalize departure summaries", label: /^Finalize global fine departure chunk \d+\/\d+$/i },
  { stage: "Fine transport · grouped authorities", label: /^Publish grouped Stage-B and face-band transport authorities$/i },
  { stage: "Fine transport · velocity cache", label: /^Publish complete global fine velocity cache \d+\/\d+$/i },
  { stage: "Fine transport", label: /(?:fine characteristic|fine trajector|fine departure|global fine (?:transport|dispatches)|direct Stage-B transport|grouped direct Stage-B|power trajectory Stage-B)/i },
  { stage: "Fine summaries", label: /fine summar/i },
  { stage: "Fine redistance / volume", label: /(?:Fine redistance|fine level-set JFA|JFA closest-point redistance|global (?:fine )?volume|compact coarse volume|fine overlap|Fine volume correction)/i },
  { stage: "Fine restriction", label: /(?:Fine-to-coarse restriction|fine restriction)/i },
  { stage: "Fine topology", label: /(?:global fine (?:topology|interface|seed|page|changed|publication|rollback|lifecycle)|dirty and support pages|fine topology candidate|fine seed expansion|added global fine samples|Finalize global fine publication|Settle deferred global fine publication)/i },
  { stage: "Power descriptors / topology", label: /(?:power (?:topology|descriptor)|power descriptors)/i },
  { stage: "Power faces", label: /(?:power[- ]face|Power faces|power face control|power site index|power boundary phi)/i },
  { stage: "Power velocity", label: /(?:power velocity|Stage-B row descriptors|direct Stage-B catalog authority)/i },
  { stage: "Power coarse level set", label: /(?:Power coarse level set|power coarse phi|coarse phi from fine-seed)/i },
  { stage: "Power operator / pressure rows", label: /(?:power operator|leaf pressure assembly|projected divergence|physical power-cell volumes|physical octree power volumes|preparePowerRows)/i },
  { stage: "MGPCG solve", label: /(?:MGPCG|Chebyshev|Power Galerkin)/i },
  { stage: "SPGrid V-cycle", label: /^SPGrid V-cycle\b/i },
  { stage: "Octree topology / frontier", label: /(?:octree reset and refinement|compact topology-tile|topology-tile refinement signatures|wet-frontier tile|persistent octree leaf frontier|dirty-tile frontier|dirty frontier candidates|old\/new frontier merge|octree leaf compaction|analytic octree bootstrap worklist|topology lifecycle membership|Cold octree topology)/i },
  { stage: "Power solid coupling", label: /(?:power solid|solid pressure reactions|owner-vertex solid SDF)/i },
  { stage: "Sparse octree publication", label: /(?:octree raw voxel records|octree sparse brick records|sparse voxel structural|sparse brick records)/i },
  { stage: "Diagnostics / overlays", label: /(?:octree overlay|voxel inspection|QA diagnostics|diagnostic fields)/i },
  { stage: "Uniform compatibility transport", label: /^Uniform (?:occupancy and transport preparation|velocity prediction|predicted transport preparation|reverse advection|MacCormack correction|density sharpening|sharpened-mass scatter|sharpened-mass resolve)$/i },
  { stage: "Uniform compatibility solids", label: /^Uniform (?:solid level-set relaxation|rigid-body coupling)$/i },
  // Stable fixture/category labels are part of the report's public test ABI.
  { stage: "Power faces", label: /^Power faces$/ },
  { stage: "Fine redistance / volume", label: /^Fine redistance$/ },
];

/** Resolve a raw descriptor label to its registered owning stage. */
export function powerDamComputePassStage(label: string): string | undefined {
  const value = label.trim();
  if (!value || value === "<unlabeled compute pass>") return undefined;
  if (/owner[- ]page/i.test(value)) return "Octree owner pages";
  if (/^Propagate global fine summaries level \d+$/i.test(value)) {
    return "Fine summaries · propagate hierarchy";
  }
  const spgridLevel = /^(SPGrid V-cycle\s*·\s*.+?)\s*·\s*level \d+$/i.exec(value);
  if (spgridLevel) return spgridLevel[1];
  return POWER_DAM_COMPUTE_PASS_OWNERSHIP.find((rule) => rule.label.test(value))?.stage;
}

function normalizedPassCounts(
  labels: Readonly<Record<string, PowerDamCommandBucket>>,
  steps: number,
): {
  byStage: Record<string, number>;
  byLabel: Record<string, number>;
  totalCalls: number;
  unownedCalls: number;
  unownedLabels: string[];
} {
  const byStage: Record<string, number> = {};
  const byLabel: Record<string, number> = {};
  let totalCalls = 0;
  let unownedCalls = 0;
  const unownedLabels: string[] = [];
  for (const [label, bucket] of Object.entries(labels)) {
    totalCalls += bucket.calls;
    const passes = perAdvance(bucket.calls, steps);
    byLabel[label] = passes;
    const stage = powerDamComputePassStage(label);
    if (!stage) {
      unownedCalls += bucket.calls;
      unownedLabels.push(label);
      continue;
    }
    byStage[stage] = (byStage[stage] ?? 0) + passes;
  }
  return { byStage, byLabel, totalCalls, unownedCalls, unownedLabels: unownedLabels.sort() };
}

export function summarizePowerDamPerformance(result: PowerDamResultRecord): PowerDamPerformanceSummary {
  if (!Number.isInteger(result.steps) || result.steps < 1) throw new Error("Power dam result must contain at least one step");
  if (!Number.isFinite(result.simulationWall_ms) || result.simulationWall_ms < 0) {
    throw new Error("Power dam result has an invalid simulationWall_ms");
  }
  const audit = result.gpuCommandAudit;
  const dispatches = audit?.dispatches ?? 0;
  const mgpcgDispatches = (audit?.dispatchesByPassLabel?.["Octree MGPCG solve"]?.calls ?? 0)
    + (audit?.dispatchesByPassLabel?.["SPGrid persistent small-domain MGPCG"]?.calls ?? 0);
  const trace = result.physicsTrace;
  const phaseTotals_ms: Partial<Record<PaperPhaseId, number>> = {};
  for (const phase of trace?.phases ?? []) {
    phaseTotals_ms[phase.id] = (phaseTotals_ms[phase.id] ?? 0) + phase.duration_ms;
  }
  const passCounts = normalizedPassCounts(audit?.computePassesByLabel ?? {}, result.steps);
  const computePassAttributionComplete = audit?.computePassesByLabel !== undefined
    && passCounts.totalCalls === (audit.computePasses ?? 0)
    && passCounts.unownedCalls === 0;
  const unattributedComputePasses = passCounts.unownedCalls
    + Math.abs((audit?.computePasses ?? 0) - passCounts.totalCalls);
  return {
    scenario: result.scenario,
    method: result.method,
    steps: result.steps,
    simulationWall_ms: result.simulationWall_ms,
    advanceWall_ms: perAdvance(result.simulationWall_ms, result.steps),
    validationErrorCount: result.validationErrors?.length ?? 0,
    ...(audit ? { commands: {
      dispatchesPerAdvance: perAdvance(dispatches, result.steps),
      indirectDispatchesPerAdvance: perAdvance(audit.indirectDispatches, result.steps),
      computePassesPerAdvance: perAdvance(audit.computePasses, result.steps),
      computePassesByStage: passCounts.byStage,
      computePassesByLabel: passCounts.byLabel,
      computePassAttributionComplete,
      unattributedComputePassesPerAdvance: perAdvance(unattributedComputePasses, result.steps),
      unownedComputePassLabels: passCounts.unownedLabels,
      mgpcgDispatchesPerAdvance: perAdvance(mgpcgDispatches, result.steps),
      mgpcgDispatchFraction: dispatches > 0 ? mgpcgDispatches / dispatches : 0,
      clearBytesPerAdvance: perAdvance(audit.clearBuffer?.bytes, result.steps),
      copyBytesPerAdvance: perAdvance(audit.copyBufferToBuffer?.bytes, result.steps),
    } } : {}),
    ...(result.gpuFineTimestamps ? { fineTimestamps: {
      measuredAdvances: result.gpuFineTimestamps.measuredAdvances,
      measuredPasses: result.gpuFineTimestamps.measuredPasses,
      invalidPasses: result.gpuFineTimestamps.invalidPasses,
      summedPass_ms: result.gpuFineTimestamps.summedPass_ms,
      summedPassPerAdvance_ms: perAdvance(
        result.gpuFineTimestamps.summedPass_ms,
        result.gpuFineTimestamps.measuredAdvances,
      ),
      byLabel: Object.fromEntries(Object.entries(result.gpuFineTimestamps.byLabel).map(
        ([label, bucket]) => [label, {
          ...bucket,
          totalPerAdvance_ms: perAdvance(
            bucket.total_ms,
            result.gpuFineTimestamps!.measuredAdvances,
          ),
        }],
      )),
    } } : {}),
    ...(result.gpuDataFlowManifest ? { dataFlow: result.gpuDataFlowManifest } : {}),
    ...(trace ? { physicsTrace: {
      sampleId: trace.sampleId,
      measurementSource: trace.measurementSource,
      total_ms: trace.total_ms,
      accounted_ms: performanceTraceAccounted_ms(trace),
      exact: performanceTraceIsExact(trace),
      phases: trace.phases,
      phaseTotals_ms,
    } } : {}),
  };
}

export function powerDamResultFromLine(line: string): PowerDamResultRecord | undefined {
  let parsed: unknown;
  try { parsed = JSON.parse(line); } catch { return undefined; }
  if (!parsed || typeof parsed !== "object") return undefined;
  const candidate = parsed as Partial<PowerDamResultRecord>;
  return candidate.phase === "result" && candidate.method === "octree"
    ? candidate as PowerDamResultRecord : undefined;
}

export function powerDamPerformanceFailures(
  summary: PowerDamPerformanceSummary,
  limits: PowerDamPerformanceLimits,
): string[] {
  const failures: string[] = [];
  if (limits.maximumAdvanceWall_ms !== undefined
    && summary.advanceWall_ms > limits.maximumAdvanceWall_ms) {
    failures.push(`advance wall ${summary.advanceWall_ms.toFixed(2)} ms exceeds ${limits.maximumAdvanceWall_ms.toFixed(2)} ms`);
  }
  if (limits.maximumDispatchesPerAdvance !== undefined && !summary.commands) {
    failures.push("dispatches/advance unavailable for configured gate");
  } else if (limits.maximumDispatchesPerAdvance !== undefined && summary.commands
    && summary.commands.dispatchesPerAdvance > limits.maximumDispatchesPerAdvance) {
    failures.push(`dispatches/advance ${summary.commands.dispatchesPerAdvance.toFixed(1)} exceeds ${limits.maximumDispatchesPerAdvance.toFixed(1)}`);
  }
  if (limits.maximumComputePassesPerAdvance !== undefined && !summary.commands) {
    failures.push("compute passes/advance unavailable for configured gate");
  } else if (limits.maximumComputePassesPerAdvance !== undefined && summary.commands) {
    if (!summary.commands.computePassAttributionComplete) {
      failures.push(`compute passes/stage attribution incomplete (${summary.commands.unattributedComputePassesPerAdvance.toFixed(1)} unattributed/advance)`);
    }
    if (summary.commands.computePassesPerAdvance > limits.maximumComputePassesPerAdvance) {
      failures.push(`compute passes/advance ${summary.commands.computePassesPerAdvance.toFixed(1)} exceeds ${limits.maximumComputePassesPerAdvance.toFixed(1)}`);
    }
  }
  const pressureSystem_ms = summary.physicsTrace?.phaseTotals_ms["pressure-system"];
  if (limits.maximumPressureNonSolve_ms !== undefined && pressureSystem_ms === undefined) {
    failures.push("pressure-system phase unavailable for configured gate");
  } else if (limits.maximumPressureNonSolve_ms !== undefined
    && pressureSystem_ms !== undefined
    && pressureSystem_ms > limits.maximumPressureNonSolve_ms) {
    failures.push(`pressure-system ${pressureSystem_ms.toFixed(2)} ms exceeds ${limits.maximumPressureNonSolve_ms.toFixed(2)} ms`);
  }
  return failures;
}
