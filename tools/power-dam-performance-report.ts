import {
  performanceTraceAccounted_ms,
  performanceTraceIsExact,
  type PaperPhaseId,
  type PerformanceTrace,
} from "../lib/performance-trace";
import type { GPUDataFlowManifest } from "./webgpu-data-flow-manifest";
import type { OctreeWorkSnapshot } from "../lib/webgpu-octree-work-accounting";
import {
  buildPowerDamPressureKernelProfile,
  type PowerDamPressureKernelProfile,
} from "./power-dam-pressure-kernel-profile";

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

export interface PowerDamPassTimestampReport {
  readonly capturedCommandBuffers: number;
  readonly measuredPasses: number;
  readonly invalidPasses: number;
  readonly capacityOverflows: number;
  readonly summedPass_ms: number;
  /** False means Dawn was free to merge passes into shared Metal encoders, so
   * every label is really its encoder's total charged to the encoder's last
   * pass. Older records predate the flag and read as unisolated, which is what
   * they were. */
  readonly encoderIsolated?: boolean;
  /** Every labelled `compute()` opened its own pass, so a label's ms covers
   * exactly the dispatches recorded under that label. Older records predate the
   * flag and read as unisolated, which is what they were. */
  readonly labelIsolated?: boolean;
  /** Optional label prefixes retained by a targeted micro-stage capture. */
  readonly labelPrefixes?: readonly string[];
  /** GPU-timeline span of the captured command buffers. */
  readonly span_ms?: number;
  /** `summedPass_ms / span_ms`; see the smoke runner for what it certifies. */
  readonly coverageRatio?: number;
  readonly byLabel: Readonly<Record<string, PowerDamFineTimestampBucket>>;
}

export interface PowerDamResultRecord {
  readonly scenario: string;
  readonly method: string;
  readonly phase: "result";
  readonly steps: number;
  readonly simulationWall_ms: number;
  readonly simulatedTime_s?: number;
  readonly pressureRelativeResidual?: number;
  readonly volumeDrift?: number;
  readonly stabilityEnvelope?: {
    readonly maximumPressureRelativeResidual?: number;
    readonly maximumExactVolumeDrift?: number;
    readonly maximumProjectionEnergyRatio?: number;
    /** Number of paired pre/post projection energy observations behind the maximum. */
    readonly projectionEnergySampleCount?: number;
  };
  readonly octreeWorkAccounting?: OctreeWorkSnapshot;
  /** Exact cause when a work-accounting stage failed its validity gate. */
  readonly octreeWorkAccountingBlocker?: string;
  readonly compactMechanicalEnergyCheckpoints?: readonly {
    readonly time_s: number;
    readonly mechanicalEnergyRetentionRatio: number;
    readonly publicationValid: boolean;
    readonly rowCount: number;
    readonly reconstructedRows: number;
    readonly invalidRows: number;
    readonly liquidCellCount: number;
    readonly finiteLiquidCellCount: number;
  }[];
  readonly validationErrors?: readonly string[];
  readonly gpuCommandAudit?: PowerDamCommandAudit;
  readonly gpuFineTimestamps?: PowerDamFineTimestampReport;
  readonly gpuPassTimestamps?: PowerDamPassTimestampReport;
  readonly algorithmDiagnostics?: Readonly<Record<string, unknown>>;
  readonly gpuDataFlowManifest?: GPUDataFlowManifest;
  readonly physicsTrace?: PerformanceTrace;
  /** Terminal state counters already published by the smoke runner. These are
   * snapshots, not sums, so a differential window retains the later value. */
  readonly activeSampleCount?: number;
  readonly globalFineActiveBricks?: number;
  readonly globalFineDesiredBricks?: number;
  /** Physical page capacity and the logical brick domain. Their ratio against
   * the active count is the fine-band occupancy the narrow-band design exists
   * to keep small; see docs/FINE_BAND_DENSITY_PLAN.md. */
  readonly globalFineLevelSetResidentBrickCapacity?: number;
  readonly globalFineLevelSetLogicalBrickCount?: number;
  readonly globalFineTransportSegmentCount?: number;
  readonly structuredAirSupportRows?: number;
  readonly structuredAirSupportCells?: number;
  readonly structuredAirSupportCapacity?: number;
  readonly structuredAirSupportFaceItems?: number;
  readonly structuredAirSupportSeedFaces?: number;
  readonly structuredAirSupportMarchDepth?: number;
  readonly quadtreePressureIterationsUsed?: number;
  readonly quadtreePressureIterationBudget?: number;
  readonly quadtreePressureIterationHardBudget?: number;
  readonly measurementWindow?: {
    readonly kind: "paired-prefix-difference";
    readonly startStep: number;
    readonly endStep: number;
  };
}

export interface PowerDamPerformanceSummary {
  readonly scenario: string;
  readonly method: string;
  readonly steps: number;
  readonly simulationWall_ms: number;
  readonly advanceWall_ms: number;
  readonly validationErrorCount: number;
  readonly measurementWindow?: PowerDamResultRecord["measurementWindow"];
  /** Existing runner telemetry available without another GPU readback. Active
   * sets and pressure counts describe the terminal measured advance. */
  readonly terminalCounters?: {
    readonly activeSamples?: number;
    readonly activeFineBricks?: number;
    readonly desiredFineBricks?: number;
    readonly fineBrickCapacity?: number;
    readonly logicalFineBricks?: number;
    /** activeFineBricks / logicalFineBricks. See docs/FINE_BAND_DENSITY_PLAN.md. */
    readonly fineBandOccupancy?: number;
    readonly transportSegments?: number;
    readonly airSupportRows?: number;
    readonly airSupportCells?: number;
    readonly airSupportCapacity?: number;
    readonly airSupportFaceItems?: number;
    readonly airSupportSeedFaces?: number;
    readonly airSupportMarchDepth?: number;
    readonly pressureIterationsExecuted?: number;
    readonly pressureIterationsScheduled?: number;
    readonly pressureIterationsHardLimit?: number;
  };
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
  /** Non-invasive beginning/end timestamps attached to the existing compute
   * passes of a bounded number of recurring command buffers. */
  readonly passTimestamps?: PowerDamPassTimestampReport;
  /** Targeted, label-isolated pressure solve attribution derived from the raw
   * pass timestamps. Present only when pressure labels were captured. */
  readonly pressureKernelProfile?: PowerDamPressureKernelProfile;
  readonly algorithmDiagnostics?: Readonly<Record<string, unknown>>;
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

const finiteCounter = (value: number | undefined): number | undefined =>
  value !== undefined && Number.isFinite(value) && value >= 0 ? value : undefined;

/** Active fine bricks as a fraction of the logical brick lattice. The paper's
 * narrow band sits near 0.06; anything approaching 1.0 means the "sparse"
 * lattice has been materialized densely. */
const fineBandOccupancy = (
  active: number | undefined,
  logical: number | undefined,
): number | undefined => {
  const activeBricks = finiteCounter(active);
  const logicalBricks = finiteCounter(logical);
  if (activeBricks === undefined || logicalBricks === undefined || logicalBricks === 0) {
    return undefined;
  }
  return activeBricks / logicalBricks;
};

function subtractCounter(
  later: number | undefined,
  earlier: number | undefined,
  label: string,
): number | undefined {
  if (later === undefined && earlier === undefined) return undefined;
  if (later === undefined || earlier === undefined) {
    throw new Error(`Power dam window cannot subtract ${label}: counter availability differs`);
  }
  const difference = later - earlier;
  if (!Number.isFinite(difference) || difference < 0) {
    throw new Error(`Power dam window cannot subtract ${label}: cumulative counter decreased`);
  }
  return difference;
}

function subtractBucket(
  later: PowerDamCommandBucket | undefined,
  earlier: PowerDamCommandBucket | undefined,
  label: string,
): PowerDamCommandBucket | undefined {
  if (!later && !earlier) return undefined;
  return {
    calls: subtractCounter(later?.calls, earlier?.calls, `${label} calls`)!,
    bytes: subtractCounter(later?.bytes, earlier?.bytes, `${label} bytes`)!,
  };
}

function subtractBucketMap(
  later: Readonly<Record<string, PowerDamCommandBucket>> | undefined,
  earlier: Readonly<Record<string, PowerDamCommandBucket>> | undefined,
  label: string,
): Readonly<Record<string, PowerDamCommandBucket>> | undefined {
  if (!later && !earlier) return undefined;
  if (!later || !earlier) {
    throw new Error(`Power dam window cannot subtract ${label}: bucket availability differs`);
  }
  const result: Record<string, PowerDamCommandBucket> = {};
  for (const key of new Set([...Object.keys(later), ...Object.keys(earlier)])) {
    const bucket = subtractBucket(later[key] ?? { calls: 0, bytes: 0 },
      earlier[key] ?? { calls: 0, bytes: 0 }, `${label} ${key}`)!;
    if (bucket.calls !== 0 || bucket.bytes !== 0) result[key] = bucket;
  }
  return result;
}

function subtractCommandAudit(
  later: PowerDamCommandAudit | undefined,
  earlier: PowerDamCommandAudit | undefined,
): PowerDamCommandAudit | undefined {
  if (!later && !earlier) return undefined;
  if (!later || !earlier) {
    throw new Error("Power dam window cannot subtract command audit: availability differs");
  }
  return {
    clearBuffer: subtractBucket(later.clearBuffer, earlier.clearBuffer, "clearBuffer"),
    copyBufferToBuffer: subtractBucket(later.copyBufferToBuffer, earlier.copyBufferToBuffer,
      "copyBufferToBuffer"),
    computePasses: subtractCounter(later.computePasses, earlier.computePasses, "computePasses"),
    dispatches: subtractCounter(later.dispatches, earlier.dispatches, "dispatches"),
    indirectDispatches: subtractCounter(later.indirectDispatches, earlier.indirectDispatches,
      "indirectDispatches"),
    computePassesByLabel: subtractBucketMap(later.computePassesByLabel,
      earlier.computePassesByLabel, "computePassesByLabel"),
    dispatchesByPassLabel: subtractBucketMap(later.dispatchesByPassLabel,
      earlier.dispatchesByPassLabel, "dispatchesByPassLabel"),
  };
}

/**
 * Isolate a trailing benchmark window when the smoke runner only exposes
 * cumulative counters. Both records start from the same deterministic scene;
 * subtracting the settle-prefix run from the longer run excludes warmup work.
 *
 * Fine timestamps and data-flow manifests are intentionally not subtracted:
 * their current reports contain distributions/identities that cannot be
 * reconstructed exactly from two cumulative aggregates.
 */
export function powerDamResultWindow(
  settlePrefix: PowerDamResultRecord,
  complete: PowerDamResultRecord,
): PowerDamResultRecord {
  if (settlePrefix.scenario !== complete.scenario || settlePrefix.method !== complete.method) {
    throw new Error("Power dam window records must have the same scenario and method");
  }
  if (settlePrefix.steps < 1 || complete.steps <= settlePrefix.steps) {
    throw new Error("Power dam window requires a non-empty suffix after the settle prefix");
  }
  const simulationWall_ms = complete.simulationWall_ms - settlePrefix.simulationWall_ms;
  if (!Number.isFinite(simulationWall_ms) || simulationWall_ms < 0) {
    throw new Error("Power dam window wall time decreased across cumulative runs");
  }
  return {
    scenario: complete.scenario,
    method: complete.method,
    phase: "result",
    steps: complete.steps - settlePrefix.steps,
    simulationWall_ms,
    simulatedTime_s: complete.simulatedTime_s === undefined || settlePrefix.simulatedTime_s === undefined
      ? undefined : complete.simulatedTime_s - settlePrefix.simulatedTime_s,
    validationErrors: [
      ...(settlePrefix.validationErrors ?? []),
      ...(complete.validationErrors ?? []),
    ],
    gpuCommandAudit: subtractCommandAudit(
      complete.gpuCommandAudit,
      settlePrefix.gpuCommandAudit,
    ),
    // A generic trace is one terminal sampled advance, so the complete run's
    // trace belongs to the suffix even though it is not a window average.
    physicsTrace: complete.physicsTrace,
    pressureRelativeResidual: complete.pressureRelativeResidual,
    volumeDrift: complete.volumeDrift,
    stabilityEnvelope: complete.stabilityEnvelope,
    octreeWorkAccounting: complete.octreeWorkAccounting,
    compactMechanicalEnergyCheckpoints: complete.compactMechanicalEnergyCheckpoints?.filter(
      (sample) => sample.time_s > (settlePrefix.simulatedTime_s ?? Number.POSITIVE_INFINITY),
    ),
    activeSampleCount: complete.activeSampleCount,
    globalFineActiveBricks: complete.globalFineActiveBricks,
    globalFineDesiredBricks: complete.globalFineDesiredBricks,
    globalFineLevelSetResidentBrickCapacity: complete.globalFineLevelSetResidentBrickCapacity,
    globalFineLevelSetLogicalBrickCount: complete.globalFineLevelSetLogicalBrickCount,
    globalFineTransportSegmentCount: complete.globalFineTransportSegmentCount,
    structuredAirSupportRows: complete.structuredAirSupportRows,
    structuredAirSupportCells: complete.structuredAirSupportCells,
    structuredAirSupportCapacity: complete.structuredAirSupportCapacity,
    structuredAirSupportFaceItems: complete.structuredAirSupportFaceItems,
    structuredAirSupportSeedFaces: complete.structuredAirSupportSeedFaces,
    structuredAirSupportMarchDepth: complete.structuredAirSupportMarchDepth,
    quadtreePressureIterationsUsed: complete.quadtreePressureIterationsUsed,
    quadtreePressureIterationBudget: complete.quadtreePressureIterationBudget,
    quadtreePressureIterationHardBudget: complete.quadtreePressureIterationHardBudget,
    measurementWindow: {
      kind: "paired-prefix-difference",
      startStep: settlePrefix.steps,
      endStep: complete.steps,
    },
  };
}

interface PowerDamComputePassOwnershipRule {
  readonly stage: string;
  readonly label: RegExp;
}

/** Closed ownership table for compute-pass labels emitted by the octree power
 * lane. Rules may normalize bounded loop/chunk suffixes, but there is
 * deliberately no raw-label fallback: a newly introduced label must register
 * its owning module here or the pass-budget audit fails closed. */
const POWER_DAM_COMPUTE_PASS_OWNERSHIP: readonly PowerDamComputePassOwnershipRule[] = [
  // GPUDynamicTracePhaseRecorder (`lib/performance-trace.ts` writeBoundary)
  // emits one single-dispatch marker pass per semantic phase while a segmented
  // profile is captured: exactly "GPU trace start" and `${phase.label} complete`.
  // Matched first so a marker never lands in the physics stage it names.
  { stage: "Profiler phase boundaries", label: /^(?:GPU trace start|.+ complete)$/ },
  { stage: "Octree owner pages", label: /owner[- ]page|owner pages|analytic octree owner generation/i },
  { stage: "Power energy ledger", label: /^Power energy ledger\b/i },
  { stage: "Power cell volumes", label: /power-cell volumes/i },
  { stage: "Fine seed adapter", label: /octree fine-seed|octree interface candidates|FineSeedLeaf to global fine seeds|Seed global fine bricks from (?:FineSeedLeaf candidates|every interface leaf)|fine-seed brick residency/i },
  { stage: "Structured velocity publication", label: /(?:direct structured|structured velocity publication|structured cell velocities|six structured velocity families|direct Section 6\.3 rows)/i },
  { stage: "Structured boundary coefficients", label: /(?:structured boundary|Classify fine-over-coarse liquid rows|Resolve canonical (?:free-surface|solid) boundary slots)/i },
  // `lib/webgpu-octree-structured-dynamics.ts` suffixes a class index onto its
  // per-family advect/commit/force/project/RHS/reconstruct labels.
  { stage: "Structured velocity dynamics", label: /(?:structured dynamics|structured family class|structured divergence RHS class|Reconstruct projected structured rows|Transfer accepted velocity to changed topology faces)/i },
  // Emitted only while a trace is active: one single-dispatch marker pass per
  // physics phase (lib/webgpu-uniform-eulerian.ts). Owned so a --profile capture
  // does not report them as unattributed work.
  { stage: "Profiler phase boundaries", label: /^Physics activity (?:boundary \d+|frame (?:begin|end))$/i },
  // C2 widened the single-workgroup cubic band scatter into an indirect
  // (seed x halo-cell) dispatch; the new pass needs the same owner.
  { stage: "Fine topology", label: /^Scatter recurring fine-band seed halos$/i },
  { stage: "Section 5 air-velocity support", label: /(?:air-support|Section 5 closest faces|Extrapolate structured ordinary faces)/i },
  { stage: "Fine transport · prepare trajectory chunks", label: /^Prepare global fine trajectory chunk \d+\/\d+$/i },
  { stage: "Fine transport · advance trajectories", label: /^Advance global fine trajectories \d+\/\d+$/i },
  { stage: "Fine transport · sample departure chunks", label: /^Sample global fine departure chunk \d+\/\d+$/i },
  { stage: "Fine transport · summarize departure chunks", label: /^Summarize global fine departure chunk \d+\/\d+$/i },
  { stage: "Fine transport · finalize departure summaries", label: /^Finalize global fine departure chunk \d+\/\d+$/i },
  { stage: "Fine transport · velocity cache", label: /^Publish complete global fine velocity cache \d+\/\d+$/i },
  { stage: "Fine transport", label: /(?:fine characteristic|fine trajector|fine departure|global fine (?:transport|dispatches)|structured (?:velocity|fine) transport|fine transport substeps|Advect fine phi|Commit structured fine phi and phase delta|Compact structured fine topology delta)/i },
  // Both hyphenated ("fine-summary delta") and spaced ("coarse summary ranks")
  // forms are authored by webgpu-octree-fine-levelset-summary-direct.ts.
  { stage: "Fine summaries", label: /(?:fine|coarse)[- ]summar/i },
  { stage: "Fine redistance / volume", label: /(?:Fine redistance|fine level-set JFA|JFA closest-point redistance|global (?:fine )?volume|compact coarse volume|fine overlap|Fine volume correction)/i },
  { stage: "Fine restriction", label: /(?:Fine-to-coarse restriction|fine restriction)/i },
  { stage: "Fine topology", label: /(?:global fine (?:topology|interface|seed|page|changed|publication|rollback|lifecycle|flags)|dirty and support pages|fine topology candidate|fine seed expansion|added global fine samples|Finalize global fine publication|Settle deferred (?:global fine publication|rejected fine work payload)|sparse fine band|fine-band logical identity)/i },
  { stage: "Power descriptors / topology", label: /(?:power (?:topology|descriptor)|power descriptors)/i },
  { stage: "Power coarse level set", label: /(?:Power coarse level set|power coarse phi|coarse phi from fine-seed)/i },
  { stage: "Structured pressure rows", label: /(?:structured pressure|leaf pressure assembly|projected divergence|prepareStructuredRows)/i },
  { stage: "MGPCG solve", label: /(?:MGPCG|Chebyshev)/i },
  { stage: "Section 4.3 preconditioner", label: /^Section 4\.3\b/i },
  { stage: "SPGrid V-cycle", label: /^SPGrid V-cycle\b/i },
  { stage: "SPGrid accurate A2", label: /^SPGrid accurate A2\b/i },
  { stage: "SPGrid Section 6.3 apply", label: /^SPGrid Section 6\.3\b/i },
  { stage: "Octree topology / frontier", label: /(?:octree reset and refinement|compact topology-tile|topology-tile refinement signatures|wet-frontier tile|persistent octree leaf frontier|dirty-tile frontier|dirty frontier candidates|old\/new frontier merge|octree leaf compaction|analytic octree bootstrap worklist|topology lifecycle membership|Cold octree topology|inactive topology epoch|topology ready-commit gate|accepted topology row|topology attempt generation|octree pressure-row candidate)/i },
  { stage: "Structured solid coupling", label: /(?:structured solid|solid pressure reactions|owner-vertex solid SDF)/i },
  { stage: "Sparse octree publication", label: /(?:octree raw voxel records|octree sparse brick records|sparse voxel structural|sparse brick records)/i },
  { stage: "Diagnostics / overlays", label: /(?:octree overlay|voxel inspection|QA diagnostics|diagnostic fields|octree inspection)/i },
  { stage: "Uniform compatibility transport", label: /^Uniform (?:occupancy and transport preparation|velocity prediction|predicted transport preparation|reverse advection|MacCormack correction|density sharpening|sharpened-mass scatter|sharpened-mass resolve)$/i },
  { stage: "Uniform compatibility solids", label: /^Uniform (?:solid level-set relaxation|rigid-body coupling)$/i },
  // Stable fixture/category labels are part of the report's public test ABI.
  { stage: "Structured velocity publication", label: /^Structured velocity publication$/ },
  { stage: "Fine redistance / volume", label: /^Fine redistance$/ },
];

/** Placeholder the command audit substitutes for a descriptor-less pass. Every
 * site that opens one today is inside `OctreePipelinedMGPCG.encode`
 * (`lib/webgpu-octree-pipelined-mgpcg.ts`: the `broker.compute()` after the
 * live-dispatch fence, after the initial indirect-tail fence, and after each
 * iteration's convergence-tail fence). `PassBroker.compute` drops a descriptor
 * when a pass is already open, so only those post-fence calls actually open an
 * unlabeled pass. Bucketed as its own stage rather than folded into "MGPCG
 * solve": if any other module ever opens an unlabeled pass the count still
 * shows up here instead of silently inflating the solve. Delete this stage
 * once those three calls carry labels. */
const POWER_DAM_UNLABELED_PASS_STAGE = "MGPCG solve · unlabeled broker pass";

/** Resolve a raw descriptor label to its registered owning stage. */
export function powerDamComputePassStage(label: string): string | undefined {
  const value = label.trim();
  if (!value) return undefined;
  if (value === "<unlabeled compute pass>") return POWER_DAM_UNLABELED_PASS_STAGE;
  if (/owner[- ]page/i.test(value)) return "Octree owner pages";
  if (/^Propagate global fine summaries level \d+$/i.test(value)) {
    return "Fine summaries · propagate hierarchy";
  }
  const spgridLevel = /^(SPGrid V-cycle\s*·\s*.+?)\s*·\s*level \d+$/i.exec(value);
  if (spgridLevel) return spgridLevel[1];
  return POWER_DAM_COMPUTE_PASS_OWNERSHIP.find((rule) => rule.label.test(value))?.stage;
}

/** Owning stages that together are the pressure solve: the pipelined MGPCG
 * driver, its Section 4.3 hybrid preconditioner shell, and the SPGrid V-cycle /
 * accurate-A2 / Section 6.3 operator applies those drive. Keyed on the resolved
 * stage so the ownership table above stays the single label registry — the old
 * key ("Octree MGPCG solve") had not been emitted for several refactors and
 * silently reported a 0.0 solve while the solve owned most of the frame. */
const POWER_DAM_MGPCG_SOLVE_STAGE =
  /^(?:MGPCG solve|Section 4\.3 preconditioner|SPGrid (?:V-cycle|accurate A2|Section 6\.3 apply))\b/;

/** True when a resolved owning stage belongs to the pressure solve. */
export function powerDamIsMgpcgSolveStage(stage: string): boolean {
  return POWER_DAM_MGPCG_SOLVE_STAGE.test(stage);
}

/** Dispatches recorded inside compute passes owned by the pressure solve. */
function mgpcgSolveDispatches(
  labels: Readonly<Record<string, PowerDamCommandBucket>> | undefined,
): number {
  let calls = 0;
  for (const [label, bucket] of Object.entries(labels ?? {})) {
    const stage = powerDamComputePassStage(label);
    if (stage !== undefined && powerDamIsMgpcgSolveStage(stage)) calls += bucket.calls;
  }
  return calls;
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
  const mgpcgDispatches = mgpcgSolveDispatches(audit?.dispatchesByPassLabel);
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
  const pressureKernelProfile = result.gpuPassTimestamps
    ? buildPowerDamPressureKernelProfile(result.gpuPassTimestamps)
    : undefined;
  return {
    scenario: result.scenario,
    method: result.method,
    steps: result.steps,
    simulationWall_ms: result.simulationWall_ms,
    advanceWall_ms: perAdvance(result.simulationWall_ms, result.steps),
    validationErrorCount: result.validationErrors?.length ?? 0,
    measurementWindow: result.measurementWindow,
    ...([
      result.activeSampleCount,
      result.globalFineActiveBricks,
      result.globalFineDesiredBricks,
      result.globalFineLevelSetResidentBrickCapacity,
      result.globalFineLevelSetLogicalBrickCount,
      result.globalFineTransportSegmentCount,
      result.structuredAirSupportRows,
      result.structuredAirSupportCells,
      result.structuredAirSupportCapacity,
      result.structuredAirSupportFaceItems,
      result.structuredAirSupportSeedFaces,
      result.structuredAirSupportMarchDepth,
      result.quadtreePressureIterationsUsed,
      result.quadtreePressureIterationBudget,
      result.quadtreePressureIterationHardBudget,
    ].some((value) => finiteCounter(value) !== undefined) ? { terminalCounters: {
      activeSamples: finiteCounter(result.activeSampleCount),
      activeFineBricks: finiteCounter(result.globalFineActiveBricks),
      desiredFineBricks: finiteCounter(result.globalFineDesiredBricks),
      fineBrickCapacity: finiteCounter(result.globalFineLevelSetResidentBrickCapacity),
      logicalFineBricks: finiteCounter(result.globalFineLevelSetLogicalBrickCount),
      fineBandOccupancy: fineBandOccupancy(
        result.globalFineActiveBricks, result.globalFineLevelSetLogicalBrickCount),
      transportSegments: finiteCounter(result.globalFineTransportSegmentCount),
      ...(finiteCounter(result.structuredAirSupportRows) === undefined ? {}
        : { airSupportRows: finiteCounter(result.structuredAirSupportRows) }),
      ...(finiteCounter(result.structuredAirSupportCells) === undefined ? {}
        : { airSupportCells: finiteCounter(result.structuredAirSupportCells) }),
      ...(finiteCounter(result.structuredAirSupportCapacity) === undefined ? {}
        : { airSupportCapacity: finiteCounter(result.structuredAirSupportCapacity) }),
      ...(finiteCounter(result.structuredAirSupportFaceItems) === undefined ? {}
        : { airSupportFaceItems: finiteCounter(result.structuredAirSupportFaceItems) }),
      ...(finiteCounter(result.structuredAirSupportSeedFaces) === undefined ? {}
        : { airSupportSeedFaces: finiteCounter(result.structuredAirSupportSeedFaces) }),
      ...(finiteCounter(result.structuredAirSupportMarchDepth) === undefined ? {}
        : { airSupportMarchDepth: finiteCounter(result.structuredAirSupportMarchDepth) }),
      pressureIterationsExecuted: finiteCounter(result.quadtreePressureIterationsUsed),
      pressureIterationsScheduled: finiteCounter(result.quadtreePressureIterationBudget),
      pressureIterationsHardLimit: finiteCounter(result.quadtreePressureIterationHardBudget),
    } } : {}),
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
    ...(result.gpuPassTimestamps ? {
      passTimestamps: result.gpuPassTimestamps,
      ...(pressureKernelProfile ? { pressureKernelProfile } : {}),
    } : {}),
    ...(result.algorithmDiagnostics ? { algorithmDiagnostics: result.algorithmDiagnostics } : {}),
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
