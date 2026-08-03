/** Host-visible accounting for the GPU command graph. Simulation decisions
 * never consume these counters; they exist to make regressions attributable.
 *
 * Missing observations are null, never zero. A zero means the runtime
 * positively observed no work; null means the instrumentation is incomplete.
 */

export const OCTREE_WORK_STAGES = [
  "topology-publication",
  "forces-and-solids",
  "fine-transport",
  "rhs",
  "pressure-solve",
  "projection",
  "fine-redistance",
  "candidate-build",
] as const;

export type OctreeWorkStage = typeof OCTREE_WORK_STAGES[number];
export type OctreeAllocationClass = "authoritative" | "scratch";

export interface OctreeStageWork {
  readonly scheduledLanes: number | null;
  readonly activeLanes: number | null;
  readonly activePages: number | null;
  readonly logicalPages: number | null;
  readonly worksets: number | null;
  readonly encodedIterations: number | null;
  readonly executedIterations: number | null;
  readonly reductionPasses: number | null;
  /** Estimate supplied by the encoder from the actual kernel ABI. */
  readonly estimatedBytesMoved: number | null;
  /** Derived only when both bytes moved and active lanes were observed. */
  readonly estimatedBytesMovedPerActiveElement: number | null;
}

type OctreeObservedStageMetric = Exclude<
  keyof OctreeStageWork,
  "estimatedBytesMovedPerActiveElement"
>;
export type OctreeStageWorkObservation = Partial<{
  [K in OctreeObservedStageMetric]: Exclude<OctreeStageWork[K], null>;
}>;

export interface OctreeWorkSnapshot {
  readonly topologyEpoch: number;
  readonly catalogStamps: number | null;
  readonly solverIterations: number | null;
  readonly reductionPasses: number | null;
  readonly scheduledLanes: number | null;
  readonly activeLanes: number | null;
  readonly activePages: number | null;
  readonly logicalPages: number | null;
  readonly worksetCount: number | null;
  readonly estimatedBytesMoved: number | null;
  readonly activeLaneRatio: number | null;
  readonly stagesComplete: boolean;
  readonly missingStageMetrics: Readonly<Record<OctreeWorkStage, readonly (keyof OctreeStageWork)[]>>;
  readonly stages: Readonly<Record<OctreeWorkStage, OctreeStageWork>>;
  readonly allocatedBytesByAuthority: Readonly<Record<string, number>>;
  readonly allocatedScratchBytesByArena: Readonly<Record<string, number>>;
  readonly authoritativeBytes: number;
  readonly scratchBytes: number;
  readonly allocatedBytes: number;
  readonly allocationInventoryComplete: boolean;
}

const EMPTY_STAGE: OctreeStageWork = Object.freeze({
  scheduledLanes: null,
  activeLanes: null,
  activePages: null,
  logicalPages: null,
  worksets: null,
  encodedIterations: null,
  executedIterations: null,
  reductionPasses: null,
  estimatedBytesMoved: null,
  estimatedBytesMovedPerActiveElement: null,
});

const STAGE_METRICS = Object.freeze(Object.keys(EMPTY_STAGE) as (keyof OctreeStageWork)[]);

function count(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
  return value;
}

function finiteNonNegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative finite number`);
  }
  return value;
}

function emptyStages(): Record<OctreeWorkStage, OctreeStageWork> {
  return Object.fromEntries(
    OCTREE_WORK_STAGES.map((stage) => [stage, EMPTY_STAGE]),
  ) as Record<OctreeWorkStage, OctreeStageWork>;
}

function completeSum(
  stages: Readonly<Record<OctreeWorkStage, OctreeStageWork>>,
  metric: keyof OctreeStageWork,
): number | null {
  let total = 0;
  for (const stage of OCTREE_WORK_STAGES) {
    const value = stages[stage][metric];
    if (value === null) return null;
    total += value;
  }
  return total;
}

export class OctreeWorkAccounting {
  private topologyEpoch = 0;
  private catalogStamps: number | null = null;
  private solverIterations: number | null = null;
  private stages = emptyStages();
  private readonly authorityBytes = new Map<string, number>();
  private readonly scratchBytes = new Map<string, number>();
  private allocationInventoryComplete = false;

  beginSubstep(): void {
    this.catalogStamps = null;
    this.solverIterations = null;
    this.stages = emptyStages();
  }

  publishTopologyEpoch(epoch: number): void {
    count(epoch, "Topology epoch");
    if (epoch < this.topologyEpoch) {
      throw new RangeError("Topology epochs must be monotonic");
    }
    this.topologyEpoch = epoch;
  }

  recordStage(stage: OctreeWorkStage, work: OctreeStageWorkObservation): void {
    const previous = this.stages[stage];
    const next: OctreeStageWork = Object.freeze({
      scheduledLanes: work.scheduledLanes === undefined
        ? previous.scheduledLanes
        : count(work.scheduledLanes, `${stage} scheduled lanes`),
      activeLanes: work.activeLanes === undefined
        ? previous.activeLanes
        : count(work.activeLanes, `${stage} active lanes`),
      activePages: work.activePages === undefined
        ? previous.activePages
        : count(work.activePages, `${stage} active pages`),
      logicalPages: work.logicalPages === undefined
        ? previous.logicalPages
        : count(work.logicalPages, `${stage} logical pages`),
      worksets: work.worksets === undefined
        ? previous.worksets
        : count(work.worksets, `${stage} worksets`),
      encodedIterations: work.encodedIterations === undefined
        ? previous.encodedIterations
        : count(work.encodedIterations, `${stage} encoded iterations`),
      executedIterations: work.executedIterations === undefined
        ? previous.executedIterations
        : count(work.executedIterations, `${stage} executed iterations`),
      reductionPasses: work.reductionPasses === undefined
        ? previous.reductionPasses
        : count(work.reductionPasses, `${stage} reduction passes`),
      estimatedBytesMoved: work.estimatedBytesMoved === undefined
        ? previous.estimatedBytesMoved
        : finiteNonNegative(work.estimatedBytesMoved, `${stage} estimated bytes moved`),
      estimatedBytesMovedPerActiveElement: null,
    });
    if (next.activeLanes !== null && next.scheduledLanes !== null
        && next.activeLanes > next.scheduledLanes) {
      throw new RangeError(`${stage} active lanes cannot exceed scheduled lanes`);
    }
    if (next.activePages !== null && next.logicalPages !== null
        && next.activePages > next.logicalPages) {
      throw new RangeError(`${stage} active pages cannot exceed logical pages`);
    }
    if (next.executedIterations !== null && next.encodedIterations !== null
        && next.executedIterations > next.encodedIterations) {
      throw new RangeError(`${stage} executed iterations cannot exceed encoded iterations`);
    }
    this.stages[stage] = Object.freeze({
      ...next,
      estimatedBytesMovedPerActiveElement:
        next.estimatedBytesMoved === null || next.activeLanes === null
          ? null
          : next.activeLanes === 0
            ? (next.estimatedBytesMoved === 0 ? 0 : null)
            : next.estimatedBytesMoved / next.activeLanes,
    });
  }

  /** Marks a stage that was positively observed to perform no work. */
  recordIdleStage(stage: OctreeWorkStage): void {
    this.recordStage(stage, {
      scheduledLanes: 0,
      activeLanes: 0,
      activePages: 0,
      logicalPages: 0,
      worksets: 0,
      encodedIterations: 0,
      executedIterations: 0,
      reductionPasses: 0,
      estimatedBytesMoved: 0,
    });
  }

  recordCatalogStamps(stamps: number): void {
    this.catalogStamps = count(stamps, "Catalog stamps");
  }

  recordSolver(iterations: number, reductionPasses: number, encodedIterations?: number): void {
    this.solverIterations = count(iterations, "Solver iterations");
    const stage = this.stages["pressure-solve"];
    this.recordStage("pressure-solve", {
      executedIterations: iterations,
      ...(encodedIterations === undefined ? {} : { encodedIterations }),
      reductionPasses,
      ...(stage.scheduledLanes === null ? {} : { scheduledLanes: stage.scheduledLanes }),
      ...(stage.activeLanes === null ? {} : { activeLanes: stage.activeLanes }),
      ...(stage.activePages === null ? {} : { activePages: stage.activePages }),
      ...(stage.logicalPages === null ? {} : { logicalPages: stage.logicalPages }),
      ...(stage.worksets === null ? {} : { worksets: stage.worksets }),
      ...(stage.estimatedBytesMoved === null ? {} : { estimatedBytesMoved: stage.estimatedBytesMoved }),
    });
  }

  /** Consume one fail-closed pressure telemetry decode after its GPU readback. */
  recordPressureSolveReport(report: OctreePressureSolveWorkReport): void {
    this.recordStage("pressure-solve", report.stage);
    this.solverIterations = report.outer.executedIterations;
  }

  setAllocationBytes(name: string, allocationClass: OctreeAllocationClass, bytes: number): void {
    if (name.length === 0) throw new RangeError("Allocation name must not be empty");
    const inventory = allocationClass === "authoritative" ? this.authorityBytes : this.scratchBytes;
    inventory.set(name, count(bytes, `${name} bytes`));
    this.allocationInventoryComplete = false;
  }

  setAuthorityBytes(authority: string, bytes: number): void {
    this.setAllocationBytes(authority, "authoritative", bytes);
  }

  setScratchBytes(arena: string, bytes: number): void {
    this.setAllocationBytes(arena, "scratch", bytes);
  }

  /** The caller asserts it has registered every persistent allocation. */
  sealAllocationInventory(): void {
    this.allocationInventoryComplete = true;
  }

  snapshot(): OctreeWorkSnapshot {
    const stages = Object.freeze({ ...this.stages });
    const missingStageMetrics = Object.freeze(Object.fromEntries(
      OCTREE_WORK_STAGES.map((stage) => [
        stage,
        Object.freeze(STAGE_METRICS.filter((metric) => stages[stage][metric] === null)),
      ]),
    ) as Record<OctreeWorkStage, readonly (keyof OctreeStageWork)[]>);
    const stagesComplete = Object.values(missingStageMetrics).every((metrics) => metrics.length === 0);
    const scheduledLanes = completeSum(stages, "scheduledLanes");
    const activeLanes = completeSum(stages, "activeLanes");
    const activePages = completeSum(stages, "activePages");
    const logicalPages = completeSum(stages, "logicalPages");
    const worksetCount = completeSum(stages, "worksets");
    const estimatedBytesMoved = completeSum(stages, "estimatedBytesMoved");
    const reductionPasses = completeSum(stages, "reductionPasses");
    const allocatedBytesByAuthority = Object.freeze(
      Object.fromEntries([...this.authorityBytes].sort(([a], [b]) => a.localeCompare(b))),
    );
    const allocatedScratchBytesByArena = Object.freeze(
      Object.fromEntries([...this.scratchBytes].sort(([a], [b]) => a.localeCompare(b))),
    );
    const authoritativeBytes = Object.values(allocatedBytesByAuthority)
      .reduce((sum, bytes) => sum + bytes, 0);
    const scratchBytes = Object.values(allocatedScratchBytesByArena)
      .reduce((sum, bytes) => sum + bytes, 0);
    return Object.freeze({
      topologyEpoch: this.topologyEpoch,
      catalogStamps: this.catalogStamps,
      solverIterations: this.solverIterations,
      reductionPasses,
      scheduledLanes,
      activeLanes,
      activePages,
      logicalPages,
      worksetCount,
      estimatedBytesMoved,
      activeLaneRatio: scheduledLanes === null || activeLanes === null
        ? null
        : scheduledLanes === 0 ? 1 : activeLanes / scheduledLanes,
      stagesComplete,
      missingStageMetrics,
      stages,
      allocatedBytesByAuthority,
      allocatedScratchBytesByArena,
      authoritativeBytes,
      scratchBytes,
      allocatedBytes: authoritativeBytes + scratchBytes,
      allocationInventoryComplete: this.allocationInventoryComplete,
    });
  }
}

export const OCTREE_PRESSURE_WORK_LAYER_NAMES = [
  "class-applies",
  "hybrid-shell",
  "spgrid-vcycle",
  "persistent-coarse-solve",
  "outer-pipelined-mgpcg",
] as const;
export type OctreePressureWorkLayerName = typeof OCTREE_PRESSURE_WORK_LAYER_NAMES[number];

export interface OctreePressureLayerWork {
  /** Payload invocations launched by nonzero direct/indirect records. The
   * one-thread publishers that gate those records are control traffic, not
   * scheduled row lanes, and are deliberately excluded. */
  readonly scheduledLanes: number;
  /** Invocations doing useful solver work after the fail-closed control gate. */
  readonly activeLanes: number;
  readonly activePages: number;
  readonly worksets: number;
  readonly encodedIterations: number;
  readonly executedIterations: number;
  /** Dispatch commands present in the immutable command graph. */
  readonly encodedDispatches: number;
  /** Payload commands whose direct/indirect dimensions launch at least one
   * workgroup. Control-only record publishers are deliberately excluded. */
  readonly executedDispatches: number;
  readonly reductionPasses: number;
  readonly estimatedBytesMoved: number;
}

export interface OctreePressureSolveGPUControls {
  /** Accepted resolved publication: error, first error, rows, epoch, bank, slots. */
  readonly acceptedRows?: Uint32Array;
  /** Four accepted-bank seven-word class workset headers. */
  readonly classWorksets?: readonly Uint32Array[];
  /** Four class intersections of the compact Section 4.3 L2 shell. */
  readonly hybridBandWorksets?: readonly Uint32Array[];
  /** Twelve words per level followed by valid/published-row lifecycle words. */
  readonly spgridDispatch?: Uint32Array;
  /** Outer merged-recurrence control record. */
  readonly outerControl?: Uint32Array;
  /** Four four-word indirect records per encoded outer iteration. */
  readonly outerIndirectTail?: Uint32Array;
  /** Optional one-workgroup persistent control when that solver is selected. */
  readonly persistentControl?: Uint32Array;
}

export interface OctreePressureSolveAccountingPlan {
  readonly rowCapacity: number;
  readonly maximumOuterIterations: number;
  /** Outer scalar reductions fold their 128-row blocks inside one workgroup. */
  readonly combinedReductionDrains?: boolean;
  readonly classWorkgroupSize?: 64;
  readonly maximumNeighborSlots: number;
  readonly classApplyEncodedDispatches: number;
  readonly hybridBoundarySweeps: number;
  readonly hybridEncodedCorrectionDispatches: number;
  readonly spgridLevelCount: number;
  readonly spgridEncodedCorrectionDispatches: number;
  readonly spgridLevelCapacities: readonly number[];
  readonly persistentMaximumIterations?: number;
  readonly persistentEnabled?: boolean;
  readonly reductionLanes?: 128;
  readonly reductionPartialCount: number;
}

export interface OctreePressureSolveWorkReport {
  readonly epoch: number;
  readonly liveRows: number;
  readonly outer: Readonly<{
    encodedIterations: number;
    executedIterations: number;
    initialReductionPasses: 1;
    outerReductionPasses: number;
    zeroedTailDispatches: number;
    tailExecutedRowLanes: 0;
  }>;
  readonly layers: Readonly<Record<OctreePressureWorkLayerName, OctreePressureLayerWork | null>>;
  readonly stage: Required<OctreeStageWorkObservation>;
}

export interface OctreePressureSolveWorkDecode {
  readonly report: OctreePressureSolveWorkReport | null;
  readonly blocker: string | null;
}

/** Complete observational packet copied from immutable GPU publications after
 * queue completion. Optional fields stay absent when the corresponding
 * authority is disabled or has not published; absence never means zero work. */
export interface OctreeRuntimeWorkGPUControls extends OctreePressureSolveGPUControls {
  /** Four accepted dynamic-boundary row classes consumed by RHS/reconstruction. */
  readonly runtimeClassWorksets?: readonly Uint32Array[];
  /** Four accepted dynamic-boundary family classes consumed by force/projection. */
  readonly runtimeFamilyWorksets?: readonly Uint32Array[];
  /** Accepted row and family-slot indirect records, two vec3u records. */
  readonly liveRowDispatch?: Uint32Array;
  readonly ownerControl?: Uint32Array;
  readonly fineWorkset?: Uint32Array;
  /** Source generation workset consumed by the last encoded transport. */
  readonly fineTransportWorkset?: Uint32Array;
  readonly fineTopologyControl?: Uint32Array;
  readonly fineTransportControl?: Uint32Array;
  /** GPU governor: error, executed substeps, substep dt, displacement, then enable words. */
  readonly fineTransportGovernor?: Uint32Array;
  readonly fineRedistanceControl?: Uint32Array;
}

export interface OctreeRuntimeWorkAccountingPlan {
  readonly pressure: OctreePressureSolveAccountingPlan;
  readonly ownerLogicalPages: number;
  readonly ownerBytesPerPage: number;
  readonly fineLogicalPages?: number;
  readonly fineSamplesPerPage?: 64;
  readonly fineMaximumPages?: number;
  readonly fineTransportEncodedIterations?: number;
  readonly fineRedistanceEncodedIterations?: number;
}

const READY_VALIDATED = 3;
const WORKSET_WORDS = Object.freeze({ epoch: 0, count: 1, capacity: 2, flags: 3,
  x: 4, y: 5, z: 6 });
const OUTER = Object.freeze({ error: 0, converged: 1, iterations: 2, reductions: 3,
  rows: 4, zeroed: 5, published: 20 });

function blocked(blocker: string): OctreePressureSolveWorkDecode {
  return Object.freeze({ report: null, blocker });
}

function safeProduct(values: ArrayLike<number>): number | null {
  let product = 1;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (!Number.isSafeInteger(value) || value < 0) return null;
    product *= value;
    if (!Number.isSafeInteger(product)) return null;
  }
  return product;
}

function layer(values: OctreePressureLayerWork): OctreePressureLayerWork {
  for (const [name, value] of Object.entries(values)) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`Pressure ${name} estimate must be a non-negative safe integer`);
    }
  }
  return Object.freeze(values);
}

function invalidPressurePlan(plan: OctreePressureSolveAccountingPlan): string | null {
  const positive = [
    [plan.rowCapacity, "row capacity"],
    [plan.maximumOuterIterations, "outer iteration budget"],
    [plan.maximumNeighborSlots, "neighbor-slot bound"],
    [plan.classApplyEncodedDispatches, "class-apply dispatch count"],
    [plan.spgridLevelCount, "SPGrid level count"],
    [plan.spgridEncodedCorrectionDispatches, "SPGrid correction dispatch count"],
    [plan.reductionPartialCount, "reduction partial count"],
  ] as const;
  for (const [value, label] of positive) {
    if (!Number.isSafeInteger(value) || value < 1) return `${label} is invalid`;
  }
  const nonNegative = [
    [plan.hybridBoundarySweeps, "hybrid boundary-sweep count"],
    [plan.hybridEncodedCorrectionDispatches, "hybrid correction dispatch count"],
  ] as const;
  for (const [value, label] of nonNegative) {
    if (!Number.isSafeInteger(value) || value < 0) return `${label} is invalid`;
  }
  if (plan.classWorkgroupSize !== undefined && plan.classWorkgroupSize !== 64) {
    return "class-apply workgroup size is not the production ABI";
  }
  if (plan.reductionLanes !== undefined && plan.reductionLanes !== 128) {
    return "outer reduction width is not the production ABI";
  }
  if (plan.reductionPartialCount !== Math.ceil(plan.rowCapacity / 128)) {
    return "outer reduction partial count does not match row capacity";
  }
  if (plan.spgridLevelCapacities.length !== plan.spgridLevelCount
    || plan.spgridLevelCapacities.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    return "SPGrid level capacities are invalid";
  }
  if (plan.persistentEnabled
    && (!Number.isSafeInteger(plan.persistentMaximumIterations)
      || plan.persistentMaximumIterations! < 1)) {
    return "persistent iteration budget is invalid";
  }
  return null;
}

/**
 * Decode already-read GPU controls into a deterministic pressure-stage record.
 * This function is observational only: no simulation decision may consume it.
 */
function decodeOctreePressureSolveWorkUnchecked(
  controls: OctreePressureSolveGPUControls,
  plan: OctreePressureSolveAccountingPlan,
): OctreePressureSolveWorkDecode {
  const invalidPlan = invalidPressurePlan(plan);
  if (invalidPlan !== null) return blocked(`pressure accounting plan is invalid: ${invalidPlan}`);
  const accepted = controls.acceptedRows;
  if (!accepted || accepted.length < 6) return blocked("accepted-row control unavailable");
  if (accepted[0] !== 0 || accepted[1] !== 0xffff_ffff
    || accepted[3] === 0 || accepted[4]! > 1) {
    return blocked("accepted-row control is invalid or unpublished");
  }
  const liveRows = accepted[2]!, epoch = accepted[3]!;
  if (liveRows === 0 || liveRows > plan.rowCapacity) {
    return blocked("accepted live-row count is outside the planned capacity");
  }
  const outer = controls.outerControl;
  if (!outer || outer.length <= OUTER.published) return blocked("outer MGPCG control unavailable");
  if (outer[OUTER.error] !== 0 || outer[OUTER.converged] !== 1
    || outer[OUTER.published] !== 1 || outer[OUTER.rows] !== liveRows) {
    return blocked("outer MGPCG control did not publish a successful solve");
  }
  const iterations = outer[OUTER.iterations]!, reductions = outer[OUTER.reductions]!;
  const requiredReductions = iterations === 0 ? 1 : 2 * iterations;
  if (iterations > plan.maximumOuterIterations || reductions !== requiredReductions) {
    return blocked("outer MGPCG iteration/reduction control is inconsistent");
  }
  const requiredZeroedTail = iterations === 0
    ? 4 * plan.maximumOuterIterations
    : 2 + 4 * (plan.maximumOuterIterations - iterations);
  if (outer[OUTER.zeroed] !== requiredZeroedTail) {
    return blocked("outer convergence tail was not proven zero-work");
  }
  const rowGroups = Math.ceil(liveRows / 64);
  const liveReductionPartials = Math.ceil(liveRows / (plan.reductionLanes ?? 128));
  if (!plan.persistentEnabled) {
    const indirect = controls.outerIndirectTail;
    if (!indirect || indirect.length < plan.maximumOuterIterations * 16) {
      return blocked("outer indirect-tail snapshot unavailable");
    }
    const zeroRecord = (record: number) => indirect[record * 4] === 0
      && indirect[record * 4 + 1] === 0 && indirect[record * 4 + 2] === 0
      && indirect[record * 4 + 3] === 0;
    const liveRecord = (record: number, x: number) => indirect[record * 4] === x
      && indirect[record * 4 + 1] === 1 && indirect[record * 4 + 2] === 1
      && indirect[record * 4 + 3] === 0;
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const base = iteration * 4;
      const terminal = iteration + 1 === iterations;
      if (!liveRecord(base, rowGroups)
        || (terminal
          ? !zeroRecord(base + 1)
          : !liveRecord(base + 1, liveReductionPartials))
        || !liveRecord(base + 2, liveReductionPartials)
        || (terminal ? !zeroRecord(base + 3) : !liveRecord(base + 3, rowGroups))) {
        return blocked("outer executed indirect records do not match the planned solve");
      }
    }
    for (let iteration = iterations; iteration < plan.maximumOuterIterations; iteration += 1) {
      for (let stage = 0; stage < 4; stage += 1) {
        if (!zeroRecord(iteration * 4 + stage)) {
          return blocked("outer future convergence-tail record is not zero-work");
        }
      }
    }
  }

  const worksets = controls.classWorksets;
  if (!worksets || worksets.length !== 4) return blocked("resolved class worksets unavailable");
  let classRows = 0, classScheduledPerApply = 0, liveClassDispatchesPerApply = 0;
  for (const header of worksets) {
    if (header.length < 7 || header[WORKSET_WORDS.epoch] !== epoch
      || (header[WORKSET_WORDS.flags]! & READY_VALIDATED) !== READY_VALIDATED
      || (header[WORKSET_WORDS.flags]! & ~7) !== 0
      || header[WORKSET_WORDS.capacity]! > plan.rowCapacity
      || header[WORKSET_WORDS.count]! > header[WORKSET_WORDS.capacity]!) {
      return blocked("resolved class workset is invalid or stale");
    }
    const groups = safeProduct([header[WORKSET_WORDS.x]!, header[WORKSET_WORDS.y]!,
      header[WORKSET_WORDS.z]!]);
    if (groups === null) return blocked("resolved class dispatch is invalid");
    if (groups !== Math.ceil(header[WORKSET_WORDS.count]! / (plan.classWorkgroupSize ?? 64))) {
      return blocked("resolved class dispatch does not cover its compact row count");
    }
    classRows += header[WORKSET_WORDS.count]!;
    classScheduledPerApply += groups * (plan.classWorkgroupSize ?? 64);
    liveClassDispatchesPerApply += Number(groups > 0);
  }
  if (classRows !== liveRows) return blocked("resolved class worksets do not partition live rows");
  const bandWorksets = controls.hybridBandWorksets;
  if (!bandWorksets || bandWorksets.length !== 4) {
    return blocked("Section 4.3 compact class intersections unavailable");
  }
  let bandRows = 0, bandScheduledPerApply = 0, liveBandDispatchesPerApply = 0;
  for (const header of bandWorksets) {
    if (header.length < 7 || header[WORKSET_WORDS.epoch] !== epoch
      || header[WORKSET_WORDS.flags] !== READY_VALIDATED
      || header[WORKSET_WORDS.capacity]! > plan.rowCapacity
      || header[WORKSET_WORDS.count]! > header[WORKSET_WORDS.capacity]!) {
      return blocked("Section 4.3 compact class intersection is invalid or stale");
    }
    const groups = safeProduct(header.subarray(WORKSET_WORDS.x, WORKSET_WORDS.z + 1));
    if (groups === null || groups !== Math.ceil(header[WORKSET_WORDS.count]! / 64)) {
      return blocked("Section 4.3 compact class dispatch is invalid");
    }
    bandRows += header[WORKSET_WORDS.count]!;
    bandScheduledPerApply += groups * 64;
    liveBandDispatchesPerApply += Number(groups > 0);
  }
  if (bandRows > liveRows) return blocked("Section 4.3 compact shell exceeds live rows");

  const dispatch = controls.spgridDispatch;
  const dispatchWords = plan.spgridLevelCount * 12 + 2;
  if (!dispatch || dispatch.length < dispatchWords) return blocked("SPGrid dispatch snapshot unavailable");
  const lifecycle = plan.spgridLevelCount * 12;
  if (dispatch[lifecycle] !== 1 || dispatch[lifecycle + 1] !== liveRows) {
    return blocked("SPGrid hierarchy is invalid or stale");
  }
  let levelRows = 0, transferRows = 0, pagesPerCycle = 0;
  let selectedScheduledLanes = 0, transferScheduledLanes = 0;
  let executedSPGridDispatchesPerCorrection = 3;
  for (let levelIndex = 0; levelIndex < plan.spgridLevelCount; levelIndex += 1) {
    const base = levelIndex * 12, count = dispatch[base]!;
    const transfers = dispatch[base + 1]!, pages = dispatch[base + 8]!;
    if (count > plan.spgridLevelCapacities[levelIndex]!
      || pages > plan.spgridLevelCapacities[levelIndex]!) {
      return blocked("SPGrid level count exceeds its published capacity");
    }
    const selectedGroups = safeProduct(dispatch.subarray(base + 2, base + 5));
    const parentGroups = safeProduct(dispatch.subarray(base + 5, base + 8));
    const pageGroups = safeProduct(dispatch.subarray(base + 9, base + 12));
    const parentSlots = levelIndex + 1 < plan.spgridLevelCount
      ? dispatch[(levelIndex + 1) * 12]!
      : 0;
    if (selectedGroups === null || parentGroups === null || pageGroups === null
      || selectedGroups !== Math.ceil(count / 64)
      // Restriction is one cooperative workgroup per coarse parent slot. The
      // transfer-record count bounds the chain walked inside those groups; it
      // is not the indirect dispatch width.
      || parentGroups !== parentSlots || pageGroups !== pages) {
      return blocked("SPGrid indirect dispatch does not match its compact publication");
    }
    levelRows += count;
    selectedScheduledLanes += selectedGroups * 64;
    executedSPGridDispatchesPerCorrection += Number(selectedGroups > 0);
    if (levelIndex + 1 < plan.spgridLevelCount) {
      transferRows += transfers;
      transferScheduledLanes += parentGroups * 64;
      pagesPerCycle += 2 * pages;
      executedSPGridDispatchesPerCorrection += 2 * Number(pages > 0)
        + Number(parentGroups > 0) + Number(selectedGroups > 0);
    } else {
      executedSPGridDispatchesPerCorrection += Number(selectedGroups > 0);
    }
  }
  if (dispatch[(plan.spgridLevelCount - 1) * 12] !== 1) {
    return blocked("SPGrid exact bottom is not a published single-cell solve");
  }

  const priorUpdates = Math.max(0, iterations - 1);
  const encodedPreconditionerApplications = 1 + plan.maximumOuterIterations;
  const preconditionerApplications = 1 + iterations;
  const fullEncodedOperatorApplications = 2 + plan.maximumOuterIterations
    + encodedPreconditionerApplications;
  // Direct-curvature PCG applies A(direction) only after a nonterminal
  // residual update. The terminal converged update zeroes that indirect
  // record, so credit prior updates rather than every completed iteration.
  const fullOperatorApplications = 2 + priorUpdates + preconditionerApplications;
  const bandAppliesPerPreconditioner = 2 * plan.hybridBoundarySweeps - 1;
  const bandEncodedOperatorApplications = bandAppliesPerPreconditioner
    * encodedPreconditionerApplications;
  const bandOperatorApplications = bandAppliesPerPreconditioner
    * preconditionerApplications;
  const encodedOperatorApplications = fullEncodedOperatorApplications
    + bandEncodedOperatorApplications;
  const operatorApplications = fullOperatorApplications + bandOperatorApplications;
  const classBytesPerRow = 4 * (4 + 2 * plan.maximumNeighborSlots);
  const classApplies = layer({
    // Every nested apply has a one-thread publisher, but convergence makes
    // its payload records exactly zero. Credit only the observed nonzero
    // applications; encodedIterations/encodedDispatches retain graph size.
    scheduledLanes: classScheduledPerApply * fullOperatorApplications
      + bandScheduledPerApply * bandOperatorApplications,
    activeLanes: liveRows * fullOperatorApplications
      + bandRows * bandOperatorApplications,
    activePages: 0,
    worksets: 4 * operatorApplications,
    encodedIterations: encodedOperatorApplications,
    executedIterations: operatorApplications,
    encodedDispatches: plan.classApplyEncodedDispatches * encodedOperatorApplications,
    executedDispatches: liveClassDispatchesPerApply * fullOperatorApplications
      + liveBandDispatchesPerApply * bandOperatorApplications,
    reductionPasses: 0,
    estimatedBytesMoved: classRows * classBytesPerRow * fullOperatorApplications
      + bandRows * classBytesPerRow * bandOperatorApplications,
  });
  // Four row-wide stages (zero sweep, inner residual, M1 seeding, publication),
  // 2k-1 compact shell smooths, and the singleton convergence-gate publisher
  // that republishes the row and union-class records for this application.
  const hybridRowPasses = 2 * plan.hybridBoundarySweeps + 4;
  const capacityRowLanes = Math.ceil(plan.rowCapacity / 64) * 64;
  const liveRowLanes = Math.ceil(liveRows / 64) * 64;
  const fullHybridPasses = 4;
  const bandHybridPasses = 2 * plan.hybridBoundarySweeps - 1;
  const hybrid = layer({
    scheduledLanes: (liveRowLanes * fullHybridPasses
      + bandScheduledPerApply * bandHybridPasses) * preconditionerApplications,
    activeLanes: (liveRows * fullHybridPasses
      + bandRows * bandHybridPasses) * preconditionerApplications,
    activePages: 0,
    worksets: preconditionerApplications,
    encodedIterations: encodedPreconditionerApplications,
    executedIterations: preconditionerApplications,
    encodedDispatches: hybridRowPasses * encodedPreconditionerApplications,
    executedDispatches: hybridRowPasses * preconditionerApplications,
    reductionPasses: 0,
    estimatedBytesMoved: (liveRows * fullHybridPasses
      + bandRows * bandHybridPasses) * 16 * preconditionerApplications,
  });
  const spgrid = layer({
    scheduledLanes: (3 * capacityRowLanes + 2 * selectedScheduledLanes
      + transferScheduledLanes + pagesPerCycle * 128) * preconditionerApplications,
    activeLanes: (3 * liveRows + 2 * levelRows + transferRows
      + pagesPerCycle * 128) * preconditionerApplications,
    activePages: pagesPerCycle * preconditionerApplications,
    worksets: plan.spgridLevelCount * preconditionerApplications,
    encodedIterations: encodedPreconditionerApplications,
    executedIterations: preconditionerApplications,
    encodedDispatches: plan.spgridEncodedCorrectionDispatches
      * (1 + plan.maximumOuterIterations),
    executedDispatches: executedSPGridDispatchesPerCorrection
      * preconditionerApplications,
    reductionPasses: 0,
    estimatedBytesMoved: (pagesPerCycle * (12_000 + 256 * 4) + levelRows * 24)
      * preconditionerApplications,
  });

  let persistent: OctreePressureLayerWork | null = null;
  if (plan.persistentEnabled) {
    const control = controls.persistentControl;
    if (!control || control.length < 5) return blocked("persistent solve control unavailable");
    if (control[0] !== 0 || control[1] !== 1 || control[4] !== liveRows
      || control[2]! > (plan.persistentMaximumIterations ?? 12)) {
      return blocked("persistent solve control is invalid or unconverged");
    }
    const persistentIterations = control[2]!;
    persistent = layer({
      scheduledLanes: 256,
      activeLanes: Math.min(liveRows, 256),
      activePages: 0,
      worksets: 1,
      encodedIterations: plan.persistentMaximumIterations ?? 12,
      executedIterations: persistentIterations,
      encodedDispatches: 1,
      executedDispatches: 1,
      reductionPasses: persistentIterations + 1,
      estimatedBytesMoved: liveRows * (persistentIterations * 48 + 64),
    });
  }
  const partialLanes = liveReductionPartials * (plan.reductionLanes ?? 128);
  const combinedReductionDrains = plan.combinedReductionDrains === true;
  const outerPartialReductions = combinedReductionDrains
    ? 1
    : iterations + 1 + priorUpdates;
  const outerLayer = layer({
    scheduledLanes: 2 + liveRowLanes * (4 + iterations + priorUpdates)
      + partialLanes * outerPartialReductions
      + (plan.reductionLanes ?? 128) * (2 * plan.maximumOuterIterations + 1),
    activeLanes: 2 + liveRows * (4 + iterations + priorUpdates)
      + partialLanes * outerPartialReductions
      + (plan.reductionLanes ?? 128) * reductions,
    activePages: 0,
    worksets: 1,
    encodedIterations: plan.maximumOuterIterations,
    executedIterations: iterations,
    encodedDispatches: 8 + (combinedReductionDrains ? 4 : 6)
      * plan.maximumOuterIterations,
    executedDispatches: 8 + 2 * plan.maximumOuterIterations
      + (combinedReductionDrains ? 1 : 2) * (iterations + priorUpdates),
    reductionPasses: reductions,
    estimatedBytesMoved: liveRows * (56 + iterations * 72)
      + liveReductionPartials * 32 * (combinedReductionDrains ? 1 : reductions),
  });
  const layers = Object.freeze({
    "class-applies": classApplies,
    "hybrid-shell": hybrid,
    "spgrid-vcycle": spgrid,
    "persistent-coarse-solve": persistent,
    "outer-pipelined-mgpcg": outerLayer,
  });
  const present = Object.values(layers).filter((value): value is OctreePressureLayerWork => value !== null);
  const total = (key: keyof OctreePressureLayerWork) => present.reduce((sum, item) => sum + item[key], 0);
  const stage: Required<OctreeStageWorkObservation> = Object.freeze({
    scheduledLanes: total("scheduledLanes"), activeLanes: total("activeLanes"),
    activePages: total("activePages"), logicalPages: total("activePages"),
    worksets: total("worksets"), encodedIterations: plan.maximumOuterIterations,
    executedIterations: iterations, reductionPasses: total("reductionPasses"),
    estimatedBytesMoved: total("estimatedBytesMoved"),
  });
  return Object.freeze({ report: Object.freeze({ epoch, liveRows,
    outer: Object.freeze({ encodedIterations: plan.maximumOuterIterations,
      executedIterations: iterations, initialReductionPasses: 1 as const,
      outerReductionPasses: reductions - 1, zeroedTailDispatches: outer[OUTER.zeroed]!,
      tailExecutedRowLanes: 0 as const }), layers, stage }), blocker: null });
}

export function decodeOctreePressureSolveWork(
  controls: OctreePressureSolveGPUControls,
  plan: OctreePressureSolveAccountingPlan,
): OctreePressureSolveWorkDecode {
  try {
    return decodeOctreePressureSolveWorkUnchecked(controls, plan);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown accounting failure";
    return blocked(`pressure accounting controls or estimates are invalid: ${detail}`);
  }
}

const OWNER_READY = 1;
const OWNER_INVALID_STATUS = (1 << 2) | (1 << 3) | (1 << 6);
const OWNER_MAGIC = 0x4f57_4e52;

function decodeCompactWorksets(headers: readonly Uint32Array[] | undefined, epoch: number): {
  scheduledLanes: number; activeLanes: number; worksets: number;
} | null {
  if (!headers || headers.length === 0) return null;
  let scheduledLanes = 0, activeLanes = 0;
  for (const header of headers) {
    if (header.length < 7 || header[0] !== epoch || header[1]! > header[2]!
      || (header[3]! & READY_VALIDATED) !== READY_VALIDATED
      || header[5] !== 1 || header[6] !== 1) return null;
    const groups = safeProduct(header.subarray(4, 7));
    if (groups === null || groups !== Math.ceil(header[1]! / 64)) return null;
    scheduledLanes += groups * 64;
    activeLanes += header[1]!;
  }
  return { scheduledLanes, activeLanes, worksets: headers.length };
}

/**
 * Apply one post-submit GPU capture to the observational stage ledger.
 * Every populated value is decoded from a compact GPU authority. Metrics not
 * proved by those controls remain null in the accounting snapshot.
 */
export function recordOctreeRuntimeGPUWork(
  accounting: OctreeWorkAccounting,
  controls: OctreeRuntimeWorkGPUControls,
  plan: OctreeRuntimeWorkAccountingPlan,
): OctreePressureSolveWorkDecode {
  const pressure = decodeOctreePressureSolveWork(controls, plan.pressure);
  if (pressure.report) accounting.recordPressureSolveReport(pressure.report);

  const accepted = controls.acceptedRows;
  const acceptedValid = accepted !== undefined && accepted.length >= 6
    && accepted[0] === 0 && accepted[1] === 0xffff_ffff
    && accepted[3] !== 0 && accepted[4]! <= 1
    && accepted[2]! <= plan.pressure.rowCapacity;
  if (acceptedValid) {
    const epoch = accepted![3]!, liveRows = accepted![2]!;
    const liveDispatch = controls.liveRowDispatch;
    const acceptedSlots = accepted![5]!;
    const liveDispatchValid = liveDispatch !== undefined && liveDispatch.length >= 6
      && liveDispatch[0] === Math.ceil(liveRows / 64)
      && liveDispatch[1] === 1 && liveDispatch[2] === 1
      && liveDispatch[3] === Math.ceil(acceptedSlots / 64)
      && liveDispatch[4] === 1 && liveDispatch[5] === 1;
    accounting.publishTopologyEpoch(epoch);
    accounting.recordCatalogStamps(accepted![5]!);
    const resolved = decodeCompactWorksets(controls.runtimeClassWorksets, epoch);
    const families = decodeCompactWorksets(controls.runtimeFamilyWorksets, epoch);
    const owner = controls.ownerControl;
    const ownerValid = owner !== undefined && owner.length >= 16 && owner[15] === OWNER_MAGIC
      && owner[2] === 0 && owner[12] === 0 && owner[7] !== 0 && owner[1]! <= owner[3]!
      && owner[4] === plan.ownerLogicalPages && owner[1]! <= plan.ownerLogicalPages
      && (owner[10]! & OWNER_READY) !== 0 && (owner[10]! & OWNER_INVALID_STATUS) === 0;
    const ownerPages = ownerValid ? owner![1]! : null;
    if (liveDispatchValid && resolved && resolved.activeLanes === liveRows) {
      const compactRows = { scheduledLanes: resolved.scheduledLanes,
        activeLanes: resolved.activeLanes,
        activePages: ownerPages ?? 0, logicalPages: plan.ownerLogicalPages,
        worksets: resolved.worksets, encodedIterations: 1, executedIterations: 1,
        reductionPasses: 0,
        estimatedBytesMoved: liveRows * 4 * (4 + 2 * plan.pressure.maximumNeighborSlots) };
      if (ownerPages !== null) accounting.recordStage("rhs", compactRows);
      if (ownerPages !== null && families) accounting.recordStage("projection", {
        scheduledLanes: compactRows.scheduledLanes + families.scheduledLanes,
        activeLanes: compactRows.activeLanes + families.activeLanes,
        activePages: ownerPages, logicalPages: plan.ownerLogicalPages,
        worksets: compactRows.worksets + families.worksets,
        encodedIterations: 1, executedIterations: 1, reductionPasses: 0,
        estimatedBytesMoved: compactRows.estimatedBytesMoved
          + families.activeLanes * 24 + liveRows * 16,
      });
    }
    if (liveDispatchValid && families && ownerPages !== null) accounting.recordStage("forces-and-solids", {
      ...families, activePages: ownerPages, logicalPages: plan.ownerLogicalPages,
      encodedIterations: 1, executedIterations: 1, reductionPasses: 0,
      estimatedBytesMoved: families.activeLanes * 16,
    });
  }

  const owner = controls.ownerControl;
  if (owner && owner.length >= 16 && owner[15] === OWNER_MAGIC && owner[2] === 0
    && owner[12] === 0 && owner[7] !== 0 && owner[1]! <= owner[3]!
    && owner[4] === plan.ownerLogicalPages && owner[1]! <= plan.ownerLogicalPages
    && (owner[10]! & OWNER_READY) !== 0 && (owner[10]! & OWNER_INVALID_STATUS) === 0) {
    accounting.recordStage("topology-publication", {
      scheduledLanes: 1, activeLanes: 1,
      activePages: owner[1]!, logicalPages: owner[4]!, worksets: 1,
      encodedIterations: 1, executedIterations: 1, reductionPasses: 0,
      estimatedBytesMoved: 64 + owner[1]! * plan.ownerBytesPerPage,
    });
  }

  const fine = controls.fineWorkset, fineTopology = controls.fineTopologyControl;
  const fineValid = fine !== undefined && fine.length >= 7 && fine[0] !== 0
    && fine[1]! <= fine[2]! && (fine[3]! & READY_VALIDATED) === READY_VALIDATED
    && fine[5] === 1 && fine[6] === 1
    && fineTopology !== undefined && fineTopology.length >= 8
    && fineTopology[0] === 0 && fineTopology[4] === 1 && fineTopology[5] === 0;
  if (fineValid) {
    const groups = safeProduct(fine!.subarray(4, 7));
    const logicalPages = plan.fineLogicalPages;
    const finePages = fineTopology![2]!;
    if (logicalPages !== undefined && finePages <= logicalPages && groups !== null
      && groups === Math.ceil(fine![1]! / 64)) {
      accounting.recordStage("candidate-build", {
        scheduledLanes: groups * 64, activeLanes: finePages,
        activePages: finePages, logicalPages, worksets: 1,
        encodedIterations: 1, executedIterations: 1, reductionPasses: 0,
        estimatedBytesMoved: finePages * 40,
      });
    }
    const transport = controls.fineTransportControl;
    const transportWorkset = controls.fineTransportWorkset;
    const governor = controls.fineTransportGovernor;
    const transportWorksetValid = transportWorkset !== undefined && transportWorkset.length >= 7
      && transportWorkset[0] !== 0 && transportWorkset[1]! <= transportWorkset[2]!
      && (transportWorkset[3]! & READY_VALIDATED) === READY_VALIDATED
      && transportWorkset[5] === 1 && transportWorkset[6] === 1;
    if (transportWorksetValid
      && transport && transport.length >= 8 && transport[3] === 1
      && transport[0] === 0 && transport[1] === 0
      && transport[6] === 0 && transport[7] === 0
      && governor && governor.length >= 4 && governor[0] === 0
      && governor[1]! >= 1
      && governor[1]! <= (plan.fineTransportEncodedIterations ?? 0)
      && logicalPages !== undefined && transportWorkset![1]! <= logicalPages) {
      const executed = governor[1]!, samples = plan.fineSamplesPerPage ?? 64;
      const scheduled = transportWorkset![1]! * samples * executed;
      if (transport[2]! <= scheduled) {
      accounting.recordStage("fine-transport", {
        scheduledLanes: scheduled, activeLanes: transport[2]!, activePages: transportWorkset![1]!,
        logicalPages, worksets: 1,
        encodedIterations: plan.fineTransportEncodedIterations!, executedIterations: executed,
        reductionPasses: executed,
        estimatedBytesMoved: transport[2]! * 24,
      });
      }
    }
    const redistance = controls.fineRedistanceControl;
    if (redistance && redistance.length >= 9 && redistance[0] === 0
      && redistance[3] === 1 && redistance[4] === 0
      && logicalPages !== undefined && redistance[8]! <= logicalPages
      && plan.fineRedistanceEncodedIterations !== undefined) {
      const samples = plan.fineSamplesPerPage ?? 64;
      const iterations = redistance[8] === 0 ? 0 : plan.fineRedistanceEncodedIterations;
      const scheduled = redistance[8]! * samples * iterations;
      accounting.recordStage("fine-redistance", {
        scheduledLanes: scheduled, activeLanes: scheduled,
        activePages: redistance[8]!, logicalPages, worksets: 2,
        encodedIterations: plan.fineRedistanceEncodedIterations,
        executedIterations: iterations, reductionPasses: iterations,
        estimatedBytesMoved: scheduled * 16,
      });
    }
  } else if (plan.fineLogicalPages === undefined) {
    accounting.recordIdleStage("candidate-build");
    accounting.recordIdleStage("fine-transport");
    accounting.recordIdleStage("fine-redistance");
  }
  return pressure;
}

export type OctreeProductionSourceRole = "host" | "shader";

export interface OctreeSourceViolation {
  readonly source: string;
  readonly line: number;
  readonly rule:
    | "capacity-dispatch"
    | "unbounded-lookup"
    | "galerkin-rap"
    | "primary-particle-transport"
    | "floating-velocity-scatter"
    | "retired-shader-terminology"
    | "deleted-production-module"
    | "general-timestep-authority"
    | "capacity-scan"
    | "nested-capacity-scan"
    | "capacity-indirect-args";
  readonly excerpt: string;
}

interface SourceCheck {
  readonly rule: OctreeSourceViolation["rule"];
  readonly expression: RegExp;
}

const DELETED_PRODUCTION_MODULE =
  /\b(?:webgpu-octree-mgpcg|webgpu-octree-power-galerkin(?:-persistent3)?|octree-power-galerkin)(?:\.ts)?\b|\b(?:WebGPUOctreeMGPCG|WebGPUOctreePowerGalerkin(?:Persistent3)?|OctreePowerGalerkin)\b/g;

const HOST_SOURCE_CHECKS: readonly SourceCheck[] = Object.freeze([
  {
    rule: "deleted-production-module",
    expression: DELETED_PRODUCTION_MODULE,
  },
  {
    rule: "unbounded-lookup",
    expression: /while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\)/g,
  },
  {
    rule: "galerkin-rap",
    expression: /\b(?:assembleGalerkin|refreshGalerkin|tripleProduct|galerkinRAP)\b/gi,
  },
  {
    rule: "primary-particle-transport",
    expression: /\b(?:WebGPUOctreeParticles|octree-particles|primary\s+(?:FLIP|APIC)|P2G|G2P)\b/gi,
  },
  {
    rule: "floating-velocity-scatter",
    expression: /atomicAdd\s*\(\s*&[^,\n]*(?:velocity|momentum|faceVelocity)[^,\n]*/gi,
  },
]);

const SHADER_SOURCE_CHECKS: readonly SourceCheck[] = Object.freeze([
  ...HOST_SOURCE_CHECKS,
  {
    rule: "unbounded-lookup",
    // WGSL's unbounded spelling is `loop { ... }`, rather than JavaScript's
    // `while (true)`. Limit the guard to lookup/search loops so fixed-width
    // reductions and explicitly terminating CAS retries are not mislabeled.
    expression: /\bloop\s*\{(?=[\s\S]{0,512}\b(?:lookup|search|probe)\b)/gi,
  },
]);

/** Identifiers whose value is fixed at allocation time or by domain extent. A
 * launch shaped by one of these costs O(capacity) or O(domain) forever, however
 * little fluid exists -- the violation Bet 1 ("existence is free") exists to
 * prevent. Capacities may size *buffers*; they may never size *launches*.
 *
 * Three spelling holes made this gate report clean while the violations it
 * names were present, so all three are now closed by shape, not by enumeration:
 *
 *  - `\w*Capacity` replaces a hard-coded five-name list. Every capacity
 *    introduced after that list was written was exempt by default --
 *    `candidateCapacity`, `identityCapacity`, `hierarchyKeyCapacity`,
 *    `sourceCapacity`, `scanRecordCapacity`, `sparseCandidateCapacity` among
 *    them, 26 sites in the octree sources alone.
 *  - `logical(?:Cell|Brick|Page|Domain)\w*` replaces a trailing `\b`, which
 *    matched the bare `logicalBrick` but not `logicalBrickCount` -- and
 *    `logicalBrickCount` is the spelling the code actually uses.
 *  - `[Cc]apacity` replaces `Capacity`. A field spelled `this.capacity` (rather
 *    than `this.rowCapacity`) was exempt: four of the six capacity-shaped
 *    launches in `webgpu-fluid-brick-residency.ts` are `dispatchWorkgroups(
 *    bricks)` where `bricks = Math.ceil(this.capacity / 64)`.
 *
 * `domainVolume` was absent entirely, so the `ceil(domainVolume/256)` family
 * that Bet 1 names as its headline violation could never have been flagged. */
const CAPACITY_AUTHORITY =
  /\b(?:\w*[Cc]apacity|maximumResidentBricks|logical(?:Cell|Brick|Page|Domain)\w*|total\w*Count|brickCount|pageDirectoryWords|domainVolume|levelStride|dims\.n[xyz])\b/;

function escapedIdentifier(value: string): string {
  return value.replace(/[\^$.*+?()[\]{}|\\]/g, "\\$&");
}

/** Reads a balanced argument list starting just after an opening parenthesis.
 * Argument lists here routinely span lines, so a line-bounded scan would clip
 * the workgroup count off exactly the multi-line calls that carry it. */
function balancedArguments(source: string, openIndex: number): { text: string; end: number } {
  let cursor = openIndex, depth = 1;
  while (cursor < source.length && depth > 0) {
    const character = source[cursor];
    if (character === "(" || character === "[" || character === "{") depth += 1;
    else if (character === ")" || character === "]" || character === "}") depth -= 1;
    cursor += 1;
  }
  return { text: source.slice(openIndex, depth === 0 ? cursor - 1 : cursor), end: cursor };
}

/** Locally-bound functions that dispatch. A capacity that reaches one of these
 * as an argument is a capacity-shaped launch just as surely as one written into
 * `dispatchWorkgroups` directly -- and this is the spelling the octree encoders
 * actually use: `run(pipeline, entries, label, Math.ceil(capacity / 64), …)`,
 * where `run` closes over the pass and calls `pass.dispatchWorkgroups(x, y)`.
 * Scanning only literal `dispatchWorkgroups(` argument text reported this file
 * clean while every one of those launches was capacity-shaped. */
function dispatchingHelpers(source: string): Set<string> {
  const helpers = new Set<string>();
  const definition = /(?:\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)(?:\s*:[^=]+)?\s*=>\s*\{|\bfunction\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*(?::[^{]+)?\{)/g;
  for (let match = definition.exec(source); match; match = definition.exec(source)) {
    const name = match[1] ?? match[2]!;
    const open = source.indexOf("{", match.index + match[0].length - 1);
    if (open < 0) continue;
    const body = source.slice(open, matchingBrace(source, open));
    if (/\bdispatchWorkgroups\s*\(/.test(body)) helpers.add(name);
  }
  return helpers;
}

/** Follow simple host aliases before inspecting direct-dispatch arguments.
 * Renaming capacity-derived work to groups or this.workgroups must not turn a
 * capacity launch into an apparently live-work launch. */
function capacityDispatchViolations(sourceName: string, source: string): OctreeSourceViolation[] {
  const aliases = new Set<string>();
  const assignments = [...source.matchAll(
    /(?:\b(?:const|let|var)\s+)?((?:this\.)?[A-Za-z_$][\w$]*)\s*=\s*([^;\n]+)/g,
  )];
  let changed = true;
  while (changed) {
    changed = false;
    for (const match of assignments) {
      const target = match[1]!, expression = match[2]!;
      const derivesFromCapacity = CAPACITY_AUTHORITY.test(expression)
        || [...aliases].some((alias) => new RegExp(
          "(?:^|[^A-Za-z0-9_$])" + escapedIdentifier(alias) + "(?:$|[^A-Za-z0-9_$])",
        ).test(expression));
      if (derivesFromCapacity && !aliases.has(target)) {
        aliases.add(target); changed = true;
      }
    }
  }
  const derivesFromCapacity = (expression: string): boolean =>
    CAPACITY_AUTHORITY.test(expression)
      || [...aliases].some((alias) => new RegExp(
        "(?:^|[^A-Za-z0-9_$])" + escapedIdentifier(alias) + "(?:$|[^A-Za-z0-9_$])",
      ).test(expression));

  const violations: OctreeSourceViolation[] = [];
  // `dispatchWorkgroupsIndirect` is deliberately not matched *here*: an indirect
  // launch is shaped by whatever was published into the args buffer, and only
  // the direct call and the local helpers that wrap it carry a host-authored
  // workgroup count on the host side.
  //
  // That exemption was itself a hole, and it is now closed on the shader side by
  // `capacityIndirectArgumentViolations`: a GPU kernel may perfectly well write
  // a capacity into an indirect-args buffer, and then the indirect launch is a
  // capacity-shaped launch wearing an indirect costume. See that function for
  // the evidence that made this the gate's third measured blindness.
  const helpers = dispatchingHelpers(source);
  const call = new RegExp(
    "\\b(?:dispatchWorkgroups|dispatchFor|workgroupPerItemDispatch"
      + [...helpers].map((name) => "|" + escapedIdentifier(name)).join("")
      + ")\\s*\\(", "g");
  for (let match = call.exec(source); match; match = call.exec(source)) {
    const { text, end } = balancedArguments(source, call.lastIndex);
    if (derivesFromCapacity(text)) violations.push({
      source: sourceName,
      line: sourceLine(source, match.index),
      rule: "capacity-dispatch",
      excerpt: source.slice(match.index, Math.min(end, match.index + 160))
        .replace(/\s+/g, " ").trim(),
    });
    call.lastIndex = end;
  }
  return violations;
}

/** A trip count fixed when the shader is compiled: `cls<9u`, `width<64u`. */
const FIXED_TRIP_COUNT = /[<>]=?\s*\d+[iu]?\s*[;)&|]/;

/** An induction range anchored to a lane/workgroup offset. `first..last` where
 * `first = lane*chunk` partitions the list across lanes, so total work is the
 * live count even though the clamp expression names a capacity. */
const LANE_PARTITIONED = /\b(?:lane|lid|local|tid|first|start|begin|wid|work)\w*\b/i;

/** Matches the midpoint-halving shape of a binary search, whose trip count is
 * logarithmic rather than linear. Reported separately when nested, because the
 * cost model — and the fix — is different from a linear rescan. */
const BINARY_SEARCH_BODY = /\b(?:middle|mid)\s*=\s*\w+\s*\+\s*\(|>>=\s*1u|\/\s*2u\s*;/;

function matchingBrace(source: string, open: number): number {
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}") { depth -= 1; if (depth === 0) return index; }
  }
  return source.length;
}

/**
 * Flags a scan whose trip count follows a capacity or work-list bound and that
 * runs once per work item rather than once per dispatch.
 *
 * Two shapes qualify. A capacity-bounded loop inside a non-entry helper is
 * executed by every caller, so a kernel dispatched over N items pays N × the
 * list length. A capacity-bounded loop nested inside another loop pays the same
 * product within a single invocation. Both are the quadratic behaviour the
 * narrow-band design exists to avoid; neither is caught by matching loop
 * keywords, because the loops themselves terminate.
 *
 * Compile-time trip counts and lane-partitioned ranges are excluded: they are
 * the ordinary way this codebase spreads a live list across a workgroup.
 */
function capacityScanViolations(sourceName: string, source: string): OctreeSourceViolation[] {
  const code = shaderCodeOnly(source);
  const violations: OctreeSourceViolation[] = [];
  const functionHeader = /\bfn\s+([A-Za-z_][\w]*)\s*\(/g;
  for (let match = functionHeader.exec(code); match; match = functionHeader.exec(code)) {
    const open = code.indexOf("{", match.index);
    if (open < 0) continue;
    const body = code.slice(open, matchingBrace(code, open));
    const isEntryPoint = /@compute/.test(code.slice(Math.max(0, match.index - 200), match.index));

    // Resolve capacity-derived locals to a fixed point so a bound spelled
    // through two or three assignments is still recognized.
    const aliases = new Set<string>();
    const assignments = [...body.matchAll(
      /\b(?:let|var)\s+([A-Za-z_][\w]*)\s*(?::[^=;]+)?=\s*([^;\n]+)/g)];
    for (let changed = true; changed;) {
      changed = false;
      for (const assignment of assignments) {
        const target = assignment[1]!, expression = assignment[2]!;
        const derived = CAPACITY_AUTHORITY.test(expression)
          || [...aliases].some((alias) => new RegExp(
            "(?:^|[^A-Za-z0-9_$])" + escapedIdentifier(alias) + "(?:$|[^A-Za-z0-9_$])",
          ).test(expression));
        if (derived && !aliases.has(target)) { aliases.add(target); changed = true; }
      }
    }
    const capacityDerived = (expression: string): boolean =>
      CAPACITY_AUTHORITY.test(expression)
      || [...aliases].some((alias) => new RegExp(
        "(?:^|[^A-Za-z0-9_$])" + escapedIdentifier(alias) + "(?:$|[^A-Za-z0-9_$])",
      ).test(expression));

    const loopHeader = /\b(for|while|loop)\b\s*[({]/g;
    const loops: Array<{ start: number; end: number; bound: string }> = [];
    for (let loop = loopHeader.exec(body); loop; loop = loopHeader.exec(body)) {
      let bound = "", bodyOpen: number;
      if (loop[1] === "loop") {
        // A WGSL `loop` carries its bound in a `break` condition, so inspect a
        // bounded prefix of the body rather than a header.
        bodyOpen = body.indexOf("{", loop.index);
        bound = body.slice(bodyOpen, Math.min(body.length, bodyOpen + 300));
      } else {
        const paren = body.indexOf("(", loop.index);
        let depth = 0, cursor = paren;
        for (; cursor < body.length; cursor += 1) {
          if (body[cursor] === "(") depth += 1;
          else if (body[cursor] === ")") { depth -= 1; if (depth === 0) break; }
        }
        bound = body.slice(paren, cursor + 1);
        bodyOpen = body.indexOf("{", cursor);
      }
      if (bodyOpen < 0) continue;
      loops.push({ start: loop.index, end: matchingBrace(body, bodyOpen), bound });
    }
    for (const loop of loops) {
      if (!capacityDerived(loop.bound)) continue;
      if (FIXED_TRIP_COUNT.test(loop.bound) || LANE_PARTITIONED.test(loop.bound)) continue;
      const nested = loops.some((other) =>
        other !== loop && other.start < loop.start && other.end > loop.end);
      if (!nested && isEntryPoint) continue; // one sweep per dispatch is the intended shape
      const text = body.slice(loop.start, loop.end);
      violations.push({
        source: sourceName,
        line: sourceLine(source, open + loop.start),
        rule: nested && BINARY_SEARCH_BODY.test(text)
          ? "nested-capacity-scan" : "capacity-scan",
        excerpt: text.slice(0, 120).replace(/\s+/g, " ").trim(),
      });
    }
  }
  return violations;
}

/** Storage arrays whose words a host later binds as `INDIRECT` dispatch
 * arguments. Recognized by name shape rather than by an enumerated list,
 * because an enumerated list is exactly what let the two earlier holes in this
 * file stay open: `candidateDispatch`, `dispatchArgs`, `indirectDispatch`,
 * `publishedDispatch`, `classDispatch`, `rowDispatch`, `commitRowsDispatch`,
 * and the bare `dispatch` / `indirect` spellings all match. */
const INDIRECT_ARGS_TARGET_NAME = /dispatch|indirect|schedule|args/i;

/** A literal argument can never carry a capacity. */
const SHADER_LITERAL_ARGUMENT = /^\s*(?:[0-9]+(?:\.[0-9]*)?[uifh]?|true|false)\s*$/;

/** Reads that can only produce a value the GPU itself published this step. */
const GPU_PUBLISHED_INTRINSIC =
  /\b(?:atomic(?:Load|Add|Sub|Max|Min|And|Or|Xor|Exchange|CompareExchangeWeak)|workgroupUniformLoad|arrayLength)\s*\(/;

interface ShaderFunction {
  readonly name: string;
  readonly parameters: readonly string[];
  readonly bodyStart: number;
  readonly bodyEnd: number;
  readonly body: string;
}

function shaderFunctions(code: string): readonly ShaderFunction[] {
  const functions: ShaderFunction[] = [];
  const header = /\bfn\s+([A-Za-z_][\w]*)\s*\(/g;
  for (let match = header.exec(code); match; match = header.exec(code)) {
    const open = code.indexOf("{", match.index);
    if (open < 0) continue;
    const close = matchingBrace(code, open);
    const signature = balancedArguments(code, header.lastIndex).text;
    functions.push({
      name: match[1]!,
      parameters: splitArguments(signature)
        .map((parameter) => /^\s*(?:@\w+(?:\([^)]*\))?\s*)*([A-Za-z_]\w*)\s*:/.exec(parameter)?.[1])
        .filter((parameter): parameter is string => parameter !== undefined),
      bodyStart: open, bodyEnd: close, body: code.slice(open, close),
    });
  }
  return functions;
}

/** WGSL `select(falseValue, trueValue, predicate)`: the third operand decides
 * *whether* the count applies, never *what* it is. A live predicate around a
 * capacity is still a capacity-shaped launch on the step it fires, so the
 * predicate is removed before the value is judged. */
function withoutSelectPredicates(expression: string): string {
  let text = expression;
  for (let guard = 0; guard < 8; guard += 1) {
    const call = /\bselect\s*\(/g;
    const match = call.exec(text);
    if (!match) break;
    const { text: inner, end } = balancedArguments(text, call.lastIndex);
    const parts = splitArguments(inner);
    if (parts.length !== 3) break;
    const replaced = `(${parts[0]!},${parts[1]!})`;
    const next = text.slice(0, match.index) + replaced + text.slice(end);
    if (next === text) break;
    text = next;
  }
  return text;
}

function declaredNames(code: string, expression: RegExp): Set<string> {
  const names = new Set<string>();
  expression.lastIndex = 0;
  for (let match = expression.exec(code); match; match = expression.exec(code)) {
    names.add(match[1]!);
  }
  return names;
}

function containsIdentifier(expression: string, name: string): boolean {
  return new RegExp(
    "(?:^|[^A-Za-z0-9_])" + escapedIdentifier(name) + "(?:$|[^A-Za-z0-9_])",
  ).test(expression);
}

/** Splits a balanced argument list on its top-level commas. */
function splitArguments(text: string): string[] {
  const parts: string[] = [];
  let depth = 0, start = 0;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === "(" || character === "[" || character === "{") depth += 1;
    else if (character === ")" || character === "]" || character === "}") depth -= 1;
    else if (character === "," && depth === 0) {
      parts.push(text.slice(start, index)); start = index + 1;
    }
  }
  if (text.slice(start).trim().length > 0 || parts.length > 0) parts.push(text.slice(start));
  return parts;
}

/**
 * The gate's third measured blindness, and the reason this rule exists.
 *
 * `capacityDispatchViolations` exempts `dispatchWorkgroupsIndirect` because "an
 * indirect launch is shaped by whatever the GPU published into the args buffer,
 * which is the compliant form". That reasoning silently assumes the *publisher*
 * used a live count. It need not: `prepareCandidateSchedules` in
 * `webgpu-octree-spgrid-vcycle.ts` writes `brickItems = p.totals.y` (the host's
 * `plan.brickCount`, a pure O(domain) sum over levels) and `logicalPageItems =
 * physicalPageItems = p.totals.z` (`plan.pageDirectoryBytes / 4`) into the
 * candidate schedule, and four recurring launches consume those records. The
 * gate reported that clean.
 *
 * Two authorities make a published item count host-authored:
 *
 *  - the same `CAPACITY_AUTHORITY` vocabulary the direct rule uses, and
 *  - **any read of a uniform block**. A uniform is written by the host, and the
 *    host may not read back a live count (`hostSchedulingUsesReadback: false`),
 *    so a uniform-derived item count is an allocation-time constant by
 *    construction. This is what catches `p.totals.y`, which no capacity
 *    vocabulary contained — and enumerating it would have re-created the hole.
 *
 * Three things suppress a report, because each one means the count is not
 * host-authored at this site:
 *
 *  - a **storage read or atomic load** anywhere in the value: that is a count
 *    the GPU published this step, which is the compliant form. `min(published,
 *    capacity)` is bounded by the published count and stays clean.
 *  - a **function parameter**: a publisher helper such as
 *    `writeCandidateSchedule(record, items, lanes)` clamps whatever it is
 *    handed, so the judgement belongs at its call site, which the transitive
 *    publisher closure reaches.
 *  - a **literal**, an index, or a workgroup-size divisor: none is an item
 *    count.
 *
 * Known limit, deliberately taken: when one name is assigned a live count on
 * one path and a capacity on another, the live assignment wins and the site is
 * not reported. Reporting it would flag every `min(capacity, live)` clamp in
 * the tree. Record the limit rather than closing it by exemption list.
 */
function capacityIndirectArgumentViolations(
  sourceName: string, source: string,
): OctreeSourceViolation[] {
  const code = shaderCodeOnly(source);
  const storages = declaredNames(code, /\bvar\s*<\s*storage[^>]*>\s*([A-Za-z_]\w*)\s*:/g);
  const targets = new Set([...storages].filter((name) => INDIRECT_ARGS_TARGET_NAME.test(name)));
  if (targets.size === 0) return [];
  const uniforms = declaredNames(code, /\bvar\s*<\s*uniform\s*>\s*([A-Za-z_]\w*)\s*:/g);
  const functions = shaderFunctions(code);

  const targetNames = [...targets].map(escapedIdentifier).join("|");
  const writeExpression = new RegExp(
    "\\b(?:" + targetNames + ")\\s*\\[([^\\]]*)\\]\\s*=(?!=)\\s*([^;\\n]+)", "g");
  const atomicWriteExpression = new RegExp(
    "\\batomicStore\\s*\\(\\s*&\\s*(?:" + targetNames + ")\\s*\\[", "g");

  // A publisher writes indirect words itself, or hands its arguments to one
  // that does. The transitive closure is what makes `writeCandidateSchedule`'s
  // caller — where the capacity actually enters — the reported site.
  const publishers = new Set<string>();
  for (const shaderFunction of functions) {
    writeExpression.lastIndex = 0;
    atomicWriteExpression.lastIndex = 0;
    if (writeExpression.test(shaderFunction.body)
      || atomicWriteExpression.test(shaderFunction.body)) publishers.add(shaderFunction.name);
  }
  for (let changed = true; changed;) {
    changed = false;
    for (const shaderFunction of functions) {
      if (publishers.has(shaderFunction.name)) continue;
      for (const publisher of publishers) {
        if (new RegExp("\\b" + escapedIdentifier(publisher) + "\\s*\\(").test(shaderFunction.body)) {
          publishers.add(shaderFunction.name); changed = true; break;
        }
      }
    }
  }

  // A function is a live source when it reads storage it does not merely
  // publish into. Stripping its own assignment targets first keeps a publisher
  // helper (which only *writes* the indirect array) from claiming to produce a
  // GPU-published count and silencing all of its call sites.
  const liveFunctions = new Set<string>();
  const readsStorage = (body: string): boolean => {
    const reads = body.replace(
      new RegExp("\\b(?:" + [...storages].map(escapedIdentifier).join("|")
        + ")\\s*\\[[^\\]]*\\]\\s*=(?!=)", "g"), " ");
    return GPU_PUBLISHED_INTRINSIC.test(reads)
      || [...storages].some((name) => containsIdentifier(reads, name));
  };
  for (const shaderFunction of functions) {
    if (readsStorage(shaderFunction.body)) liveFunctions.add(shaderFunction.name);
  }
  for (let changed = true; changed;) {
    changed = false;
    for (const shaderFunction of functions) {
      if (liveFunctions.has(shaderFunction.name)) continue;
      for (const live of liveFunctions) {
        if (new RegExp("\\b" + escapedIdentifier(live) + "\\s*\\(").test(shaderFunction.body)) {
          liveFunctions.add(shaderFunction.name); changed = true; break;
        }
      }
    }
  }

  const violations: OctreeSourceViolation[] = [];
  const report = (offset: number, excerpt: string): void => {
    violations.push({
      source: sourceName,
      line: sourceLine(source, offset),
      rule: "capacity-indirect-args",
      excerpt: excerpt.slice(0, 140).replace(/\s+/g, " ").trim(),
    });
  };

  for (const shaderFunction of functions) {
    const body = shaderFunction.body;
    // Declarations and plain re-assignments both carry a capacity forward, and
    // the SPGrid publisher uses the second form: `var brickItems = 0u;` then
    // `brickItems = p.totals.y;` inside the dirty-epoch branch.
    const aliases = new Set<string>();
    const published = new Set<string>(shaderFunction.parameters);
    const assignments = [...body.matchAll(
      /(?:\b(?:let|var)\s+)?\b([A-Za-z_]\w*)\s*(?::\s*[A-Za-z_][\w<>,\s]*?)?=(?!=)\s*([^;\n]+)/g)]
      .map((match) => [match[1]!, withoutSelectPredicates(match[2]!)] as const);
    const gpuPublished = (expression: string): boolean =>
      GPU_PUBLISHED_INTRINSIC.test(expression)
      || [...storages].some((name) => containsIdentifier(expression, name))
      || [...liveFunctions].some((name) => new RegExp(
        "\\b" + escapedIdentifier(name) + "\\s*\\(").test(expression))
      || [...published].some((name) => containsIdentifier(expression, name));
    const capacityDerived = (expression: string): boolean =>
      CAPACITY_AUTHORITY.test(expression)
      || [...uniforms].some((name) => new RegExp(
        "(?:^|[^A-Za-z0-9_])" + escapedIdentifier(name) + "\\s*\\.").test(expression))
      || [...aliases].some((alias) => containsIdentifier(expression, alias));
    const hostAuthored = (raw: string): boolean => {
      const expression = withoutSelectPredicates(raw);
      return capacityDerived(expression) && !gpuPublished(expression);
    };
    for (let changed = true; changed;) {
      changed = false;
      for (const [target, expression] of assignments) {
        if (targets.has(target)) continue;
        if (!published.has(target) && gpuPublished(expression)) {
          published.add(target); aliases.delete(target); changed = true; continue;
        }
        if (!aliases.has(target) && !published.has(target) && capacityDerived(expression)) {
          aliases.add(target); changed = true;
        }
      }
    }

    writeExpression.lastIndex = 0;
    for (let match = writeExpression.exec(body); match; match = writeExpression.exec(body)) {
      if (hostAuthored(match[2]!)) report(shaderFunction.bodyStart + match.index, match[0]);
    }
    for (const publisher of publishers) {
      const call = new RegExp("\\b" + escapedIdentifier(publisher) + "\\s*\\(", "g");
      for (let match = call.exec(body); match; match = call.exec(body)) {
        const { text, end } = balancedArguments(body, match.index + match[0].length);
        call.lastIndex = end;
        for (const argument of splitArguments(text)) {
          if (SHADER_LITERAL_ARGUMENT.test(argument)) continue;
          if (!hostAuthored(argument)) continue;
          report(shaderFunction.bodyStart + match.index,
            `${publisher}(${text}) publishes host-authored ${argument.trim()}`);
          break;
        }
      }
    }
  }
  return violations;
}

const RETIRED_SHADER_IDENTIFIER =
  /\b[A-Za-z_][A-Za-z0-9_]*\b/g;

/** General variable-width geometry/operator streams forbidden by the
 * structured six-family cutover. Case-local handles and explicitly named
 * structured families do not match these declarations. */
const GENERAL_TIMESTEP_SHADER_AUTHORITY =
  /\bvar\s*<\s*storage\b[^>]*>\s*(?:faces?|powerFaces|candidateFaces|committedFaces|faceList|faceRecords?|incidences?|incidenceRows?|csr|csrRows|csrEntries|leafEntries|matrixEntries)\s*:|\bvar\s*<\s*storage\b[^>]*>\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*array\s*<\s*(?:PowerFace|FaceRecord|Incidence|LeafEntry|CsrEntry)\b/gi;

/** Removes WGSL comments without changing offsets or line numbers. */
function shaderCodeOnly(source: string): string {
  const characters = source.split("");
  let index = 0;
  while (index < characters.length) {
    if (characters[index] === "/" && characters[index + 1] === "/") {
      characters[index] = " "; characters[index + 1] = " "; index += 2;
      while (index < characters.length && characters[index] !== "\n") {
        characters[index] = " "; index += 1;
      }
      continue;
    }
    if (characters[index] === "/" && characters[index + 1] === "*") {
      characters[index] = " "; characters[index + 1] = " "; index += 2;
      while (index < characters.length) {
        if (characters[index] === "*" && characters[index + 1] === "/") {
          characters[index] = " "; characters[index + 1] = " "; index += 2; break;
        }
        if (characters[index] !== "\n") characters[index] = " ";
        index += 1;
      }
      continue;
    }
    index += 1;
  }
  return characters.join("");
}

function sourceLine(source: string, offset: number): number {
  let line = 1;
  for (let index = 0; index < offset; index += 1) {
    if (source.charCodeAt(index) === 10) line += 1;
  }
  return line;
}

/** Audits one production source unit. Deliberate exceptions must live outside
 * the production-octree scan set; inline suppressions are intentionally not
 * supported because they turn architectural violations into comments. */
export function auditOctreeProductionSource(
  sourceName: string,
  source: string,
  role: OctreeProductionSourceRole,
): readonly OctreeSourceViolation[] {
  const checks = role === "shader" ? SHADER_SOURCE_CHECKS : HOST_SOURCE_CHECKS;
  const violations: OctreeSourceViolation[] = role === "host"
    ? capacityDispatchViolations(sourceName, source) : [];
  DELETED_PRODUCTION_MODULE.lastIndex = 0;
  const deletedName = DELETED_PRODUCTION_MODULE.exec(sourceName);
  if (deletedName) {
    violations.push({
      source: sourceName,
      line: 1,
      rule: "deleted-production-module",
      excerpt: deletedName[0],
    });
  }
  for (const { rule, expression } of checks) {
    expression.lastIndex = 0;
    for (let match = expression.exec(source); match; match = expression.exec(source)) {
      violations.push({
        source: sourceName,
        line: sourceLine(source, match.index),
        rule,
        excerpt: match[0],
      });
      if (match[0].length === 0) expression.lastIndex += 1;
    }
  }
  if (role === "shader") {
    violations.push(...capacityScanViolations(sourceName, source));
    violations.push(...capacityIndirectArgumentViolations(sourceName, source));
    const code = shaderCodeOnly(source);
    RETIRED_SHADER_IDENTIFIER.lastIndex = 0;
    for (let match = RETIRED_SHADER_IDENTIFIER.exec(code); match;
      match = RETIRED_SHADER_IDENTIFIER.exec(code)) {
      if (!/(?:fallback|legacy|compatibility)/i.test(match[0])) continue;
      violations.push({
        source: sourceName,
        line: sourceLine(source, match.index),
        rule: "retired-shader-terminology",
        excerpt: match[0],
      });
    }
    GENERAL_TIMESTEP_SHADER_AUTHORITY.lastIndex = 0;
    for (let match = GENERAL_TIMESTEP_SHADER_AUTHORITY.exec(code); match;
      match = GENERAL_TIMESTEP_SHADER_AUTHORITY.exec(code)) {
      violations.push({
        source: sourceName,
        line: sourceLine(source, match.index),
        rule: "general-timestep-authority",
        excerpt: match[0],
      });
    }
  }
  return violations.sort((a, b) => a.line - b.line || a.rule.localeCompare(b.rule));
}
