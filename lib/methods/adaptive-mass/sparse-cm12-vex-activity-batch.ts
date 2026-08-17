/** Append-only resident adapter for the production A4D2 + VEX1 cutover batch. */
import {
  SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER,
  createSparseCM12ProductionActivityInitialWords,
  createSparseCM12ProductionActivityLayout,
  type SparseCM12ProductionActivityLayout,
} from "./sparse-cm12-production-activity";
import {
  SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT,
  createSparseCM12VelocityExtensionInitialWords,
  createSparseCM12VelocityExtensionResidentLayouts,
  type SparseCM12VelocityExtensionLayout,
  type SparseCM12VelocityExtensionStateLayout,
} from "./sparse-cm12-velocity-extension";

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_MAGIC = 0x5641_4231; // VAB1
export const SPARSE_CM12_VEX_ACTIVITY_BATCH_VERSION = 1;
export const SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT_WORDS = 15;

const align64 = (value: number): number => Math.ceil(value / 64) * 64;

export type SparseCM12VexActivityBatchPacket =
  "a4d2Classify" | "a4d2Scan" | "a4d2Rebuild" | "a4d2Census" | "vexSerial";

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT = Object.freeze({
  a4d2Classify: Object.freeze({ offsetWords: 0, sizeWords: 3 }),
  a4d2Scan: Object.freeze({ offsetWords: 3, sizeWords: 3 }),
  a4d2Rebuild: Object.freeze({ offsetWords: 6, sizeWords: 3 }),
  a4d2Census: Object.freeze({ offsetWords: 9, sizeWords: 3 }),
  /** Serial scratch packet: copy only after the preceding indirect dispatch. */
  vexSerial: Object.freeze({ offsetWords: 12, sizeWords: 3 }),
} as const);

export interface SparseCM12VexActivityBatchLayout {
  readonly activityBaseWords: number;
  readonly productionActivity: SparseCM12ProductionActivityLayout;
  readonly velocityExtension: SparseCM12VelocityExtensionLayout;
  readonly velocityState: SparseCM12VelocityExtensionStateLayout;
  readonly totalActivityWords: number;
  readonly totalStateFloats: number;
  readonly indirectWords: number;
}

/**
 * Appends to the final resident activity/state tails. It deliberately does not
 * reconstruct FCA/FPL/FPP/PCM/PCF layouts, so concurrent authority additions
 * before either tail cannot overlap this batch.
 */
export function createSparseCM12VexActivityBatchLayout(options: {
  readonly activityTailWords: number;
  readonly stateTailFloats: number;
  readonly brickCapacity: number;
  readonly cellCapacity: number;
  readonly brickFineResolution?: 4 | 8 | 16;
}): SparseCM12VexActivityBatchLayout {
  const activityBaseWords = align64(options.activityTailWords);
  const productionActivity = createSparseCM12ProductionActivityLayout({
    baseWords: activityBaseWords,
    brickCapacity: options.brickCapacity,
    brickFineResolution: options.brickFineResolution ?? 16,
  });
  const velocity = createSparseCM12VelocityExtensionResidentLayouts({
    activityTailWords: productionActivity.totalWords,
    stateTailFloats: options.stateTailFloats,
    cellCapacity: options.cellCapacity,
  });
  return Object.freeze({
    activityBaseWords,
    productionActivity,
    velocityExtension: velocity.activity,
    velocityState: velocity.state,
    totalActivityWords: velocity.activity.totalWords,
    totalStateFloats: velocity.state.floatCount,
    indirectWords: SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT_WORDS,
  });
}

/** Upload only this append-only slice at construction. Runtime scheduling and
 * generation publication are GPU-owned. */
export function createSparseCM12VexActivityBatchInitialWords(
  layout: SparseCM12VexActivityBatchLayout,
): Uint32Array {
  const result = new Uint32Array(layout.totalActivityWords - layout.activityBaseWords);
  result.set(createSparseCM12ProductionActivityInitialWords(layout.productionActivity));
  result.set(createSparseCM12VelocityExtensionInitialWords(layout.velocityExtension),
    layout.velocityExtension.headerBaseWords - layout.activityBaseWords);
  return result;
}

export interface SparseCM12VexActivityBatchIndirectCopy {
  readonly id: string;
  readonly packet: SparseCM12VexActivityBatchPacket;
  readonly sourceWord: number;
  readonly destinationWord: number;
  readonly words: 3;
  readonly reuseAfter?: string;
}

/** Header-to-indirect snapshots. VEX sources share one destination packet but
 * are separated by an indirect dispatch and an encoder-ordering seam. */
export function createSparseCM12VexActivityBatchIndirectCopies(
  layout: SparseCM12VexActivityBatchLayout,
): readonly SparseCM12VexActivityBatchIndirectCopy[] {
  const a = layout.productionActivity.baseWords;
  const v = layout.velocityExtension.headerBaseWords;
  const target = SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT;
  return Object.freeze([
    { id: "copy-a4d2-classify", packet: "a4d2Classify",
      sourceWord: a + SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.classifyIndirectX,
      destinationWord: target.a4d2Classify.offsetWords, words: 3 },
    { id: "copy-a4d2-scan", packet: "a4d2Scan",
      sourceWord: a + SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.scanIndirectX,
      destinationWord: target.a4d2Scan.offsetWords, words: 3 },
    { id: "copy-a4d2-rebuild", packet: "a4d2Rebuild",
      sourceWord: a + SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuildIndirectX,
      destinationWord: target.a4d2Rebuild.offsetWords, words: 3 },
    { id: "copy-a4d2-census", packet: "a4d2Census",
      sourceWord: a + SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.censusIndirectX,
      destinationWord: target.a4d2Census.offsetWords, words: 3 },
    { id: "copy-vex-roots", packet: "vexSerial",
      sourceWord: v + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchX,
      destinationWord: target.vexSerial.offsetWords, words: 3 },
    { id: "copy-vex-frontier-a", packet: "vexSerial",
      sourceWord: v + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchX,
      destinationWord: target.vexSerial.offsetWords, words: 3,
      reuseAfter: "seed-or-prior-frontier-dispatch" },
    { id: "copy-vex-frontier-b", packet: "vexSerial",
      sourceWord: v + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBDispatchX,
      destinationWord: target.vexSerial.offsetWords, words: 3,
      reuseAfter: "prior-frontier-dispatch" },
    { id: "copy-vex-blast", packet: "vexSerial",
      sourceWord: v + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastDispatchX,
      destinationWord: target.vexSerial.offsetWords, words: 3,
      reuseAfter: "final-frontier-dispatch" },
  ] satisfies SparseCM12VexActivityBatchIndirectCopy[]);
}

export type SparseCM12VexActivityDispatch =
  | Readonly<{ kind: "fixed"; x: number; y: 1; z: 1 }>
  | Readonly<{ kind: "batch-indirect"; packet: SparseCM12VexActivityBatchPacket }>
  | Readonly<{ kind: "construction-authority-indirect" }>;

export interface SparseCM12VexActivityPipelineDescriptor {
  readonly key: string;
  readonly entryPoint: string;
  readonly dispatch: SparseCM12VexActivityDispatch;
  readonly constants?: Readonly<Record<string, number>>;
  readonly constructionOnly?: true;
  readonly runtimeSelectable?: false;
  readonly outputSelectable?: false;
}

const fixed = (x = 1): SparseCM12VexActivityDispatch =>
  Object.freeze({ kind: "fixed", x, y: 1, z: 1 });
const packet = (value: SparseCM12VexActivityBatchPacket): SparseCM12VexActivityDispatch =>
  Object.freeze({ kind: "batch-indirect", packet: value });

/** Descriptors are consumed by GPUCompilationManager; the adapter never calls
 * createComputePipeline or createComputePipelineAsync directly. */
export function createSparseCM12VexActivityBatchPipelineDescriptors():
readonly SparseCM12VexActivityPipelineDescriptor[] {
  const descriptors: SparseCM12VexActivityPipelineDescriptor[] = [
    { key: "beginProductionActivity", entryPoint: "beginProductionActivity", dispatch: fixed() },
    { key: "buildProductionActivityTriggers", entryPoint: "buildProductionActivityTriggers",
      dispatch: packet("a4d2Classify") },
    { key: "classifyProductionActivityBricks", entryPoint: "classifyProductionActivityBricks",
      dispatch: packet("a4d2Classify") },
    { key: "scanProductionActivityBrickBlocks", entryPoint: "scanProductionActivityBrickBlocks",
      dispatch: packet("a4d2Scan") },
    { key: "scanProductionActivityBlockSums", entryPoint: "scanProductionActivityBlockSums",
      dispatch: fixed() },
    { key: "scatterProductionActivityBricks", entryPoint: "scatterProductionActivityBricks",
      dispatch: packet("a4d2Scan") },
    { key: "rebuildProductionActivityBricks", entryPoint: "rebuildProductionActivityBricks",
      dispatch: packet("a4d2Rebuild") },
    { key: "reduceProductionActivityCensusBlocks",
      entryPoint: "reduceProductionActivityCensusBlocks", dispatch: packet("a4d2Census") },
    { key: "commitProductionActivityCensus", entryPoint: "commitProductionActivityCensus",
      dispatch: fixed() },
    { key: "acceptProductionActivity", entryPoint: "acceptProductionActivity", dispatch: fixed() },
    { key: "beginVelocityExtensionCandidate", entryPoint: "beginVelocityExtensionCandidate",
      dispatch: fixed() },
    { key: "reopenVelocityExtensionPlanForInjection",
      entryPoint: "reopenVelocityExtensionPlanForInjection", dispatch: fixed() },
    { key: "sealVelocityExtensionRoots", entryPoint: "sealVelocityExtensionRoots",
      dispatch: fixed() },
    { key: "seedVelocityExtensionRoots", entryPoint: "seedVelocityExtensionRoots",
      dispatch: packet("vexSerial") },
    { key: "sealVelocityExtensionSeedFrontier", entryPoint: "sealVelocityExtensionSeedFrontier",
      dispatch: fixed() },
    { key: "finalizeVelocityExtensionBlast", entryPoint: "finalizeVelocityExtensionBlast",
      dispatch: fixed() },
    { key: "beginVelocityExtensionExecution", entryPoint: "beginVelocityExtensionExecution",
      dispatch: fixed() },
    { key: "initializeVelocityExtensionCandidates",
      entryPoint: "initializeVelocityExtensionCandidates", dispatch: packet("vexSerial") },
    { key: "commitVelocityExtensionCandidates", entryPoint: "commitVelocityExtensionCandidates",
      dispatch: packet("vexSerial") },
    { key: "finalizeVelocityExtensionCandidate", entryPoint: "finalizeVelocityExtensionCandidate",
      dispatch: fixed() },
    { key: "bootstrapVelocityExtensionRoots", entryPoint: "bootstrapVelocityExtensionRoots",
      dispatch: Object.freeze({ kind: "construction-authority-indirect" }),
      constructionOnly: true, runtimeSelectable: false, outputSelectable: false },
  ];
  for (let depth = 1; depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH; depth += 1) {
    descriptors.push(
      { key: `prepareVelocityExtensionFrontier${depth}`,
        entryPoint: "prepareVelocityExtensionFrontier", dispatch: fixed(),
        constants: Object.freeze({ EXTENSION_FRONTIER_DEPTH: depth }) },
      { key: `expandVelocityExtensionFrontier${depth}`,
        entryPoint: "expandVelocityExtensionFrontier", dispatch: packet("vexSerial"),
        constants: Object.freeze({ EXTENSION_FRONTIER_DEPTH: depth }) },
      { key: `sealVelocityExtensionFrontier${depth}`,
        entryPoint: "sealVelocityExtensionFrontier", dispatch: fixed(),
        constants: Object.freeze({ EXTENSION_FRONTIER_DEPTH: depth }) },
      { key: `advanceVelocityExtensionCandidates${depth}`,
        entryPoint: "advanceVelocityExtensionCandidates", dispatch: packet("vexSerial"),
        constants: Object.freeze({ EXTENSION_RECURRENCE_DEPTH: depth }) },
    );
  }
  return Object.freeze(descriptors);
}

export interface SparseCM12VexActivityBatchScheduleStep {
  readonly id: string;
  readonly phase: "velocity-extension" | "activity-reduction" | "receipt";
  readonly operation: "pipeline" | "copy-indirect" | "producer-seam" | "acceptance-seam";
  readonly pipeline?: string;
  readonly copy?: string;
  readonly after: readonly string[];
  readonly constructionOnly?: true;
}

/** Construction-only publication. The resident encodes this from
 * encodeInitialPresentation, never from the recurring frame schedule. */
export function createSparseCM12VexActivityBatchConstructionSchedule():
readonly SparseCM12VexActivityBatchScheduleStep[] {
  const steps: SparseCM12VexActivityBatchScheduleStep[] = [
    { id: "vex-construction-bootstrap", phase: "velocity-extension",
      operation: "pipeline", pipeline: "bootstrapVelocityExtensionRoots",
      after: [], constructionOnly: true },
    { id: "vex-construction-begin-plan", phase: "velocity-extension",
      operation: "pipeline", pipeline: "beginVelocityExtensionCandidate",
      after: ["vex-construction-bootstrap"], constructionOnly: true },
    { id: "vex-construction-seal-roots", phase: "velocity-extension",
      operation: "pipeline", pipeline: "sealVelocityExtensionRoots",
      after: ["vex-construction-begin-plan"], constructionOnly: true },
    { id: "vex-construction-copy-roots", phase: "velocity-extension",
      operation: "copy-indirect", copy: "copy-vex-roots",
      after: ["vex-construction-seal-roots"], constructionOnly: true },
    { id: "vex-construction-seed", phase: "velocity-extension",
      operation: "pipeline", pipeline: "seedVelocityExtensionRoots",
      after: ["vex-construction-copy-roots"], constructionOnly: true },
    { id: "vex-construction-seal-seed", phase: "velocity-extension",
      operation: "pipeline", pipeline: "sealVelocityExtensionSeedFrontier",
      after: ["vex-construction-seed"], constructionOnly: true },
  ];
  let prior = "vex-construction-seal-seed";
  for (let depth = 1; depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH; depth += 1) {
    const prepare = `vex-construction-prepare-${depth}`;
    const copy = `vex-construction-copy-frontier-${depth}`;
    const expand = `vex-construction-expand-${depth}`;
    const seal = `vex-construction-seal-frontier-${depth}`;
    steps.push(
      { id: prepare, phase: "velocity-extension", operation: "pipeline",
        pipeline: `prepareVelocityExtensionFrontier${depth}`, after: [prior],
        constructionOnly: true },
      { id: copy, phase: "velocity-extension", operation: "copy-indirect",
        copy: depth % 2 === 1 ? "copy-vex-frontier-a" : "copy-vex-frontier-b",
        after: [prepare], constructionOnly: true },
      { id: expand, phase: "velocity-extension", operation: "pipeline",
        pipeline: `expandVelocityExtensionFrontier${depth}`, after: [copy],
        constructionOnly: true },
      { id: seal, phase: "velocity-extension", operation: "pipeline",
        pipeline: `sealVelocityExtensionFrontier${depth}`, after: [expand],
        constructionOnly: true },
    );
    prior = seal;
  }
  steps.push(
    { id: "vex-construction-finalize-blast", phase: "velocity-extension",
      operation: "pipeline", pipeline: "finalizeVelocityExtensionBlast",
      after: [prior], constructionOnly: true },
    { id: "vex-construction-copy-blast", phase: "velocity-extension",
      operation: "copy-indirect", copy: "copy-vex-blast",
      after: ["vex-construction-finalize-blast"], constructionOnly: true },
  );
  return Object.freeze(steps);
}

/** Mechanical encoder order. Every indirect count is copied GPU-to-GPU after
 * its producer kernel; neither frame parity nor a work count reaches the CPU. */
export function createSparseCM12VexActivityBatchSchedule():
readonly SparseCM12VexActivityBatchScheduleStep[] {
  const steps: SparseCM12VexActivityBatchScheduleStep[] = [];
  const push = (step: SparseCM12VexActivityBatchScheduleStep) => { steps.push(step); };
  push({ id: "vex-begin-execution", phase: "velocity-extension", operation: "pipeline",
    pipeline: "beginVelocityExtensionExecution", after: [] });
  push({ id: "vex-initialize-candidates", phase: "velocity-extension", operation: "pipeline",
    pipeline: "initializeVelocityExtensionCandidates", after: ["vex-begin-execution"] });
  let prior = "vex-initialize-candidates";
  for (let depth = 1; depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH; depth += 1) {
    const id = `vex-advance-${depth}`;
    push({ id, phase: "velocity-extension", operation: "pipeline",
      pipeline: `advanceVelocityExtensionCandidates${depth}`, after: [prior] });
    prior = id;
  }
  push({ id: "vex-commit", phase: "velocity-extension", operation: "pipeline",
    pipeline: "commitVelocityExtensionCandidates", after: [prior] });
  push({ id: "vex-accept", phase: "velocity-extension", operation: "pipeline",
    pipeline: "finalizeVelocityExtensionCandidate", after: ["vex-commit"] });
  push({ id: "face-stage-receipt", phase: "receipt", operation: "acceptance-seam",
    after: ["vex-accept"] });

  push({ id: "a4d2-begin", phase: "activity-reduction", operation: "pipeline",
    pipeline: "beginProductionActivity", after: ["face-stage-receipt"] });
  push({ id: "a4d2-copy-classify", phase: "activity-reduction",
    operation: "copy-indirect", copy: "copy-a4d2-classify", after: ["a4d2-begin"] });
  push({ id: "a4d2-copy-scan", phase: "activity-reduction",
    operation: "copy-indirect", copy: "copy-a4d2-scan", after: ["a4d2-begin"] });
  push({ id: "a4d2-build-triggers", phase: "activity-reduction", operation: "pipeline",
    pipeline: "buildProductionActivityTriggers", after: ["a4d2-copy-classify"] });
  push({ id: "a4d2-classify", phase: "activity-reduction", operation: "pipeline",
    pipeline: "classifyProductionActivityBricks", after: ["a4d2-build-triggers"] });
  push({ id: "a4d2-scan-bricks", phase: "activity-reduction", operation: "pipeline",
    pipeline: "scanProductionActivityBrickBlocks", after: ["a4d2-classify", "a4d2-copy-scan"] });
  push({ id: "a4d2-scan-blocks", phase: "activity-reduction", operation: "pipeline",
    pipeline: "scanProductionActivityBlockSums", after: ["a4d2-scan-bricks"] });
  push({ id: "a4d2-copy-rebuild", phase: "activity-reduction",
    operation: "copy-indirect", copy: "copy-a4d2-rebuild", after: ["a4d2-scan-blocks"] });
  push({ id: "a4d2-copy-census", phase: "activity-reduction",
    operation: "copy-indirect", copy: "copy-a4d2-census", after: ["a4d2-scan-blocks"] });
  push({ id: "a4d2-scatter", phase: "activity-reduction", operation: "pipeline",
    pipeline: "scatterProductionActivityBricks", after: ["a4d2-scan-blocks"] });
  push({ id: "a4d2-rebuild", phase: "activity-reduction", operation: "pipeline",
    pipeline: "rebuildProductionActivityBricks",
    after: ["a4d2-scatter", "a4d2-copy-rebuild"] });
  push({ id: "a4d2-reduce-census", phase: "activity-reduction", operation: "pipeline",
    pipeline: "reduceProductionActivityCensusBlocks",
    after: ["a4d2-rebuild", "a4d2-copy-census"] });
  push({ id: "a4d2-commit-census", phase: "activity-reduction", operation: "pipeline",
    pipeline: "commitProductionActivityCensus", after: ["a4d2-reduce-census"] });
  push({ id: "a4d2-accept", phase: "activity-reduction", operation: "pipeline",
    pipeline: "acceptProductionActivity", after: ["a4d2-commit-census"] });
  push({ id: "vex-producer-roots", phase: "velocity-extension", operation: "producer-seam",
    after: ["a4d2-accept"] });
  push({ id: "vex-begin-next-plan", phase: "velocity-extension", operation: "pipeline",
    pipeline: "beginVelocityExtensionCandidate", after: ["vex-producer-roots"] });
  push({ id: "vex-seal-roots", phase: "velocity-extension", operation: "pipeline",
    pipeline: "sealVelocityExtensionRoots", after: ["vex-begin-next-plan"] });
  push({ id: "vex-copy-roots", phase: "velocity-extension", operation: "copy-indirect",
    copy: "copy-vex-roots", after: ["vex-seal-roots"] });
  push({ id: "vex-seed", phase: "velocity-extension", operation: "pipeline",
    pipeline: "seedVelocityExtensionRoots", after: ["vex-copy-roots"] });
  push({ id: "vex-seal-seed", phase: "velocity-extension", operation: "pipeline",
    pipeline: "sealVelocityExtensionSeedFrontier", after: ["vex-seed"] });
  prior = "vex-seal-seed";
  for (let depth = 1; depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH; depth += 1) {
    const prepare = `vex-prepare-frontier-${depth}`;
    const copyId = `vex-copy-frontier-${depth}`;
    const expand = `vex-expand-frontier-${depth}`;
    const seal = `vex-seal-frontier-${depth}`;
    push({ id: prepare, phase: "velocity-extension", operation: "pipeline",
      pipeline: `prepareVelocityExtensionFrontier${depth}`, after: [prior] });
    push({ id: copyId, phase: "velocity-extension", operation: "copy-indirect",
      copy: depth % 2 === 1 ? "copy-vex-frontier-a" : "copy-vex-frontier-b",
      after: [prepare] });
    push({ id: expand, phase: "velocity-extension", operation: "pipeline",
      pipeline: `expandVelocityExtensionFrontier${depth}`, after: [copyId] });
    push({ id: seal, phase: "velocity-extension", operation: "pipeline",
      pipeline: `sealVelocityExtensionFrontier${depth}`, after: [expand] });
    prior = seal;
  }
  push({ id: "vex-finalize-next-blast", phase: "velocity-extension", operation: "pipeline",
    pipeline: "finalizeVelocityExtensionBlast", after: [prior] });
  push({ id: "vex-copy-next-blast", phase: "velocity-extension", operation: "copy-indirect",
    copy: "copy-vex-blast", after: ["vex-finalize-next-blast"] });
  push({ id: "fpl-activity-receipt", phase: "receipt", operation: "acceptance-seam",
    after: ["vex-copy-next-blast"] });
  return Object.freeze(steps);
}

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_PRODUCER_HOOKS = Object.freeze({
  vexRoots: SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT,
  activity: Object.freeze([
    "cm12ActivityCandidateGeneration", "cm12ActivityCandidateBrickCount",
    "cm12ActivityCandidateListGeneration", "cm12ActivityCandidateBrickInvocation",
    "cm12ActivityTopologyGeneration", "cm12ActivityFramePlanGeneration",
    "cm12ActivityBrickTopologySignature", "cm12ActivityBrickTopologyChanged",
    "cm12ActivityBuildTileTrigger", "cm12ActivityRebuildExactTile",
    "cm12ActivityExpectedTileContributionCount", "cm12ActivityExpectedTileCheck",
    "cm12ActivityPublishFramePlanRoot", "cm12ActivityPublishExactBrick",
  ]),
  provenance: Object.freeze([
    "cm12BatchVelocityExtensionRoot", "cm12BatchVelocityExtensionClosure",
    "cm12BatchVelocityExtensionScheduled", "cm12BatchVelocityExtensionOwner",
    "cm12BatchVelocityExtensionFault",
  ]),
} as const);

export interface SparseCM12BatchGridReceipt {
  readonly stage: 0 | 1 | 2 | 3 | 4 | 5;
  readonly owner: "VEX1" | "SAW1" | "PCF1" | "FPP1";
  readonly requirement: string;
  readonly direct: "required";
  readonly closure: "required";
  readonly disposition: "executed-or-skipped";
  readonly generation: "accepted-frame";
  readonly absent: "unknown-magenta";
}

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_REQUIRED_FIELDS = Object.freeze([
  "directMaskLow", "directMaskHigh", "closureMaskLow", "closureMaskHigh",
  "executedMaskLow", "executedMaskHigh", "skippedMaskLow", "skippedMaskHigh",
  "causeMask", "maximumClosureDepth", "producerGeneration", "consumerGeneration",
  "executedCount", "skippedCount", "unknownCount", "uncoveredWriteCount", "faultCount",
] as const);

/** FramePlan must not be exposed until every entry is present for one accepted
 * generation. This prevents a partially cut-over overlay from reporting reuse. */
export const SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_RECEIPTS = Object.freeze([
  { stage: 0, owner: "VEX1",
    requirement: "direct roots + closure depth + scheduled/executed or skipped",
    direct: "required", closure: "required", disposition: "executed-or-skipped",
    generation: "accepted-frame", absent: "unknown-magenta" },
  { stage: 1, owner: "SAW1", requirement: "mass direct/closure + executed/skipped",
    direct: "required", closure: "required", disposition: "executed-or-skipped",
    generation: "accepted-frame", absent: "unknown-magenta" },
  { stage: 2, owner: "SAW1", requirement: "gamma direct/closure + executed/skipped",
    direct: "required", closure: "required", disposition: "executed-or-skipped",
    generation: "accepted-frame", absent: "unknown-magenta" },
  { stage: 3, owner: "SAW1", requirement: "surface direct/closure + executed/skipped",
    direct: "required", closure: "required", disposition: "executed-or-skipped",
    generation: "accepted-frame", absent: "unknown-magenta" },
  { stage: 4, owner: "PCF1", requirement: "coefficient direct/closure + executed/skipped",
    direct: "required", closure: "required", disposition: "executed-or-skipped",
    generation: "accepted-frame", absent: "unknown-magenta" },
  { stage: 5, owner: "FPP1", requirement: "page direct/closure + published/skipped",
    direct: "required", closure: "required", disposition: "executed-or-skipped",
    generation: "accepted-frame", absent: "unknown-magenta" },
] satisfies readonly SparseCM12BatchGridReceipt[]);

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_QA = Object.freeze([
  { id: "legacy-eight-sweep-extension", constructionOnly: true,
    runtimeSelectable: false, outputSelectable: false },
  { id: "legacy-full-activity-measurement", constructionOnly: true,
    runtimeSelectable: false, outputSelectable: false },
] as const);

/** Existing resident entry points acquired only when constructing a distinct QA
 * solver. Production never constructs or branches to these descriptors. */
export function createSparseCM12VexActivityBatchQAOraclePipelineDescriptors():
readonly SparseCM12VexActivityPipelineDescriptor[] {
  const qaDispatch = Object.freeze({ kind: "construction-authority-indirect" } as const);
  return Object.freeze([
    { key: "qaExtrapolateTransportVelocityToSource",
      entryPoint: "extrapolateTransportVelocityToSource", dispatch: qaDispatch,
      constructionOnly: true, runtimeSelectable: false, outputSelectable: false },
    { key: "qaExtrapolateTransportVelocityToDestination",
      entryPoint: "extrapolateTransportVelocityToDestination", dispatch: qaDispatch,
      constructionOnly: true, runtimeSelectable: false, outputSelectable: false },
    { key: "qaMeasureBrickActivity", entryPoint: "measureBrickActivity",
      dispatch: qaDispatch, constructionOnly: true,
      runtimeSelectable: false, outputSelectable: false },
    { key: "qaFinalizeVelocityExtensionClock",
      entryPoint: "finalizeLegacyVelocityExtensionClockForQA", dispatch: fixed(),
      constructionOnly: true, runtimeSelectable: false, outputSelectable: false },
  ]);
}

export const SPARSE_CM12_VEX_ACTIVITY_BATCH_QA_RECEIPT = Object.freeze({
  steps: 5,
  requiredExactFields: Object.freeze([
    "density", "velocity", "pressure", "divergence", "gamma", "internalTransport",
    "presentationPayload",
  ]),
  diagnosticFields: Object.freeze(["topology", "worksets"]),
  provenanceFields: Object.freeze([
    "vexAcceptedGeneration", "vexFaultCount", "vexUncoveredWriteCount",
    "a4d2AcceptedGeneration", "a4d2FaultCount", "a4d2UncoveredWriteCount",
    "fplAcceptedGeneration", "fplUnknownCount", "firstMismatchStep", "firstMismatchField",
  ]),
} as const);
