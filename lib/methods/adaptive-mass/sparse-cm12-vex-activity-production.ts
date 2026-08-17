/** Static production composition contract for FCA/FPL/FPP + SAW-B + A4D2/VEX1. */
import { createSparseCM12FramePlanLayout } from "../../core/sparse-cm12-frame-plan";
import { createSparseCM12FrameControl } from "./sparse-cm12-frame-control";
import { createSparseCM12FramePlanPresentationLayout } from
  "./sparse-cm12-frame-plan-presentation";
import { createSparseCM12ProductionActivityLayout } from
  "./sparse-cm12-production-activity";
import { createSparseCM12ScalarWorkAuthority } from
  "./sparse-cm12-scalar-work-authority";
import { createSparseCM12VelocityExtensionResidentLayouts } from
  "./sparse-cm12-velocity-extension";
import { createSparseCM12ResidentScalarAuthorityLayout } from
  "./webgpu-sparse-cm12-scalar-authority";

export const SPARSE_CM12_VEX_ACTIVITY_PRODUCTION_VERSION = 1;

export type SparseCM12ProductionArena =
  "activity" | "state" | "topology" | "saw" | "indirect";

export interface SparseCM12ProductionRegion {
  readonly name: string;
  readonly arena: SparseCM12ProductionArena;
  readonly buffer: string;
  readonly start: number;
  readonly end: number;
  readonly unit: "u32" | "f32";
}

export type SparseCM12DispatchAuthority =
  "fixed-one" | "local-brick-packet" | "local-tile-packet"
  | "local-cell-packet" | "local-row-packet" | "dedicated-indirect";

export type SparseCM12LifecyclePhase =
  "accepted" | "collecting" | "classified" | "sealed" | "executing"
  | "receipted" | "committed";

export interface SparseCM12ProductionStage {
  readonly id: string;
  readonly authority: "FCA1" | "FPL1" | "FPP1" | "SAW1" | "A4D2" | "VEX1"
  | "PCF1";
  readonly phase: SparseCM12LifecyclePhase;
  readonly dispatch: SparseCM12DispatchAuthority;
  readonly after: readonly string[];
  readonly coalescingGroup?: string;
  readonly barrierAfter?: boolean;
  readonly fplStage?: number;
  readonly receipt: "none" | "root" | "direct" | "closure" | "executed" | "skipped";
}

export interface SparseCM12ProductionQAPath {
  readonly id: string;
  readonly constructionOnly: true;
  readonly runtimeSelectable: false;
  readonly outputSelectable: false;
}

export interface SparseCM12VexActivityProductionManifest {
  readonly version: number;
  readonly profile: "B16/P16";
  readonly requiredVexRoots: readonly string[];
  readonly requiredActivityHooks: readonly string[];
  readonly requiredVexHooks: readonly string[];
  readonly stages: readonly SparseCM12ProductionStage[];
  readonly qaPaths: readonly SparseCM12ProductionQAPath[];
  readonly forbiddenDispatchTokens: readonly string[];
}

export const SPARSE_CM12_VEX_ACTIVITY_PRODUCTION_MANIFEST:
SparseCM12VexActivityProductionManifest = Object.freeze({
  version: SPARSE_CM12_VEX_ACTIVITY_PRODUCTION_VERSION,
  profile: "B16/P16",
  requiredVexRoots: Object.freeze([
    "construction bootstrap", "final projected/collocated velocity",
    "liquid classification or density authority",
    "topology or extrapolation-edge ownership", "liquid injection",
    "moving-solid velocity/open-fraction update",
  ]),
  requiredActivityHooks: Object.freeze([
    "cm12ActivityCandidateGeneration", "cm12ActivityCandidateBrickCount",
    "cm12ActivityCandidateListGeneration", "cm12ActivityCandidateBrickInvocation",
    "cm12ActivityTopologyGeneration", "cm12ActivityFramePlanGeneration",
    "cm12ActivityBrickTopologySignature", "cm12ActivityBrickTopologyChanged",
    "cm12ActivityBuildTileTrigger", "cm12ActivityRebuildExactTile",
    "cm12ActivityExpectedTileContributionCount", "cm12ActivityExpectedTileCheck",
    "cm12ActivityPublishFramePlanRoot", "cm12ActivityPublishExactBrick",
  ]),
  requiredVexHooks: Object.freeze([
    "VelocityExtensionRoot", "VelocityExtensionClosure",
    "VelocityExtensionScheduled", "VelocityExtensionOwner",
    "VelocityExtensionFault",
  ]),
  forbiddenDispatchTokens: Object.freeze([
    "acceptedTemplateCellInvocation", "acceptedTemplateRowInvocation",
    "dispatchAcceptedCells", "dispatchAcceptedRows", "globalAcceptedCells",
    "globalAcceptedRows", "pressureRowInvocation",
  ]),
  stages: Object.freeze([
    { id: "fca-seal", authority: "FCA1", phase: "sealed", dispatch: "fixed-one",
      after: [], receipt: "none", barrierAfter: true },
    { id: "vex-seal-roots", authority: "VEX1", phase: "sealed",
      dispatch: "fixed-one", after: ["fca-seal"], receipt: "direct", fplStage: 0,
      barrierAfter: true },
    { id: "vex-seed", authority: "VEX1", phase: "executing",
      dispatch: "local-cell-packet", after: ["vex-seal-roots"],
      coalescingGroup: "vex-recurrence", receipt: "root", fplStage: 0 },
    { id: "vex-eight-edge-recurrence", authority: "VEX1", phase: "executing",
      dispatch: "local-cell-packet", after: ["vex-seed"],
      coalescingGroup: "vex-recurrence", receipt: "closure", fplStage: 0,
      barrierAfter: true },
    { id: "face-stage-receipt", authority: "FPL1", phase: "collecting",
      dispatch: "local-cell-packet", after: ["vex-eight-edge-recurrence"],
      receipt: "executed", fplStage: 0, barrierAfter: true },
    { id: "saw-copy-option-b", authority: "SAW1", phase: "collecting",
      dispatch: "local-tile-packet", after: ["face-stage-receipt"],
      coalescingGroup: "saw-plan", receipt: "none" },
    { id: "saw-plan-option-b", authority: "SAW1", phase: "classified",
      dispatch: "local-tile-packet", after: ["saw-copy-option-b"],
      coalescingGroup: "saw-plan", receipt: "root", barrierAfter: true },
    { id: "saw-mass-execution", authority: "SAW1", phase: "executing",
      dispatch: "dedicated-indirect", after: ["saw-plan-option-b"],
      receipt: "executed", fplStage: 1, barrierAfter: true },
    { id: "saw-gamma-execution", authority: "SAW1", phase: "executing",
      dispatch: "dedicated-indirect", after: ["saw-mass-execution"],
      receipt: "executed", fplStage: 2, barrierAfter: true },
    { id: "saw-surface-execution", authority: "SAW1", phase: "executing",
      dispatch: "dedicated-indirect", after: ["saw-gamma-execution"],
      receipt: "executed", fplStage: 3, barrierAfter: true },
    { id: "pcf-pressure-coefficient-receipt", authority: "PCF1", phase: "receipted",
      dispatch: "local-row-packet", after: ["saw-surface-execution"],
      receipt: "executed", fplStage: 4, barrierAfter: true },
    { id: "a4d2-trigger-gather", authority: "A4D2", phase: "collecting",
      dispatch: "local-brick-packet", after: ["pcf-pressure-coefficient-receipt"],
      coalescingGroup: "a4d2-plan", receipt: "root" },
    { id: "a4d2-classify-compact", authority: "A4D2", phase: "classified",
      dispatch: "local-brick-packet", after: ["a4d2-trigger-gather"],
      coalescingGroup: "a4d2-plan", receipt: "none", barrierAfter: true },
    { id: "a4d2-local-rebuild", authority: "A4D2", phase: "executing",
      dispatch: "dedicated-indirect", after: ["a4d2-classify-compact"],
      receipt: "executed", barrierAfter: true },
    { id: "a4d2-census-accept", authority: "A4D2", phase: "committed",
      dispatch: "dedicated-indirect", after: ["a4d2-local-rebuild"],
      receipt: "executed", barrierAfter: true },
    { id: "fpl-seal", authority: "FPL1", phase: "sealed", dispatch: "fixed-one",
      after: ["a4d2-census-accept"], receipt: "none", barrierAfter: true },
    { id: "fpp-page-execution", authority: "FPP1", phase: "executing",
      dispatch: "dedicated-indirect", after: ["fpl-seal"], receipt: "executed",
      fplStage: 5, barrierAfter: true },
    { id: "fpp-commit", authority: "FPP1", phase: "committed",
      dispatch: "fixed-one", after: ["fpp-page-execution"], receipt: "executed",
      fplStage: 5 },
  ] satisfies readonly SparseCM12ProductionStage[]),
  qaPaths: Object.freeze([
    { id: "legacy-eight-sweep-extension", constructionOnly: true,
      runtimeSelectable: false, outputSelectable: false },
    { id: "legacy-full-activity-measurement", constructionOnly: true,
      runtimeSelectable: false, outputSelectable: false },
    { id: "scalar-full-path-oracle", constructionOnly: true,
      runtimeSelectable: false, outputSelectable: false },
  ] satisfies readonly SparseCM12ProductionQAPath[]),
});

export interface SparseCM12VexActivityProductionComposition {
  readonly activity: {
    readonly framePlan: ReturnType<typeof createSparseCM12FramePlanLayout>;
    readonly presentation: ReturnType<typeof createSparseCM12FramePlanPresentationLayout>;
    readonly scalarCandidate: ReturnType<typeof createSparseCM12ResidentScalarAuthorityLayout>;
    readonly productionActivity: ReturnType<typeof createSparseCM12ProductionActivityLayout>;
    readonly velocityExtension: ReturnType<typeof createSparseCM12VelocityExtensionResidentLayouts>["activity"];
    readonly totalWords: number;
  };
  readonly state: ReturnType<typeof createSparseCM12VelocityExtensionResidentLayouts>["state"];
  readonly topology: ReturnType<typeof createSparseCM12FrameControl>["layout"];
  readonly scalarAuthority: ReturnType<typeof createSparseCM12ScalarWorkAuthority>["layout"];
  readonly indirect: Readonly<Record<string, Readonly<{ startWords: number; words: number }>>>;
  readonly regions: readonly SparseCM12ProductionRegion[];
}

const align64 = (value: number): number => Math.ceil(value / 64) * 64;

/** Compose after immutable resident prefixes. Prefixes include every currently
 * accepted allocation, including PCM and the pressure journal state tail. */
export function createSparseCM12VexActivityProductionComposition(options: {
  readonly activityPrefixWords: number;
  readonly statePrefixFloats: number;
  readonly topologyPrefixWords: number;
  readonly brickCapacity: number;
  readonly cellCapacity: number;
  readonly tileCapacity?: number;
  readonly cellWorkgroups: number;
  readonly rowWorkgroups: number;
  readonly bodyCapacity?: number;
}): SparseCM12VexActivityProductionComposition {
  const tileCapacity = options.tileCapacity ?? 64 * options.brickCapacity;
  if (tileCapacity !== 64 * options.brickCapacity) {
    throw new Error("production composition requires B16's 64 tiles per brick");
  }
  const framePlan = createSparseCM12FramePlanLayout({
    baseWords: align64(options.activityPrefixWords), brickCapacity: options.brickCapacity,
    brickFineResolution: 16, packetCount: 6,
  });
  const presentation = createSparseCM12FramePlanPresentationLayout({
    baseWords: framePlan.totalWords, pageCapacity: options.brickCapacity,
    brickFineResolution: 16, pageResolution: 16, packetIndex: 5,
  });
  const scalarCandidate = createSparseCM12ResidentScalarAuthorityLayout({
    baseWords: presentation.totalWords, tileCapacity,
  });
  const productionActivity = createSparseCM12ProductionActivityLayout({
    baseWords: align64(scalarCandidate.totalWords), brickCapacity: options.brickCapacity,
    brickFineResolution: 16,
  });
  const velocity = createSparseCM12VelocityExtensionResidentLayouts({
    activityTailWords: productionActivity.totalWords,
    stateTailFloats: options.statePrefixFloats,
    cellCapacity: options.cellCapacity,
  });
  const frameControl = createSparseCM12FrameControl({
    baseWords: align64(options.topologyPrefixWords), cellWorkgroups: options.cellWorkgroups,
    rowWorkgroups: options.rowWorkgroups, bodyCapacity: options.bodyCapacity ?? 0,
    d4Capable: true, rigidCapable: (options.bodyCapacity ?? 0) > 0,
    boundaryCapable: true,
  });
  const scalarAuthority = createSparseCM12ScalarWorkAuthority({ tileCapacity,
    brickFineResolution: 16, presentationPageResolution: 16 }).layout;
  const indirect = Object.freeze({
    fca: Object.freeze({ startWords: 0, words: 42 }),
    fpl: Object.freeze({ startWords: 42, words: 18 }),
    fpp: Object.freeze({ startWords: 60, words: 3 }),
    sawStages: Object.freeze({ startWords: 63, words: 9 }),
    a4d2: Object.freeze({ startWords: 72, words: 12 }),
    // One serial triplet is copied only after the previous indirect dispatch
    // has completed: root, frontier A/B, and blast never overlap in flight.
    vexSerial: Object.freeze({ startWords: 84, words: 3 }),
  });
  const regions: SparseCM12ProductionRegion[] = [
    { name: "FPL1", arena: "activity", buffer: "activity", start: framePlan.baseWords,
      end: framePlan.totalWords, unit: "u32" },
    { name: "FPP1", arena: "activity", buffer: "activity", start: presentation.baseWords,
      end: presentation.totalWords, unit: "u32" },
    { name: "SCA1 option-B receipts", arena: "activity", buffer: "activity",
      start: scalarCandidate.candidateBaseWords, end: scalarCandidate.totalWords, unit: "u32" },
    { name: "A4D2", arena: "activity", buffer: "activity", start: productionActivity.baseWords,
      end: productionActivity.totalWords, unit: "u32" },
    { name: "VEX1 work", arena: "activity", buffer: "activity",
      start: velocity.activity.headerBaseWords, end: velocity.activity.totalWords, unit: "u32" },
    { name: "VEX1 accepted velocity", arena: "state", buffer: "state",
      start: velocity.state.acceptedVelocityFloatBase, end: velocity.state.floatCount, unit: "f32" },
    { name: "FCA1", arena: "topology", buffer: "topologyArena",
      start: frameControl.layout.baseWords, end: frameControl.layout.totalWords, unit: "u32" },
    { name: "SAW1 option-B planner", arena: "saw", buffer: "scalarAuthority",
      start: 0, end: scalarAuthority.totalWords, unit: "u32" },
    ...Object.entries(indirect).map(([name, range]) => ({
      name: `${name} indirect`, arena: "indirect" as const,
      buffer: "productionIndirectArguments", start: range.startWords,
      end: range.startWords + range.words, unit: "u32" as const,
    })),
  ];
  return Object.freeze({
    activity: Object.freeze({ framePlan, presentation, scalarCandidate,
      productionActivity, velocityExtension: velocity.activity,
      totalWords: velocity.activity.totalWords }),
    state: velocity.state, topology: frameControl.layout, scalarAuthority,
    indirect, regions: Object.freeze(regions),
  });
}
