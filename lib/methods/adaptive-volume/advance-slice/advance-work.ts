/**
 * What one advance actually costs to encode and to run.
 *
 * The stage registry beside this folder says what each stage *is*; the encoder
 * says what each stage *does*. This is the third thing, and the only one that
 * was not already written down: for every sub-seam, the dispatches it encodes
 * and the rule that sizes each one. From that a scene's accepted-cell count,
 * row count and brick count become a count of workgroups — which is the figure
 * that moves when a scene grows, and the figure a stage timing has to be read
 * against.
 *
 * Two counts, deliberately kept apart:
 *
 *   - **encoded dispatches** — call sites the command encoder writes. A
 *     zero-workgroup indirect dispatch is still one of these, and the
 *     difference between the two counts is where that shows.
 *   - **executed workgroups** — what the GPU actually runs.
 *
 * ## What drifts and where it breaks
 *
 * | The encoder changes | This file must | Breaks at |
 * |---|---|---|
 * | a stage is added or renamed | gain or rename its entry | the exhaustive table type |
 * | a sub-seam is added or moved | change that stage's seams | `advance-slice.test.ts` |
 * | a sub-seam is renamed | rename its `id` | `SparseCM12ResidentSubstage<Stage>` |
 *
 * The sizing rules are the lab's own reading of the encoder's dispatch
 * helpers, not a receipt: they model the shape of the work, and they are not
 * a substitute for a trace.
 */
import type {
  SPARSE_CM12_RESIDENT_STAGES,
  SparseCM12ResidentSubstage,
} from "../webgpu-sparse-cm12-resident";

/**
 * The stages the production graph still encodes.
 *
 * `SparseCM12ResidentStageId` also carries the three retired scalar stages, so
 * it is the wrong key for a table of work that is actually done. Taking the
 * live list instead means a retirement removes an entry here rather than
 * leaving a named zero behind.
 */
export type AdvanceStageId = (typeof SPARSE_CM12_RESIDENT_STAGES)[number];

/** Workgroup size the resident's `dispatch` helpers round against. */
export const ADVANCE_WORKGROUP_SIZE = 64;
/** `SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE`. */
export const ADVANCE_TRUE_RESIDUAL_CADENCE = 8;
/** Packets the host encodes before it maps the progress word. */
export const ADVANCE_TRANSPORT_CHUNK = 8;

const groups = (items: number): number =>
  Math.max(1, Math.ceil(items / ADVANCE_WORKGROUP_SIZE));

/** The scene figures every sizing rule is a function of. */
export interface AdvanceWorkModel {
  readonly cells: number;
  readonly rows: number;
  readonly bricks: number;
  readonly rungs: number;
  readonly pressureCells: number;
  readonly rowsPerCell: number;
  readonly deltaBricks: number;
  readonly deltaCells: number;
  readonly vexPackets: number;
  readonly markers: number;
  readonly microsteps: number;
  readonly packets: number;
  readonly pressureIterations: number;
  readonly gates: Readonly<Record<AdvanceGate, boolean>>;
}

export type AdvanceGate = "solids" | "inflow" | "tracers" | "world" | "unfrozen";

/** How one dispatch is sized. The label is what the lab shows beside it. */
export interface AdvanceDispatchKind {
  readonly label: string;
  readonly workgroups: (m: AdvanceWorkModel) => number;
}

export const ADVANCE_DISPATCH_KINDS = {
  one: { label: "singleton", workgroups: () => 1 },
  cell: { label: "accepted cells", workgroups: m => groups(m.cells) },
  row: { label: "accepted rows", workgroups: m => groups(m.rows) },
  leaf: { label: "accepted leaves", workgroups: m => groups(m.bricks) },
  brickWG: { label: "one workgroup per brick", workgroups: m => m.bricks },
  brickLn: { label: "one lane per brick", workgroups: m => groups(m.bricks) },
  pcell: { label: "pressure cells", workgroups: m => groups(m.pressureCells) },
  member: { label: "membership words", workgroups: m => groups(m.pressureCells / 32) },
  dcell: { label: "dirty cell worklist", workgroups: m => groups(m.deltaCells) },
  drow: { label: "dirty row worklist", workgroups: m => groups(m.deltaCells * m.rowsPerCell) },
  vex: { label: "VEX packets", workgroups: m => m.vexPackets },
  tiles: { label: "policy tiles", workgroups: m => groups(m.bricks) },
  delta: { label: "topology-delta cells", workgroups: m => groups(m.deltaCells) },
  dltRow: { label: "topology-delta rows", workgroups: m => groups(m.deltaCells * m.rowsPerCell) },
  shCell: { label: "shadow cells", workgroups: m => groups(m.deltaCells) },
  shRow: { label: "shadow rows", workgroups: m => groups(m.deltaCells * m.rowsPerCell) },
  shLeaf: { label: "shadow leaves", workgroups: m => groups(m.deltaBricks) },
  frontier: { label: "frontier neighbours", workgroups: m => groups(m.bricks * 6) },
  worldDir: { label: "world directory", workgroups: m => groups(m.bricks * 4) },
  pages: { label: "topology pages", workgroups: m => groups(m.bricks) },
  faceTile: { label: "interior face tiles", workgroups: m => groups(m.rows) },
  seamPk: { label: "seam packets", workgroups: m => groups(m.rows * 0.1) },
  source: { label: "source ledger", workgroups: m => groups(m.bricks) },
  tracer: { label: "markers", workgroups: m => groups(m.markers) },
  volCell: { label: "transport cells", workgroups: m => groups(m.cells) },
  volFace: { label: "transport faces", workgroups: m => groups(m.rows) },
  volOne: { label: "singleton", workgroups: () => 1 },
  copy: { label: "buffer copy", workgroups: () => 0 },
} as const satisfies Record<string, AdvanceDispatchKind>;

export type AdvanceDispatchKindId = keyof typeof ADVANCE_DISPATCH_KINDS;

export interface AdvanceKernel {
  /** The WGSL entry point, or the command for a copy. */
  readonly name: string;
  readonly kind: AdvanceDispatchKindId;
  /** Fixed repeat count inside the seam, such as a closure's pass budget. */
  readonly repeats?: number;
  /** Encoded only when this scene capability is present. */
  readonly gate?: AdvanceGate;
  /** Encoded once per rung of the ladder the scene carries. */
  readonly perRung?: boolean;
  /** Encoded once per stage even inside a repeating seam. */
  readonly once?: boolean;
  /** Indirect, and zeroed on every pass but the converging one. */
  readonly commit?: boolean;
  /** Not a dispatch: a copy or clear that still costs command time. */
  readonly isCopy?: boolean;
  /**
   * A host helper rather than a named WGSL entry point — it still opens a pass
   * and dispatches, but the encoder calls it by function name.
   */
  readonly host?: boolean;
  readonly note?: string;
}

const kernel = (
  name: string, kind: AdvanceDispatchKindId,
  options?: Omit<AdvanceKernel, "name" | "kind">,
): AdvanceKernel => ({ name, kind, ...options });
const bufferCopy = (name: string): AdvanceKernel =>
  ({ name, kind: "copy", isCopy: true });

export interface AdvanceSeamWork<Stage extends AdvanceStageId> {
  /** The ABI sub-seam, or null for an interval the encoder does not close. */
  readonly id: SparseCM12ResidentSubstage<Stage> | null;
  /** Shown in place of the registry's sub-seam label when `id` is null. */
  readonly label?: string;
  /** What the seam does — the registry names it, this says why it is there. */
  readonly note: string;
  readonly kernels: readonly AdvanceKernel[];
}

export interface AdvanceStageWork<Stage extends AdvanceStageId> {
  /**
   * Stages whose encoded work is a loop rather than a list.
   *
   * `pressure` repeats its body once per iteration of the solver budget and
   * keeps the tail encoded past convergence; `transport` repeats its packet
   * body once per microstep-times-limiter-pass and zeroes its commit
   * dispatches on every pass but the converging one.
   */
  readonly loop?: "pressure" | "transport";
  readonly notes?: readonly AdvanceNoteId[];
  readonly seams: readonly AdvanceSeamWork<Stage>[];
}

export type AdvanceWorkTable = {
  readonly [Stage in AdvanceStageId]: AdvanceStageWork<Stage>;
};

/**
 * One seam and one stage with the stage parameter erased, for iteration.
 *
 * The typed table is what keeps a sub-seam id honest; a reader walking all
 * fifteen stages does not need that guarantee a second time, and a union of
 * fifteen differently-parameterised arrays is not something `.map` can be
 * called on. `advanceStageWork` is the one place the parameter is dropped.
 */
export interface AnyAdvanceSeamWork {
  readonly id: string | null;
  readonly label?: string;
  readonly note: string;
  readonly kernels: readonly AdvanceKernel[];
}

export interface AnyAdvanceStageWork {
  readonly loop?: "pressure" | "transport";
  readonly notes?: readonly AdvanceNoteId[];
  readonly seams: readonly AnyAdvanceSeamWork[];
}

export function advanceStageWork(stage: AdvanceStageId): AnyAdvanceStageWork {
  return ADVANCE_WORK[stage];
}

export type AdvanceNoteId =
  | "topology-flip" | "continuation" | "unclosed-seams"
  | "zero-workgroup" | "incremental-not-blind" | "one-diagonal";

/**
 * The order the encoder writes the stages in, which is not the order the
 * registry declares them in: stages 9-15 sit inside `encodeAfterTransport`,
 * so geometric transport is eighth in the frame and tenth in the file.
 */
export const ADVANCE_STAGE_ORDER = [
  "transport-velocity-extension", "face-preparation", "body-forces",
  "pressure-topology", "pressure-rhs", "pressure-solve", "velocity-projection",
  "conservative-transport", "tracer-advection", "scalar-publication",
  "activity-measurement", "resolution-planning", "candidate-transfer",
  "brick-retirement", "presentation-publication",
] as const satisfies readonly AdvanceStageId[];

/** A stage missing from the order above is a type error here. */
type OrderedStage = typeof ADVANCE_STAGE_ORDER[number];
const ADVANCE_ORDER_IS_COMPLETE: AdvanceStageId extends OrderedStage
  ? true : never = true;
void ADVANCE_ORDER_IS_COMPLETE;

export const ADVANCE_WORK: AdvanceWorkTable = {
  "transport-velocity-extension": {
    seams: [
      {
        id: "frame-control-authority",
        note: "Translate host inputs and persistent receipts into a fixed set of indirect families, then refresh the PLIC interface cache.",
        kernels: [
          kernel("beginSparseCM12FrameControl", "one"),
          kernel("publishSparseCM12FrameBodyAuthority", "one"),
          kernel("sealSparseCM12FrameControl", "one"),
          bufferCopy("frame-control indirect families → 12 × family count bytes"),
          kernel("publishSparseCM12MovingSolidActivity", "cell", { gate: "solids" }),
          kernel("sparseCM12FrameControlNoop", "one", { repeats: 2, note: "body + row bypass" }),
          kernel("seedGeometricVolumeDestination", "cell"),
          kernel("refreshGeometricInterface", "cell"),
          kernel("extendGeometricInterface", "cell"),
          kernel("publishGeometricTransportFrontierSource", "cell"),
          kernel("beginGeometricSolidSnapshot", "one", { gate: "solids" }),
          kernel("snapshotGeometricSolidCells", "cell", { gate: "solids" }),
          kernel("snapshotGeometricSolidRows", "row", { gate: "solids" }),
          kernel("captureGeometricSolidCells", "cell", { gate: "solids" }),
          kernel("activateGeometricSolidMotion", "one", { gate: "solids" }),
          kernel("publishSparseCM12MovingSolidActivity", "cell", { gate: "solids" }),
        ],
      },
      {
        id: "velocity-extension-mask-initialization",
        note: "Compile the accepted packet schedule, cached by topology generation, and clear packet validity.",
        kernels: [
          kernel("beginSparseCM12VelocityExtensionSchedule", "one"),
          kernel("compileSparseCM12VelocityExtensionSchedule", "leaf"),
          kernel("sealSparseCM12VelocityExtensionSchedule", "one"),
          bufferCopy("packet schedule header → indirect arguments"),
          kernel("initializeVelocityExtensionPackets", "vex"),
        ],
      },
      {
        id: "velocity-extension-sweeps",
        note: "Eight sweeps at increasing depth. Sweep 8 publishes the effective transport velocity the fluxes ride.",
        kernels: [
          kernel("advanceVelocityExtensionPackets", "vex", { repeats: 8, note: "depth 1…8" }),
        ],
      },
      {
        id: "transport-packet-authority",
        note: "No dispatches. The scalar packet families belonged to retired CM12 transport; face prediction now reads the VEX support cache and volume owns its own subface list.",
        kernels: [],
      },
    ],
  },
  "face-preparation": {
    seams: [
      {
        id: "face-support-publication",
        note: "Retire the previous generation's face-velocity support and republish it per brick.",
        kernels: [
          kernel("seedGeometricVolumeDestination", "cell"),
          kernel("clearSparseCM12RetiredFaceVelocitySupport", "brickWG"),
          kernel("publishSparseCM12FaceVelocitySupport", "brickWG"),
        ],
      },
      {
        id: "accepted-face-row-preparation",
        note: "One compact pass over accepted rows: RK2 trace, finest-incident sampling, mixed-resolution subface selection.",
        kernels: [
          kernel("prepareSparseCM12AcceptedFaceRows", "row"),
        ],
      },
    ],
  },
  "body-forces": {
    seams: [
      {
        id: null, label: "(no sub-seams)",
        note: "One interval. The source chain below is encoded only for scenes that declare an inflow.",
        kernels: [
          kernel("forceFaces", "row"),
          kernel("beginContinuousGeometricSource", "one"),
          kernel("initializeContinuousGeometricSource", "cell"),
          kernel("connectContinuousGeometricSource", "row", { repeats: 32, gate: "inflow", note: "closure pass" }),
          kernel("compressContinuousGeometricSource", "cell", { repeats: 32, gate: "inflow", note: "closure pass" }),
          kernel("sealContinuousGeometricSource", "row", { gate: "inflow" }),
          kernel("gatherContinuousGeometricSourceWeights", "source", { gate: "inflow" }),
          kernel("prepareContinuousGeometricSourceBudget", "one", { gate: "inflow" }),
          kernel("publishContinuousGeometricSourceRates", "source", { gate: "inflow" }),
          kernel("finalizeContinuousGeometricSourceRates", "one", { gate: "inflow" }),
        ],
      },
    ],
  },
  "pressure-topology": {
    notes: ["topology-flip", "incremental-not-blind"],
    seams: [
      {
        id: "ptr-setup-brick-plan",
        note: "Open the cell, row and cache epochs and plan the brick frontier the repair will walk.",
        kernels: [
          kernel("refreshGeometricInterface", "cell"),
          kernel("extendGeometricInterface", "cell"),
          kernel("beginCanonicalPressureCells", "one"),
          kernel("beginCanonicalPressureRows", "one"),
          kernel("beginPersistentPressureCache", "one"),
          kernel("planPressureMembershipEpoch", "one"),
          kernel("beginSparseCM12PressureTopologyRepair", "one"),
          kernel("finalizeSparseCM12PressureTopologyBrickFrontier", "one"),
          bufferCopy("bootstrap cell worklist → indirect arguments"),
        ],
      },
      {
        id: "pcm-cell-publication",
        note: "Bootstrap worklist plus a full accepted-cell dirty scan. The scan runs whether or not anything changed.",
        kernels: [
          kernel("classifyPressureCells", "dcell"),
          kernel("classifyDirtyPressureCells", "cell", { note: "full accepted scan" }),
          kernel("finalizeCanonicalPressureCellFrontier", "one"),
          bufferCopy("repair frontier → indirect arguments"),
          kernel("repairCanonicalPressureCellLeaves", "dcell"),
          kernel("finalizeCanonicalPressureCells", "one"),
        ],
      },
      {
        id: "pcm-row-publication",
        note: "Rows are repaired by tile: mark, compile leaders, then publish only the dirty canonical rows.",
        kernels: [
          kernel("markCanonicalPressureRowRepairTiles", "row"),
          kernel("compileCanonicalPressureRowRepairTiles", "tiles"),
          kernel("sealCanonicalPressureRowRepairTiles", "one"),
          bufferCopy("row repair control → indirect arguments"),
          kernel("compileDirtyCanonicalPressureRows", "drow"),
          kernel("finalizeCanonicalPressureRows", "one"),
        ],
      },
      {
        id: "pca-fine-publication",
        note: "Fine coefficients are the whole production cache publication — cell ids, membership bits, coefficients, then the freeze.",
        kernels: [
          bufferCopy("execution-image cell indirect"),
          kernel("publishFrozenPressureCellIds", "pcell"),
          kernel("publishFrozenPressureMembership", "member"),
          kernel("publishFrozenPressureCoefficients", "pcell"),
          kernel("finalizePersistentPressureFineCache", "one"),
        ],
      },
      {
        id: "pca-coarse-repair",
        note: "Retired. The brick-aggregate cache publishes no numerical work; the sub-seam remains so its timestamp stays a named zero rather than silently folding into a neighbour.",
        kernels: [],
      },
      {
        id: "pca-hierarchy-and-freeze",
        note: "Retired for the same reason. The aggregate correction was not an SPD map, so the solve keeps a single positive diagonal inverse instead.",
        kernels: [],
      },
      {
        id: "pei-publication",
        note: "Snapshot the accepted generation into the pressure execution image every solve consumer dispatches against.",
        kernels: [
          kernel("finalizeSparseCM12PressureExecutionImage", "one"),
          bufferCopy("execution-image cell + solve indirect (2 copies)"),
        ],
      },
      {
        id: "ptr-commit-and-prepare-pressure",
        note: "Close the repair transaction, immediately reopen the journal for this frame's later producers, and build the per-cell RHS terms.",
        kernels: [
          kernel("finalizeSparseCM12BoundedPressureTopologyRepair", "one"),
          kernel("beginSparseCM12PressureTopologyRepair", "one"),
          kernel("preparePressure", "pcell"),
        ],
      },
    ],
  },
  "pressure-rhs": {
    notes: ["one-diagonal"],
    seams: [
      {
        id: null, label: "(no sub-seams)",
        note: "The seed can already satisfy the tolerance, so the device-side dispatch gate is published before the first iteration block.",
        kernels: [
          kernel("beginPressureSolve", "one"),
          kernel("initializePCG", "pcell"),
          kernel("initializeJacobiDirection", "pcell"),
          kernel("reduceInitialize", "one"),
          kernel("measureTrueResidual", "pcell"),
          kernel("reduceInitialTrueResidual", "one"),
          kernel("initializePipelinedImage", "pcell"),
          kernel("reducePipelinedInitialize", "one"),
          kernel("publishPressureSolveDispatchGate", "one"),
          bufferCopy("dispatch gate → cell + solve indirect (2 copies)"),
        ],
      },
    ],
  },
  "pressure-solve": {
    loop: "pressure",
    notes: ["zero-workgroup", "one-diagonal"],
    seams: [
      {
        id: null, label: "per iteration",
        note: "Three dispatches. The reduction is one workgroup; the two cell passes are the whole cost.",
        kernels: [
          kernel("updatePipelinedState", "pcell"),
          kernel("applyPipelinedImage", "pcell"),
          kernel("reducePipelinedIteration", "one"),
        ],
      },
      {
        id: null, label: "every 8th iteration",
        note: "Fresh true residual, curvature-loss restart and a republished dispatch gate. Once the tolerance is met the tail dispatches stay encoded but skip their arithmetic.",
        kernels: [
          kernel("measureGuardedTrueResidual", "pcell"),
          kernel("reduceGuardedTrueResidual", "one"),
          kernel("restartPCGAfterCurvatureLoss", "pcell"),
          kernel("initializeJacobiRecoveryDirection", "pcell"),
          kernel("reduceCurvatureRecovery", "one"),
          kernel("applyPipelinedRecovery", "pcell"),
          kernel("reducePipelinedRecovery", "one"),
          kernel("publishPressureSolveDispatchGate", "one"),
          bufferCopy("dispatch gate → cell + solve indirect (2 copies)"),
        ],
      },
      {
        id: null, label: "close",
        note: "Always a fresh b−Ap. No convergence or performance receipt may rely on the recursive residual.",
        kernels: [
          kernel("restorePressureSolveDispatches", "one"),
          bufferCopy("restored gate → cell + solve indirect (2 copies)"),
          kernel("measureTrueResidual", "pcell"),
          kernel("reduceFinalTrueResidual", "one"),
        ],
      },
    ],
  },
  "velocity-projection": {
    seams: [
      {
        id: null, label: "(no sub-seams)",
        note: "Three compiled face-address programs — interior tiles, seams, sparse air — plus a dynamic row pass for everything the compiled program cannot address.",
        kernels: [
          kernel("beginIncrementalActivity", "one"),
          kernel("projectSparseCM12InteriorFaceTiles", "faceTile"),
          kernel("projectSparseCM12SeamFacePackets", "seamPk"),
          kernel("projectSparseCM12SparseAirFacePackets", "seamPk"),
          kernel("projectSparseCM12DynamicFaceRows", "row"),
          kernel("collocateAndDiagnose", "cell"),
          kernel("reduceDivergenceDiagnostics", "one"),
          kernel("publishSparseCM12FrameFaceOutput", "one"),
        ],
      },
    ],
  },
  "conservative-transport": {
    loop: "transport",
    notes: ["continuation", "unclosed-seams", "zero-workgroup"],
    seams: [
      {
        id: "transport-trace",
        note: "Subface compilation and the CFL plan. Declared in the stage ABI but never closed in the encoder, so its interval folds into transport-gather.",
        kernels: [
          kernel("beginGeometricVolumeTransport", "one"),
          kernel("compileGeometricVolumeSubfaces", "row"),
          kernel("compileGeometricVolumeCellFaces", "cell"),
          kernel("initializeGeometricVolumeCells", "cell"),
          kernel("beginGeometricTransportEnvelope", "one"),
          kernel("gatherGeometricTransportMaterialBounds", "cell"),
          kernel("gatherGeometricTransportVelocityBounds", "row"),
          kernel("sealGeometricTransportEnvelope", "one"),
          kernel("sealGeometricVolumePlan", "one", { note: "microsteps = max(1, ⌈2·CFL⌉), fault above 128" }),
          kernel("publishVolumeDispatches", "one", { host: true }),
        ],
      },
      {
        id: "transport-scatter",
        note: "One packet: a PLIC flux pass and one relaxation of the shared low-flux limiter. Packets repeat until the limiter's volume audit passes — up to 1024 passes, or 128 while a solid is moving. Also never closed in the encoder.",
        kernels: [
          kernel("reconstructGeometricVolumeInterface", "volCell"),
          kernel("computeGeometricVolumeFluxes", "volFace"),
          kernel("beginGeometricLowFluxLimits", "volOne"),
          kernel("initializeGeometricLowFluxLimits", "volCell"),
          kernel("updateGeometricLowFluxLimits", "volCell"),
          kernel("commitGeometricLowFluxLimits", "volCell"),
          kernel("advanceGeometricLowFluxLimits", "volOne"),
          kernel("publishVolumeDispatches", "volOne", { host: true }),
        ],
      },
      {
        id: "transport-gather",
        note: "The commit half of a packet. Its indirect counts are zero on every pass except the one where the limiter converged — the dispatches are still encoded and still cost a launch.",
        kernels: [
          kernel("applyGeometricLowFluxFactors", "volFace", { commit: true }),
          kernel("initializeGeometricClosingComponents", "volCell", { commit: true, gate: "solids" }),
          kernel("allocateGeometricClosingResidual", "volCell", { commit: true, gate: "solids" }),
          kernel("computeGeometricVolumeLimits", "volCell", { commit: true }),
          kernel("limitGeometricVolumeFluxes", "volFace", { commit: true }),
          kernel("validateGeometricVolumeCells", "volCell", { commit: true }),
          kernel("commitGeometricVolumeCells", "volCell", { commit: true }),
          kernel("advanceGeometricVolumeSubstep", "volOne"),
          kernel("publishVolumeDispatches", "volOne", { host: true }),
          kernel("finishGeometricVolumeTransport", "one", { once: true }),
          kernel("reexpressGeometricSolidRows", "row", { gate: "solids", once: true }),
          kernel("finishGeometricSolidPublication", "one", { gate: "solids", once: true }),
        ],
      },
    ],
  },
  "tracer-advection": {
    seams: [
      {
        id: null, label: "(no sub-seams)",
        note: "Every off-to-on transition re-seeds, so enabling the view colours the liquid as it is at that moment.",
        kernels: [
          kernel("seedTracers", "tracer", { gate: "tracers", note: "on the first frame after enabling" }),
          kernel("advanceTracers", "tracer", { gate: "tracers" }),
        ],
      },
    ],
  },
  "scalar-publication": {
    seams: [
      {
        id: "final-scalar-mask-publication",
        note: "One workgroup per brick slot writes the packet masks the next advance's AEI authority compiles from.",
        kernels: [
          kernel("beginSparseCM12FinalScalarMasks", "one"),
          kernel("publishSparseCM12FinalScalarMasks", "brickWG"),
          kernel("sealSparseCM12FinalScalarMasks", "one"),
        ],
      },
      {
        id: null, label: "(stage remainder)",
        note: "The frame receipt and a final interface refresh on the published density.",
        kernels: [
          kernel("publishSparseCM12FrameScalarOutput", "one"),
          kernel("refreshGeometricInterface", "cell"),
          kernel("extendGeometricInterface", "cell"),
        ],
      },
    ],
  },
  "activity-measurement": {
    seams: [
      {
        id: "dirty-brick-mask-publication",
        note: "One workgroup per brick for the scalar comparison; one lane per brick for the topology bits.",
        kernels: [
          kernel("markIncrementalActivityScalarBricks", "brickWG"),
          kernel("markIncrementalActivityTopology", "brickLn"),
          kernel("finalizeIncrementalActivityMasks", "one"),
        ],
      },
      {
        id: "brick-activity-measurement",
        note: "Energy and curvature per brick, over the incremental-activity brick count.",
        kernels: [
          kernel("measureBrickActivity", "brickWG"),
        ],
      },
      {
        id: "brick-activity-census-and-history",
        note: "Age the hysteresis history and seal the census the planner reads.",
        kernels: [
          kernel("ageIncrementalActivityHistory", "brickLn"),
          kernel("finalizeIncrementalActivityCensus", "one"),
        ],
      },
      {
        id: "sparse-world-frontier-allocation",
        note: "Only when the scene carries a solid-occupancy layout: scan the six-neighbour frontier and allocate directory slots for pages the liquid is about to reach.",
        kernels: [
          kernel("allocateSparseWorldFrontier", "frontier", { gate: "world" }),
          kernel("finalizeSparseWorldDirectoryAllocations", "worldDir", { gate: "world" }),
        ],
      },
      {
        id: null, label: "(stage remainder)",
        note: "Synthesize the page bodies the allocation reserved.",
        kernels: [
          kernel("synthesizeSparseWorldFrontierPages", "pages", { gate: "world" }),
        ],
      },
    ],
  },
  "resolution-planning": {
    seams: [
      {
        id: "liquid-frontier-classification",
        note: "Which accepted bricks sit on the liquid frontier — one workgroup per brick slot.",
        kernels: [
          kernel("classifyAcceptedLiquidFrontier", "brickWG"),
        ],
      },
      {
        id: "refinement-policy-classification",
        note: "Score each policy tile against the energy, curvature, travel and thin-feature thresholds.",
        kernels: [
          kernel("classifyRefinementPolicyTiles", "tiles"),
        ],
      },
      {
        id: "initial-resolution-plan",
        note: "A first rung per brick on the 1 / 2 / 4 / 8 ladder, before grading.",
        kernels: [
          kernel("planBrickResolution", "brickLn"),
        ],
      },
      {
        id: "frontier-activation-and-retirement",
        note: "Activate swept pages, reserve the transport face support, enforce the dynamic seam floor, and mark unsupported empty bricks for retirement.",
        kernels: [
          kernel("activateSweptFrontierPages", "brickWG"),
          kernel("reserveGeometricTransportFaceSupport", "brickLn"),
          kernel("enforceGeometricDynamicSeamFloor", "brickLn"),
          kernel("retireUnsupportedEmptyBricks", "brickWG", { gate: "unfrozen" }),
        ],
      },
      {
        id: "resolution-grading-and-validation",
        note: "One 2:1 grading pass per ladder rung — log₂(brick resolution) of them — then a candidate validation.",
        kernels: [
          kernel("closeRefinementPolicyTileResolution", "tiles", { perRung: true }),
          kernel("closePlannedResolution", "brickLn", { perRung: true }),
          kernel("validateCandidateResolution", "brickLn"),
        ],
      },
      {
        id: "candidate-page-allocation-and-synthesis",
        note: "Schedule the per-frame brick budget and certify the candidate faces.",
        kernels: [
          kernel("scheduleTopologyPreparation", "one"),
          kernel("certifyGeometricTopologyFaces", "brickWG"),
          kernel("sealGeometricTopologyFaces", "brickLn"),
        ],
      },
      {
        id: null, label: "(stage remainder)",
        note: "Build the shadow worklists and publish five indirect command copies for the following transaction.",
        kernels: [
          kernel("clearShadowRowMembership", "shRow"),
          kernel("beginShadowTopology", "one"),
          kernel("buildShadowLeafWorklist", "one"),
          bufferCopy("shadow leaf manifest → indirect arguments"),
          kernel("buildShadowStructureWorklist", "shLeaf"),
          kernel("finalizeShadowWorklists", "one"),
          bufferCopy("worklist headers → 4 further indirect copies"),
        ],
      },
    ],
  },
  "candidate-transfer": {
    notes: ["topology-flip"],
    seams: [
      {
        id: "candidate-field-transfer",
        note: "Density, gamma and momentum into the shadow slots. Refinement intersects the parent plane with each child's bounds; coarsening sums accepted extensive amounts.",
        kernels: [
          kernel("transferCandidateCellsFromTopologyDelta", "delta"),
        ],
      },
      {
        id: "candidate-face-reconstruction",
        note: "Exterior faces rebuilt for the new rungs.",
        kernels: [
          kernel("transferCandidateFacesFromTopologyDelta", "delta"),
        ],
      },
      {
        id: "candidate-face-validation",
        note: "Every shadow face checked before anything may depend on it.",
        kernels: [
          kernel("validateCandidateShadowFaces", "shRow"),
        ],
      },
      {
        id: "candidate-effects-preflight",
        note: "Census what the delta will change, so the transaction can be rejected before any publication.",
        kernels: [
          kernel("beginSparseCM12TopologyEffectsPreflight", "one"),
          kernel("recordCandidateTopologyEffectsFromTopologyDelta", "delta"),
          kernel("finalizeSparseCM12TopologyEffectsPreflight", "one"),
        ],
      },
      {
        id: "candidate-ibo-construction",
        note: "Compile the interned-boundary delta — the shared boundary identities the next frame's operators intern against.",
        kernels: [
          kernel("beginSparseCM12InternedBoundaryDelta", "one"),
          kernel("compileSparseCM12InternedBoundaryDelta", "delta"),
        ],
      },
      {
        id: "candidate-ibo-validation",
        note: "An independent semantic validation of the delta packets, not a re-run of the construction.",
        kernels: [
          kernel("finalizeSparseCM12ISAChangedSetReceipt", "one"),
          kernel("validateSparseCM12InternedBoundaryDeltaPackets", "delta"),
          kernel("finalizeSparseCM12InternedBoundaryDelta", "one"),
        ],
      },
      {
        id: "candidate-tei-compilation",
        note: "Compile the transport execution image shadow the next advance dispatches against.",
        kernels: [
          kernel("compileSparseCM12TransportExecutionImageShadow", "delta"),
        ],
      },
      {
        id: "candidate-authorization",
        note: "One GPU-authored decision: the whole transaction is accepted or none of it is.",
        kernels: [
          kernel("validateAndAuthorizeShadowTopology", "one"),
        ],
      },
      {
        id: "candidate-ptr-publication",
        note: "No dispatches of its own — the PTR effects were journalled as the producers ran.",
        kernels: [],
      },
      {
        id: "candidate-effects-seal",
        note: "Seal the authorized effects and close the publication.",
        kernels: [
          kernel("sealSparseCM12AuthorizedTopologyEffects", "one"),
          kernel("finishSparseCM12TopologyEffectsPublication", "one"),
        ],
      },
      {
        id: "candidate-state-publication",
        note: "The flip. Fields, membership, frontier acceptance and the world execution image all become the accepted generation here.",
        kernels: [
          kernel("publishCandidateTopologyDeltaFromWorklist", "delta"),
          kernel("connectSparseWorldFrontierPages", "one"),
          kernel("publishCandidateShadowFaces", "shRow"),
          kernel("finalizeAuthorizedShadowTopology", "one"),
          kernel("publishSparseWorldFrontierAcceptance", "one"),
          kernel("compileSparseWorldFrontierExecutionImage", "one"),
        ],
      },
      {
        id: "candidate-image-replay",
        note: "Replay the retired generation's images so nothing downstream reads a slot that has moved.",
        kernels: [
          kernel("replaySparseCM12TransportExecutionImageRetired", "delta"),
          kernel("replaySparseCM12InternedBoundaryDelta", "delta"),
        ],
      },
    ],
  },
  "brick-retirement": {
    seams: [
      {
        id: null, label: "(no sub-seams)",
        note: "Runs after the census, because the commit happens after it.",
        kernels: [
          kernel("markIncrementalActivityPostTopology", "brickLn"),
          kernel("finalizeIncrementalActivityMasks", "one"),
        ],
      },
    ],
  },
  "presentation-publication": {
    seams: [
      {
        id: null, label: "(no sub-seams)",
        note: "Allocate, sort, publish the plan, then retire and compact the all-air pages of the retiring generation before the next plan is accepted.",
        kernels: [
          kernel("allocateSparseCM12PresentationPages", "brickLn"),
          kernel("sortSparseCM12PresentationPageDirectory", "one"),
          kernel("encodeFramePlanPresentation", "brickWG", { host: true, note: "FPP1 nested encode" }),
          kernel("retireSparseCM12PresentationPages", "brickLn"),
          kernel("compactSparseCM12PresentationPageDirectory", "one"),
          kernel("commitSparseCM12FrameControl", "one"),
        ],
      },
    ],
  },
};

/* ---- the scenes the lab scales its figures to ---------------------- */

export interface AdvanceWorkScene {
  readonly label: string;
  /** Where the figures came from, so a stale capture cannot pass as live. */
  readonly provenance: string;
  readonly cells: number;
  readonly rows: number;
  readonly bricks: number;
  readonly rungs: number;
  readonly gates: Readonly<Record<AdvanceGate, boolean>>;
}

const gates = (
  over: Partial<Record<AdvanceGate, boolean>>,
): Record<AdvanceGate, boolean> => ({
  solids: false, inflow: false, tracers: false, world: true, unfrozen: true, ...over,
});

export const ADVANCE_WORK_SCENES = {
  mini32: {
    label: "mini32",
    provenance: "B8 · 32×16×32 symmetric expansion, captured 2026-09-11 at HEAD",
    cells: 13_696, rows: 43_232, bricks: 32, rungs: 3,
    gates: gates({}),
  },
  ocean: {
    label: "ocean-seiche",
    provenance: "B16 · terminal work from the 2026-09-10 stage-cost capture",
    cells: 537_220, rows: 1_574_783, bricks: 509, rungs: 4,
    gates: gates({}),
  },
  dam: {
    label: "moving dam",
    provenance: "illustrative: a moving solid and an inflow at mini32's scale",
    cells: 60_000, rows: 190_000, bricks: 140, rungs: 3,
    gates: gates({ solids: true, inflow: true }),
  },
} as const satisfies Record<string, AdvanceWorkScene>;

export type AdvanceWorkSceneId = keyof typeof ADVANCE_WORK_SCENES;

export interface AdvanceWorkInputs {
  readonly scene: AdvanceWorkScene;
  /** Pressure iteration budget. The tail stays encoded past convergence. */
  readonly pressureIterations: number;
  /** Peak CFL over the advance; the microstep plan is `ceil(2 * CFL)`. */
  readonly cfl: number;
  /** Limiter passes per microstep. */
  readonly limiterPasses: number;
  /** Fraction of resident bricks whose rung moved this advance. */
  readonly churn: number;
  readonly markers: number;
}

/** Resolve the scene and the live readings into one set of sizing figures. */
export function advanceWorkModel(inputs: AdvanceWorkInputs): AdvanceWorkModel {
  const { scene } = inputs;
  const microsteps = Math.min(128, Math.max(1, Math.ceil(2 * inputs.cfl)));
  const deltaBricks = Math.min(64, Math.max(0, scene.bricks * inputs.churn));
  return {
    cells: scene.cells,
    rows: scene.rows,
    bricks: scene.bricks,
    rungs: scene.rungs,
    /* leaves carrying a pressure row, as a share of accepted cells */
    pressureCells: scene.cells * 0.52,
    rowsPerCell: scene.rows / Math.max(1, scene.cells),
    deltaBricks,
    deltaCells: deltaBricks * (scene.cells / Math.max(1, scene.bricks)),
    vexPackets: Math.max(1, Math.ceil(scene.cells / ADVANCE_WORKGROUP_SIZE)),
    markers: inputs.markers,
    microsteps,
    packets: microsteps * inputs.limiterPasses,
    pressureIterations: inputs.pressureIterations,
    gates: scene.gates,
  };
}

export interface AdvanceCost {
  /** Dispatch call sites the encoder writes. */
  readonly dispatches: number;
  /** Workgroups the GPU runs. */
  readonly workgroups: number;
}

const ZERO: AdvanceCost = { dispatches: 0, workgroups: 0 };
const add = (a: AdvanceCost, b: AdvanceCost): AdvanceCost => ({
  dispatches: a.dispatches + b.dispatches,
  workgroups: a.workgroups + b.workgroups,
});

function kernelCost(
  entry: AdvanceKernel, m: AdvanceWorkModel, repeats: number,
): AdvanceCost {
  if (entry.gate && !m.gates[entry.gate]) return ZERO;
  if (entry.isCopy) return ZERO;
  let count = entry.repeats ?? 1;
  if (entry.perRung) count *= m.rungs;
  count *= entry.once ? 1 : repeats;
  const each = ADVANCE_DISPATCH_KINDS[entry.kind].workgroups(m);
  return { dispatches: count, workgroups: count * each };
}

/**
 * One seam's cost, or null for a seam inside the pressure loop — the solver's
 * three labels describe the loop's phases, not three separable intervals.
 */
export function advanceSeamCost(
  stage: AnyAdvanceStageWork, seam: AnyAdvanceSeamWork,
  index: number, m: AdvanceWorkModel,
): AdvanceCost | null {
  if (stage.loop === "pressure") return null;
  const repeats = stage.loop === "transport" && index > 0 ? m.packets : 1;
  return seam.kernels.reduce((total, entry) => {
    const cost = kernelCost(entry, m, repeats);
    /* a commit kernel is encoded every pass and executes on the converging one */
    const workgroups = stage.loop === "transport" && entry.commit
      ? (cost.workgroups / Math.max(1, m.packets)) * m.microsteps
      : cost.workgroups;
    return add(total, { dispatches: cost.dispatches, workgroups });
  }, ZERO);
}

/**
 * One stage's cost.
 *
 * The pressure solve is counted from the encoder's own loop rather than from
 * a seam list: three dispatches per iteration, a seven-dispatch true-residual
 * guard on every eighth iteration but the last, and a three-dispatch close.
 */
export function advanceStageCost(
  stage: AnyAdvanceStageWork, m: AdvanceWorkModel,
): AdvanceCost {
  if (stage.loop === "pressure") {
    const cells = ADVANCE_DISPATCH_KINDS.pcell.workgroups(m);
    const iterations = m.pressureIterations;
    const guards = Math.max(0,
      Math.ceil(iterations / ADVANCE_TRUE_RESIDUAL_CADENCE) - 1);
    return {
      dispatches: iterations * 3 + guards * 8 + 3,
      workgroups: iterations * (cells * 2 + 1)
        + guards * (cells * 4 + 4) + (cells + 2),
    };
  }
  return stage.seams.reduce((total, seam, index) =>
    add(total, advanceSeamCost(stage, seam, index, m) ?? ZERO), ZERO);
}

/** Every stage's cost, in encode order. */
export function advanceCosts(m: AdvanceWorkModel): readonly AdvanceCost[] {
  return ADVANCE_STAGE_ORDER.map(id => advanceStageCost(ADVANCE_WORK[id], m));
}

/**
 * What the work model shows that the stage registry does not say.
 *
 * Each of these is a reading of the encoder that only becomes visible once
 * encoded dispatches and executed workgroups are counted apart. They are filed
 * against the stages they concern rather than collected in a list, so they
 * surface when someone is looking at the stage they are about.
 */
export const ADVANCE_NOTES: Readonly<Record<AdvanceNoteId, {
  readonly heading: string;
  readonly body: string;
}>> = {
  "topology-flip": {
    heading: "The topology flip lands in the next advance",
    body: "Candidate transfer commits at the frame tail, but pressure topology already ran at the head. This advance's commit is the next advance's pressure-repair input, never this one's — so a stage that reports a topology generation is reporting the one accepted at the end of the previous frame.",
  },
  continuation: {
    heading: "Transport can span several command submissions",
    body: "With a packet chunk of 8 the host encodes eight packets, submits, maps a four-word progress buffer, and only then decides whether to encode more. The frame is a continuation, not one command buffer. A direct command-encoder caller instead gets a fixed schedule of 512 packets.",
  },
  "unclosed-seams": {
    heading: "Two sub-seams are declared but never closed",
    body: "transport-trace and transport-scatter appear in the stage ABI and carry phase labels, but the encoder only calls closeSubstage(\"transport-gather\"). All three labels resolve to a single measured interval, so the trace and limiter halves of transport are not separately timed.",
  },
  "zero-workgroup": {
    heading: "A zero-workgroup indirect dispatch is still a launch",
    body: "The commit kernels in a transport packet have their indirect counts zeroed on every pass but the converging one, and the pressure solve keeps its full tail encoded after the tolerance is met. Encoded dispatches and executed workgroups diverge sharply — switch the strip between them to see by how much.",
  },
  "incremental-not-blind": {
    heading: "Pressure topology is incremental but not change-blind",
    body: "The repair is seeded from the prior generation and walks dirty worklists, yet classifyDirtyPressureCells is a full accepted-cell scan and the frozen cache publication is sized by every pressure cell. A scene with no topology churn does not take pressure topology to zero.",
  },
  "one-diagonal": {
    heading: "The preconditioner is one positive diagonal",
    body: "The brick-aggregate and hierarchy sub-seams survive as named zeros. Their correction was not an SPD map — it lost positive curvature before runtime pages entered the accepted topology — so the solve uses a single Jacobi inverse, uniform across authored and runtime cells.",
  },
};
