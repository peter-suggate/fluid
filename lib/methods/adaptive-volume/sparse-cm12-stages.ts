import { correctionStageControl } from "./correction-controls";
import { adaptivityStageControl } from "./features/adaptivity/definition";
import { algorithmStageControl } from "./features/algorithms/definition";
import type { AdvanceSliceDeclaration } from "./features/advance-slice/definition";
/**
 * Every Sparse CM12 stage, described once.
 *
 * The resident encoder owns the stage ABI — the ordered stage ids and the
 * sub-seams each stage closes (`SPARSE_CM12_RESIDENT_STAGES`,
 * `SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES`). This is everything *about* those
 * stages that is not a dispatch: what the SIM diagram says of each, the trace
 * phase its seam is timed under, the phase of every sub-seam inside it, and
 * the lens on it. One entry per stage, keyed by the resident's own id, so the
 * diagram node, the timing partition and the lens can never be three
 * differently spelled lists.
 *
 * ## What drifts and where it breaks
 *
 * | The encoder changes | This file must | Breaks at |
 * |---|---|---|
 * | a stage is added or renamed | gain or rename its entry | the exhaustive `satisfies` |
 * | a sub-seam is added, removed or moved | change that stage's `substages` | `SparseCM12ResidentSubstage<Stage>` |
 * | a lens is filed under another stage | move it | `lens.stage` must equal the key |
 * | a body closes a seam that is not its own | — | the typed `closeSubstage` in `encode` |
 *
 * Only the *prose* — labels, tips, chips — is beyond the type checker, which
 * is why it lives beside the thing it describes rather than in a component.
 *
 * This module is a leaf on purpose: it imports the resident's types and
 * nothing of its runtime, so the resident may import it (through the lens
 * roster) without a cycle. Chips therefore read resolved parameter values
 * from the panel's context rather than the resident's default constants.
 */
import type {
  FluidPipelineContext,
  FluidPipelineStage,
  FluidPipelineTip,
  FluidStageControl,
} from "../../core/fluid-pipeline";
import type { GPUTimestampPhase } from "../../core/performance-trace";
import type { AnyStageLens, StageLens } from "../../core/stage-lens";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import { formatSparseCM12PressureCutoverAuthorities } from
  "./sparse-cm12-pressure-cutover-observability";
import type {
  SparseCM12ResidentStageId,
  SparseCM12ResidentSubstage,
} from "./webgpu-sparse-cm12-resident";

/** The diagram's bands, in the order the advance first enters each. */
export const SPARSE_CM12_STAGE_BANDS = Object.freeze({
  transport: "Transport velocity + geometric volume transport",
  momentum: "Momentum prediction",
  pressure: "Composite pressure projection + receipts",
  adaptivity: "Activity census + candidate topology",
  output: "Sparse presentation publication",
});
export type SparseCM12StageBand = keyof typeof SPARSE_CM12_STAGE_BANDS;

interface SparseCM12StageDeclarationBase<Stage extends SparseCM12ResidentStageId> {
  /** The diagram node's name. The id is the resident's stage id. */
  readonly label: string;
  readonly band: SparseCM12StageBand;
  readonly side: "left" | "right";
  /**
   * The phase the stage's own seam is timed under.
   *
   * For a stage with sub-seams this is whatever follows its last sub-seam —
   * usually nothing, and it reads as such — because each sub-seam owns its own
   * disjoint interval. The stage's figure on the diagram is the sum.
   */
  readonly phase: GPUTimestampPhase;
  /**
   * Shader entry points and non-shader command work bracketed by this row.
   *
   * Truth-sensitive rows use this manifest to derive their UI timing detail.
   * The advance-partition contract compares it with the encoder source, so a
   * dispatch cannot be added or removed without updating what the UI says.
   */
  readonly timedWork?: SparseCM12TimedWorkManifest;
  /**
   * The lens on this stage, or the written-down decision that there is none.
   * The lens's own `stage` must be this key: a lens filed under the wrong
   * stage is a type error here, not a ◎ that opens the wrong picture.
   */
  readonly lens: (AnyStageLens & { readonly stage: Stage }) | null;
  /**
   * How the 2-D advance slice reads this stage: its caption, the marks the
   * picture may put on a cell, and which of the loop's four readings it falls
   * under. Declaration only — what this stage *is*, said once more in the
   * terms a reader watching one cell needs, never a drawing. The lab joins it
   * with its own `draw`/`holds` table in `advance-lab/lenses.ts`.
   *
   * Absent exactly on the stages the production graph no longer encodes: a
   * retired stage has nothing to show, and giving it a caption would put a
   * lens in the strip for work that never runs.
   */
  readonly slice?: AdvanceSliceDeclaration;
  readonly tip: FluidPipelineTip;
  /** The short factual chip under the label. Never a description. */
  readonly chip: (context: FluidPipelineContext) => string;
  readonly controls?: readonly FluidStageControl[];
  /** Optional live gate rendered as the stage lamp in the SIM panel. */
  readonly toggle?: FluidPipelineStage["toggle"];
}

export interface SparseCM12TimedWorkGroup {
  /** Short, user-facing description shown in the stage timing tooltip. */
  readonly label: string;
  /** Unique WGSL entry points invoked by this group. */
  readonly entryPoints: readonly string[];
}

export interface SparseCM12TimedWorkManifest {
  readonly groups: readonly SparseCM12TimedWorkGroup[];
  /** Copies/clears that consume time inside the seam but are not shaders. */
  readonly commandCopies?: number;
}

const activityTimedWork = Object.freeze({
  groups: Object.freeze([
    {
      label: "dirty-brick mask publication",
      entryPoints: Object.freeze([
        "markIncrementalActivityScalarBricks",
        "markIncrementalActivityTopology",
        "finalizeIncrementalActivityMasks",
      ]),
    },
    {
      label: "brick activity census and history",
      entryPoints: Object.freeze([
        "measureBrickActivity",
        "ageIncrementalActivityHistory",
        "finalizeIncrementalActivityCensus",
      ]),
    },
    {
      label: "sparse-world frontier allocation",
      entryPoints: Object.freeze([
        "allocateSparseWorldFrontier",
        "finalizeSparseWorldDirectoryAllocations",
      ]),
    },
    {
      label: "sparse-world frontier page synthesis",
      entryPoints: Object.freeze([
        "synthesizeSparseWorldFrontierPages",
      ]),
    },
  ]),
} satisfies SparseCM12TimedWorkManifest);

const candidatePlanTimedWork = Object.freeze({
  groups: Object.freeze([
    {
      label: "accepted-liquid frontier classification",
      entryPoints: Object.freeze([
        "classifyAcceptedLiquidFrontier",
      ]),
    },
    {
      label: "refinement-policy tile classification",
      entryPoints: Object.freeze([
        "compileSparseCM12RefinementPolicyTileLeaders",
        "classifyRefinementPolicyTiles",
      ]),
    },
    {
      label: "initial resolution plan",
      entryPoints: Object.freeze([
        "planBrickResolution",
      ]),
    },
    {
      label: "frontier activation and retirement",
      entryPoints: Object.freeze([
        "activateSweptFrontierPages",
        "retireUnsupportedEmptyBricks",
      ]),
    },
    {
      label: "one 2:1 grading pass per rung and candidate validation",
      entryPoints: Object.freeze([
        "closeRefinementPolicyTileResolution",
        "closePlannedResolution",
        "validateCandidateResolution",
      ]),
    },
    {
      label: "budget scheduling of backed candidate topology",
      entryPoints: Object.freeze([
        "scheduleTopologyPreparation",
        "certifyGeometricTopologyFaces",
        "sealGeometricTopologyFaces",
      ]),
    },
    {
      label: "shadow row, leaf and structure worklist construction",
      entryPoints: Object.freeze([
        "clearShadowRowMembership",
        "beginShadowTopology",
        "buildShadowLeafWorklist",
        "buildShadowStructureWorklist",
        "finalizeShadowWorklists",
      ]),
    },
  ]),
  commandCopies: 5,
} satisfies SparseCM12TimedWorkManifest);

/**
 * A stage's declaration, shaped by its sub-seams.
 *
 * A stage that closes sub-seams must name a phase for every one of them, and
 * only those; a stage that closes none may not declare any. Both halves are
 * `SparseCM12ResidentSubstage<Stage>`, which the resident derives from its own
 * table, so moving a sub-seam between stages in the encoder breaks exactly the
 * two entries it affects.
 */
export type SparseCM12StageDeclaration<Stage extends SparseCM12ResidentStageId> =
  SparseCM12StageDeclarationBase<Stage>
  & ([SparseCM12ResidentSubstage<Stage>] extends [never]
    ? { readonly substages?: undefined }
    : {
      readonly substages: Readonly<
        Record<SparseCM12ResidentSubstage<Stage>, GPUTimestampPhase>
      >;
    });

export type SparseCM12StageDeclarations = {
  readonly [Stage in SparseCM12ResidentStageId]: SparseCM12StageDeclaration<Stage>;
};

/** Any one declaration with its stage parameter erased, for iteration. */
export interface SparseCM12AnyStageDeclaration
  extends SparseCM12StageDeclarationBase<SparseCM12ResidentStageId> {
  readonly substages?: Readonly<Record<string, GPUTimestampPhase>>;
}

const number = (value: unknown): number | undefined => {
  const parsed = typeof value === "number" ? value
    : typeof value === "string" ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : undefined;
};
const fixed = (value: unknown, digits: number): string => {
  const parsed = number(value);
  return parsed === undefined ? "—" : parsed.toFixed(digits);
};

/** Latest queue-confirmed Sparse CM12 pressure work, for the SIM frame panel. */
export function adaptiveMassPressureIterationReadout(
  info: Pick<GPUEulerianInfo,
    "pressureIterationsExecuted" | "pressureIterationsEncoded"> | null | undefined,
  requestedBudget: unknown,
): string {
  const encoded = info?.pressureIterationsEncoded ?? number(requestedBudget);
  const executed = info?.pressureIterationsExecuted;
  if (executed === undefined) return `— / ${encoded === undefined ? "—" : Math.round(encoded)}`;
  const ceiling = encoded === undefined ? executed : Math.max(executed, Math.round(encoded));
  return `${executed} / ${ceiling}`;
}

/**
 * Pressure repair consumes the topology accepted before this advance. The
 * topology committed at the end of this advance is intentionally a separate
 * final line because it can only affect the next pressure-topology sample.
 */
export function adaptiveMassPressureTopologyChip(
  info: Pick<GPUEulerianInfo,
    "adaptivePressureTopologyAttribution"
    | "adaptiveAcceptedSameLevelCoarseRowCount"
    | "adaptiveAcceptedMixedSeamRowCount"
    | "adaptiveMixedSeamFaceCount"> | null | undefined,
): string {
  const receipt = info?.adaptivePressureTopologyAttribution;
  if (!receipt) {
    return "Input topology: awaiting paired diagnostics receipt"
      + "\nPressure work attribution unavailable; end-frame commits are not relabelled";
  }
  const input = receipt.status === "matched"
    ? `Input topology gen ${receipt.inputTopologyGeneration ?? "?"}`
      + ` · prior commit ${(receipt.priorCommittedBrickCount ?? 0).toLocaleString()} bricks`
    : "Input topology: UNAVAILABLE (unobserved generation change)";
  const work = `Matched work: accepted ${receipt.acceptedCellCount.toLocaleString()} cells / `
    + `${receipt.acceptedRowCount.toLocaleString()} rows · pressure `
    + `${receipt.pressureCellCount.toLocaleString()} / `
    + `${receipt.pressureActiveRowCount.toLocaleString()}`;
  const pcm = `PCM gen ${receipt.pcmCellAcceptedGeneration}/`
    + `${receipt.pcmRowAcceptedGeneration} · cell dirty leaves `
    + `${receipt.pcmCellDirtyLeafCount} · row words `
    + `${receipt.pcmRowPublishedWordCount}`
    + ` · ${receipt.pcmMatched ? "matched" : "FAULT/INCOMPLETE"}`;
  const authorities = formatSparseCM12PressureCutoverAuthorities(
    receipt.authorities, receipt.inputTopologyGeneration,
  );
  const next = `End-frame → topology gen ${receipt.currentEndFrameTopologyGeneration}`
    + ` · ${receipt.currentEndFrameCommittedBrickCount.toLocaleString()} committed bricks`
    + " (next repair input)";
  const structure = `Rows: ${(info?.adaptiveAcceptedSameLevelCoarseRowCount ?? 0)
    .toLocaleString()} same-level coarse · ${(info?.adaptiveAcceptedMixedSeamRowCount
      ?? info?.adaptiveMixedSeamFaceCount ?? 0).toLocaleString()} mixed seams`;
  return `${input}\n${work}\n${pcm}\n${authorities}\n${structure}\n${next}`;
}

const activityOnly = (context: FluidPipelineContext) =>
  context.values.selectorMode === "activity" || context.values.selectorMode === "coarse-first";

/**
 * The registry. Keys are the resident's stage ids, in encode order, and the
 * `satisfies` is what makes a new or renamed stage a compile error here.
 */
export const SPARSE_CM12_STAGES = Object.freeze({
  "transport-velocity-extension": {
    label: "Velocity extension", band: "transport", side: "left",
    phase: { id: "velocity-extrapolation", label: "Velocity extension stage remainder" },
    substages: {
      "frame-control-authority": {
        id: "velocity-extrapolation",
        label: "Frame-control authority + moving-solid activity",
      },
      "velocity-extension-mask-initialization": {
        id: "velocity-extrapolation",
        label: "VEX2 cached schedule + packet-mask initialization",
      },
      "velocity-extension-sweeps": {
        id: "velocity-extrapolation",
        label: "VEX2 eight scheduled packet sweeps + fused commit",
      },
      "transport-packet-authority": {
        id: "velocity-extrapolation",
        label: "AEI packet-authority construction",
      },
    },
    lens: null,
    slice: {
      caption: "Eight packet sweeps push face velocity out of the liquid into the empty band, so transport has a defined velocity everywhere it might sweep. A ghost row is one no liquid cell touches.",
      keys: [
        { id: "extended-ghost-row", tone: "transport", label: "extended ghost row",
          note: "the sweeps wrote velocity here: no liquid cell touches this row" },
        { id: "carried-velocity", tone: "amber", label: "carried velocity",
          note: "liquid here carries its own face velocity into the sweep" },
      ],
      loopStep: 2,
    },
    tip: {
      summary: "Prepares frame controls, motion bounds and activity bookkeeping. Optionally stages extra sparse support needed by moving liquid. Caches accepted packet addresses by topology generation, initializes liquid velocities and clears air velocities, then optionally runs eight sweeps to extend velocity into air. Support lookahead and extension sweeps can be disabled independently.",
      reads: "accepted velocity, liquid level set, accepted topology and body/boundary inputs",
      writes: "sealed frame control, staged sparse support and transport velocity cache",
      feeds: "face preparation and geometric volume transport",
    },
    controls: [algorithmStageControl("velocityExtension"), algorithmStageControl("preflightSupport")],
    chip: (context) => `${context.values.velocityExtension === "off"
      ? "liquid seeds only · 0 extension sweeps"
      : "8 cached-packet sweeps"} · support lookahead ${context.values.preflightSupport === "off" ? "off" : "on"}`,
  },
  "face-preparation": {
    label: "Face preparation", band: "transport", side: "right",
    phase: { id: "power-topology", label: "Composite face preparation + oriented transport rows" },
    substages: {
      "face-support-publication": {
        id: "power-topology",
        label: "Resident face-velocity support clear + publication",
      },
      "accepted-face-row-preparation": {
        id: "power-topology",
        label: "Compact accepted face preparation",
      },
    },
    lens: null,
    slice: {
      caption: "Every row is re-cut against the solids. The stored face velocity already folds in the aperture as u = a·u_fluid + (1−a)·u_wall — flux code must never multiply by a twice.",
      keys: [
        { id: "closed-row", tone: "solidEdge", label: "closed row · a = 0",
          note: "a row of this cell is shut against the solid, so no flux crosses it" },
        { id: "partly-open-row", tone: "momentum", label: "partly open row",
          note: "a row of this cell is part solid; the stored u already folds that aperture in" },
      ],
      loopStep: 2,
    },
    tip: {
      summary: "Traces accepted faces with RK2 through extended velocity, then samples the source staggered face field on the finest incident lattice. Physical subface overlap selects mixed-resolution samples. Dry support, moving cut faces and uncertified exterior patches use the extended velocity field. Explicit coarse regions can enlarge trajectory sampling without coarsening the advected face field.",
      reads: "source face velocity, extended trajectory velocity, accepted cells and composite row topology",
      writes: "oriented face transport rows",
      feeds: "geometric volume transport",
    },
    chip: (context) => context.info
      ? `${context.info.fluidBrickResidentCount ?? 0} resident bricks · supported rows`
      : "supported rows",
  },
  "conservative-transport": {
    label: "Adaptive level set and volume transport", band: "transport", side: "left",
    phase: {
      id: "fine-sdf-advection",
      label: "Adaptive phi and whole-frame volume transport",
    },
    substages: {
      "transport-coupling": {
        id: "fine-sdf-advection", label: "Phi advection and geometric transport coupling",
      },
      "liquid-capacity-balancing": {
        id: "fine-sdf-advection", label: "Liquid receiver capacity balancing",
      },
      "transport-gather": {
        id: "fine-sdf-advection", label: "Conservative volume gather and commit",
      },
    },
    lens: null,
    slice: {
      caption: "Volume moves as swept prisms cut from the PLIC polygon and handed across one shared subface. Each arrow is a paired debit and credit; a marked row is one the bounded limiter had to cut back.",
      keys: [
        { id: "swept-flux", tone: "transport", label: "swept flux",
          note: "volume crossed a row of this cell as a swept prism this advance" },
        { id: "limiter-clipped", tone: "alarm", label: "limiter clipped",
          note: "the bounded limiter had to cut a row of this cell back to keep V inside K" },
      ],
      loopStep: 3,
    },
    tip: {
      summary: "Advects adaptive phi, builds a sparse donor/receiver coupling, and balances overfull liquid receivers while preserving donor totals. Gathers extensive liquid volume once; any excess remaining after the bounded balancing stays explicit for the pressure source.",
      reads: "adaptive phi, extensive liquid volume, cell capacity, projected face velocity and compiled topology",
      writes: "advected adaptive phi, conservative liquid volume and transport receipts",
      feeds: "pressure, adaptivity and presentation publication",
    },
    controls: [
      algorithmStageControl("surfaceSharpening"),
      algorithmStageControl("sharpeningStrength"),
      algorithmStageControl("distanceSweeps"),
      algorithmStageControl("returnPasses"),
    ],
    chip: (context) => `adaptive phi · whole-frame volume · ${
      context.values.surfaceSharpening === "off" ? "no sharpening"
        : `sharpen ${fixed(context.values.sharpeningStrength, 2)}x`}`,
  },
  "tracer-advection": {
    label: "Marker advection", band: "transport", side: "right",
    phase: { id: "other", label: "Fluid marker advection along the transport characteristic" },
    lens: null,
    slice: {
      caption: "Markers ride the same published transport velocity the volume does. They carry no mass — they exist so a colour or an age can be read back out of the flow.",
      keys: [
        { id: "marker", tone: "adaptivity", label: "marker",
          note: "a marker rides this cell; it carries no mass, only what was written on it" },
      ],
      loopStep: 4,
    },
    tip: {
      summary: "Presentation-only markers integrated through the extended velocity that supplies the geometric face fluxes. Encoded only while the marker view is on, so this reads zero on an ordinary frame.",
      reads: "extended transport velocity, accepted density",
      writes: "marker positions and their live flags",
      feeds: "the marker overlay, and nothing in the physics",
    },
    chip: () => "view only — zero when off",
  },
  "gamma-diffusion": {
    label: "Gamma diffusion", band: "transport", side: "left",
    phase: { id: "fine-sdf-advection", label: "Gamma diffusion row-owned snapshot iterations" },
    lens: null,
    tip: {
      summary: "Sec. 3.4 step 8 as configurable stable immutable-snapshot iterations, each a transport-row-authority scatter followed by an accepted-cell finalize. Every composite subface contributes paired antisymmetric fixed-point rho/gamma receipts, so mass is conserved and no dimensional sweep order remains.",
      reads: "transported density and gamma",
      writes: "conditioned density and gamma",
      feeds: "surface sharpening",
    },
    toggle: {
      param: "gammaDiffusion", on: "on", off: "off",
      hint: "Toggle CM12 Sec. 3.4 gamma diffusion. Conservative transport and the sparse scalar-publication chain remain active when it is off.",
    },
    controls: [
      correctionStageControl("gammaDiffusionStrength", "gammaDiffusion"),
      correctionStageControl("gammaDiffusionIterations", "gammaDiffusion"),
    ],
    chip: (context) => context.values.gammaDiffusion === "off"
      ? "disabled · transported scalars pass through"
      : `${context.values.gammaDiffusionIterations ?? 1} passes · ${fixed(context.values.gammaDiffusionStrength ?? 1, 2)}× dose`,
  },
  "surface-sharpening": {
    label: "Surface sharpening", band: "transport", side: "right",
    phase: { id: "fine-sdf-redistance", label: "Surface sharpening stage remainder" },
    substages: {
      "sharpening-receipt-setup": {
        id: "fine-sdf-redistance", label: "Sharpening receipt/indirect setup",
      },
      "sharpening-dose": {
        id: "fine-sdf-redistance", label: "Sharpening density-gradient dose",
      },
      "sharpening-transform": {
        id: "fine-sdf-redistance", label: "Sharpening gradient trace + mass scatter",
      },
      "sharpening-finalize": {
        id: "fine-sdf-redistance", label: "Sharpening scalar finalization + dependency publication",
      },
    },
    lens: null,
    tip: {
      summary: "Sec. 3.5's density correction and Algorithm 2's local mass return on the shared transport packet authority: receipt setup, a fused dose/TEI fixed-point mass transform, then scalar finalization. Excess-density capacity repair and final-scalar publication are separate following stages.",
      reads: "transported density and gamma, solid fractions",
      writes: "conditioned density and gamma",
      feeds: "density capacity repair and scalar publication",
    },
    toggle: {
      param: "surfaceSharpening", on: "on", off: "off",
      hint: "Toggle CM12 Sec. 3.5 Algorithm 2 surface sharpening. Final sparse scalar publication remains active when it is off.",
    },
    controls: [
      correctionStageControl("sharpeningTau", "surfaceSharpening"),
      {
        kind: "param-range",
        param: "sharpeningStrength",
        label: "Sharpening strength",
        unit: "dose",
        min: 0, max: 4, step: 0.05, digits: 2,
        hint: "Multiplier of Algorithm 2's per-step removed-density dose. One is the paper dose; values above one strengthen it. Removal is limited to available density.",
        enabled: (context) => context.values.surfaceSharpening !== "off",
      },
      {
        kind: "param-range",
        param: "sharpeningDistance",
        label: "Trace distance",
        unit: "cells",
        min: 0.1, max: 3.1, step: 0.1, digits: 1,
        hint: "Algorithm 2's D. Both CM12 lanes default to the 2.1-cell reference value; the paper explores 1.1-3.1 cells and reads increasing it as surface tension.",
        enabled: (context) => context.values.surfaceSharpening !== "off",
      },
      {
        kind: "param-range",
        param: "sharpeningTraceSteps",
        label: "Trace substeps",
        unit: "substeps",
        min: 1, max: 16, step: 1, digits: 0,
        hint: "Forward-Euler substeps the trace may spend, at half a cell each. Reach is min(D, half the substeps), so at the default seven the distance is what binds and lowering these is a separate, shorter-trace ablation.",
        enabled: (context) => context.values.surfaceSharpening !== "off",
      },
    ],
    chip: (context) => context.values.surfaceSharpening === "off"
      ? "Algorithm 2 disabled · sparse publication remains"
      : `CM12 sharpening · ${fixed(context.values.sharpeningStrength, 2)} dose · D ${
        fixed(context.values.sharpeningDistance, 1)} cells · ${
        fixed(context.values.sharpeningTraceSteps, 0)} substeps`,
  },
  "density-capacity-repair": {
    label: "Density capacity repair", band: "transport", side: "left",
    phase: { id: "fine-sdf-redistance", label: "Density capacity repair stage remainder" },
    substages: {
      "density-capacity-repair": {
        id: "fine-sdf-redistance", label: "Conservative density-capacity repair",
      },
    },
    lens: null,
    toggle: {
      param: "densityCapacityRepair", on: "on", off: "off",
      hint: "Redistribute excess density independently of sharpening. Off leaves the conserved excess for pressure recovery.",
    },
    controls: [
      correctionStageControl("densityCapacityRepairStrength", "densityCapacityRepair"),
      correctionStageControl("densityCapacityRepairIterations", "densityCapacityRepair"),
    ],
    tip: {
      summary: "Relays excess mass through open neighbouring faces using paired conservative debits and credits. Strength sets the fraction moved per pass; pass count sets the available relay distance.",
      reads: "final density and solid fractions",
      writes: "redistributed density",
      feeds: "scalar publication and pressure volume recovery",
    },
    chip: context => context.values.densityCapacityRepair === "off" ? "disabled · excess retained"
      : `${context.values.densityCapacityRepairIterations ?? 8} passes · ${fixed(context.values.densityCapacityRepairStrength ?? 1, 2)}× dose`,
  },
  "scalar-publication": {
    label: "Scalar publication", band: "transport", side: "left",
    phase: { id: "other", label: "Scalar output publication" },
    substages: {
      "final-scalar-mask-publication": {
        id: "fine-sdf-redistance", label: "FSM1 final-scalar packet-mask publication",
      },
    },
    lens: null,
    slice: {
      caption: "What this advance actually changed. Only these cells enter the dirty worklists the adaptivity band walks — everything unlit is carried forward untouched.",
      keys: [
        { id: "volume-changed", tone: "output", label: "volume changed",
          note: "this advance moved volume here, so the cell enters the dirty worklists" },
      ],
      loopStep: 4,
    },
    tip: {
      summary: "Publishes final-scalar packet masks and the completed scalar output without modifying density or gamma. This publication runs even when every correction is disabled.",
      reads: "completed scalar stage coverage",
      writes: "final-scalar packet masks and frame scalar output receipt",
      feeds: "body-force prediction",
      gate: "accepted frame control",
    },
    chip: () => "scalar output receipt",
  },
  "body-forces": {
    label: "Body forces", band: "momentum", side: "left",
    phase: { id: "velocity-advection", label: "Body-force prediction" },
    lens: null,
    slice: {
      caption: "Gravity lands on the rows, not the cells — one add per row that touches liquid. Nothing else in the advance writes velocity without being projected afterwards.",
      keys: [
        { id: "row-taking-gravity", tone: "momentum", label: "row taking g·dt",
          note: "gravity lands on this row once, and the projection answers for it" },
      ],
      loopStep: 2,
    },
    tip: {
      summary: "Applies gravity and scene acceleration on the accepted face rows before projection.",
      reads: "transported face velocity, scene acceleration",
      writes: "predicted face velocity",
      feeds: "pressure RHS",
    },
    chip: () => "accepted face rows",
  },
  "pressure-topology": {
    label: "Pressure topology", band: "pressure", side: "left",
    phase: { id: "pressure-system", label: "Pressure topology stage remainder" },
    substages: {
      "ptr-setup-brick-plan": {
        id: "pressure-system", label: "PTR setup, seed and brick plan",
      },
      "pcm-cell-publication": {
        id: "pressure-system", label: "PCM canonical cell publication",
      },
      "pcm-row-publication": {
        id: "pressure-system", label: "Direct pressure-row membership publication",
      },
      "pca-fine-publication": {
        id: "pressure-system", label: "PEI direct coefficients + PCA frontier",
      },
      "pca-coarse-repair": {
        id: "pressure-system", label: "PCA brick + aggregate-edge repair",
      },
      "pca-hierarchy-and-freeze": {
        id: "pressure-system", label: "PCA hierarchy repair + frozen coarse publication",
      },
      "pei-publication": {
        id: "pressure-system", label: "PEI canonical pressure publication",
      },
      "ptr-commit-and-prepare-pressure": {
        id: "pressure-system", label: "PTR commit, reopen + pressure preparation",
      },
    },
    lens: null,
    slice: {
      caption: "The compact leaf set this solve runs on. The repair is incremental — seeded from the previous generation, walked over dirty worklists — but the classify pass is still a full accepted-cell scan.",
      keys: [
        { id: "pressure-cell", tone: "pressure", label: "pressure cell",
          note: "over half full, so this cell is a leaf in the set the solve runs on" },
        { id: "two-to-one-port", tone: "adaptivity", label: "2:1 port",
          note: "this brick meets a horizontal neighbour a rung away; the seam is a 2:1 port" },
      ],
      loopStep: 2,
    },
    tip: {
      summary: "Incremental. Seeds the persistent pressure cache and the bounded topology repair from the prior accepted generation, classifies only changed cells and rows — bootstrap plus dirty worklists — with ghost-fluid theta at sparse air, repairs the brick, aggregate-edge and hierarchy caches, and assembles one symmetric GᵀWG operator across regular and 2:1 faces. Its timestamp is attributed to the topology generation accepted at the end of the prior advance; this advance's later commit is reported separately as next-frame input.",
      reads: "prior end-frame topology receipt, conditioned atlas, density-derived phi, temporal cell/row worklists and matched PCM generations",
      writes: "active rows, diagonal, nullspace components, repaired pressure cache",
      feeds: "this advance's RHS construction; the later topology commit feeds the next advance instead",
    },
    chip: (context) => adaptiveMassPressureTopologyChip(context.info),
  },
  "pressure-rhs": {
    label: "RHS + PCG initialization", band: "pressure", side: "right",
    phase: {
      id: "pressure-system",
      label: "Finite-volume divergence RHS + compatibility projection",
    },
    lens: null,
    slice: {
      caption: "Divergence of the extended face field, one row per canonical incidence. Blue is compressing, red expanding; a converged solve drives every one of them to zero.",
      keys: [
        { id: "negative-divergence", tone: "pressure", label: "negative divergence",
          note: "the cell is compressing; a converged solve drives this to zero" },
        { id: "positive-divergence", tone: "alarm", label: "positive divergence",
          note: "the cell is expanding; a converged solve drives this to zero" },
      ],
      loopStep: 2,
    },
    tip: {
      summary: "Builds the finite-volume divergence RHS from predicted face flux, geometric source volume and moving-solid capacity change, with enclosed components projected onto their compatible quotient space. It applies the brick-aggregate + hierarchy preconditioner once for the initial direction, reduces the initial true residual and primes the pipelined image the solve iterates on.",
      reads: "predicted face velocity, active pressure rows, pressure cache, source and solid-capacity rates",
      writes: "compatible RHS, initial direction, pipelined solver image",
      feeds: "sparse MGPCG",
    },
    chip: () => "compatible flux · sources + solid-volume change",
  },
  "pressure-solve": {
    label: "Pressure solve", band: "pressure", side: "left",
    phase: { id: "pressure-solve", label: "One-reduction sparse MGPCG pressure solve" },
    lens: null,
    slice: {
      caption: "The solved pressure. One reduction per iteration, a single positive Jacobi diagonal as the preconditioner, and a true-residual guard every eighth iteration — the tail stays encoded whether or not it has converged.",
      keys: [
        { id: "high-pressure", tone: "pressure", label: "high pressure",
          note: "solved pressure, above the free-surface value" },
        { id: "free-surface", tone: "ground", label: "free surface · p = 0",
          note: "water with no solved pressure over it: the Dirichlet value the surface is pinned to" },
      ],
      loopStep: 2,
    },
    tip: {
      summary: "Pipelined conjugate gradient with one uniform positive Jacobi preconditioner and one reduction per iteration. A guarded true-residual reduction after each fixed eight-iteration block gates later arithmetic or restarts the direction after curvature loss, and a final true residual closes the stage.",
      reads: "canonical incidence rows, diagonal, compatible RHS",
      writes: "compact leaf pressure, residual receipts",
      feeds: "velocity projection",
    },
    controls: [
      {
        kind: "param-range",
        param: "pressureIterations",
        label: "Iteration budget",
        unit: "iterations",
        min: 8, max: 256, step: 8, digits: 0,
        hint: "Maximum pipelined PCG iterations. Reducing this is the most direct pressure/frame-time tradeoff.",
      },
      {
        kind: "param-range",
        param: "pressureRelativeTolerance",
        label: "Early-stop residual",
        unit: "rel. L2",
        min: 0, max: 1, step: 0.001, digits: 3,
        hint: "Checks a fresh relative L2 residual every eight iterations. Once met, the fixed tail dispatches remain encoded but skip their arithmetic. Zero runs every budgeted iteration.",
      },
      {
        kind: "readout",
        label: "Iterations executed / encoded",
        value: (context) => adaptiveMassPressureIterationReadout(
          context.info, context.values.pressureIterations),
        hint: "Iterations that performed solver arithmetic in the latest queue-confirmed frame, followed by that frame's adaptive encoded ceiling. The slider remains the hard maximum.",
      },
    ],
    chip: (context) => {
      const tolerance = number(context.values.pressureRelativeTolerance) ?? 0;
      const iterations = fixed(context.values.pressureIterations, 0);
      const executed = context.info?.pressureIterationsExecuted;
      const encoded = context.info?.pressureIterationsEncoded ?? iterations;
      const work = executed === undefined ? `${iterations} max`
        : `${executed}/${encoded} PCG iterations`;
      return tolerance > 0
        ? `${work} · rel ${tolerance.toFixed(3)}`
        : `${work} · fixed`;
    },
  },
  "velocity-projection": {
    label: "Velocity projection", band: "pressure", side: "right",
    phase: {
      id: "velocity-projection",
      label: "Projected transport velocity extension",
    },
    substages: {
      "projection-faces": {
        id: "velocity-projection",
        label: "Composite pressure-gradient face projection + divergence",
      },
      "projected-frontier-commit": {
        id: "velocity-projection",
        label: "Projected transport frontier commit",
      },
      "projected-velocity-extension": { id: "velocity-projection", label: "Projected velocity extension" },
      "air-band-correction": { id: "velocity-projection", label: "Air-band velocity correction" },
      "momentum-snapshot": { id: "velocity-projection", label: "Pre-remesh momentum snapshot" },
      "projected-topology-rebuild": {
        id: "velocity-projection",
        label: "Projected commit compiled-topology rebuild",
      },
    },
    lens: null,
    slice: {
      caption: "Grey is the field entering the projection, amber the divergence-free field leaving it. The difference is the pressure gradient, applied one row at a time.",
      keys: [
        { id: "before-projection", tone: "muted", label: "before projection",
          note: "the grey arrow: the field entering the projection, gravity already in it" },
        { id: "after-projection", tone: "amber", label: "after projection",
          note: "the amber arrow: the divergence-free field leaving it" },
      ],
      loopStep: 2,
    },
    tip: {
      summary: "Advances the incremental-activity clock, then projects the compiled dirty/pressure row masks directly through the same composite rows that built the divergence, conservative 2:1 ports and sparse-air boundaries included. Collocation publishes divergence maxima during its existing incidence traversal; rigid-body reaction and frame face output follow.",
      reads: "predicted face velocity, pressure, dirty bricks",
      writes: "projected face and collocated velocity, divergence receipts, frame face output",
      feeds: "activity measurement and the next frame's velocity extension",
    },
    controls: [algorithmStageControl("airExtension")],
    chip: (context) => context.info?.maxDivergenceAfter_s === undefined
      ? "G/D shared rows · touched faces only"
      : `|div|∞ ${context.info.maxDivergenceAfter_s.toExponential(2)} s⁻¹`,
  },
  "activity-measurement": {
    label: "Activity census + frontier", band: "adaptivity", side: "left",
    phase: {
      id: "power-topology",
      label: "Sparse-world frontier page synthesis",
    },
    substages: {
      "dirty-brick-mask-publication": {
        id: "power-topology",
        label: "Incremental activity dirty-brick mask publication",
      },
      "brick-activity-measurement": {
        id: "power-topology",
        label: "Brick activity measurement and curvature",
      },
      "brick-activity-census-and-history": {
        id: "power-topology",
        label: "Brick activity census and history",
      },
      "sparse-world-frontier-allocation": {
        id: "power-topology",
        label: "Sparse-world frontier allocation",
      },
    },
    timedWork: activityTimedWork,
    lens: null,
    slice: {
      caption: "One score per brick, from interface presence and peak speed. This is the only number the resolution policy reads — geometry and motion, never an authored region.",
      keys: [
        { id: "high-activity", tone: "adaptivity", label: "high activity",
          note: "interface presence and peak speed, scored; the only number the resolution policy reads" },
      ],
      loopStep: 4,
    },
    tip: {
      summary: "This interval is larger than its historical ‘activity measurement’ name implied. It publishes scalar/topology masks, measures and ages brick activity, seals the census and, when dynamic sparse-world growth is enabled, scans and synthesizes frontier pages.",
      reads: "conditioned density, momentum, previous records, dirty-brick worklist",
      writes: "activity masks, per-brick score/history and discovered sparse-world pages",
      feeds: "candidate topology planning",
    },
    chip: (context) => `${context.info?.adaptiveActivityMeasuredBrickCount ?? 0} measured · masks/history/frontier included`,
  },
  "resolution-planning": {
    label: "Candidate topology build", band: "adaptivity", side: "right",
    phase: {
      id: "power-topology",
      label: "Shadow row, leaf and structure worklist construction",
    },
    substages: {
      "liquid-frontier-classification": {
        id: "power-topology",
        label: "Accepted-liquid frontier classification",
      },
      "refinement-policy-classification": {
        id: "power-topology",
        label: "Refinement-policy tile classification",
      },
      "initial-resolution-plan": {
        id: "power-topology",
        label: "Initial resolution plan",
      },
      "frontier-activation-and-retirement": {
        id: "power-topology",
        label: "Frontier activation and retirement",
      },
      "resolution-grading-and-validation": {
        id: "power-topology",
        label: "Repeated 2:1 grading and candidate validation",
      },
      "candidate-page-allocation-and-synthesis": {
        id: "power-topology",
        label: "Schedule backed candidate topology",
      },
    },
    timedWork: candidatePlanTimedWork,
    lens: null,
    slice: {
      caption: "The activity score becomes a target rung on the dyadic ladder — 1, 2, 4 or 8 cells per brick edge — then 2:1 grading pulls in any neighbour sitting more than one rung away.",
      keys: [
        { id: "target-rung", tone: "adaptivity", label: "target rung",
          note: "the rung this brick is planned to carry, after 2:1 grading pulled its neighbours in" },
      ],
      loopStep: 4,
    },
    tip: {
      summary: "This is a candidate-topology construction interval, not just a policy decision. It scores and grades resolutions, consumes generation-stamped surface-output proofs, activates and retires pages, schedules the budget, allocates and synthesizes candidate cells, builds shadow row/leaf/structure worklists and publishes five indirect command copies for the following transaction.",
      reads: "transported density, momentum, policy history and accepted surface-output proofs",
      writes: "score/reason/proof history, urgent/ordinary queues, candidate levels, shadow worklists",
      feeds: "candidate transfer",
    },
    controls: [
      {
        kind: "readout", label: "Live topology",
        value: (context) => `GPU TOPOLOGY GEN ${context.info?.adaptiveTopologyShadowGeneration ?? 0}`,
        hint: "Accepted GPU-owned cell, pressure-row and field generation. Frame submission stays synchronous. Larger topology changes are prepared between accepted frames.",
      },
      adaptivityStageControl("selectorMode"),
      adaptivityStageControl("energyThreshold"),
      adaptivityStageControl("curvatureTolerance"),
      adaptivityStageControl("surfaceQuietEpochs"),
      adaptivityStageControl("surfaceFineRings"),
      adaptivityStageControl("finestTravelCells"),
      adaptivityStageControl("fourTravelCells"),
      adaptivityStageControl("twoTravelCells"),
      adaptivityStageControl("frontLookaheadSteps"),
      adaptivityStageControl("thinFeatureCells"),
      adaptivityStageControl("thinFeatureDensity"),
      adaptivityStageControl("residencyDensity"),
      adaptivityStageControl("residencyMassFineCells"),
      adaptivityStageControl("surfaceDensityMinimum"),
      adaptivityStageControl("detailTolerance"),
      adaptivityStageControl("surfaceDisplacementToleranceCells"),
      adaptivityStageControl("topologyCadenceSteps"),
      adaptivityStageControl("prepareBricksPerFrame"),
      adaptivityStageControl("promoteEpochs"),
      adaptivityStageControl("demoteEpochs"),
      adaptivityStageControl("promoteScore"),
      adaptivityStageControl("demoteScore"),
      adaptivityStageControl("emergencyScore"),
    ],
    chip: (context) => `${context.values.selectorMode === "coarse-first"
      ? "coarse first · energy + deformation + representability"
      : activityOnly(context)
      ? `surface proof + activity · plan every ${fixed(context.values.topologyCadenceSteps, 0)} steps`
      : "surface distance · direct 1³ bulk"} · grade/allocate/shadow · ${
      fixed(context.values.prepareBricksPerFrame, 0)}/frame`,
  },
  "candidate-transfer": {
    label: "Candidate transfer", band: "adaptivity", side: "left",
    phase: { id: "power-topology", label: "Candidate transfer stage remainder" },
    substages: {
      "candidate-field-transfer": {
        id: "power-topology", label: "Candidate scalar + momentum field transfer",
      },
      "candidate-face-reconstruction": {
        id: "power-topology", label: "Candidate exterior-face reconstruction",
      },
      "candidate-face-validation": {
        id: "power-topology", label: "Candidate shadow-face validation",
      },
      "candidate-effects-preflight": {
        id: "power-topology", label: "Candidate effects census + preflight",
      },
      "candidate-ibo-construction": {
        id: "power-topology", label: "Candidate IBO delta construction",
      },
      "candidate-ibo-validation": {
        id: "power-topology", label: "Candidate independent IBO semantic validation",
      },
      "candidate-tei-compilation": {
        id: "power-topology", label: "Candidate TEI delta compilation",
      },
      "candidate-authorization": {
        id: "power-topology", label: "Candidate transaction authorization",
      },
      "candidate-ptr-publication": {
        id: "power-topology", label: "Candidate PTR effects publication",
      },
      "candidate-effects-seal": {
        id: "power-topology", label: "Candidate effects receipt seal",
      },
      "candidate-state-publication": {
        id: "power-topology", label: "Candidate fields + membership publication",
      },
      "candidate-image-replay": {
        id: "power-topology", label: "Candidate retired-image replay",
      },
    },
    lens: null,
    slice: {
      caption: "Bricks whose rung moved this advance. The shadow topology is built beside the live one and committed as a single transaction at the frame tail — so this flip is the next advance's input, never this one's.",
      keys: [
        { id: "rung-changed", tone: "output", label: "rung changed",
          note: "this brick's rung moved; the flip commits at the frame tail, so it is the next advance's input" },
      ],
      loopStep: 4,
    },
    tip: {
      summary: "One transaction over the topology delta resolution planning built: density, gamma, momentum and exterior faces transfer into double-buffered shadow slots; shadow faces are validated; the effects census and preflight run; the interned-boundary (IBO) delta is built and independently validated and the transport execution image (TEI) shadow compiled; the transaction is authorized; PTR effects publish; the receipt seals; fields and membership publish; retired images replay. That end-frame flip is input to the next advance's pressure topology, never this advance's.",
      reads: "shadow worklists, candidate levels, accepted cell and face state",
      writes: "shadow leaf/face storage, IBO + TEI deltas, pressure-row worklists, conservation receipts and accepted-generation metadata",
      feeds: "transport, pressure, projection, diagnostics and presentation through the next frame's indirect dispatches",
    },
    chip: (context) => `${context.info?.adaptiveTopologyUrgentQueuedBrickCount ?? 0} urgent · ${context.info?.adaptiveTopologyOrdinaryQueuedBrickCount ?? 0} queued · end-frame generation ${context.info?.adaptiveTopologyShadowGeneration ?? 0} → next pressure repair`,
  },
  "brick-retirement": {
    label: "Post-commit activity mask", band: "adaptivity", side: "right",
    phase: { id: "adaptive-publication", label: "Post-topology activity-mask publication" },
    lens: null,
    slice: {
      caption: "A brick holding no liquid and no source is released back to the atlas. The hatched bricks pay nothing this frame — the sparse set is the lit region plus its band, and no more.",
      keys: [
        { id: "retired-brick", tone: "muted", label: "retired brick",
          note: "no liquid and no source: this brick is released back to the atlas" },
        { id: "resident-brick", tone: "liquid", label: "resident brick",
          note: "still holding liquid, so this brick pays for the frame" },
      ],
      loopStep: 4,
    },
    tip: {
      summary: "Marks every brick the topology commit changed in the post-topology activity mask, so the next advance's direct face and activity transforms select exactly the bricks that moved. The decision to retire an unsupported empty brick is taken in resolution planning; this stage publishes the retired and reshaped brick bits.",
      reads: "committed topology, incremental-activity state",
      writes: "post-commit generation-stamped brick mask",
      feeds: "the next advance's face preparation and activity measurement",
    },
    chip: () => "post-commit brick mask",
  },
  "presentation-publication": {
    label: "Presentation pages", band: "output", side: "left",
    controls: [{
      kind: "param-choice", param: "presentationSurface", label: "Surface reconstruction",
      options: [
        { value: "rdf", label: "Shared RDF", hint: "Publish one watertight shared distance field reconstructed from accepted VOF fractions and PLIC normals. Transport remains PLIC." },
        { value: "plic", label: "Legacy PLIC field", hint: "Publish the previous independently extended PLIC distance field for comparison. Transport remains PLIC." },
      ],
    }, {
      kind: "param-choice", param: "presentationColumnHeight", label: "Column height",
      options: [
        { value: "auto", label: "Auto", hint: "Use liquid volume to set surface height only in coarse columns filled continuously from the physical floor to a single surface. Keep the interface surface elsewhere. Applies on the next simulation step." },
        { value: "on", label: "On", hint: "Use a broader local column check, including fine columns and some floating liquid. Keep the interface surface elsewhere. Applies on the next simulation step." },
        { value: "off", label: "Off", hint: "Use the interface surface throughout. Applies on the next simulation step without resetting." },
      ],
    }, {
      kind: "param-choice", param: "surfaceMeshRefinement", label: "Mesh refinement",
      options: [
        { value: "1", label: "×1", hint: "One target subdivision per accepted surface-cell edge." },
        { value: "2", label: "×2", hint: "Two target subdivisions per accepted surface-cell edge." },
        { value: "4", label: "×4", hint: "Four target subdivisions per accepted surface-cell edge." },
      ],
    }],
    phase: { id: "adaptive-publication", label: "Encode compact sparse presentation pages" },
    lens: null,
    slice: {
      caption: "The surface the renderer receives: the PLIC segments of every interface cell, stitched across brick boundaries at whatever rung each brick happens to be carrying.",
      keys: [
        { id: "published-interface", tone: "output", label: "published interface",
          note: "a PLIC chord in this cell is part of the surface the renderer receives" },
      ],
      loopStep: 4,
    },
    tip: {
      summary: "Classifies which bricks the renderer can see, publishes their compact level-set pages in place, proves whether each accepted B8 surface remains representable at B4, and commits frame control. Nothing is expanded to a dense field and nothing crosses to the host.",
      reads: "committed sparse authority",
      writes: "compact level-set brick pages, presentation classification and generation-stamped surface proofs",
      feeds: "renderer level-set/grid consumers and the next topology plan",
    },
    chip: (context) => `${context.info?.fluidBrickResidentCount ?? 0} pages · resident`,
  },
} satisfies SparseCM12StageDeclarations);

/** The tap names a stage's lens declares, or `never` when it has no lens. */
export type SparseCM12StageTapName<Stage extends SparseCM12ResidentStageId> =
  (typeof SPARSE_CM12_STAGES)[Stage]["lens"] extends
    StageLens<infer _Id, infer _Publications, infer Taps, infer _Header, infer _Programs>
    ? keyof Taps & string
    : never;

/** One stage's declaration with its parameter erased. */
export function sparseCM12Stage(stage: SparseCM12ResidentStageId): SparseCM12AnyStageDeclaration {
  return SPARSE_CM12_STAGES[stage];
}

/**
 * The phase a sub-seam is timed under.
 *
 * Typed so a caller can only ask about a sub-seam the stage owns; the cast
 * inside is the one place the conditional declaration type is erased.
 */
export function sparseCM12SubstagePhase<Stage extends SparseCM12ResidentStageId>(
  stage: Stage,
  substage: SparseCM12ResidentSubstage<Stage>,
): GPUTimestampPhase {
  const phase = sparseCM12Stage(stage).substages?.[substage];
  if (!phase) {
    throw new Error(`Sparse Geometric (CM12) stage ${stage} declares no phase for sub-seam ${substage}`);
  }
  return phase;
}
