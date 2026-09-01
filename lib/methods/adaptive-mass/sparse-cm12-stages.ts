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
  transport: "Transport velocity + conservative CM12 transport",
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
      label: "brick activity census, D4 fold and history",
      entryPoints: Object.freeze([
        "measureBrickActivity",
        "preserveActivityHorizontalD4",
        "commitActivityHorizontalD4",
        "ageIncrementalActivityHistory",
        "finalizeIncrementalActivityCensus",
      ]),
    },
    {
      label: "sparse-world frontier allocation",
      entryPoints: Object.freeze([
        "allocateSparseWorldFrontier",
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
      label: "budget scheduling, candidate-page allocation and cell synthesis",
      entryPoints: Object.freeze([
        "scheduleTopologyPreparation",
        "allocateCandidateTopologyPages",
        "synthesizeCandidateCellPages",
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
  commandCopies: 4,
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
  context.values.selectorMode === "activity";

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
        label: "VEX2 packet-mask initialization",
      },
      "velocity-extension-sweeps": {
        id: "velocity-extrapolation",
        label: "VEX2 eight direct packet sweeps + fused commit",
      },
      "transport-packet-authority": {
        id: "velocity-extrapolation",
        label: "AEI packet-authority construction",
      },
    },
    lens: null,
    tip: {
      summary: "FCA1 seals the frame's body and boundary authority. VEX2 then initializes packet validity and runs eight direct packet sweeps over the accepted topology image; sweep 8 publishes the effective transport velocity. Last, the AEI transport packet authority is compiled from the prior frame's final-scalar masks.",
      reads: "projected face velocity, accepted topology image, prior final-scalar packet masks",
      writes: "sealed frame control, extended transport velocity cache, transport packet families",
      feeds: "face preparation and conservative transport",
    },
    chip: () => "FCA1 · VEX2 8 direct sweeps · AEI packets",
  },
  "face-preparation": {
    label: "Face preparation", band: "transport", side: "right",
    phase: { id: "power-topology", label: "Composite face preparation + oriented transport rows" },
    substages: {
      "face-support-publication": {
        id: "power-topology",
        label: "Dense face-velocity support clear + publication",
      },
      "dirty-face-row-preparation": {
        id: "power-topology",
        label: "Dirty oriented face-row preparation",
      },
    },
    lens: null,
    tip: {
      summary: "Clears face-velocity support on retired bricks, republishes per-brick face-velocity support, and prepares the oriented regular and 2:1 face-port transport rows — for dirty bricks only, so a stable submerged brick costs nothing here.",
      reads: "extended transport velocity, incremental-activity dirty bricks, composite row topology",
      writes: "oriented face transport rows",
      feeds: "coupled conservative transport",
    },
    chip: (context) => context.info
      ? `${context.info.fluidBrickResidentCount ?? 0} resident bricks · dirty rows only`
      : "dirty rows only",
  },
  "conservative-transport": {
    label: "Mass + gamma + momentum transport", band: "transport", side: "left",
    phase: {
      id: "fine-sdf-advection",
      label: "Conservative transport",
    },
    substages: {
      "transport-trace": {
        id: "fine-sdf-advection", label: "CM12 transport trace + sharpening catalog publication",
      },
      "transport-scatter": { id: "fine-sdf-advection", label: "CM12 transport deficit scatter" },
      "transport-gather": { id: "fine-sdf-advection", label: "CM12 transport conservative gather" },
    },
    lens: null,
    tip: {
      summary: "Moves density, gamma and all three momentum components through the same oriented composite face fluxes in one conservative transaction — trace, deficit scatter, conservative gather — with exact donor/receiver cancellation across 2:1 seams; momentum's share cannot be separated without double counting.",
      reads: "density, gamma, momentum, oriented face rows",
      writes: "transported density, gamma and momentum; sharpening cell catalog",
      feeds: "gamma diffusion and surface sharpening",
    },
    chip: () => "trace · scatter · gather",
  },
  "tracer-advection": {
    label: "Marker advection", band: "transport", side: "right",
    phase: { id: "other", label: "Fluid marker advection along the transport characteristic" },
    lens: null,
    tip: {
      summary: "Presentation-only markers integrated forward along the same characteristic, through the same extended transport velocity, that the conservative transport traces backward. Encoded only while the marker view is on, so this reads zero on an ordinary frame.",
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
      summary: "Sec. 3.4 step 8 as two stable immutable-snapshot iterations, each a transport-row-authority scatter followed by an accepted-cell finalize. Every composite subface contributes paired antisymmetric fixed-point rho/gamma receipts, so mass is conserved and no dimensional sweep order remains.",
      reads: "transported density and gamma",
      writes: "conditioned density and gamma",
      feeds: "surface sharpening",
    },
    toggle: {
      param: "gammaDiffusion", on: "on", off: "off",
      hint: "Toggle CM12 Sec. 3.4 gamma diffusion. Conservative transport and the sparse scalar-publication chain remain active when it is off.",
    },
    chip: (context) => context.values.gammaDiffusion === "off"
      ? "disabled · transported scalars pass through"
      : "2 × row scatter + cell resolve",
  },
  "surface-sharpening": {
    label: "Surface sharpening", band: "transport", side: "right",
    phase: { id: "fine-sdf-redistance", label: "Surface sharpening stage remainder" },
    substages: {
      "sharpening-receipt-setup": {
        id: "fine-sdf-redistance", label: "Sharpening receipt/indirect setup",
      },
      "sharpening-transform": {
        id: "fine-sdf-redistance", label: "Sharpening dose + TEI mass scatter",
      },
      "sharpening-finalize": {
        id: "fine-sdf-redistance", label: "Sharpening scalar finalization + dependency publication",
      },
      "density-capacity-repair": {
        id: "fine-sdf-redistance", label: "Conservative density-capacity repair",
      },
      "final-scalar-mask-publication": {
        id: "fine-sdf-redistance", label: "FSM1 final-scalar packet-mask publication",
      },
    },
    lens: null,
    tip: {
      summary: "Sec. 3.5's density correction and Algorithm 2's local mass return on the shared transport packet authority: receipt setup, a fused dose/TEI fixed-point mass transform, then scalar finalization. The stage then conservatively enforces per-cell open-volume capacity and publishes FSM1 final-scalar packet masks.",
      reads: "transported density and gamma, solid fractions",
      writes: "conditioned density and gamma, final-scalar packet masks",
      feeds: "symmetry authority and activity measurement",
    },
    toggle: {
      param: "surfaceSharpening", on: "on", off: "off",
      hint: "Toggle CM12 Sec. 3.5 Algorithm 2 surface sharpening. Final sparse scalar publication remains active when it is off.",
    },
    controls: [
      {
        kind: "param-range",
        param: "sharpeningStrength",
        label: "Sharpening strength",
        unit: "dose",
        min: 0, max: 1, step: 0.05, digits: 2,
        hint: "Fraction of Algorithm 2's per-step removed-density dose. One is the paper dose; reducing it tempers sharpening without disabling gamma diffusion.",
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
  "symmetry-authority": {
    label: "D4 symmetry authority", band: "transport", side: "left",
    phase: { id: "other", label: "Horizontal D4 symmetry authority" },
    lens: null,
    tip: {
      summary: "Folds the conditioned scalars onto the horizontal D4 orbit over the frame-control D4 families, so a scene authored symmetric stays bit-identical under the group action instead of drifting apart one rounding step at a time, then publishes the frame's scalar output.",
      reads: "conditioned density and gamma",
      writes: "D4-folded density and gamma, frame scalar output",
      feeds: "body-force prediction",
      gate: "the authored scene and every injected drop are horizontally D4 symmetric; otherwise the bypass family runs and the fold is a no-op",
    },
    chip: () => "horizontal orbit fold · scalar output",
  },
  "body-forces": {
    label: "Body forces", band: "momentum", side: "left",
    phase: { id: "velocity-advection", label: "Body-force prediction" },
    lens: null,
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
    tip: {
      summary: "Builds the finite-volume divergence RHS with enclosed components projected onto their compatible quotient space, applies the brick-aggregate + hierarchy preconditioner once for the initial direction, reduces the initial true residual and primes the pipelined image the solve iterates on.",
      reads: "predicted face velocity, active pressure rows, pressure cache",
      writes: "compatible RHS, initial direction, pipelined solver image",
      feeds: "sparse MGPCG",
    },
    chip: () => "D = −M⁻¹GᵀW · first preconditioner sweep",
  },
  "pressure-solve": {
    label: "Pressure solve", band: "pressure", side: "left",
    phase: { id: "pressure-solve", label: "One-reduction sparse MGPCG pressure solve" },
    lens: null,
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
    phase: { id: "velocity-projection", label: "Composite pressure-gradient projection" },
    lens: null,
    tip: {
      summary: "Advances the incremental-activity clock, then projects the compiled dirty/pressure row masks directly through the same composite rows that built the divergence, conservative 2:1 ports and sparse-air boundaries included. Collocation publishes divergence maxima during its existing incidence traversal; the face D4 fold, rigid-body reaction and frame face output follow.",
      reads: "predicted face velocity, pressure, dirty bricks",
      writes: "projected face and collocated velocity, divergence receipts, frame face output",
      feeds: "activity measurement and the next frame's velocity extension",
    },
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
      "brick-activity-census-and-history": {
        id: "power-topology",
        label: "Brick activity census, D4 fold and history",
      },
      "sparse-world-frontier-allocation": {
        id: "power-topology",
        label: "Sparse-world frontier allocation",
      },
    },
    timedWork: activityTimedWork,
    lens: null,
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
        label: "Candidate-page allocation and cell synthesis",
      },
    },
    timedWork: candidatePlanTimedWork,
    lens: null,
    tip: {
      summary: "This is a candidate-topology construction interval, not just a policy decision. It scores and grades resolutions, consumes generation-stamped surface-output proofs, activates and retires pages, schedules the budget, allocates and synthesizes candidate cells, builds shadow row/leaf/structure worklists and publishes four indirect command copies for the following transaction.",
      reads: "transported density, momentum, policy history and accepted surface-output proofs",
      writes: "score/reason/proof history, urgent/ordinary queues, candidate levels, shadow worklists",
      feeds: "candidate transfer",
    },
    controls: [
      {
        kind: "readout", label: "Live topology",
        value: (context) => `GPU TOPOLOGY GEN ${context.info?.adaptiveTopologyShadowGeneration ?? 0}`,
        hint: "Accepted GPU-owned cell, pressure-row and field generation. advanceTo never reads scheduling state back or rebuilds topology on the host.",
      },
      {
        kind: "param-choice", param: "selectorMode", label: "Criterion",
        options: [
          { value: "surface", label: "SURFACE", hint: "Surface/thin liquid is fine; submerged liquid requests 1³ and only 2:1 closure grades it." },
          { value: "activity", label: "ACTIVITY + PROOF", hint: "Calm surfaces may reach 4³ only after the accepted presentation output proves the merge; flooded deep bulk keeps the full 8/4/2/1 ladder." },
        ],
      },
      {
        kind: "param-range", param: "surfaceFineRings", label: "Initial fine band",
        unit: " bricks", min: 1, max: 8, step: 1, digits: 0,
        hint: "Structural/rebuild control: occupied face-distance rings initialized at the ladder maximum around the authored surface.",
      },
      {
        kind: "param-range", param: "finestTravelCells", label: "Finest travel",
        unit: " cells/step", min: 0.05, max: 4, step: 0.05, digits: 2,
        enabled: activityOnly,
        hint: "Maximum occupied-cell displacement needed to target the ladder maximum.",
      },
      {
        kind: "param-range", param: "fourTravelCells", label: "4³ travel",
        unit: " cells/step", min: 0, max: 2, step: 0.05, digits: 2,
        enabled: activityOnly,
        hint: "Displacement needed to retain at least 4³.",
      },
      {
        kind: "param-range", param: "twoTravelCells", label: "2³ travel",
        unit: " cells/step", min: 0, max: 1, step: 0.025, digits: 3,
        enabled: activityOnly,
        hint: "Displacement needed to retain at least 2³; slower calm bulk may target 1³.",
      },
      {
        kind: "param-range", param: "frontLookaheadSteps", label: "Front lookahead",
        unit: " steps", min: 1, max: 32, step: 1, digits: 0,
        hint: "Accepted steps swept ahead when a surface characteristic selects missing-solid world pages.",
      },
      {
        kind: "param-range", param: "thinFeatureCells", label: "Thin floor",
        unit: " cells", min: 0.25, max: 8, step: 0.25, digits: 2,
        hint: "Two-sided represented liquid thinner than this targets the ladder maximum.",
      },
      {
        kind: "param-range", param: "thinFeatureDensity", label: "Thin density",
        unit: " ρ", min: 0, max: 0.25, step: 0.005, digits: 3,
        hint: "Minimum density allowed to pin thin geometry; zero means the CM12 dry threshold.",
      },
      {
        kind: "param-range", param: "residencyDensity", label: "Region density",
        unit: " ρ", min: 0.000_01, max: 0.05, step: 0.001, digits: 3,
        hint: "Minimum cell density that keeps a sparse region populated after interface support leaves it.",
      },
      {
        kind: "param-range", param: "residencyMassFineCells", label: "Region mass",
        unit: " cells", min: 0, max: 8, step: 0.25, digits: 2,
        hint: "Integrated liquid mass needed to keep a region populated; one cell rejects subcell fragments.",
      },
      {
        kind: "param-range", param: "surfaceDensityMinimum", label: "Surface low",
        unit: " ρ", min: 0, max: 0.49, step: 0.01, digits: 2,
        hint: "Low bound of partial-density surface evidence.",
      },
      {
        kind: "param-range", param: "surfaceDensityMaximum", label: "Surface high",
        unit: " ρ", min: 0.51, max: 1, step: 0.01, digits: 2,
        hint: "High bound of partial-density surface evidence.",
      },
      {
        kind: "param-range", param: "detailTolerance", label: "Detail tolerance",
        unit: " ρ", min: 0.005, max: 0.5, step: 0.005, digits: 3,
        enabled: activityOnly,
        hint: "2x2x2 restriction error allowed before enclosed-bulk detail vetoes demotion. Surface demotion is governed separately by the accepted-output proof.",
      },
      {
        kind: "param-range", param: "surfaceDisplacementToleranceCells",
        label: "Surface displacement", unit: " cells",
        min: 0, max: 8, step: 0.05, digits: 2,
        enabled: activityOnly,
        hint: "Maximum rho=.5 edge-crossing movement accepted when proving that a B8 surface can be represented at B4.",
      },
      {
        kind: "param-range", param: "surfaceNormalToleranceDegrees",
        label: "Surface normal", unit: "°",
        min: 0, max: 90, step: 1, digits: 0,
        enabled: activityOnly,
        hint: "Maximum narrow-band normal-angle change accepted by the B8-to-B4 presentation proof.",
      },
      {
        kind: "param-range", param: "topologyCadenceSteps", label: "Epoch cadence",
        unit: " steps", min: 1, max: 32, step: 1, digits: 0,
        enabled: activityOnly,
        hint: "Accepted steps between quiet-history updates. One evaluates settling every step; each accepted merge still moves only one rung.",
      },
      {
        kind: "param-range", param: "prepareBricksPerFrame", label: "Work budget",
        unit: " bricks/frame", min: 1, max: 256, step: 1, digits: 0,
        hint: "Ordinary topology preparations started per frame. Surface/thin-fluid refinement has a separate urgent lane and does not wait behind coarsening.",
      },
      {
        kind: "param-range", param: "promoteEpochs", label: "Promote hold",
        unit: " epochs", min: 1, max: 16, step: 1, digits: 0,
        enabled: activityOnly,
        hint: "Hot epochs required for non-emergency activity promotion.",
      },
      {
        kind: "param-range", param: "demoteEpochs", label: "Demote hold",
        unit: " epochs", min: 1, max: 32, step: 1, digits: 0,
        enabled: activityOnly,
        hint: "Quiet epochs required for each one-rung bulk merge; at the surface this is an independent run of fresh accepted-output proofs.",
      },
      {
        kind: "param-range", param: "promoteScore", label: "Promote score",
        min: 0, max: 1, step: 0.025, digits: 3,
        enabled: activityOnly,
        hint: "Normalized activity needed for a hot epoch.",
      },
      {
        kind: "param-range", param: "demoteScore", label: "Demote score",
        min: 0, max: 1, step: 0.025, digits: 3,
        enabled: activityOnly,
        hint: "Maximum normalized activity allowed for a quiet epoch.",
      },
      {
        kind: "param-range", param: "emergencyScore", label: "Emergency score",
        min: 0, max: 1, step: 0.025, digits: 3,
        enabled: activityOnly,
        hint: "Normalized activity that bypasses promotion persistence.",
      },
    ],
    chip: (context) => `${activityOnly(context)
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
    phase: { id: "adaptive-publication", label: "Post-topology D4 + activity-mask publication" },
    lens: null,
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
    phase: { id: "adaptive-publication", label: "Encode compact sparse presentation pages" },
    lens: null,
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
    throw new Error(`Sparse CM12 stage ${stage} declares no phase for sub-seam ${substage}`);
  }
  return phase;
}
