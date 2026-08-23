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
  adaptivity: "Brick activity + candidate resolution",
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
   * The lens on this stage, or the written-down decision that there is none.
   * The lens's own `stage` must be this key: a lens filed under the wrong
   * stage is a type error here, not a ◎ that opens the wrong picture.
   */
  readonly lens: (AnyStageLens & { readonly stage: Stage }) | null;
  readonly tip: FluidPipelineTip;
  /** The short factual chip under the label. Never a description. */
  readonly chip: (context: FluidPipelineContext) => string;
  readonly controls?: readonly FluidStageControl[];
}

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
    + `${receipt.pcmRowAcceptedGeneration} · dirty leaves `
    + `${receipt.pcmCellDirtyLeafCount}/${receipt.pcmRowDirtyLeafCount}`
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
      label: "Transport row + sharpening cell authority finalization",
    },
    substages: {
      "transport-authority-setup": {
        id: "fine-sdf-advection", label: "CM12 transport authority setup",
      },
      "transport-trace": {
        id: "fine-sdf-advection", label: "CM12 transport trace + sharpening catalog publication",
      },
      "transport-scatter": { id: "fine-sdf-advection", label: "CM12 transport deficit scatter" },
      "transport-gather": { id: "fine-sdf-advection", label: "CM12 transport conservative gather" },
      "producer-mask-compilation": {
        id: "fine-sdf-advection", label: "AEI producer-mask seal + compilation",
      },
    },
    lens: null,
    tip: {
      summary: "Moves density, gamma and all three momentum components through the same oriented composite face fluxes in one conservative transaction — trace, deficit scatter, conservative gather — with exact donor/receiver cancellation across 2:1 seams; momentum's share cannot be separated without double counting. The trailing sub-seams seal the producer masks and compile the transport-row and VEX-root masks the following stages dispatch over.",
      reads: "density, gamma, momentum, oriented face rows",
      writes: "transported density, gamma and momentum; producer masks",
      feeds: "gamma diffusion, surface sharpening and the next VEX roots",
    },
    chip: () => "trace · scatter · gather · mask compile",
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
    chip: () => "2 × row scatter + cell resolve",
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
      "solid-excess-repair": {
        id: "fine-sdf-redistance", label: "Cut-cell solid-excess redistribution",
      },
      "final-scalar-mask-publication": {
        id: "fine-sdf-redistance", label: "FSM1 final-scalar packet-mask publication",
      },
    },
    lens: null,
    tip: {
      summary: "Sec. 3.5's density correction and Algorithm 2's local mass return on the shared transport packet authority: receipt setup, a fused dose/TEI fixed-point mass transform, then scalar finalization. The stage then redistributes cut-cell solid excess and publishes FSM1 final-scalar packet masks.",
      reads: "transported density and gamma, solid fractions",
      writes: "conditioned density and gamma, final-scalar packet masks",
      feeds: "symmetry authority and activity measurement",
    },
    controls: [
      {
        kind: "param-range",
        param: "sharpeningDistance",
        label: "Trace distance",
        unit: "cells",
        min: 0.1, max: 3.1, step: 0.1, digits: 1,
        hint: "Algorithm 2's D. The paper explores 1.1-3.1 cells and reads increasing it as surface tension; the sparse lane sits at the top of that range because a shorter trace strands removed mass on the tall side walls.",
      },
      {
        kind: "param-range",
        param: "sharpeningTraceSteps",
        label: "Trace substeps",
        unit: "substeps",
        min: 1, max: 16, step: 1, digits: 0,
        hint: "Forward-Euler substeps the trace may spend, at half a cell each. Reach is min(D, half the substeps), so at the default seven the distance is what binds and lowering these is a separate, shorter-trace ablation.",
      },
    ],
    chip: (context) => `CM12 sharpening · D ${fixed(context.values.sharpeningDistance, 1)} cells · ${
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
    phase: { id: "pressure-system", label: "Composite pressure topology + ghost-fluid rows" },
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
      summary: "Pipelined conjugate gradient with a brick-aggregate + hierarchy multigrid preconditioner and one reduction per iteration. A guarded true-residual reduction on a fixed cadence restarts the direction after curvature loss, and a final true residual closes the stage.",
      reads: "G rows, W, diagonal, compatible RHS, persistent pressure cache",
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
        min: 0, max: 0.1, step: 0.001, digits: 3,
        hint: "Stops further PCG arithmetic after the relative L2 residual reaches this value. Zero runs every budgeted iteration.",
      },
    ],
    chip: (context) => {
      const tolerance = number(context.values.pressureRelativeTolerance) ?? 0;
      const iterations = fixed(context.values.pressureIterations, 0);
      return tolerance > 0
        ? `${iterations} max · rel ${tolerance.toFixed(3)}`
        : `${iterations} PCG iterations · fixed`;
    },
  },
  "velocity-projection": {
    label: "Velocity projection", band: "pressure", side: "right",
    phase: { id: "velocity-projection", label: "Composite pressure-gradient projection" },
    lens: null,
    tip: {
      summary: "Advances the incremental-activity clock, then projects the compiled dirty/pressure row masks directly through the same composite rows that built the divergence, conservative 2:1 ports and sparse-air boundaries included. Collocation and diagnosis follow, then the face D4 fold, the rigid-body reaction and the frame face output publication.",
      reads: "predicted face velocity, pressure, dirty bricks",
      writes: "projected face and collocated velocity, frame face output",
      feeds: "projection diagnostics and the next frame's velocity extension",
    },
    chip: (context) => context.info?.maxDivergenceAfter_s === undefined
      ? "G/D shared rows · touched faces only"
      : `|div|∞ ${context.info.maxDivergenceAfter_s.toExponential(2)} s⁻¹`,
  },
  "projection-diagnostics": {
    label: "Projection diagnostics", band: "pressure", side: "left",
    phase: { id: "other", label: "Projection residual + divergence + energy receipts" },
    lens: null,
    tip: {
      summary: "Measures post-projection divergence per accepted cell and reduces the frame's physics receipts — residual, divergence, energy identity, mass and gamma conservation — into the published info record.",
      reads: "projected state and solver residual",
      writes: "GPUEulerianInfo acceptance telemetry",
    },
    chip: () => "divergence · residual · conservation",
  },
  "activity-measurement": {
    label: "Activity measurement", band: "adaptivity", side: "left",
    phase: { id: "power-topology", label: "Compact brick activity measurement" },
    lens: null,
    tip: {
      summary: "Marks scalar- and topology-dirty bricks into the incremental worklist, measures one activity record per dirty brick — occupied travel, surface and thin-feature evidence, restriction detail, the density moments the planner scores — then ages the history and seals the census.",
      reads: "conditioned density, momentum, previous records, dirty-brick worklist",
      writes: "per-brick score, reasons and epoch history",
      feeds: "resolution planning",
    },
    chip: (context) => `${context.info?.adaptiveActivityMeasuredBrickCount ?? 0} bricks measured`,
  },
  "resolution-planning": {
    label: "Resolution planning", band: "adaptivity", side: "right",
    phase: { id: "power-topology", label: "Hysteretic resolution planning + 2:1 candidate grading" },
    lens: null,
    tip: {
      summary: "Plans each brick's resolution — the default surface-distance selector keeps interfaces and thin liquid fine and sends deeply submerged bricks to the coarsest rung the 2:1-closed ladder permits; surface + activity additionally refines moving or detailed liquid — then activates swept receivers, retires unsupported empty bricks, grades the plan to 2:1 closure one pass per ladder rung, validates it, schedules budgeted topology preparation, allocates candidate pages and builds the shadow leaf and structure worklists the transfer consumes.",
      reads: "transported density, momentum, policy history",
      writes: "score/reason history, urgent/ordinary queues, candidate levels, shadow worklists",
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
          { value: "activity", label: "ACTIVITY", hint: "Free surfaces stay fine; calm flooded deep bulk merges." },
        ],
      },
      {
        kind: "param-choice", param: "receiverFloor", label: "Created floor",
        options: [
          { value: "auto", label: "AUTO", hint: "Boundary-fed dams use 4³; interior sources may use 1³." },
          { value: "1", label: "1³" }, { value: "2", label: "2³" },
          { value: "4", label: "4³" }, { value: "8", label: "8³" },
        ],
        hint: "Structural capacity bootstrap. The GPU topology scheduler subsequently splits or merges every created receiver.",
      },
      {
        kind: "param-range", param: "surfaceFineRings", label: "Initial fine band",
        unit: " bricks", min: 1, max: 8, step: 1, digits: 0,
        hint: "Structural/rebuild control: occupied face-distance rings initialized at the ladder maximum around the authored surface.",
      },
      {
        kind: "param-range", param: "receiverSupportRings", label: "Receiver reach",
        unit: " bricks", min: 1, max: 24, step: 1, digits: 0,
        hint: "Structural/rebuild control: radius of sparse receiver capacity reserved around authored liquid.",
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
        hint: "Accepted steps swept ahead when a surface characteristic selects receiver support.",
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
        hint: "2x2x2 restriction error allowed before enclosed-bulk detail vetoes demotion. Free surfaces retain their independent fine floor.",
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
        hint: "Quiet epochs required for each one-rung merge.",
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
      ? `surface + activity · plan every ${fixed(context.values.topologyCadenceSteps, 0)} steps`
      : "surface distance · direct 1³ bulk"} · ${
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
    label: "Brick retirement", band: "adaptivity", side: "right",
    phase: { id: "adaptive-publication", label: "Dry-brick retirement + retained-atlas conditioning" },
    lens: null,
    tip: {
      summary: "Marks every brick the topology commit changed into the post-topology incremental-activity worklist and seals it, so the next advance's face preparation and activity measurement visit exactly the bricks that moved. The decision to retire an unsupported empty brick is taken in resolution planning; this stage is where the retired and reshaped bricks are handed to the dirty set.",
      reads: "committed topology, incremental-activity state",
      writes: "post-commit dirty-brick worklist and its indirect arguments",
      feeds: "the next advance's face preparation and activity measurement",
    },
    chip: () => "post-commit dirty worklist",
  },
  "presentation-publication": {
    label: "Presentation pages", band: "output", side: "left",
    phase: { id: "adaptive-publication", label: "Encode compact sparse presentation pages" },
    lens: null,
    tip: {
      summary: "Classifies which bricks the renderer can see, publishes their compact level-set pages in place and commits frame control. Nothing is expanded to a dense field and nothing crosses to the host.",
      reads: "committed sparse authority",
      writes: "compact level-set brick pages and presentation classification",
      feeds: "renderer level-set and grid consumers",
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
