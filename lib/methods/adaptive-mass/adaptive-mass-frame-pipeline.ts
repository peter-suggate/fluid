import type {
  FluidPipelineGraph,
  FluidPipelineStage,
} from "../../core/fluid-pipeline";
import { gpuPhysicsPerformanceActivityFrameId } from "../../core/performance-activity";
import {
  CPUPerformanceTrace,
  GPUStageTimestampRecorder,
  partitionPerformanceTrace,
  type GPUTimestampPhase,
  type PerformanceTrace,
} from "../../core/performance-trace";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import {
  SPARSE_CM12_RESIDENT_FINAL_STAGE,
  SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
  SPARSE_CM12_SHARPENING_TRACE_STEPS,
  type SparseCM12ResidentStageId,
  type SparseCM12ResidentStageSeams,
} from "./webgpu-sparse-cm12-resident";

/** The capture cadence matches the uniform reference's observatory lane. */
export const ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS = 100;

/**
 * The advance partition: one phase per resident stage, in encode order.
 *
 * Both lanes emit exactly these labels. The GPU boundary chain closes a stage
 * on the pass that opens its successor, so a phase here is that stage's own
 * hardware execution time; the CPU chain closes the same stage at the same
 * seam over host command construction, and carries the panel when a device has
 * no timestamp queries. A diagram node names its label, which is what keeps
 * the picture and the measurement from drifting apart.
 *
 * `commandEncoding` and `upload` bracket the chain on the CPU lane only —
 * planning before the first stage and closure after the last. No GPU work
 * lives in either, so no diagram node claims them.
 */
export const ADAPTIVE_MASS_ADVANCE_PHASE = Object.freeze({
  commandEncoding: {
    id: "command-encoding",
    label: "Advance planning + trace setup",
  },
  velocityExtension: {
    id: "velocity-extrapolation",
    label: "Transport velocity extension into the sparse air band",
  },
  faceTopology: {
    id: "power-topology",
    label: "Composite face preparation + oriented transport rows",
  },
  coupledTransport: {
    id: "fine-sdf-advection",
    label: "Coupled conservative mass + gamma + momentum transport",
  },
  gammaDiffusion: {
    id: "fine-sdf-advection",
    label: "Gamma diffusion forward and reverse axis sweeps",
  },
  surfaceConditioning: {
    id: "fine-sdf-redistance",
    label: "CM12 surface sharpening + gamma conditioning",
  },
  symmetryAuthority: {
    id: "other",
    label: "Horizontal D4 symmetry authority",
  },
  force: {
    id: "velocity-advection",
    label: "Body-force prediction",
  },
  pressureTopology: {
    id: "pressure-system",
    label: "Composite pressure topology + ghost-fluid rows",
  },
  pressureRhs: {
    id: "pressure-system",
    label: "Finite-volume divergence RHS + compatibility projection",
  },
  pressureSolve: {
    id: "pressure-solve",
    label: "Matrix-free Jacobi-PCG pressure solve",
  },
  pressureProjection: {
    id: "velocity-projection",
    label: "Composite pressure-gradient projection",
  },
  pressureDiagnostics: {
    id: "other",
    label: "Projection residual + divergence + energy receipts",
  },
  activityMeasurement: {
    id: "power-topology",
    label: "Compact brick activity measurement",
  },
  resolutionPlanning: {
    id: "power-topology",
    label: "Hysteretic resolution planning + 2:1 candidate grading",
  },
  candidateTransfer: {
    id: "power-topology",
    label: "Candidate cell and face conservative transfer",
  },
  retentionConditioning: {
    id: "adaptive-publication",
    label: "Dry-brick retirement + retained-atlas conditioning",
  },
  materialization: {
    id: "adaptive-publication",
    label: "Encode compact sparse presentation pages",
  },
  upload: {
    id: "adaptive-publication",
    label: "Command closure + queue submission",
  },
  queueCompletion: {
    id: "adaptive-publication",
    label: "GPU-resident frame queue completion",
  },
} satisfies Record<string, GPUTimestampPhase>);

/**
 * Resident stage id to trace phase. Exhaustive by type: a stage added to the
 * encoder's partition does not compile until it has a phase to be timed under.
 */
export const ADAPTIVE_MASS_RESIDENT_STAGE_PHASE: Readonly<Record<
  SparseCM12ResidentStageId,
  GPUTimestampPhase
>> = Object.freeze({
  "transport-velocity-extension": ADAPTIVE_MASS_ADVANCE_PHASE.velocityExtension,
  "face-preparation": ADAPTIVE_MASS_ADVANCE_PHASE.faceTopology,
  "conservative-transport": ADAPTIVE_MASS_ADVANCE_PHASE.coupledTransport,
  "gamma-diffusion": ADAPTIVE_MASS_ADVANCE_PHASE.gammaDiffusion,
  "surface-sharpening": ADAPTIVE_MASS_ADVANCE_PHASE.surfaceConditioning,
  "symmetry-authority": ADAPTIVE_MASS_ADVANCE_PHASE.symmetryAuthority,
  "body-forces": ADAPTIVE_MASS_ADVANCE_PHASE.force,
  "pressure-topology": ADAPTIVE_MASS_ADVANCE_PHASE.pressureTopology,
  "pressure-rhs": ADAPTIVE_MASS_ADVANCE_PHASE.pressureRhs,
  "pressure-solve": ADAPTIVE_MASS_ADVANCE_PHASE.pressureSolve,
  "velocity-projection": ADAPTIVE_MASS_ADVANCE_PHASE.pressureProjection,
  "projection-diagnostics": ADAPTIVE_MASS_ADVANCE_PHASE.pressureDiagnostics,
  "activity-measurement": ADAPTIVE_MASS_ADVANCE_PHASE.activityMeasurement,
  "resolution-planning": ADAPTIVE_MASS_ADVANCE_PHASE.resolutionPlanning,
  "candidate-transfer": ADAPTIVE_MASS_ADVANCE_PHASE.candidateTransfer,
  "brick-retirement": ADAPTIVE_MASS_ADVANCE_PHASE.retentionConditioning,
  "presentation-publication": ADAPTIVE_MASS_ADVANCE_PHASE.materialization,
});

export interface AdaptiveMassFrameCaptureResult {
  readonly cpuTrace: PerformanceTrace;
  /**
   * The hardware boundary chain, when the device could carry one. `undefined`
   * resolves mean the sample came back undecodable and the caller should stop
   * asking this device for hardware boundaries.
   */
  readonly hardwareTrace?: Promise<PerformanceTrace | undefined>;
  /** Always observed, and the answer whenever the chain above has none. */
  readonly queueTrace: Promise<PerformanceTrace | undefined>;
  readonly identity: NonNullable<GPUEulerianInfo["physicsCaptureIdentity"]>;
}

/**
 * One advance, observed three ways from a single set of seams.
 *
 * The seams are the resident encoder's own stage partition, so all three lanes
 * describe the same boundaries: hardware timestamps spliced into the frame's
 * passes, host command-construction intervals, and the queue-wall interval
 * that still covers the advance when a device has no timestamp queries. The
 * recorder owns no physics and is safe to omit entirely.
 */
export class AdaptiveMassFrameCapture {
  private readonly cpu: CPUPerformanceTrace;
  private gpu?: GPUStageTimestampRecorder;
  private encoder?: GPUCommandEncoder;
  private queueStartedAt_ms?: number;
  private closed = false;

  constructor(
    readonly sampleId: number,
    readonly context: string,
    private readonly clock: () => number = () => performance.now(),
  ) {
    this.cpu = new CPUPerformanceTrace(
      sampleId,
      context,
      ADAPTIVE_MASS_ADVANCE_PHASE.commandEncoding,
      clock,
    );
  }

  /**
   * Adopt the frame's encoder, and the hardware chain when one was supplied.
   *
   * The returned encoder must be used everywhere the raw one went: stage
   * boundaries ride the passes opened through it, which is what makes them
   * free. Passing no recorder leaves the encoder untouched.
   */
  instrument(
    encoder: GPUCommandEncoder,
    gpu?: GPUStageTimestampRecorder,
  ): GPUCommandEncoder {
    if (this.encoder) throw new Error("Sparse CM12 frame capture already instrumented");
    this.gpu = gpu;
    this.encoder = gpu ? gpu.instrument(encoder) : encoder;
    gpu?.begin();
    this.cpu.completePhase(ADAPTIVE_MASS_ADVANCE_PHASE.commandEncoding);
    return this.encoder;
  }

  /**
   * The encoder's stage seams: one stage closes on every lane at once.
   *
   * The final stage is the exception. Its hardware boundary is armed before
   * its pass begins so that pass's own end-of-pass counter closes the chain;
   * a marker pass after it measures whenever the driver felt like running it.
   * The host lane has no such hazard and closes it at the seam like any other.
   */
  readonly residentStageSeams: SparseCM12ResidentStageSeams = {
    close: (stage) => {
      const phase = ADAPTIVE_MASS_RESIDENT_STAGE_PHASE[stage];
      this.cpu.completePhase(phase);
      if (this.encoder && stage !== SPARSE_CM12_RESIDENT_FINAL_STAGE) {
        this.gpu?.completePhase(this.encoder, phase);
      }
    },
    openFinal: (stage) => {
      this.gpu?.completeFinalPhaseOnNextPass(ADAPTIVE_MASS_RESIDENT_STAGE_PHASE[stage]);
    },
  };

  /** Stage the query readback and start the queue clock, just before submit. */
  closeCommands(): void {
    if (this.queueStartedAt_ms !== undefined) {
      throw new Error("Sparse CM12 queue capture already started");
    }
    if (this.encoder) this.gpu?.resolve(this.encoder);
    this.queueStartedAt_ms = this.clock();
  }

  /** Close after the command buffer has been submitted. */
  finish(queue: Pick<GPUQueue, "onSubmittedWorkDone">): AdaptiveMassFrameCaptureResult {
    if (this.closed) throw new Error("Sparse CM12 frame capture already closed");
    if (this.queueStartedAt_ms === undefined) {
      throw new Error("Sparse CM12 queue capture was not started");
    }
    this.closed = true;
    this.cpu.completePhase(ADAPTIVE_MASS_ADVANCE_PHASE.upload);
    const cpuTrace = this.cpu.finishCompletedPhases();
    const queueStartedAt_ms = this.queueStartedAt_ms;
    const queueTrace = queue.onSubmittedWorkDone().then(() => {
      const end_ms = Math.max(queueStartedAt_ms, this.clock());
      return partitionPerformanceTrace({
        sampleId: this.sampleId,
        domain: "gpu",
        lane: "physics",
        context: `${this.context}:queue-wall-fallback`,
        measurementSource: "gpu-queue-wall",
        capturedAt_ms: end_ms,
        start_ms: queueStartedAt_ms,
        end_ms,
        intervals: [{
          ...ADAPTIVE_MASS_ADVANCE_PHASE.queueCompletion,
          start_ms: queueStartedAt_ms,
          end_ms,
        }],
      });
    // The hardware chain usually wins, leaving this observation unread; a
    // rejection here must not surface as an unhandled one.
    }).catch(() => undefined);
    return {
      cpuTrace,
      ...(this.gpu ? { hardwareTrace: this.gpu.read() } : {}),
      queueTrace,
      identity: {
        sampleId: this.sampleId,
        context: this.context,
        frameId: gpuPhysicsPerformanceActivityFrameId({
          sampleId: this.sampleId,
          context: this.context,
        }),
      },
    };
  }
}

const alwaysOn = () => "on" as const;
const stage = (
  definition: Omit<FluidPipelineStage, "state">,
): FluidPipelineStage => ({ ...definition, state: alwaysOn });

const ADAPTIVE_MASS_FLUID_STAGES: readonly FluidPipelineStage[] = [
  stage({
    id: "transport-velocity-extension", band: "transport", side: "left",
    label: "Velocity extension",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.velocityExtension.label],
    tip: {
      summary: "Seeds the transport velocity from the projected liquid field and extends it outward through eight alternating source/destination sweeps, so a semi-Lagrangian trace that leaves the liquid still lands on a defined velocity.",
      reads: "projected collocated velocity, density",
      writes: "transport velocity over liquid and its sparse air band",
      feeds: "composite face preparation",
    },
    chip: () => "8 alternating sweeps",
  }),
  stage({
    id: "receiver-topology", band: "transport", side: "right",
    label: "Face preparation",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.faceTopology.label],
    tip: {
      summary: "Resolves the extended velocity onto the canonical regular and 2:1 face-port rows this advance transports through, so donor and receiver read one oriented flux rather than two independent samples.",
      reads: "extended transport velocity, composite row topology",
      writes: "oriented face transport rows",
      feeds: "coupled conservative transport",
    },
    chip: (context) => context.info
      ? `${context.info.fluidBrickResidentCount ?? 0} resident bricks`
      : "resident + one-face receivers",
  }),
  stage({
    id: "scalar-transport", band: "transport", side: "left",
    label: "Mass + gamma transport",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.coupledTransport.label],
    tip: {
      summary: "Moves density integral and gamma integral through the same oriented composite face fluxes, with exact donor/receiver cancellation across 2:1 seams.",
      reads: "density, gamma, face velocity",
      writes: "transported density and gamma",
      feeds: "retention and pressure classification",
    },
    chip: (context) => `${context.info?.lastSubsteps ?? 1} CFL substep${context.info?.lastSubsteps === 1 ? "" : "s"}`,
  }),
  stage({
    id: "velocity-transport", band: "transport", side: "right",
    label: "Momentum transport",
    phaseLabels: [],
    costInsideStage: "scalar-transport",
    tip: {
      summary: "Transports all three momentum components in the same conservative face transaction as density and gamma; its cost cannot be separated without double counting.",
      reads: "cell momentum, shared face sweep",
      writes: "transported collocated velocity",
      feeds: "body-force prediction",
    },
    chip: () => "shared conservative transaction",
  }),
  stage({
    id: "gamma-diffusion", band: "transport", side: "left",
    label: "Gamma diffusion",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.gammaDiffusion.label],
    tip: {
      summary: "Sec. 3.4 step 8: Jacobi within an axis, Gauss-Seidel between axes. Three forward and three reverse sweeps are averaged into one order-independent result, which is what keeps the diffused gamma from inheriting the sweep direction.",
      reads: "transported gamma",
      writes: "diffused gamma",
      feeds: "surface sharpening",
    },
    chip: () => "6 axis sweeps · averaged",
  }),
  stage({
    id: "surface-conditioning", band: "transport", side: "right",
    label: "Surface conditioning",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.surfaceConditioning.label],
    tip: {
      summary: "Applies the shared CM12 surface-conditioning transaction on compact resident leaves before activity is measured. Sec. 3.5's density correction and Algorithm 2's local mass return are one dispatch pair here rather than the uniform lane's two separable stages, so both trace controls live on this stage.",
      reads: "transported density and gamma",
      writes: "conditioned density and gamma",
      feeds: "activity measurement and resolution planning",
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
    chip: (context) => `CM12 sharpening · D ${Number(
      context.values.sharpeningDistance ?? SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
    ).toFixed(1)} cells · ${Number(
      context.values.sharpeningTraceSteps ?? SPARSE_CM12_SHARPENING_TRACE_STEPS,
    ).toFixed(0)} substeps`,
  }),
  stage({
    id: "symmetry-authority", band: "transport", side: "left",
    label: "D4 symmetry authority",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.symmetryAuthority.label],
    tip: {
      summary: "Folds the conditioned scalars onto the horizontal D4 orbit so a scene authored symmetric stays bit-identical under the group action instead of drifting apart one rounding step at a time.",
      reads: "conditioned density and gamma",
      writes: "D4-folded density and gamma",
      feeds: "body-force prediction",
      gate: "the authored scene and every injected drop are horizontally D4 symmetric",
    },
    chip: () => "horizontal orbit fold",
  }),
  stage({
    id: "activity-measurement", band: "adaptivity", side: "left",
    label: "Activity measurement",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.activityMeasurement.label],
    tip: {
      summary: "Advances the accepted-step clock and measures one activity record per compact brick: occupied travel, surface evidence, thin-feature evidence, restriction detail and the density moments the planner scores.",
      reads: "conditioned density, momentum, previous records",
      writes: "per-brick score, reasons and epoch history",
      feeds: "resolution planning",
    },
    chip: (context) => `${context.info?.adaptiveActivityMeasuredBrickCount ?? 0} bricks measured`,
  }),
  stage({
    id: "resolution-planning", band: "adaptivity", side: "right",
    label: "Resolution planning",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.resolutionPlanning.label],
    tip: {
      summary: "Measures compact brick activity every accepted step and authors full 1³/2³/4³/8³ candidate requests. Surface, thin fluid, fast bulk, and predicted fronts target fine; calm bulk demotes hysteretically. Accepted in-run topology publication is not implemented yet, so structural receiver controls remain the physical guard.",
      reads: "transported density, momentum, policy history",
      writes: "score/reason history and candidate 1³/2³/4³/8³ levels",
      feeds: "candidate conservative transfer (publication pending)",
    },
    chip: (context) => `${context.info?.adaptiveFineBrickCount ?? 0} fine · ${context.info?.adaptiveCoarseBrickCount ?? 0} coarse`,
  }),
  stage({
    id: "candidate-transfer", band: "adaptivity", side: "left",
    label: "Candidate transfer",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.candidateTransfer.label],
    tip: {
      summary: "Restricts and prolongs cells and faces into each brick's candidate resolution and writes the mass, gamma and momentum error receipts that decide whether that candidate could be accepted.",
      reads: "candidate levels, conditioned cell and face state",
      writes: "candidate leaf storage and per-brick transfer receipts",
      feeds: "dry-brick retirement",
    },
    chip: (context) => `${context.info?.adaptiveActivityHotBrickCount ?? 0} hot · ${context.info?.adaptiveActivityQuietBrickCount ?? 0} quiet`,
  }),
  stage({
    id: "retention-conditioning", band: "adaptivity", side: "right",
    label: "Retain + condition",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.retentionConditioning.label],
    tip: {
      summary: "Retires unsupported dry bricks logically and clears their stale fields. Packed cell/row capacity remains immutable; this stage does not publish candidate resolutions as a rebuilt accepted atlas.",
      reads: "transported density/gamma/momentum",
      writes: "active-bit history, cleared residue, and retirement receipts",
      feeds: "forces and projection",
    },
    chip: () => "zero persistent dry bricks",
  }),
  stage({
    id: "body-forces", band: "momentum", side: "left",
    label: "Body forces",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.force.label],
    tip: {
      summary: "Applies gravity and future acceleration terms on compact resident cells and composite faces before projection.",
      reads: "advected velocity, scene acceleration",
      writes: "predicted velocity",
      feeds: "pressure RHS",
    },
    chip: () => "compact resident faces",
  }),
  stage({
    id: "pressure-topology", band: "pressure", side: "left",
    label: "Pressure topology",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.pressureTopology.label],
    tip: {
      summary: "Classifies liquid rows, applies ghost-fluid theta at sparse air, and assembles one symmetric GᵀWG operator across regular and 2:1 faces.",
      reads: "conditioned atlas and density-derived phi",
      writes: "active rows, diagonal, nullspace components",
      feeds: "RHS construction",
    },
    chip: (context) => `${context.info?.adaptiveMixedSeamFaceCount ?? 0} mixed rows`,
  }),
  stage({
    id: "pressure-rhs", band: "pressure", side: "right",
    label: "Divergence RHS",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.pressureRhs.label],
    tip: {
      summary: "Builds the finite-volume divergence RHS and projects enclosed components onto their exact compatible quotient space.",
      reads: "predicted face velocity, active pressure rows",
      writes: "compatible pressure RHS",
      feeds: "Jacobi-PCG",
    },
    chip: () => "D = −M⁻¹GᵀW",
  }),
  stage({
    id: "pressure-solve", band: "pressure", side: "left",
    label: "Pressure solve",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.pressureSolve.label],
    tip: {
      summary: "Solves the matrix-free symmetric composite pressure system with a Jacobi-preconditioned conjugate-gradient iteration.",
      reads: "G rows, W, diagonal, compatible RHS",
      writes: "compact leaf pressure",
      feeds: "velocity projection",
    },
    chip: (context) => `${context.info?.pressureIterations ?? 0} PCG iterations`,
  }),
  stage({
    id: "pressure-projection", band: "pressure", side: "right",
    label: "Velocity projection",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.pressureProjection.label],
    tip: {
      summary: "Subtracts the pressure correction through the same composite face rows used to build divergence, including conservative 2:1 ports.",
      reads: "predicted velocity, pressure",
      writes: "projected face and collocated velocity",
      feeds: "persistent next-step authority",
    },
    chip: (context) => context.info?.maxDivergenceAfter_s === undefined
      ? "G/D shared rows"
      : `|div|∞ ${context.info.maxDivergenceAfter_s.toExponential(2)} s⁻¹`,
  }),
  stage({
    id: "projection-receipts", band: "pressure", side: "left",
    label: "Physics receipts",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.pressureDiagnostics.label],
    tip: {
      summary: "Publishes pressure residual, post-projection divergence, energy identity, mass/gamma conservation and sparse work receipts from the committed authority.",
      reads: "projected state and solver residual",
      writes: "GPUEulerianInfo acceptance telemetry",
    },
    chip: () => "mass · gamma · residual · divergence",
  }),
  stage({
    id: "presentation-publication", band: "output", side: "left",
    label: "Presentation pages",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.materialization.label],
    tip: {
      summary: "Classifies which bricks the renderer can see and publishes their compact level-set pages in place. Nothing is expanded to a dense field and nothing crosses to the host, so consumers read the same sparse authority physics stepped.",
      reads: "committed sparse authority",
      writes: "compact level-set brick pages and presentation classification",
      feeds: "renderer level-set and grid consumers",
    },
    chip: (context) => `${context.info?.fluidBrickResidentCount ?? 0} pages · resident`,
  }),
];

/** Method-owned diagram; no core or UI module imports adaptive implementation. */
export const ADAPTIVE_MASS_FLUID_PIPELINE: FluidPipelineGraph = Object.freeze({
  methodId: "adaptive-mass",
  // Encode order, so the diagram reads down the advance the solver encodes.
  bands: [
    { id: "transport", label: "Transport velocity + conservative CM12 transport" },
    { id: "momentum", label: "Momentum prediction" },
    { id: "pressure", label: "Composite pressure projection + receipts" },
    { id: "adaptivity", label: "Brick activity + candidate resolution" },
    { id: "output", label: "Sparse presentation publication" },
  ],
  stages: ADAPTIVE_MASS_FLUID_STAGES,
});
