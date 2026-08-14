import type {
  FluidPipelineGraph,
  FluidPipelineStage,
} from "../../core/fluid-pipeline";
import { gpuPhysicsPerformanceActivityFrameId } from "../../core/performance-activity";
import {
  CPUPerformanceTrace,
  partitionPerformanceTrace,
  type GPUTimestampPhase,
  type PerformanceTrace,
} from "../../core/performance-trace";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { SparseAtlasProjectionStageId } from "./sparse-atlas-composite-projection";
import type { SparseAtlasDynamicsStageId } from "./sparse-atlas-dynamics";

/** The capture cadence matches the uniform reference's observatory lane. */
export const ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS = 100;

/**
 * Exact trailing seams for the current sparse CPU authority.
 *
 * Scalar density/gamma and momentum intentionally share one conservative
 * face transaction. Giving them separate timings would double count the same
 * loop, so the graph exposes both algorithmic stages but marks momentum as a
 * term inside the coupled transport stage.
 */
export const ADAPTIVE_MASS_ADVANCE_PHASE = Object.freeze({
  receiverTopology: {
    id: "power-topology",
    label: "Sparse receiver activation + composite face topology",
  },
  coupledTransport: {
    id: "fine-sdf-advection",
    label: "Coupled conservative mass + gamma + momentum transport",
  },
  retentionConditioning: {
    id: "adaptive-publication",
    label: "Dry-brick retirement + retained-atlas conditioning",
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
  stateCommit: {
    id: "adaptive-publication",
    label: "Sparse authority commit + conservation receipts",
  },
  materialization: {
    id: "adaptive-publication",
    label: "Dense presentation field materialization",
  },
  upload: {
    id: "adaptive-publication",
    label: "WebGPU texture upload + queue enqueue",
  },
  queueCompletion: {
    id: "adaptive-publication",
    label: "WebGPU upload queue completion",
  },
} satisfies Record<string, GPUTimestampPhase>);

const DYNAMICS_PHASE: Readonly<Partial<Record<
  SparseAtlasDynamicsStageId,
  GPUTimestampPhase
>>> = Object.freeze({
  "receiver-topology": ADAPTIVE_MASS_ADVANCE_PHASE.receiverTopology,
  "coupled-transport": ADAPTIVE_MASS_ADVANCE_PHASE.coupledTransport,
  "retain-rebuild": ADAPTIVE_MASS_ADVANCE_PHASE.retentionConditioning,
  force: ADAPTIVE_MASS_ADVANCE_PHASE.force,
  // Projection is partitioned by the nested projection-stage seams below.
});

const PROJECTION_PHASE: Readonly<Record<
  SparseAtlasProjectionStageId,
  GPUTimestampPhase
>> = Object.freeze({
  topology: ADAPTIVE_MASS_ADVANCE_PHASE.pressureTopology,
  rhs: ADAPTIVE_MASS_ADVANCE_PHASE.pressureRhs,
  solve: ADAPTIVE_MASS_ADVANCE_PHASE.pressureSolve,
  projection: ADAPTIVE_MASS_ADVANCE_PHASE.pressureProjection,
  diagnostics: ADAPTIVE_MASS_ADVANCE_PHASE.pressureDiagnostics,
});

export interface AdaptiveMassFrameCaptureResult {
  readonly cpuTrace: PerformanceTrace;
  readonly queueTrace: Promise<PerformanceTrace>;
  readonly identity: NonNullable<GPUEulerianInfo["physicsCaptureIdentity"]>;
}

/**
 * One exhaustive CPU partition plus the independently observed WebGPU queue
 * interval for one adaptive advance. Construction/warmup never enters either
 * lane. The recorder owns no physics and is therefore safe to omit entirely
 * when instrumentation is off.
 */
export class AdaptiveMassFrameCapture {
  private readonly cpu: CPUPerformanceTrace;
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
      ADAPTIVE_MASS_ADVANCE_PHASE.receiverTopology,
      clock,
    );
  }

  readonly completeDynamicsStage = (stage: SparseAtlasDynamicsStageId): void => {
    const phase = DYNAMICS_PHASE[stage];
    if (phase) this.cpu.completePhase(phase);
  };

  readonly completeProjectionStage = (stage: SparseAtlasProjectionStageId): void => {
    this.cpu.completePhase(PROJECTION_PHASE[stage]);
  };

  /** Close work after the projection receipt and before dense publication. */
  completeStateCommit(): void {
    this.cpu.completePhase(ADAPTIVE_MASS_ADVANCE_PHASE.stateCommit);
  }

  completeMaterialization(): void {
    this.cpu.completePhase(ADAPTIVE_MASS_ADVANCE_PHASE.materialization);
  }

  /** Start immediately before the first queue.writeTexture call. */
  beginQueueUpload(): void {
    if (this.queueStartedAt_ms !== undefined) {
      throw new Error("Sparse CM12 queue capture already started");
    }
    this.queueStartedAt_ms = this.clock();
  }

  /** Close after all upload calls have been enqueued. */
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
    });
    return {
      cpuTrace,
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
    id: "receiver-topology", band: "topology", side: "left",
    label: "Receiver activation",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.receiverTopology.label],
    tip: {
      summary: "Activates only face-local receiver bricks needed by this CFL-bounded advance, then builds the canonical regular and 2:1 face-port rows.",
      reads: "resident atlas, face-normal velocity",
      writes: "transient support atlas, composite G rows",
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
    id: "retention-conditioning", band: "topology", side: "right",
    label: "Retain + condition",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.retentionConditioning.label],
    tip: {
      summary: "Retires dry bricks, rebuilds the compact atlas and remaps surviving velocity without allocating work for empty bricks.",
      reads: "transported density/gamma/momentum",
      writes: "next resident atlas and face state",
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
    id: "projection-receipts", band: "output", side: "left",
    label: "Physics receipts",
    phaseLabels: [
      ADAPTIVE_MASS_ADVANCE_PHASE.pressureDiagnostics.label,
      ADAPTIVE_MASS_ADVANCE_PHASE.stateCommit.label,
    ],
    tip: {
      summary: "Publishes pressure residual, post-projection divergence, energy identity, mass/gamma conservation and sparse work receipts from the committed authority.",
      reads: "projected state and solver residual",
      writes: "GPUEulerianInfo acceptance telemetry",
    },
    chip: () => "mass · gamma · residual · divergence",
  }),
  stage({
    id: "presentation-materialization", band: "output", side: "right",
    label: "Presentation fields",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.materialization.label],
    tip: {
      summary: "Expands only the renderer-facing density, phi, owner, velocity, pressure and divergence textures; dense fields never feed physics.",
      reads: "compact sparse authority",
      writes: "bounded dense presentation arrays",
      feeds: "WebGPU upload",
    },
    chip: () => "presentation-only dense bridge",
  }),
  stage({
    id: "presentation-upload", band: "output", side: "left",
    label: "Upload + queue",
    phaseLabels: [ADAPTIVE_MASS_ADVANCE_PHASE.upload.label],
    tip: {
      summary: "Packs rows, enqueues all consumer texture uploads, and observes queue completion independently from CPU physics timing.",
      reads: "dense presentation arrays",
      writes: "WebGPU consumer textures",
    },
    chip: () => "6 texture publications",
  }),
];

/** Method-owned diagram; no core or UI module imports adaptive implementation. */
export const ADAPTIVE_MASS_FLUID_PIPELINE: FluidPipelineGraph = Object.freeze({
  methodId: "adaptive-mass",
  bands: [
    { id: "topology", label: "Sparse topology + receiver activation" },
    { id: "transport", label: "Coupled CM12 conservative transport" },
    { id: "momentum", label: "Momentum prediction" },
    { id: "pressure", label: "Composite pressure projection" },
    { id: "output", label: "Receipts + presentation publication" },
  ],
  stages: ADAPTIVE_MASS_FLUID_STAGES,
});
