/**
 * Sparse CM12's advance, observed and drawn — derived from the stage registry.
 *
 * Nothing in this file names a stage, a sub-seam or a phase label by hand. The
 * resident owns the stage ABI (`SPARSE_CM12_RESIDENT_STAGES`,
 * `SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES`), the registry in
 * `sparse-cm12-stages.ts` owns one declaration per stage (phases, diagram
 * copy, lens), and this module only folds the two together into the shapes
 * the capture, the SIM panel and the probes consume:
 *
 * - `AdaptiveMassFrameCapture` closes each seam on the CPU, hardware and
 *   queue-wall lanes under the phase the registry names for it;
 * - `ADAPTIVE_MASS_FLUID_PIPELINE` is the SIM diagram, one node per resident
 *   stage in encode order, each node's `phaseLabels` exactly the labels its
 *   seams emit and each node carrying its own lens;
 * - `ADAPTIVE_MASS_GPU_WORK_CHUNKS` is the exact ownership of every timestamp
 *   phase, for the stage-cost probe.
 *
 * A stage added to the encoder therefore cannot be timed, drawn or lensed
 * until it has a registry entry, and an entry cannot outlive its stage.
 */
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
  SPARSE_CM12_STAGE_BANDS,
  SPARSE_CM12_STAGES,
  sparseCM12Stage,
  sparseCM12SubstagePhase,
  type SparseCM12StageBand,
} from "./sparse-cm12-stages";
import {
  SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES,
  SPARSE_CM12_RESIDENT_STAGES,
  type SparseCM12ResidentStageId,
  type SparseCM12ResidentStageSeams,
} from "./webgpu-sparse-cm12-resident";

export { adaptiveMassPressureTopologyChip } from "./sparse-cm12-stages";

/** The capture cadence matches the uniform reference's observatory lane. */
export const ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS = 100;

/**
 * The CPU lane's brackets around the stage chain: planning before the first
 * stage, closure after the last, and the queue-wall interval that stands in
 * for the whole advance when a device has no timestamp queries. No GPU work
 * lives in any of them, so no diagram node claims them. Every other phase the
 * advance emits comes from the stage registry.
 */
export const ADAPTIVE_MASS_ADVANCE_PHASE = Object.freeze({
  commandEncoding: {
    id: "command-encoding",
    label: "Advance planning + trace setup",
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

/** Resident stage id to the phase its own seam is timed under. */
export const ADAPTIVE_MASS_RESIDENT_STAGE_PHASE: Readonly<Record<
  SparseCM12ResidentStageId,
  GPUTimestampPhase
>> = Object.freeze(Object.fromEntries(SPARSE_CM12_RESIDENT_STAGES.map(
  (stage) => [stage, SPARSE_CM12_STAGES[stage].phase],
)) as Record<SparseCM12ResidentStageId, GPUTimestampPhase>);

export interface AdaptiveMassGPUWorkChunk {
  readonly id: string;
  readonly residentStage: SparseCM12ResidentStageId;
  /** The diagram node that sums this chunk. Always the resident stage now. */
  readonly rollupStage: SparseCM12ResidentStageId;
  readonly phase: GPUTimestampPhase;
  readonly kind: "stage" | "substage";
}

/**
 * Exact ownership of every GPU timestamp phase, in encode order: each stage's
 * sub-seams, then the stage's own seam (`<stage>/remainder` when it has
 * sub-seams, the bare id otherwise). Rollups may sum these records, but no
 * phase appears twice and no chunk is fabricated.
 */
export const ADAPTIVE_MASS_GPU_WORK_CHUNKS: readonly AdaptiveMassGPUWorkChunk[] =
  Object.freeze(SPARSE_CM12_RESIDENT_STAGES.flatMap((stage): AdaptiveMassGPUWorkChunk[] => {
    const substages: readonly string[] = SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES[stage];
    return [
      ...substages.map((substage): AdaptiveMassGPUWorkChunk => ({
        id: `${stage}/${substage}`,
        residentStage: stage,
        rollupStage: stage,
        // The ABI says this sub-seam belongs to this stage; the registry is
        // typed to agree, so the erased lookup here cannot miss.
        phase: sparseCM12SubstagePhase(stage, substage as never),
        kind: "substage",
      })),
      {
        id: substages.length > 0 ? `${stage}/remainder` : stage,
        residentStage: stage,
        rollupStage: stage,
        phase: SPARSE_CM12_STAGES[stage].phase,
        kind: "stage",
      },
    ];
  }));

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
   * The encoder's stage seams: one stage closes on every lane at once, under
   * the phase the registry files it under.
   *
   * The final stage closes through the recorder's observable trailing marker.
   * This matters when the stage contains several passes and intervening copies:
   * closing on its first pass would silently omit the rest of the frame tail.
   */
  readonly residentStageSeams: SparseCM12ResidentStageSeams = {
    close: (stage) => {
      const phase = SPARSE_CM12_STAGES[stage].phase;
      this.cpu.completePhase(phase);
      if (this.encoder) this.gpu?.completePhase(this.encoder, phase);
    },
    closeSubstage: (stage, substage) => {
      const phase = sparseCM12SubstagePhase(stage, substage);
      this.cpu.completePhase(phase);
      if (this.encoder) this.gpu?.completePhase(this.encoder, phase);
    },
    anchorFinalBoundary: (source, offset) => {
      this.gpu?.anchorFinalBoundary(source, offset);
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

/**
 * The SIM diagram's nodes: one per resident stage, in encode order, built from
 * the stage's registry entry. A node's `id` is the resident's stage id — the
 * same string the trace seams, the lens roster and the `stage-lens/<stage>`
 * overlay mode use — and its `phaseLabels` are exactly the labels the stage's
 * seams emit, sub-seams first, own seam last.
 */
const ADAPTIVE_MASS_FLUID_STAGES: readonly FluidPipelineStage[] =
  SPARSE_CM12_RESIDENT_STAGES.map((id): FluidPipelineStage => {
    const entry = sparseCM12Stage(id);
    const substages: readonly string[] = SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES[id];
    return {
      id,
      band: entry.band,
      side: entry.side,
      label: entry.label,
      phaseLabels: [
        ...substages.map((substage) => sparseCM12SubstagePhase(id, substage as never).label),
        entry.phase.label,
      ],
      tip: entry.tip,
      ...(entry.controls ? { controls: entry.controls } : {}),
      ...(entry.lens ? { lens: entry.lens } : {}),
      state: alwaysOn,
      chip: entry.chip,
    };
  });

/** Bands in the order the advance first enters each, so the diagram reads down the encode. */
const ADAPTIVE_MASS_FLUID_BANDS = [...new Set(
  ADAPTIVE_MASS_FLUID_STAGES.map((stage) => stage.band as SparseCM12StageBand),
)].map((band) => ({ id: band, label: SPARSE_CM12_STAGE_BANDS[band] }));

/** Method-owned diagram; no core or UI module imports adaptive implementation. */
export const ADAPTIVE_MASS_FLUID_PIPELINE: FluidPipelineGraph = Object.freeze({
  methodId: "adaptive-mass",
  bands: ADAPTIVE_MASS_FLUID_BANDS,
  stages: ADAPTIVE_MASS_FLUID_STAGES,
});
