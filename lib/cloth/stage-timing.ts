/**
 * Per-stage GPU time of a cloth frame, for the study rail's Cloth card.
 *
 * The cloth step is one compute pass, so a sampled frame encodes into a
 * `ClothStagePasses` instead: a stand-in pass that ends the live pass and opens
 * the next whenever a pipeline of another stage is set, and the encoder under
 * it is a `GPUPassTimestampRecorder`'s, so each stage's passes carry their own
 * begin/end counters. The stage of a kernel is read from its pipeline label
 * (`Cloth <entry point>` for the solver, `ogc <kernel>` for the contact
 * detector), so neither encoder changes to be measured. Unsampled frames keep
 * the single pass.
 *
 * Per-pass pairs on Apple do not partition the frame (adjacent passes stay in
 * flight; see `GPUPassTimestampRecorder`): the stage figures are attribution,
 * `span_ms` is the frame's GPU time.
 */
import { GPUPassTimestampRecorder } from '../core/performance-trace';

export type ClothStageId = 'setup' | 'detect' | 'assemble' | 'pcg' | 'filter' | 'line-search' | 'finish' | 'other';

export const CLOTH_STAGES: readonly { readonly id: ClothStageId; readonly label: string; readonly tip: string }[] = [
  { id: 'setup', label: 'Step setup', tip: 'Grab anchors, the step\'s frame (inertial target, element frames) and the contact detector\'s frame and step begin.' },
  { id: 'detect', label: 'Contact detection', tip: 'OGC detection rounds, truncation, the decision and re-detection after each Newton iteration, and certificate verification.' },
  { id: 'assemble', label: 'Newton assembly', tip: 'Per-face terms and the gradient at the iterate, and the compiled contact linearization when it is on.' },
  { id: 'pcg', label: 'PCG solve', tip: 'The Newton direction: one dispatch per PCG iteration (or the whole solve in one when the cloth is one patch).' },
  { id: 'filter', label: 'Contact filter', tip: 'The line-search filter: each vertex\'s plane records and the largest step inside every certificate.' },
  { id: 'line-search', label: 'Line search', tip: 'Projection, trial energy, the trust-region model when on, the choice and the update; strain-limit duals.' },
  { id: 'finish', label: 'Finish', tip: 'Diagnostics, the contact state, restart scans and the step\'s final write-back.' },
  { id: 'other', label: 'Other', tip: 'Kernels this table does not name.' },
];

const SETUP = new Set(['grabBegin', 'restartScan', 'beginStep', 'frameElements', 'frame', 'nwBeginStep', 'frameBegin', 'stepBegin']);
const ASSEMBLE = new Set(['nwFaces', 'nwGradient', 'nwLinCount', 'nwLinBuild']);
const PCG = new Set(['pcgStep', 'pcgStepOverflow', 'pcgSolve']);
const LINE_SEARCH = new Set(['nwProject', 'nwTrial', 'nwModel', 'nwChoose', 'nwAccept', 'nwApply', 'nwEdgeDuals']);
const FINISH = new Set(['diagnose', 'reduceDiagnostics', 'onsetScan', 'finishStep', 'contactState']);

/** The stage a cloth pipeline belongs to, by its label. */
export function clothStageOf(pipelineLabel: string): ClothStageId {
  const solver = pipelineLabel.startsWith('Cloth '), detector = pipelineLabel.startsWith('ogc ');
  if (!solver && !detector) return 'other';
  const name = pipelineLabel.slice(solver ? 6 : 4);
  if (SETUP.has(name)) return 'setup';
  if (FINISH.has(name)) return 'finish';
  if (solver) return ASSEMBLE.has(name) ? 'assemble' : PCG.has(name) ? 'pcg' : LINE_SEARCH.has(name) ? 'line-search' : 'other';
  return name.startsWith('filter') ? 'filter' : 'detect';
}

/**
 * A compute pass that is really a run of them, one per stage change. Hand it
 * to code that encodes into a `GPUComputePassEncoder`; call `end` once.
 */
export class ClothStagePasses {
  private current?: GPUComputePassEncoder;
  private stage?: ClothStageId;
  /** Bind groups set so far, replayed into each new pass so a group set before its pipeline survives the split. */
  private readonly groups = new Map<number, [GPUBindGroup | null, Uint32Array | undefined]>();
  readonly pass: GPUComputePassEncoder;

  constructor(private readonly encoder: GPUCommandEncoder, private readonly prefix = 'cloth') {
    const setPipeline = (pipeline: GPUComputePipeline) => this.open(clothStageOf(pipeline.label)).setPipeline(pipeline);
    const setBindGroup = (index: number, group: GPUBindGroup | null, offsets?: Iterable<number>) => {
      const dynamic = offsets ? Uint32Array.from(offsets) : undefined;
      this.groups.set(index, [group, dynamic]);
      this.open(this.stage ?? 'other').setBindGroup(index, group, dynamic ?? []);
    };
    const end = () => this.end();
    const live = () => this.open(this.stage ?? 'other');
    this.pass = new Proxy({} as GPUComputePassEncoder, {
      get(_target, property) {
        if (property === 'setPipeline') return setPipeline;
        if (property === 'setBindGroup') return setBindGroup;
        if (property === 'end') return end;
        const pass = live();
        const value = Reflect.get(pass, property, pass);
        return typeof value === 'function' ? value.bind(pass) : value;
      },
    });
  }

  private open(stage: ClothStageId): GPUComputePassEncoder {
    if (this.current && stage === this.stage) return this.current;
    this.current?.end();
    this.stage = stage;
    this.current = this.encoder.beginComputePass({ label: `${this.prefix}:${stage}` });
    for (const [index, [group, offsets]] of this.groups) this.current.setBindGroup(index, group, offsets ?? []);
    return this.current;
  }

  end(): void { this.current?.end(); this.current = undefined; this.stage = undefined; }
}

export interface ClothStageTiming {
  readonly id: ClothStageId;
  readonly label: string;
  /** Summed pass time of the stage over the sampled frame. */
  readonly duration_ms: number;
  readonly passes: number;
}

export interface ClothConvergence {
  readonly newtonIterations: number;
  readonly pcgIterations: number;
  /** |g| / |m g| at the step's last Newton iterate (the residual test's ratio). */
  readonly residual: number;
  /** The last accepted step length. */
  readonly alpha: number;
}

export interface ClothDocumentFacts {
  readonly id: string;
  readonly vertices: number;
  readonly triangles: number;
  readonly edges: number;
  readonly integrator: string;
  readonly newtonIterations: number;
  readonly pcgIterations: number;
  readonly pcgTolerance?: number;
  readonly residualTolerance?: number;
  readonly trustRegion: string;
  readonly activeSet: string;
  readonly contactRadius_m: number;
  readonly convergence?: ClothConvergence;
}

/** One sampled cloth frame, published as `GPUEulerianInfo.clothPerformance`. */
export interface ClothPerformanceSample {
  readonly sampleId: number;
  readonly capturedAt_ms: number;
  /** Steps encoded in the sampled frame, over every document. */
  readonly steps: number;
  /** First pass begin to last pass end: the frame's GPU time. */
  readonly span_ms: number;
  /** Sum of the stage passes; above `span_ms` where passes overlapped. */
  readonly sum_ms: number;
  readonly stages: readonly ClothStageTiming[];
  readonly documents: readonly ClothDocumentFacts[];
  /** Why there is no figure, when there is none. */
  readonly unavailable?: string;
}

/** A sampled frame's recorder: instrument the encoder, encode through `ClothStagePasses`, resolve, read. */
export class ClothFrameTimer {
  private readonly recorder: GPUPassTimestampRecorder;
  readonly encoder: GPUCommandEncoder;
  static supported(device: GPUDevice): boolean { return GPUPassTimestampRecorder.supported(device); }

  constructor(device: GPUDevice, encoder: GPUCommandEncoder, capacity = 2048) {
    this.recorder = new GPUPassTimestampRecorder(device, capacity, 'cloth stage timestamps');
    this.encoder = this.recorder.instrument(encoder);
  }

  resolve(encoder: GPUCommandEncoder): boolean { return this.recorder.resolve(encoder); }

  /** The stage totals, in table order; undefined when the hardware returned nothing usable. */
  async read(): Promise<{ span_ms: number; sum_ms: number; stages: ClothStageTiming[] } | undefined> {
    const reading = await this.recorder.read();
    if (!reading) return undefined;
    const by = new Map<ClothStageId, { duration_ms: number; passes: number }>();
    let sum_ms = 0;
    for (const pass of reading.passes) {
      if (!pass.sampled || !pass.trusted) continue;
      const id = pass.label.slice(pass.label.indexOf(':') + 1) as ClothStageId;
      const at = by.get(id) ?? { duration_ms: 0, passes: 0 };
      at.duration_ms += pass.duration_ms; at.passes += 1; sum_ms += pass.duration_ms;
      by.set(id, at);
    }
    const stages = CLOTH_STAGES.filter(s => by.has(s.id)).map(s => ({ id: s.id, label: s.label, ...by.get(s.id)! }));
    return { span_ms: reading.span_ms, sum_ms, stages };
  }
}
