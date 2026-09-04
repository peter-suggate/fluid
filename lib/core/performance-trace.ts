/**
 * Exhaustive, non-overlapping performance accounting.
 *
 * A trace is a partition of one measured interval. Children are siblings, not
 * nested spans, so their durations can be added without double accounting.
 * Any interval not assigned by the producer is retained as "other" rather
 * than disappearing from the total.
 */
export type PerformanceDomain = "cpu" | "gpu";
export type PerformanceLane = "main-thread" | "physics" | "presentation";
export type PerformanceMeasurementSource =
  | "cpu-active-wall"
  | "gpu-hardware-timestamp"
  | "gpu-queue-wall"
  /**
   * Per-pass `beginningOfPassWriteIndex`/`endOfPassWriteIndex` pairs.
   *
   * Still genuine hardware execution time, but *not* a partition of a frame:
   * see `GPUPassTimestampRecorder` for why a boundary chain cannot be one on
   * Dawn/Metal, and what the resulting total does and does not mean.
   */
  | "gpu-pass-timestamp";

export type PaperPhaseId =
  | "frame-control"
  | "scene-upload"
  | "command-encoding"
  | "coarse-grid"
  | "power-topology"
  | "velocity-advection"
  | "pressure-system"
  | "pressure-solve"
  | "velocity-projection"
  | "velocity-extrapolation"
  | "fine-sdf-advection"
  | "fine-sdf-redistance"
  | "adaptive-publication"
  | "surface-extraction"
  | "water-caustics"
  | "svo-cone-lighting"
  | "svo-environment-gi"
  /**
   * The persistent level-0 voxel visibility cache: its demand pass and the
   * bounded population drain that follows.
   *
   * Split out of `svo-cone-lighting` so the cache is priced as itself. Both
   * dispatches used to fall inside whichever phase closed next, which charged a
   * bounded per-frame drain to the cone stage and left the cache with no cost
   * of its own to report.
   */
  | "svo-voxel-light"
  | "svo-primary"
  /**
   * The authored SDF set's own visibility passes (coverage → resolve →
   * overflow, or the direct proxy raster they replace).
   *
   * Distinct from `svo-primary`, which is the octree brick primary. The two
   * shared one id while the SDF set had only the direct fragment, and that made
   * the largest row in the hero frame — 85 % of it — invisible to every
   * aggregation that groups by phase id rather than by label.
   */
  | "svo-scene-primitive"
  /**
   * The near-field analytic band (W3): the camera-local record set that stays
   * analytic while everything behind it resolves from voxels.
   */
  | "svo-band"
  | "svo-brick-cull"
  | "svo-rigid"
  | "svo-glass"
  | "dry-scene"
  | "water-front-interface"
  | "water-back-interface"
  | "water-interfaces"
  | "optical-composite"
  | "inspection-overlay"
  | "present"
  | "other";

export interface PerformancePhaseSample {
  id: PaperPhaseId;
  label: string;
  duration_ms: number;
  /**
   * In an averaged trace: the fraction of observations that contained this
   * phase. An intermittent stage (caustics on mesh-move frames, world
   * maintenance on edit frames) reports its cost per frame that *encoded* it,
   * and this is the k/n that lets a consumer say so instead of silently
   * diluting the figure toward zero the rarer the stage ran.
   */
  encodedFraction?: number;
}

export interface PerformanceTrace {
  sampleId: number;
  domain: PerformanceDomain;
  lane: PerformanceLane;
  context: string;
  capturedAt_ms: number;
  measurementSource?: PerformanceMeasurementSource;
  total_ms: number;
  /**
   * For a pass-timestamp trace: the wall interval from the earliest sampled
   * pass begin to the latest sampled pass end — how long the GPU was busy with
   * the traced passes, overlap collapsed. `total_ms` on such a trace is the
   * end-to-end pass *sum*, which double-counts overlapped passes; shares of the
   * frame should be shares of the span.
   */
  span_ms?: number;
  phases: PerformancePhaseSample[];
}

/** Derived acceptance ledger for one independently observed lane. */
export interface PerformanceTraceAccounting {
  observedTotal_ms: number;
  accounted_ms: number;
  closureError_ms: number;
  exact: boolean;
}

export interface PerformanceInterval {
  id: PaperPhaseId;
  label: string;
  start_ms: number;
  end_ms: number;
}

const finiteDuration = (value: number) =>
  Number.isFinite(value) && value > 0 ? value : 0;

/**
 * Convert disjoint intervals inside [start, end] to an exact partition.
 * Overlap is rejected because clipping it would hide double accounting.
 */
export function partitionPerformanceTrace(input: {
  sampleId: number;
  domain: PerformanceDomain;
  lane: PerformanceLane;
  context: string;
  measurementSource?: PerformanceMeasurementSource;
  capturedAt_ms?: number;
  start_ms: number;
  end_ms: number;
  intervals: readonly PerformanceInterval[];
}): PerformanceTrace {
  const start_ms = input.start_ms;
  const end_ms = Math.max(start_ms, input.end_ms);
  const total_ms = finiteDuration(end_ms - start_ms);
  const ordered = [...input.intervals]
    .filter((interval) => interval.end_ms > interval.start_ms)
    .sort((left, right) => left.start_ms - right.start_ms);
  const phases: PerformancePhaseSample[] = [];
  let cursor = start_ms;
  let unassigned_ms = 0;
  for (const interval of ordered) {
    if (interval.start_ms < cursor - 1e-7) {
      throw new Error(`Performance intervals overlap at ${interval.label}`);
    }
    const clippedStart = Math.max(start_ms, interval.start_ms);
    const clippedEnd = Math.min(end_ms, interval.end_ms);
    if (clippedEnd <= clippedStart) continue;
    unassigned_ms += Math.max(0, clippedStart - cursor);
    phases.push({
      id: interval.id,
      label: interval.label,
      duration_ms: clippedEnd - clippedStart,
    });
    cursor = clippedEnd;
  }
  unassigned_ms += Math.max(0, end_ms - cursor);
  if (unassigned_ms > 0) {
    phases.push({ id: "other", label: "Other measured work", duration_ms: unassigned_ms });
  }
  const attributed_ms = phases.reduce((sum, phase) => sum + phase.duration_ms, 0);
  const rounding_ms = total_ms - attributed_ms;
  if (rounding_ms !== 0) {
    const other = phases.find((phase) => phase.id === "other");
    if (other) other.duration_ms += rounding_ms;
    else phases.push({ id: "other", label: "Other measured work", duration_ms: rounding_ms });
  }
  return {
    sampleId: input.sampleId,
    domain: input.domain,
    lane: input.lane,
    context: input.context,
    capturedAt_ms: input.capturedAt_ms ?? end_ms,
    measurementSource: input.measurementSource
      ?? (input.domain === "cpu" ? "cpu-active-wall" : "gpu-hardware-timestamp"),
    total_ms,
    phases,
  };
}

export function performanceTraceAccounted_ms(trace: PerformanceTrace): number {
  // Phases carry per-encode durations; a phase encoded in only k of n averaged
  // observations contributes duration * (k/n) to the frame's accounted time.
  return trace.phases.reduce(
    (sum, phase) => sum + phase.duration_ms * (phase.encodedFraction ?? 1),
    0,
  );
}

export function performanceTraceClosureError_ms(trace: PerformanceTrace, tolerance_ms = 1e-9): number {
  const error_ms = performanceTraceAccounted_ms(trace) - trace.total_ms;
  return Math.abs(error_ms) <= tolerance_ms ? 0 : error_ms;
}

export function performanceTraceIsExact(trace: PerformanceTrace, tolerance_ms = 1e-6): boolean {
  return trace.total_ms >= 0
    && trace.phases.every((phase) => Number.isFinite(phase.duration_ms) && phase.duration_ms >= 0)
    && Math.abs(performanceTraceClosureError_ms(trace, 0)) <= tolerance_ms;
}

export function performanceTraceAccounting(trace: PerformanceTrace): PerformanceTraceAccounting {
  const accounted_ms = performanceTraceAccounted_ms(trace);
  return {
    observedTotal_ms: trace.total_ms,
    accounted_ms,
    closureError_ms: performanceTraceClosureError_ms(trace),
    exact: performanceTraceIsExact(trace),
  };
}

export function performanceTraceMatchesLane(
  trace: PerformanceTrace | undefined,
  domain: PerformanceDomain,
  lane: PerformanceLane,
): trace is PerformanceTrace {
  return trace?.domain === domain && trace.lane === lane && performanceTraceIsExact(trace);
}

/**
 * Main-thread wall-time partition. `transition` closes the current phase and
 * opens the next at one shared timestamp, leaving neither gaps nor overlap.
 */
export class CPUPerformanceTrace {
  private readonly startedAt_ms: number;
  private phaseStartedAt_ms: number;
  private current: Pick<PerformancePhaseSample, "id" | "label">;
  private readonly intervals: PerformanceInterval[] = [];

  constructor(
    private readonly sampleId: number,
    private readonly context: string,
    initial: Pick<PerformancePhaseSample, "id" | "label">,
    now: () => number = () => performance.now(),
    private readonly lane: PerformanceLane = "main-thread",
  ) {
    this.clock = now;
    this.startedAt_ms = now();
    this.phaseStartedAt_ms = this.startedAt_ms;
    this.current = initial;
  }

  private readonly clock: () => number;

  transition(next: Pick<PerformancePhaseSample, "id" | "label">): void {
    const boundary_ms = this.clock();
    this.intervals.push({
      ...this.current,
      start_ms: this.phaseStartedAt_ms,
      end_ms: boundary_ms,
    });
    this.current = next;
    this.phaseStartedAt_ms = boundary_ms;
  }

  /** Close the interval since the previous boundary using the phase which was
   * just completed. This is useful for command encoders where the honest phase
   * identity is known at the trailing seam rather than before work begins. */
  completePhase(completed: Pick<PerformancePhaseSample, "id" | "label">): void {
    const boundary_ms = this.clock();
    this.intervals.push({
      ...completed,
      start_ms: this.phaseStartedAt_ms,
      end_ms: boundary_ms,
    });
    this.phaseStartedAt_ms = boundary_ms;
  }

  finish(final: Pick<PerformancePhaseSample, "id" | "label"> = this.current): PerformanceTrace {
    const completedAt_ms = this.clock();
    this.intervals.push({
      ...final,
      start_ms: this.phaseStartedAt_ms,
      end_ms: completedAt_ms,
    });
    return partitionPerformanceTrace({
      sampleId: this.sampleId,
      domain: "cpu",
      lane: this.lane,
      context: this.context,
      capturedAt_ms: completedAt_ms,
      start_ms: this.startedAt_ms,
      end_ms: completedAt_ms,
      intervals: this.intervals,
    });
  }

  /**
   * Close a trailing-boundary trace without inventing another phase.
   *
   * Command encoders name work at the seam after it was encoded. Once the last
   * seam has been recorded by `completePhase`, the partition is already closed;
   * adding `current` again would charge trace bookkeeping to an arbitrary stage.
   */
  finishCompletedPhases(): PerformanceTrace {
    return partitionPerformanceTrace({
      sampleId: this.sampleId,
      domain: "cpu",
      lane: this.lane,
      context: this.context,
      capturedAt_ms: this.phaseStartedAt_ms,
      start_ms: this.startedAt_ms,
      end_ms: this.phaseStartedAt_ms,
      intervals: this.intervals,
    });
  }
}

export interface GPUTimestampPhase {
  id: PaperPhaseId;
  label: string;
}

/**
 * Decode one ordered chain of GPU timestamp boundaries. Boundary i and i+1
 * define phase i, so the phase sum is mathematically identical to the root
 * interval and no GPU command can be counted twice.
 */
export function decodeGPUTimestampPartition(input: {
  sampleId: number;
  lane: "physics" | "presentation";
  context: string;
  capturedAt_ms?: number;
  timestamps: ArrayLike<bigint>;
  phases: readonly GPUTimestampPhase[];
}): PerformanceTrace | undefined {
  if (input.timestamps.length !== input.phases.length + 1 || input.phases.length === 0) {
    return undefined;
  }
  const origin = input.timestamps[0];
  if (origin === undefined || origin === 0n) return undefined;
  const intervals: PerformanceInterval[] = [];
  let previous = origin;
  for (let index = 0; index < input.phases.length; index += 1) {
    const next = input.timestamps[index + 1];
    if (next === undefined || next === 0n || next < previous) {
      return undefined;
    }
    intervals.push({
      ...input.phases[index],
      start_ms: Number(previous - origin) / 1e6,
      end_ms: Number(next - origin) / 1e6,
    });
    previous = next;
  }
  const total_ms = Number(previous - origin) / 1e6;
  if (!Number.isFinite(total_ms) || total_ms <= 0 || total_ms >= 10_000) return undefined;
  return partitionPerformanceTrace({
    sampleId: input.sampleId,
    domain: "gpu",
    lane: input.lane,
    context: input.context,
    capturedAt_ms: input.capturedAt_ms,
    start_ms: 0,
    end_ms: total_ms,
    intervals,
  });
}

export function averagePerformanceTraces(traces: readonly PerformanceTrace[]): PerformanceTrace | undefined {
  if (traces.length === 0) return undefined;
  const domain = traces[0].domain;
  const lane = traces[0].lane;
  const measurementSource = traces[0].measurementSource;
  if (traces.some((trace) =>
    trace.domain !== domain
    || trace.lane !== lane
    || trace.measurementSource !== measurementSource)) {
    throw new Error("Cannot average independent performance lanes");
  }
  const observations = traces.filter((trace, index) =>
    traces.findIndex((candidate) =>
      candidate.domain === trace.domain
      && candidate.lane === trace.lane
      && candidate.context === trace.context
      && candidate.sampleId === trace.sampleId
      && candidate.capturedAt_ms === trace.capturedAt_ms) === index);
  const latest = observations[observations.length - 1];
  const totals = new Map<string, PerformancePhaseSample & { count: number }>();
  for (const trace of observations) {
    for (const phase of trace.phases) {
      const key = `${phase.id}\0${phase.label}`;
      const current = totals.get(key);
      if (current) {
        current.duration_ms += phase.duration_ms;
        current.count += 1;
      } else totals.set(key, { ...phase, count: 1 });
    }
  }
  // Per encode, not per observation: a phase present in k of n samples is a
  // stage that ran k times, and dividing by n diluted it toward zero the rarer
  // it ran. The fraction is kept so a consumer can annotate intermittent rows;
  // duration × fraction reconstructs the per-observation mean.
  const phases = [...totals.values()].map(({ count, ...phase }) => ({
    ...phase,
    duration_ms: phase.duration_ms / count,
    encodedFraction: count / observations.length,
  }));
  const total_ms = observations.reduce((sum, trace) => sum + trace.total_ms, 0) / observations.length;
  const span_ms = observations.some((trace) => trace.span_ms !== undefined)
    ? observations.reduce((sum, trace) => sum + (trace.span_ms ?? trace.total_ms), 0) / observations.length
    : undefined;
  const accounted_ms = phases.reduce((sum, phase) => sum + phase.duration_ms * phase.encodedFraction, 0);
  const delta_ms = total_ms - accounted_ms;
  // Averaging independently accumulated IEEE-754 values can leave a
  // sub-nanosecond negative residual. Do not turn that harmless dust into a
  // negative "other" phase, which would make a closed trace appear invalid.
  if (Math.abs(delta_ms) > 1e-9) {
    const other = phases.find((phase) => phase.id === "other");
    if (other && other.duration_ms + delta_ms >= 0) other.duration_ms += delta_ms;
    else if (delta_ms > 0) phases.push({ id: "other", label: "Other measured work", duration_ms: delta_ms, encodedFraction: 1 });
    else {
      const largest = phases.reduce((candidate, phase) =>
        phase.duration_ms > candidate.duration_ms ? phase : candidate, phases[0]);
      if (largest && largest.duration_ms + delta_ms >= 0) largest.duration_ms += delta_ms;
    }
  }
  return { ...latest, total_ms, span_ms, phases };
}

/**
 * Sum disjoint main-thread callbacks into one active-CPU observation.
 *
 * Browser callbacks on the main thread cannot overlap, so adding their
 * individually closed wall-time partitions accounts all active frame work
 * without counting the idle interval between callbacks. Duplicate samples are
 * rejected rather than silently counted twice.
 */
export function combineMainThreadPerformanceTraces(
  traces: readonly PerformanceTrace[],
  context?: string,
): PerformanceTrace | undefined {
  if (traces.length === 0) return undefined;
  if (traces.some((trace) => !performanceTraceMatchesLane(trace, "cpu", "main-thread"))) {
    throw new Error("Only exact CPU main-thread traces can be combined");
  }
  const identities = new Set<string>();
  for (const trace of traces) {
    const identity = `${trace.context}\0${trace.sampleId}\0${trace.capturedAt_ms}`;
    if (identities.has(identity)) throw new Error("Cannot double count a main-thread trace");
    identities.add(identity);
  }
  const latest = traces.reduce((candidate, trace) =>
    trace.capturedAt_ms >= candidate.capturedAt_ms ? trace : candidate);
  const total_ms = traces.reduce((sum, trace) => sum + trace.total_ms, 0);
  const phases = traces.flatMap((trace) => trace.phases.map((phase) => ({ ...phase })));
  const accounted_ms = phases.reduce((sum, phase) => sum + phase.duration_ms, 0);
  const residual_ms = total_ms - accounted_ms;
  if (residual_ms !== 0) {
    const other = phases.find((phase) => phase.id === "other");
    if (other) other.duration_ms += residual_ms;
    else phases.push({ id: "other", label: "Other measured work", duration_ms: residual_ms });
  }
  return {
    sampleId: latest.sampleId,
    domain: "cpu",
    lane: "main-thread",
    context: context ?? latest.context,
    capturedAt_ms: latest.capturedAt_ms,
    measurementSource: "cpu-active-wall",
    total_ms,
    phases,
  };
}

/**
 * One-shot WebGPU boundary chain. The recorder deliberately owns its query
 * resources so an asynchronous map can never observe a later frame's resolve.
 */
export class GPUPerformanceTraceRecorder {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;
  private readonly encoderBreakSource: GPUBuffer;
  private readonly encoderBreakTarget: GPUBuffer;
  private markerResources?: DynamicTraceMarkerResources;
  private boundaryCount = 0;
  private disposed = false;

  static supported(device: GPUDevice): boolean {
    return device.features.has("timestamp-query");
  }

  constructor(
    private readonly device: GPUDevice,
    private readonly sampleId: number,
    private readonly lane: "physics" | "presentation",
    private readonly context: string,
    private readonly phases: readonly GPUTimestampPhase[],
  ) {
    const count = phases.length + 1;
    this.querySet = device.createQuerySet({ type: "timestamp", count });
    this.resolveBuffer = device.createBuffer({
      label: `${lane} trace resolve`,
      size: count * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      label: `${lane} trace read`,
      size: count * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.encoderBreakSource = device.createBuffer({
      label: `${lane} trace encoder break source`,
      size: 4,
      usage: GPUBufferUsage.COPY_SRC,
    });
    this.encoderBreakTarget = device.createBuffer({
      label: `${lane} trace encoder break target`,
      size: 4,
      usage: GPUBufferUsage.COPY_DST,
    });
    this.markerResources = preparedDynamicTraceMarkers.get(device);
    if (!this.markerResources) {
      void dynamicTraceMarkerResources(device).then((resources) => { this.markerResources = resources; });
    }
  }

  boundary(encoder: GPUCommandEncoder, label: string): void {
    if (this.boundaryCount > this.phases.length) throw new Error("Too many GPU trace boundaries");
    // Empty marker passes collapse to zero on Dawn/Metal. Match the dynamic
    // recorder's observable stage boundary so presentation traces work on the
    // same Apple GPU path as physics traces.
    encoder.copyBufferToBuffer(this.encoderBreakSource, 0, this.encoderBreakTarget, 0, 4);
    const marker = encoder.beginComputePass({
      label,
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex: this.boundaryCount,
      },
    });
    if (this.markerResources) {
      marker.setPipeline(this.markerResources.pipeline);
      marker.setBindGroup(0, this.markerResources.bindGroup);
      marker.dispatchWorkgroups(1);
    }
    marker.end();
    this.boundaryCount += 1;
  }

  resolve(encoder: GPUCommandEncoder): void {
    if (this.boundaryCount !== this.phases.length + 1) {
      throw new Error(`GPU trace has ${this.boundaryCount} boundaries; expected ${this.phases.length + 1}`);
    }
    const bytes = (this.phases.length + 1) * 8;
    encoder.resolveQuerySet(this.querySet, 0, this.phases.length + 1, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, bytes);
  }

  async read(): Promise<PerformanceTrace | undefined> {
    try {
      await this.readBuffer.mapAsync(GPUMapMode.READ);
      const timestamps = new BigUint64Array(this.readBuffer.getMappedRange().slice(0));
      return decodeGPUTimestampPartition({
        sampleId: this.sampleId,
        lane: this.lane,
        context: this.context,
        capturedAt_ms: performance.now(),
        timestamps,
        phases: this.phases,
      });
    } finally {
      if (this.readBuffer.mapState === "mapped") this.readBuffer.unmap();
      this.destroy();
    }
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    this.readBuffer.destroy();
    this.encoderBreakSource.destroy();
    this.encoderBreakTarget.destroy();
  }
}

interface DynamicTraceMarkerResources {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly buffer: GPUBuffer;
}

const dynamicTraceMarkers = new WeakMap<GPUDevice, Promise<DynamicTraceMarkerResources>>();
const preparedDynamicTraceMarkers = new WeakMap<GPUDevice, DynamicTraceMarkerResources>();
async function dynamicTraceMarkerResources(device: GPUDevice): Promise<DynamicTraceMarkerResources> {
  const cached = dynamicTraceMarkers.get(device);
  if (cached) return cached;
  // Callers may request instrumentation while constructing a frame recorder;
  // never let that constructor execute WGSL work on its own stack.
  await Promise.resolve();
  const shaderModule = device.createShaderModule({
    label: "dynamic trace marker",
    // An empty dispatch is dead-code eliminated by Metal/Dawn and its timestamp
    // may resolve to zero. The atomic write is the smallest observable GPU side
    // effect we can give the closing boundary.
    code: `
      struct Marker { value: atomic<u32> }
      @group(0) @binding(0) var<storage, read_write> marker: Marker;
      @compute @workgroup_size(1) fn mark() { atomicAdd(&marker.value, 1u); }
    `,
  });
  const promise = device.createComputePipelineAsync({
    label: "dynamic trace marker",
    layout: "auto",
    compute: { module: shaderModule, entryPoint: "mark" },
  }).then((pipeline) => {
    const buffer = device.createBuffer({
      label: "dynamic trace marker storage",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      label: "dynamic trace marker",
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });
    const resources = { pipeline, bindGroup, buffer };
    preparedDynamicTraceMarkers.set(device, resources);
    return resources;
  });
  dynamicTraceMarkers.set(device, promise);
  return promise;
}

/**
 * Dynamic variant for solvers whose CFL substep count and optional stages are
 * known only while encoding. Every completed phase appends one adjacent
 * boundary; repeated semantic IDs remain separate occurrences in the sample.
 */
export class DynamicGPUPerformanceTraceRecorder {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;
  /** A 4-byte copy between these at every boundary. Dawn's Metal backend
   * folds adjacent compute passes into one MTLComputeCommandEncoder; a folded
   * empty marker pass never receives a counter sample and its timestamp
   * resolves to zero. The blit forces an encoder break so every boundary is
   * observable. */
  private readonly encoderBreakSource: GPUBuffer;
  private readonly encoderBreakTarget: GPUBuffer;
  /** Apple GPUs sample timestamp counters at stage boundaries; a marker pass
   * with no dispatch can resolve zero. One empty-workgroup dispatch gives the
   * pass a stage to sample. */
  private readonly markerResources: DynamicTraceMarkerResources;
  private readonly phases: GPUTimestampPhase[] = [];
  private boundaryCount = 0;
  private disposed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly sampleId: number,
    private readonly lane: "physics" | "presentation",
    private readonly context: string,
    private readonly capacity = 128,
    markerResources: DynamicTraceMarkerResources,
  ) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: capacity });
    this.resolveBuffer = device.createBuffer({
      label: `${lane} dynamic trace resolve`,
      size: capacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      label: `${lane} dynamic trace read`,
      size: capacity * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.encoderBreakSource = device.createBuffer({
      label: `${lane} dynamic trace encoder break source`,
      size: 4,
      usage: GPUBufferUsage.COPY_SRC,
    });
    this.encoderBreakTarget = device.createBuffer({
      label: `${lane} dynamic trace encoder break target`,
      size: 4,
      usage: GPUBufferUsage.COPY_DST,
    });
    this.markerResources = markerResources;
  }

  static async create(
    device: GPUDevice,
    sampleId: number,
    lane: "physics" | "presentation",
    context: string,
    capacity = 128,
  ): Promise<DynamicGPUPerformanceTraceRecorder> {
    return new DynamicGPUPerformanceTraceRecorder(
      device, sampleId, lane, context, capacity, await dynamicTraceMarkerResources(device));
  }

  begin(encoder: GPUCommandEncoder): void {
    if (this.boundaryCount !== 0) throw new Error("GPU trace already started");
    this.writeBoundary(encoder, "GPU trace start");
  }

  completePhase(encoder: GPUCommandEncoder, phase: GPUTimestampPhase): void {
    if (this.boundaryCount === 0) throw new Error("GPU trace has not started");
    if (this.boundaryCount >= this.capacity) throw new Error("GPU trace capacity exceeded");
    this.phases.push(phase);
    this.writeBoundary(encoder, `${phase.label} complete`);
  }

  private writeBoundary(encoder: GPUCommandEncoder, label: string): void {
    // A blit between passes prevents Dawn/Metal from folding the marker into
    // the surrounding compute encoder, which would leave its timestamp zero.
    encoder.copyBufferToBuffer(this.encoderBreakSource, 0, this.encoderBreakTarget, 0, 4);
    const marker = encoder.beginComputePass({
      label,
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex: this.boundaryCount,
      },
    });
    marker.setPipeline(this.markerResources.pipeline);
    marker.setBindGroup(0, this.markerResources.bindGroup);
    marker.dispatchWorkgroups(1);
    marker.end();
    this.boundaryCount += 1;
  }

  resolve(encoder: GPUCommandEncoder): void {
    if (this.boundaryCount !== this.phases.length + 1 || this.phases.length === 0) {
      throw new Error("GPU trace is not a complete boundary chain");
    }
    const bytes = this.boundaryCount * 8;
    encoder.resolveQuerySet(this.querySet, 0, this.boundaryCount, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, bytes);
  }

  async read(): Promise<PerformanceTrace | undefined> {
    try {
      await this.readBuffer.mapAsync(GPUMapMode.READ);
      const bytes = this.boundaryCount * 8;
      const timestamps = new BigUint64Array(this.readBuffer.getMappedRange(0, bytes).slice(0));
      const trace = decodeGPUTimestampPartition({
        sampleId: this.sampleId,
        lane: this.lane,
        context: this.context,
        capturedAt_ms: performance.now(),
        timestamps,
        phases: this.phases,
      });
      if (!trace && (globalThis as { process?: { env?: Record<string, string | undefined> } })
        .process?.env?.FLUID_TRACE_DEBUG === "1") {
        console.warn(JSON.stringify({
          record: "trace-decode-failure", sampleId: this.sampleId, context: this.context,
          boundaries: this.boundaryCount, phases: this.phases.length,
          phaseLabels: this.phases.map((phase) => phase.label),
          timestamps: Array.from(timestamps, String),
        }));
      }
      return trace;
    } finally {
      if (this.readBuffer.mapState === "mapped") this.readBuffer.unmap();
      this.destroy();
    }
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    this.readBuffer.destroy();
    this.encoderBreakSource.destroy();
    this.encoderBreakTarget.destroy();
  }
}

/**
 * Stage boundaries carried by the frame's own passes.
 *
 * The recorders above buy each boundary with a whole extra compute pass. This
 * one buys it with nothing: the boundary is *armed* by `completePhase` and then
 * spliced into the `timestampWrites` of the next real pass the frame encodes,
 * through a proxy over the command encoder. A traced advance therefore adds one
 * marker pass in total — for the closing boundary, which by definition has no
 * following work — instead of one per stage, and adds no queue fence at all.
 *
 * Stages that encode no pass share their successor's boundary slot and close at
 * exactly zero, so the partition stays complete and the reported source remains
 * genuine hardware execution time.
 */
export class GPUStageTimestampRecorder {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;
  /** See DynamicGPUPerformanceTraceRecorder: Dawn/Metal folds adjacent compute
   * passes, and a folded pass never receives its counter sample. A 4-byte blit
   * ahead of a boundary-carrying pass forces the encoder break that makes the
   * sample observable. It is the only command this recorder adds per stage. */
  private readonly encoderBreakSource: GPUBuffer;
  private readonly encoderBreakTarget: GPUBuffer;
  private markerResources?: DynamicTraceMarkerResources;
  private readonly phases: GPUTimestampPhase[] = [];
  /** Query slot each boundary landed on; repeats mark an empty stage. */
  private readonly boundarySlots: number[] = [];
  private armedBoundaries = 0;
  private queryCount = 0;
  private finalPhaseClosesOnNextPass = false;
  private finalBoundaryAnchor?: { readonly source: GPUBuffer; readonly offset: number };
  private started = false;
  private resolved = false;
  private overflowed = false;
  private disposed = false;

  static supported(device: GPUDevice): boolean {
    return device.features.has("timestamp-query");
  }

  /** Compile the observable closing marker before any synchronous frame asks for it. */
  static async prepare(device: GPUDevice): Promise<void> {
    if (!this.supported(device)) return;
    await dynamicTraceMarkerResources(device);
  }

  /**
   * Whether the closing marker pipeline is compiled for this device. A recorder
   * constructed before then closes the chain with an *empty* pass, and Metal
   * writes no timestamp for a pass that does no observable work — the final
   * boundary decodes as unsampled and the whole first sample is rejected.
   * Callers that retire hardware tracing on one bad sample must not construct
   * a recorder until this is true.
   */
  static markersReady(device: GPUDevice): boolean {
    return preparedDynamicTraceMarkers.has(device);
  }

  constructor(
    private readonly device: GPUDevice,
    private readonly sampleId: number,
    private readonly lane: "physics" | "presentation",
    private readonly context: string,
    private readonly capacity = 256,
  ) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: capacity });
    this.resolveBuffer = device.createBuffer({
      label: `${lane} stage trace resolve`,
      size: capacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      label: `${lane} stage trace read`,
      size: capacity * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.encoderBreakSource = device.createBuffer({
      label: `${lane} stage trace encoder break source`,
      size: 4,
      usage: GPUBufferUsage.COPY_SRC,
    });
    this.encoderBreakTarget = device.createBuffer({
      label: `${lane} stage trace encoder break target`,
      size: 4,
      usage: GPUBufferUsage.COPY_DST,
    });
    this.markerResources = preparedDynamicTraceMarkers.get(device);
    if (!this.markerResources) {
      void dynamicTraceMarkerResources(device).then((resources) => { this.markerResources = resources; });
    }
  }

  /**
   * Wrap the frame's encoder. Every pass created through the returned encoder
   * can carry a boundary; everything else forwards untouched. Callers must use
   * this object everywhere the raw encoder went, because solver bookkeeping
   * keys off encoder identity.
   */
  instrument(encoder: GPUCommandEncoder): GPUCommandEncoder {
    const claimBoundary = (target: GPUCommandEncoder, occupied: boolean) => this.claimBoundary(target, occupied);
    return new Proxy(encoder, {
      get(target, property) {
        if (property === "beginComputePass") {
          return (descriptor?: GPUComputePassDescriptor) => {
            const writes = claimBoundary(target, descriptor?.timestampWrites !== undefined);
            return target.beginComputePass(writes ? { ...descriptor, timestampWrites: writes } : descriptor);
          };
        }
        if (property === "beginRenderPass") {
          return (descriptor: GPURenderPassDescriptor) => {
            const writes = claimBoundary(target, descriptor.timestampWrites !== undefined);
            return target.beginRenderPass(writes ? { ...descriptor, timestampWrites: writes } : descriptor);
          };
        }
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as GPUCommandEncoder;
  }

  /** Open the first stage. Its boundary lands on the frame's first pass. */
  begin(): void {
    if (this.started) throw new Error("GPU stage trace already started");
    this.started = true;
    this.armedBoundaries = 1;
  }

  /**
   * Close the named stage. The encoder is accepted so call sites read as the
   * command-adjacent seam they are, but nothing is written into it: the
   * boundary deliberately waits for the next pass that does real work.
   */
  completePhase(_encoder: GPUCommandEncoder, phase: GPUTimestampPhase): void {
    if (!this.started) throw new Error("GPU stage trace has not started");
    if (this.phases.length + 1 >= this.capacity) { this.overflowed = true; return; }
    this.phases.push(phase);
    this.armedBoundaries += 1;
  }

  /**
   * Name the final phase before its pass begins so that pass can carry both its
   * opening and closing timestamp. Metal render and compute stage clocks are
   * not reliably comparable, so a synthetic compute marker after the final
   * render pass cannot close a valid monotonic chain.
   */
  completeFinalPhaseOnNextPass(phase: GPUTimestampPhase): void {
    if (!this.started) throw new Error("GPU stage trace has not started");
    if (this.finalPhaseClosesOnNextPass) throw new Error("GPU stage trace final phase is already armed");
    if (this.phases.length + 1 >= this.capacity) { this.overflowed = true; return; }
    this.phases.push(phase);
    this.finalPhaseClosesOnNextPass = true;
  }

  /**
   * Make an otherwise independent trailing marker consume a word published by
   * the observed command tail. Dawn/Metal may schedule an unrelated marker
   * ahead of prior work; this copy creates the dependency that makes its end
   * timestamp a trustworthy final boundary.
   */
  anchorFinalBoundary(source: GPUBuffer, offset = 0): void {
    if (this.resolved) throw new Error("GPU stage trace is already resolved");
    if (offset < 0 || offset % 4 !== 0) {
      throw new RangeError("GPU stage trace final anchor offset must be nonnegative/aligned");
    }
    this.finalBoundaryAnchor = { source, offset };
  }

  /**
   * Assign every armed boundary to the pass about to begin. A descriptor that
   * already carries timestamp writes belongs to another recorder and is never
   * displaced; its boundaries stay armed for the pass after it.
   */
  private claimBoundary(encoder: GPUCommandEncoder, occupied: boolean) {
    if (this.armedBoundaries === 0 || this.disposed || this.overflowed || occupied) return undefined;
    const querySlots = this.finalPhaseClosesOnNextPass ? 2 : 1;
    if (this.queryCount + querySlots > this.capacity) { this.overflowed = true; return undefined; }
    const beginningOfPassWriteIndex = this.queryCount;
    this.queryCount += querySlots;
    for (let boundary = 0; boundary < this.armedBoundaries; boundary += 1) {
      this.boundarySlots.push(beginningOfPassWriteIndex);
    }
    this.armedBoundaries = 0;
    const endOfPassWriteIndex = this.finalPhaseClosesOnNextPass
      ? beginningOfPassWriteIndex + 1 : undefined;
    if (endOfPassWriteIndex !== undefined) {
      this.boundarySlots.push(endOfPassWriteIndex);
      this.finalPhaseClosesOnNextPass = false;
    }
    encoder.copyBufferToBuffer(this.encoderBreakSource, 0, this.encoderBreakTarget, 0, 4);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex,
      ...(endOfPassWriteIndex === undefined ? {} : { endOfPassWriteIndex }),
    };
  }

  /**
   * Close the chain and stage the readback. The trailing boundary has no
   * following work to attach to, so this is the one marker pass the recorder
   * ever encodes.
   */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.started || this.resolved) throw new Error("GPU stage trace is not open");
    this.resolved = true;
    if (this.finalPhaseClosesOnNextPass) { this.overflowed = true; return; }
    if (this.phases.length === 0) { this.overflowed = true; return; }
    // The last completed stage already armed the closing boundary. It has no
    // following work to ride, so this is the one pass the recorder encodes.
    // Metal/Dawn can resolve a final pass's *beginning* timestamp to zero even
    // when the pass performs an observable atomic write. Its end timestamp is
    // reliable, so the closing boundary owns an end-of-pass query directly.
    if (this.armedBoundaries > 0) {
      if (this.queryCount >= this.capacity) { this.overflowed = true; return; }
      const endOfPassWriteIndex = this.queryCount;
      this.queryCount += 1;
      for (let boundary = 0; boundary < this.armedBoundaries; boundary += 1) {
        this.boundarySlots.push(endOfPassWriteIndex);
      }
      this.armedBoundaries = 0;
      encoder.copyBufferToBuffer(this.encoderBreakSource, 0, this.encoderBreakTarget, 0, 4);
      if (this.finalBoundaryAnchor) {
        if (!this.markerResources) { this.overflowed = true; return; }
        encoder.copyBufferToBuffer(this.finalBoundaryAnchor.source,
          this.finalBoundaryAnchor.offset, this.markerResources.buffer, 0, 4);
      }
      const marker = encoder.beginComputePass({
        label: "GPU stage trace close",
        timestampWrites: { querySet: this.querySet, endOfPassWriteIndex },
      });
      if (this.markerResources) {
        marker.setPipeline(this.markerResources.pipeline);
        marker.setBindGroup(0, this.markerResources.bindGroup);
        marker.dispatchWorkgroups(1);
      }
      marker.end();
    }
    if (this.boundarySlots.length !== this.phases.length + 1) { this.overflowed = true; return; }
    const bytes = this.queryCount * 8;
    encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, bytes);
  }

  async read(): Promise<PerformanceTrace | undefined> {
    try {
      if (!this.resolved || this.overflowed) return undefined;
      const bytes = this.queryCount * 8;
      await this.readBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
      const resolved = new BigUint64Array(this.readBuffer.getMappedRange(0, bytes).slice(0));
      if ((globalThis as { process?: { env?: Record<string, string | undefined> } })
        .process?.env?.FLUID_TRACE_DEBUG === "1") {
        console.warn(JSON.stringify({
          record: "stage-trace", context: this.context, sampleId: this.sampleId,
          queryCount: this.queryCount, slots: this.boundarySlots,
          phases: this.phases.map((phase) => phase.label),
          resolved: Array.from(resolved, String),
        }));
      }
      return decodeGPUTimestampPartition({
        sampleId: this.sampleId,
        lane: this.lane,
        context: this.context,
        capturedAt_ms: performance.now(),
        timestamps: this.boundarySlots.map((slot) => resolved[slot] ?? 0n),
        phases: this.phases,
      });
    } finally {
      if (this.readBuffer.mapState === "mapped") this.readBuffer.unmap();
      this.destroy();
    }
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    this.readBuffer.destroy();
    this.encoderBreakSource.destroy();
    this.encoderBreakTarget.destroy();
  }
}

/** One pass, bracketed by its own pair of counters. */
export interface GPUPassTimestampSample {
  label: string;
  kind: "compute" | "render";
  /** Nanosecond-domain begin, rebased on the earliest sampled begin in the frame. */
  begin_ms: number;
  end_ms: number;
  /**
   * End minus begin. **Only a cost when `trusted`.** For a render pass on a
   * tile-based deferred GPU this is a *window*, not a duration — see `trusted`.
   */
  duration_ms: number;
  /** False when the hardware wrote no counter for this pass; see `read`. */
  sampled: boolean;
  /**
   * Whether `duration_ms` is this pass's cost.
   *
   * True for compute passes, false for render passes on a tile-based deferred
   * renderer, and the difference is measured rather than assumed. Apple's tiler
   * hoists every pass's vertex stage to the front of the frame and defers its
   * fragment stage to the back, and the counter pair brackets [vertex start,
   * fragment end] — so a render pass's window can span nearly the whole frame
   * whatever it cost. From this repository's own xctrace capture
   * (`docs/svo-render-frame-anatomy.data.js`): the deferred dry-lighting pass
   * runs its vertex stage at t=1.159 and its fragment stage at t=393.659 inside
   * a 405.6 ms frame, so its pair reads ~403 ms for 10.7 ms of work.
   *
   * Cross-checked against xctrace on the SVO render path: compute passes agree
   * (compact cone visibility, 29.62 timestamp vs 28.29 xctrace) and render
   * passes do not (live-scene primitive overflow, 29.82 vs 0.144 — 200x). The
   * tell is that the last-encoded render pass tracks the frame wall minus a
   * constant across an 8x frame-time change.
   *
   * Render-pass attribution on this hardware needs per-stage sampling, which
   * WebGPU does not expose; `tools/profile-svo-render-xctrace.ts` is the lane
   * that can answer it. Nothing here may quietly present a window as a cost.
   */
  trusted: boolean;
}

export interface GPUPassTimestampReading {
  passes: readonly GPUPassTimestampSample[];
  /**
   * Earliest sampled begin to latest sampled end. This *is* a real hardware
   * interval and is the honest answer to "how long was the GPU busy with this
   * frame": both endpoints are genuine hardware events (the first stage to
   * start and the last to finish), and overlap between passes is inside it
   * exactly once. It survives the tiler reordering that invalidates individual
   * render-pass durations, because reordering moves stages *within* the span.
   */
  span_ms: number;
  /**
   * Sum of every per-pass duration, trusted or not. Retained only so a caller
   * can see how far the render-pass windows inflate it; it is not a frame time
   * and it is not an attribution. Prefer `trustedSum_ms`.
   */
  sum_ms: number;
  /**
   * Sum over compute passes alone — the passes whose counters bracket their own
   * work. Still an over-count where adjacent compute stages stay in flight
   * (measured at up to 1.9x on this path), so it is a ratio instrument between
   * arms under identical instrumentation, never a clock.
   */
  trustedSum_ms: number;
  /** `sum_ms / span_ms`. 1 is strictly serial; above 1 is the overlap factor. */
  overlap: number;
  sampledPassCount: number;
  /** Passes whose duration is a tiler window rather than a cost. */
  untrustedPassCount: number;
}

/**
 * Per-pass GPU time, taken from each pass's own begin/end counters.
 *
 * ## Why this exists beside the boundary-chain recorders above
 *
 * `GPUPerformanceTraceRecorder`, `DynamicGPUPerformanceTraceRecorder` and
 * `GPUStageTimestampRecorder` all read a frame as one *chain*: boundary i and
 * i+1 delimit phase i, so the phases partition the whole interval by
 * construction and nothing can be double counted. That is the right model when
 * the boundaries land where they were encoded. On Dawn/Metal in the SVO render
 * path they do not — a standalone marker compute pass gets scheduled away from
 * the work it was meant to bracket, and one displaced boundary poisons the
 * entire frame rather than one phase. Measured on this path: a 95 ms frame came
 * back as five boundaries spanning 327 us with one 90 ms sample *ahead* of its
 * successor, and `decodeGPUTimestampPartition` correctly rejected all eighteen
 * attempts. That is what makes `FLUID_SVO_DRY_FRAME_PHASE_TRACE=1` and
 * `FLUID_SVO_DRY_FRAME_TIMING=gpu` abort rather than mis-report.
 *
 * A `beginningOfPassWriteIndex`/`endOfPassWriteIndex` pair has no such failure
 * mode: the two counters bracket that pass and nothing else, so a reordered
 * pass reports its own duration in the wrong place in the list rather than
 * corrupting its neighbours.
 *
 * ## What is given up, stated plainly
 *
 * Two things, and the second is the sharper one.
 *
 * **The durations no longer sum to the frame.** Apple keeps adjacent stages in
 * flight, so `sum_ms` over-counts — 191.8 ms of passes inside a 99.0 ms fence,
 * in the capture that motivated this class. `span_ms` is the frame-time answer,
 * the sums are the attribution answer, and `overlap` reports how far apart they
 * are so a caller cannot quietly treat one as the other.
 *
 * **A render pass's pair is not a duration at all.** See `trusted` on
 * `GPUPassTimestampSample`: this is a tile-based deferred GPU, the pair
 * brackets [tiler-hoisted vertex start, deferred fragment end], and the window
 * can span the frame whatever the pass cost — measured 200x wrong on this path.
 * So render-pass rows here carry `trusted: false`, they are excluded from
 * `trustedSum_ms` and from `passTimestampPerformanceTrace`, and any report that
 * shows them must mark them unavailable rather than plot them. That is not a
 * conservative stance: publishing a window as a cost has already produced one
 * wrong conclusion on this program ("deferred lighting scales 10.5x with record
 * count"; it is 2.70 ms at 1x and 6.21 ms at 10x).
 */
export class GPUPassTimestampRecorder {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;
  /** Dawn/Metal folds adjacent compute passes into one encoder, and a folded
   * pass never receives its counter sample. A 4-byte blit ahead of the pass
   * forces the encoder break that makes the sample observable. It is the only
   * command this recorder adds. */
  private readonly encoderBreakSource: GPUBuffer;
  private readonly encoderBreakTarget: GPUBuffer;
  private readonly labels: { label: string; kind: "compute" | "render" }[] = [];
  private readonly semanticPhases: Array<{
    phase: GPUTimestampPhase;
    firstPass: number;
    endPass: number;
  }> = [];
  private semanticPhaseStart = 0;
  private resolved = false;
  private disposed = false;

  static supported(device: GPUDevice): boolean {
    return device.features.has("timestamp-query");
  }

  constructor(
    private readonly device: GPUDevice,
    /** Two counters per pass. */
    private readonly capacity = 512,
    private readonly label = "pass timestamps",
  ) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: capacity });
    this.resolveBuffer = device.createBuffer({
      label: `${label} resolve`,
      size: capacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      label: `${label} read`,
      size: capacity * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.encoderBreakSource = device.createBuffer({
      label: `${label} encoder break source`, size: 4, usage: GPUBufferUsage.COPY_SRC,
    });
    this.encoderBreakTarget = device.createBuffer({
      label: `${label} encoder break target`, size: 4, usage: GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Wrap the frame's encoder; every pass opened through it carries a pair.
   * A descriptor that already asked for timestamp writes belongs to another
   * recorder and is never displaced.
   */
  instrument(encoder: GPUCommandEncoder): GPUCommandEncoder {
    const claim = (target: GPUCommandEncoder, descriptorLabel: string | undefined, occupied: boolean,
      kind: "compute" | "render") => {
      if (occupied || this.disposed || 2 * (this.labels.length + 1) > this.capacity) return undefined;
      const base = 2 * this.labels.length;
      this.labels.push({ label: descriptorLabel ?? `pass ${this.labels.length}`, kind });
      target.copyBufferToBuffer(this.encoderBreakSource, 0, this.encoderBreakTarget, 0, 4);
      return { querySet: this.querySet, beginningOfPassWriteIndex: base, endOfPassWriteIndex: base + 1 };
    };
    return new Proxy(encoder, {
      get: (target, property) => {
        if (property === "beginComputePass") {
          return (descriptor?: GPUComputePassDescriptor) => {
            const writes = claim(target, descriptor?.label, descriptor?.timestampWrites !== undefined, "compute");
            return target.beginComputePass(writes ? { ...descriptor, timestampWrites: writes } : descriptor);
          };
        }
        if (property === "beginRenderPass") {
          return (descriptor: GPURenderPassDescriptor) => {
            const writes = claim(target, descriptor.label, descriptor.timestampWrites !== undefined, "render");
            return target.beginRenderPass(writes ? { ...descriptor, timestampWrites: writes } : descriptor);
          };
        }
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as GPUCommandEncoder;
  }

  /** Close one semantic stage over the passes encoded since the prior seam. */
  completePhase(phase: GPUTimestampPhase): void {
    this.semanticPhases.push({
      phase,
      firstPass: this.semanticPhaseStart,
      endPass: this.labels.length,
    });
    this.semanticPhaseStart = this.labels.length;
  }

  /** False when the instrumented encoder opened no pass at all. */
  resolve(encoder: GPUCommandEncoder): boolean {
    if (this.labels.length === 0 || this.disposed) return false;
    this.resolved = true;
    const count = 2 * this.labels.length;
    encoder.resolveQuerySet(this.querySet, 0, count, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, count * 8);
    return true;
  }

  /**
   * Every pass, with the ones the hardware never sampled marked rather than
   * guessed at.
   *
   * A pass whose counters both come back zero was folded away or ran no stage —
   * `raster-primary` encodes a thin-glass discovery pass with an instance count
   * of zero, and that reports nothing on this hardware. Failing the whole frame
   * on it would mean the raster path could never be measured at all, so an
   * unsampled pass keeps its place in the list at zero. An *inverted* sampled
   * pair still fails the frame, because that is the hardware disagreeing with
   * itself rather than declining to answer.
   */
  async read(): Promise<GPUPassTimestampReading | undefined> {
    try {
      if (!this.resolved || this.disposed) return undefined;
      const bytes = 2 * this.labels.length * 8;
      await this.readBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
      const stamps = new BigUint64Array(this.readBuffer.getMappedRange(0, bytes).slice(0));
      let origin: bigint | undefined;
      let latest: bigint | undefined;
      for (let index = 0; index < this.labels.length; index += 1) {
        const begin = stamps[2 * index];
        const end = stamps[2 * index + 1];
        if (begin === 0n || end === 0n) continue;
        if (end < begin) return undefined;
        if (origin === undefined || begin < origin) origin = begin;
        if (latest === undefined || end > latest) latest = end;
      }
      if (origin === undefined || latest === undefined) return undefined;
      const passes: GPUPassTimestampSample[] = [];
      let sum_ms = 0;
      let trustedSum_ms = 0;
      for (let index = 0; index < this.labels.length; index += 1) {
        const { label, kind } = this.labels[index];
        const trusted = kind === "compute";
        const begin = stamps[2 * index];
        const end = stamps[2 * index + 1];
        if (begin === 0n || end === 0n) {
          passes.push({ label, kind, begin_ms: 0, end_ms: 0, duration_ms: 0, sampled: false, trusted: false });
          continue;
        }
        const duration_ms = Number(end - begin) / 1e6;
        sum_ms += duration_ms;
        if (trusted) trustedSum_ms += duration_ms;
        passes.push({
          label,
          kind,
          begin_ms: Number(begin - origin) / 1e6,
          end_ms: Number(end - origin) / 1e6,
          duration_ms,
          sampled: true,
          trusted,
        });
      }
      const span_ms = Number(latest - origin) / 1e6;
      if (!Number.isFinite(span_ms) || span_ms <= 0 || span_ms >= 10_000) return undefined;
      return {
        passes,
        span_ms,
        sum_ms,
        trustedSum_ms,
        overlap: sum_ms / span_ms,
        sampledPassCount: passes.filter(({ sampled }) => sampled).length,
        untrustedPassCount: passes.filter(({ sampled, trusted }) => sampled && !trusted).length,
      };
    } finally {
      if (this.readBuffer.mapState === "mapped") this.readBuffer.unmap();
      this.destroy();
    }
  }

  /**
   * Aggregate trustworthy pass pairs under the semantic stages that owned them.
   *
   * Render-pass pairs are tiler windows on Apple GPUs, not costs, so they are
   * excluded from every sum — but only *they* are. A stage used to be dropped
   * whole if it contained a single render pass or a single pass the hardware
   * declined to sample, which threw away the compute time beside them and, far
   * worse, made the stage indistinguishable from one that encoded nothing.
   * Downstream that silence was read as "unmeasured", and the render panel's
   * band-wall attribution handed the band's entire wall to whichever row was
   * silent. A stage now reports the part of itself that is real; what it
   * leaves out is a pass count the frame's manifest carries beside it.
   */
  async readSemanticTrace(input: {
    sampleId: number;
    lane: "physics" | "presentation";
    context: string;
    capturedAt_ms?: number;
  }): Promise<PerformanceTrace | undefined> {
    const phases = this.semanticPhases.map((group) => ({ ...group }));
    const reading = await this.read();
    if (!reading) return undefined;
    const trustedPasses: GPUPassTimestampSample[] = [];
    for (const group of phases) {
      const passes = reading.passes
        .slice(group.firstPass, group.endPass)
        .filter((pass) => pass.sampled && pass.trusted);
      if (passes.length === 0) continue;
      trustedPasses.push(...passes.map((pass) => ({ ...pass, label: group.phase.label })));
    }
    if (trustedPasses.length === 0) return undefined;
    const trustedSum_ms = trustedPasses.reduce((sum, pass) => sum + pass.duration_ms, 0);
    return passTimestampPerformanceTrace({
      reading: {
        passes: trustedPasses,
        span_ms: reading.span_ms,
        sum_ms: trustedSum_ms,
        trustedSum_ms,
        overlap: reading.span_ms > 0 ? trustedSum_ms / reading.span_ms : 0,
        sampledPassCount: trustedPasses.length,
        untrustedPassCount: 0,
      },
      sampleId: input.sampleId,
      lane: input.lane,
      context: input.context,
      capturedAt_ms: input.capturedAt_ms,
    });
  }

  /**
   * Turn pass completion timestamps into one exclusive semantic frame ledger.
   *
   * A render pass's begin/end pair is not its cost on a tile GPU: tiling can
   * begin early while fragments retire much later. Its completion timestamp is
   * still a real point on the GPU timeline. Walking semantic groups in encode
   * order and charging only the amount by which each group advances the latest
   * observed completion gives an additive critical-path partition. Concurrent
   * work already hidden behind an earlier completion is zero instead of being
   * double-counted under every overlapping stage.
   */
  async readSemanticFrontierTrace(input: {
    sampleId: number;
    lane: "physics" | "presentation";
    context: string;
    capturedAt_ms?: number;
  }): Promise<PerformanceTrace | undefined> {
    const groups = this.semanticPhases.map((group) => ({ ...group }));
    const reading = await this.read();
    if (!reading) return undefined;
    let frontier_ms = 0;
    const phases: PerformancePhaseSample[] = groups.map((group) => {
      const sampled = reading.passes
        .slice(group.firstPass, group.endPass)
        .filter((pass) => pass.sampled);
      if (sampled.length === 0) return { ...group.phase, duration_ms: 0 };
      const completed_ms = Math.max(...sampled.map((pass) => pass.end_ms));
      const duration_ms = Math.max(0, completed_ms - frontier_ms);
      frontier_ms = Math.max(frontier_ms, completed_ms);
      return { ...group.phase, duration_ms };
    });
    if (!(reading.span_ms > 0) || !Number.isFinite(frontier_ms)) return undefined;
    // The reading origin is the earliest sampled pass beginning, so the final
    // completion frontier and the observed frame span are the same interval.
    // Retain the hardware span to absorb floating-point rounding exactly.
    const accounted_ms = phases.reduce((sum, phase) => sum + phase.duration_ms, 0);
    const rounding_ms = reading.span_ms - accounted_ms;
    if (rounding_ms !== 0) {
      const phase = phases.find((entry) => entry.duration_ms > 0) ?? phases[0];
      if (phase && phase.duration_ms + rounding_ms >= 0) phase.duration_ms += rounding_ms;
    }
    return {
      sampleId: input.sampleId,
      domain: "gpu",
      lane: input.lane,
      context: input.context,
      capturedAt_ms: input.capturedAt_ms ?? performance.now(),
      measurementSource: "gpu-pass-timestamp",
      total_ms: reading.span_ms,
      phases,
    };
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    this.readBuffer.destroy();
    this.encoderBreakSource.destroy();
    this.encoderBreakTarget.destroy();
  }
}

/**
 * A per-pass reading as a `PerformanceTrace`, laid out end to end.
 *
 * The intervals are synthetic — consecutive, in encode order — because the real
 * ones overlap and `partitionPerformanceTrace` rejects overlap rather than
 * clipping it into a lie. The total is therefore `sum_ms`, not `span_ms`, and
 * the context carries `:pass-sum` so no consumer can read the total as a frame
 * time by accident. `span_ms` is what a frame time should be taken from.
 */
export function passTimestampPerformanceTrace(input: {
  reading: GPUPassTimestampReading;
  sampleId: number;
  lane: "physics" | "presentation";
  context: string;
  capturedAt_ms?: number;
  classify?: (label: string) => PaperPhaseId;
}): PerformanceTrace | undefined {
  // Compute passes only. A render pass's counter pair is a tiler window and
  // putting it in a phase would publish the frame wall under a shader's name —
  // the failure this whole file exists to make impossible.
  const sampled = input.reading.passes.filter(({ sampled, trusted }) => sampled && trusted);
  if (sampled.length === 0) return undefined;
  let cursor = 0;
  const intervals: PerformanceInterval[] = sampled.map((pass) => {
    const start_ms = cursor;
    cursor += pass.duration_ms;
    return {
      id: input.classify?.(pass.label) ?? "other",
      label: pass.label,
      start_ms,
      end_ms: cursor,
    };
  });
  const trace = partitionPerformanceTrace({
    sampleId: input.sampleId,
    domain: "gpu",
    lane: input.lane,
    context: `${input.context}:compute-pass-sum`,
    measurementSource: "gpu-pass-timestamp",
    // A pass-sum trace starts at synthetic zero; using its end as the default
    // capture time makes freshness checks discard every otherwise valid sample.
    capturedAt_ms: input.capturedAt_ms ?? performance.now(),
    start_ms: 0,
    end_ms: cursor,
    intervals,
  });
  // The span rides along: it is the honest "how long was the GPU busy" answer
  // (earliest sampled begin to latest sampled end, across every sampled pass,
  // untrusted render passes included), where the total above is a sum that
  // double-counts overlap. Shares of the frame should be shares of this.
  return { ...trace, span_ms: input.reading.span_ms };
}

/**
 * Non-invasive fallback when hardware timestamps are unsupported or invalid.
 *
 * This observes submission-to-queue-completion wall time. It is an honest GPU
 * queue observation, but cannot attribute hardware time to semantic subphases,
 * so the complete duration is retained as one explicit fallback phase.
 */
export class GPUQueueWallPerformanceTraceRecorder {
  private startedAt_ms?: number;

  constructor(
    private readonly sampleId: number,
    private readonly lane: "physics" | "presentation",
    private readonly context: string,
    private readonly clock: () => number = () => performance.now(),
  ) {}

  begin(): void {
    if (this.startedAt_ms !== undefined) throw new Error("GPU queue-wall trace already started");
    this.startedAt_ms = this.clock();
  }

  async read(queue: Pick<GPUQueue, "onSubmittedWorkDone">): Promise<PerformanceTrace> {
    if (this.startedAt_ms === undefined) throw new Error("GPU queue-wall trace has not started");
    const start_ms = this.startedAt_ms;
    await queue.onSubmittedWorkDone();
    const end_ms = Math.max(start_ms, this.clock());
    return partitionPerformanceTrace({
      sampleId: this.sampleId,
      domain: "gpu",
      lane: this.lane,
      context: `${this.context}:queue-wall-fallback`,
      measurementSource: "gpu-queue-wall",
      capturedAt_ms: end_ms,
      start_ms,
      end_ms,
      intervals: [{
        id: "other",
        label: "GPU queue completion · hardware timestamps unavailable or invalid",
        start_ms,
        end_ms,
      }],
    });
  }
}
