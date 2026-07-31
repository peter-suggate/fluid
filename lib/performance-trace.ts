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
  | "gpu-queue-wall";

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
  | "svo-primary"
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
}

export interface PerformanceTrace {
  sampleId: number;
  domain: PerformanceDomain;
  lane: PerformanceLane;
  context: string;
  capturedAt_ms: number;
  measurementSource?: PerformanceMeasurementSource;
  total_ms: number;
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
  return trace.phases.reduce((sum, phase) => sum + phase.duration_ms, 0);
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
      lane: "main-thread",
      context: this.context,
      capturedAt_ms: completedAt_ms,
      start_ms: this.startedAt_ms,
      end_ms: completedAt_ms,
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
  if (input.timestamps.length !== input.phases.length + 1 || input.phases.length === 0) return undefined;
  const origin = input.timestamps[0];
  if (origin === undefined || origin === 0n) return undefined;
  const intervals: PerformanceInterval[] = [];
  let previous = origin;
  for (let index = 0; index < input.phases.length; index += 1) {
    const next = input.timestamps[index + 1];
    if (next === undefined || next === 0n || next < previous) return undefined;
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
  const phases = [...totals.values()].map(({ count, ...phase }) => ({
    ...phase,
    duration_ms: phase.duration_ms / observations.length,
  }));
  const total_ms = observations.reduce((sum, trace) => sum + trace.total_ms, 0) / observations.length;
  const accounted_ms = phases.reduce((sum, phase) => sum + phase.duration_ms, 0);
  const delta_ms = total_ms - accounted_ms;
  // Averaging independently accumulated IEEE-754 values can leave a
  // sub-nanosecond negative residual. Do not turn that harmless dust into a
  // negative "other" phase, which would make a closed trace appear invalid.
  if (Math.abs(delta_ms) > 1e-9) {
    const other = phases.find((phase) => phase.id === "other");
    if (other && other.duration_ms + delta_ms >= 0) other.duration_ms += delta_ms;
    else if (delta_ms > 0) phases.push({ id: "other", label: "Other measured work", duration_ms: delta_ms });
    else {
      const largest = phases.reduce((candidate, phase) =>
        phase.duration_ms > candidate.duration_ms ? phase : candidate, phases[0]);
      if (largest && largest.duration_ms + delta_ms >= 0) largest.duration_ms += delta_ms;
    }
  }
  return { ...latest, total_ms, phases };
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
  private readonly markerPipeline: GPUComputePipeline;
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
    this.markerPipeline = dynamicTraceMarkerPipeline(device);
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
    marker.setPipeline(this.markerPipeline);
    marker.dispatchWorkgroups(1);
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

const dynamicTraceMarkerPipelines = new WeakMap<GPUDevice, GPUComputePipeline>();
function dynamicTraceMarkerPipeline(device: GPUDevice): GPUComputePipeline {
  const cached = dynamicTraceMarkerPipelines.get(device);
  if (cached) return cached;
  const module = device.createShaderModule({
    label: "dynamic trace marker",
    code: "@compute @workgroup_size(1) fn mark() {}",
  });
  const pipeline = device.createComputePipeline({
    label: "dynamic trace marker",
    layout: "auto",
    compute: { module, entryPoint: "mark" },
  });
  dynamicTraceMarkerPipelines.set(device, pipeline);
  return pipeline;
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
  private readonly markerPipeline: GPUComputePipeline;
  private readonly phases: GPUTimestampPhase[] = [];
  private boundaryCount = 0;
  private disposed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly sampleId: number,
    private readonly lane: "physics" | "presentation",
    private readonly context: string,
    private readonly capacity = 128,
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
    this.markerPipeline = dynamicTraceMarkerPipeline(device);
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
    marker.setPipeline(this.markerPipeline);
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
  private readonly markerPipeline: GPUComputePipeline;
  private readonly phases: GPUTimestampPhase[] = [];
  /** Query slot each boundary landed on; repeats mark an empty stage. */
  private readonly boundarySlots: number[] = [];
  private armedBoundaries = 0;
  private queryCount = 0;
  private started = false;
  private resolved = false;
  private overflowed = false;
  private disposed = false;

  static supported(device: GPUDevice): boolean {
    return device.features.has("timestamp-query");
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
    this.markerPipeline = dynamicTraceMarkerPipeline(device);
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
   * Assign every armed boundary to the pass about to begin. A descriptor that
   * already carries timestamp writes belongs to another recorder and is never
   * displaced; its boundaries stay armed for the pass after it.
   */
  private claimBoundary(encoder: GPUCommandEncoder, occupied: boolean) {
    if (this.armedBoundaries === 0 || this.disposed || this.overflowed || occupied) return undefined;
    if (this.queryCount >= this.capacity) { this.overflowed = true; return undefined; }
    const beginningOfPassWriteIndex = this.queryCount;
    this.queryCount += 1;
    for (let boundary = 0; boundary < this.armedBoundaries; boundary += 1) {
      this.boundarySlots.push(beginningOfPassWriteIndex);
    }
    this.armedBoundaries = 0;
    encoder.copyBufferToBuffer(this.encoderBreakSource, 0, this.encoderBreakTarget, 0, 4);
    return { querySet: this.querySet, beginningOfPassWriteIndex };
  }

  /**
   * Close the chain and stage the readback. The trailing boundary has no
   * following work to attach to, so this is the one marker pass the recorder
   * ever encodes.
   */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.started || this.resolved) throw new Error("GPU stage trace is not open");
    this.resolved = true;
    if (this.phases.length === 0) { this.overflowed = true; return; }
    // The last completed stage already armed the closing boundary. It has no
    // following work to ride, so this is the one pass the recorder encodes.
    if (this.armedBoundaries > 0) {
      const closing = this.claimBoundary(encoder, false);
      if (!closing) { this.overflowed = true; return; }
      const marker = encoder.beginComputePass({ label: "GPU stage trace close", timestampWrites: closing });
      marker.setPipeline(this.markerPipeline);
      marker.dispatchWorkgroups(1);
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
