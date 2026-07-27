import type {
  PerformanceMeasurementSource,
  PerformancePhaseSample,
  PerformanceTrace,
} from "./performance-trace";

/** The profiler UI renders fixed one-millisecond buckets. */
export const PERFORMANCE_ACTIVITY_SLICE_MS = 1 as const;

export type ActivityEvidence = "measured" | "reconstructed" | "idle" | "unknown";
export type ActivityResourceKind = "cpu-main" | "cpu-worker" | "gpu-aggregate" | "gpu-logical-capacity";
export type ActivityLane = "cpu-main" | "cpu-worker" | "gpu-physics" | "gpu-presentation";
export type ActivityClockDomain =
  | "cpu-performance"
  | "gpu-physics-timestamp"
  | "gpu-presentation-timestamp"
  | (string & {});

export interface ActivityFrameIdentity {
  /** Application-owned identity joining callbacks that contribute to one displayed frame. */
  frameId: string;
  /** Capture generation. Late work from an earlier enable cycle is rejected by the store. */
  generation: number;
}

export interface ActivityWorkIdentity extends ActivityFrameIdentity {
  /** Queue or CPU-task submission which produced this work. */
  submissionId?: string;
  /** Published state/texture/buffer generation consumed or produced by this work. */
  publicationId?: string;
}

/**
 * Stable join key shared by GPU readback producers and the later controller
 * publication. It deliberately excludes readback/completion wall time.
 */
export function gpuPhysicsPerformanceActivityFrameId(
  trace: Pick<PerformanceTrace, "sampleId" | "context">,
): string {
  return `gpu-physics:${encodeURIComponent(trace.context)}:${trace.sampleId}`;
}

export interface CPUPerformanceClockSnapshot {
  /** `performance.now()` coordinate. */
  now_ms: number;
  /** Unix-epoch offset for the CPU performance clock. */
  timeOrigin_ms: number;
  /** `timeOrigin_ms + now_ms`, useful for cross-context CPU alignment. */
  epoch_ms: number;
}

export function cpuPerformanceClockSnapshot(
  source: Pick<Performance, "now" | "timeOrigin"> = performance,
): CPUPerformanceClockSnapshot {
  const now_ms = source.now();
  const timeOrigin_ms = source.timeOrigin;
  return { now_ms, timeOrigin_ms, epoch_ms: timeOrigin_ms + now_ms };
}

export interface ActivityClockAlignment {
  from: ActivityClockDomain;
  to: ActivityClockDomain;
  /** `toTime = fromTime + offset_ms`. */
  offset_ms: number;
  /** Non-negative error bound; zero is allowed only for a shared clock. */
  uncertainty_ms: number;
  source: "shared-time-origin" | "calibrated" | "supplied";
}

export type ActivityClockStatus =
  | { state: "reference" }
  | { state: "synchronized"; to: ActivityClockDomain; offset_ms: number; uncertainty_ms: number; source: ActivityClockAlignment["source"] }
  | { state: "unsynchronized" };

export interface ActivityClockDescriptor {
  id: ActivityClockDomain;
  label: string;
  /** CPU performance coordinates have an epoch origin. GPU query coordinates do not. */
  epochOrigin_ms?: number;
  status: ActivityClockStatus;
}

export interface ActivityTask {
  id: string;
  label: string;
  color: string;
  lane: ActivityLane;
  /** Stable semantic stage used to group dynamically registered tasks in the UI. */
  stageId?: string;
  parentId?: string;
}

export interface ActivitySpan {
  id: string;
  kind: "task" | "observation";
  taskId?: string;
  resourceId: string;
  clockDomain: ActivityClockDomain;
  start_ms: number;
  end_ms: number;
  evidence: Exclude<ActivityEvidence, "idle" | "unknown">;
  identity: ActivityWorkIdentity;
  metadata?: Readonly<Record<string, string | number | boolean | null>>;
}

export interface ActivityEvent {
  id: string;
  kind: "begin" | "end" | "instant" | "submit" | "publish";
  taskId?: string;
  resourceId: string;
  clockDomain: ActivityClockDomain;
  at_ms: number;
  evidence: Exclude<ActivityEvidence, "idle" | "unknown">;
  identity: ActivityWorkIdentity;
  metadata?: Readonly<Record<string, string | number | boolean | null>>;
}

export interface ActivityResource {
  id: string;
  label: string;
  kind: ActivityResourceKind;
  lane: ActivityLane;
  clockDomain: ActivityClockDomain;
  /** Logical capacity index only. It is never presented as a physical GPU core ID. */
  capacitySlot?: number;
}

export interface ActivitySlice {
  index: number;
  start_ms: number;
  end_ms: number;
  evidence: ActivityEvidence;
  taskId?: string;
  /** Union of task-span overlap with this bucket, bounded to the bucket width. */
  active_ms: number;
  spanIds: string[];
  eventIds: string[];
}

export interface ActivityRow {
  resource: ActivityResource;
  windowStart_ms: number;
  windowEnd_ms: number;
  slice_ms: typeof PERFORMANCE_ACTIVITY_SLICE_MS;
  slices: ActivitySlice[];
}

export interface PerformanceActivityFrame {
  identity: ActivityFrameIdentity;
  context: string;
  /** CPU performance-clock capture coordinate and its epoch-aligned equivalent. */
  capturedAt_cpu_ms: number;
  capturedAt_epoch_ms: number;
  cpuTimeOrigin_ms: number;
  clocks: ActivityClockDescriptor[];
  tasks: ActivityTask[];
  resources: ActivityResource[];
  /** Raw evidence remains authoritative; rows/slices are a derived display index. */
  spans: ActivitySpan[];
  events: ActivityEvent[];
  rows: ActivityRow[];
  /**
   * End-to-end capture validation. Absence means the asynchronous recorder has
   * not supplied a verdict yet; it must not be interpreted as complete.
   */
  captureDiagnostics?: PerformanceActivityCaptureDiagnostics;
}

export type PerformanceActivityCaptureReason =
  | "recorder-overflow"
  | "missing-frame-begin"
  | "missing-frame-end"
  | "unprofiled-dispatch"
  | "row-limit"
  | "unprojected-event"
  | "timestamp-fallback"
  | (string & {});

/** Capture-level diagnostics are retained with the frame, not only returned to the producer. */
export interface PerformanceActivityCaptureDiagnostics {
  /** A capture is complete exactly when this collection is empty. */
  reasons: readonly PerformanceActivityCaptureReason[];
  recorderOverflowed?: boolean;
  droppedEventCount?: number;
  droppedRowCount?: number;
  unprojectedEventCount?: number;
  unprofiledDispatchCount?: number;
  unprofiledPipelineLabels?: readonly string[];
}

export interface ActivityRowWindow {
  resourceId: string;
  start_ms: number;
  end_ms: number;
}

const finiteTime = (value: number, fallback = 0) => Number.isFinite(value) ? value : fallback;

const overlap_ms = (startA: number, endA: number, startB: number, endB: number) =>
  Math.max(0, Math.min(endA, endB) - Math.max(startA, startB));

/** Deterministic task color: IDs remain visually stable when display labels change. */
export function performanceActivityTaskColor(taskId: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < taskId.length; index += 1) {
    hash ^= taskId.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  const hue = (hash >>> 0) % 360;
  const saturation = 62 + ((hash >>> 9) & 15);
  const lightness = 48 + ((hash >>> 17) & 7);
  return hslToHex(hue, saturation, lightness);
}

function hslToHex(hue: number, saturation: number, lightness: number): string {
  const s = saturation / 100;
  const l = lightness / 100;
  const chroma = (1 - Math.abs(2 * l - 1)) * s;
  const section = hue / 60;
  const x = chroma * (1 - Math.abs(section % 2 - 1));
  const [r1, g1, b1] = section < 1 ? [chroma, x, 0]
    : section < 2 ? [x, chroma, 0]
      : section < 3 ? [0, chroma, x]
        : section < 4 ? [0, x, chroma]
          : section < 5 ? [x, 0, chroma]
            : [chroma, 0, x];
  const match = l - chroma / 2;
  const byte = (value: number) => Math.round((value + match) * 255).toString(16).padStart(2, "0");
  return `#${byte(r1)}${byte(g1)}${byte(b1)}`;
}

const evidencePriority: Readonly<Record<Exclude<ActivityEvidence, "idle" | "unknown">, number>> = {
  measured: 2,
  reconstructed: 1,
};

/**
 * Derive one-millisecond display rows without discarding the raw spans/events.
 * Empty time inside an observation is idle; time outside evidence is unknown.
 */
export function slicePerformanceActivityRows(input: {
  resources: readonly ActivityResource[];
  spans: readonly ActivitySpan[];
  events?: readonly ActivityEvent[];
  windows: readonly ActivityRowWindow[];
}): ActivityRow[] {
  const events = input.events ?? [];
  return input.resources.map((resource) => {
    const window = input.windows.find((candidate) => candidate.resourceId === resource.id)
      ?? { resourceId: resource.id, start_ms: 0, end_ms: 1 };
    const windowStart_ms = finiteTime(window.start_ms);
    const windowEnd_ms = Math.max(windowStart_ms, finiteTime(window.end_ms, windowStart_ms));
    const count = Math.max(1, Math.ceil((windowEnd_ms - windowStart_ms) / PERFORMANCE_ACTIVITY_SLICE_MS));
    const spans = input.spans.filter((span) => span.resourceId === resource.id && span.end_ms > span.start_ms);
    const observations = spans.filter((span) => span.kind === "observation");
    const tasks = spans.filter((span) => span.kind === "task" && span.taskId);
    const rowEvents = events.filter((event) => event.resourceId === resource.id);
    const slices: ActivitySlice[] = [];
    for (let index = 0; index < count; index += 1) {
      const start_ms = windowStart_ms + index * PERFORMANCE_ACTIVITY_SLICE_MS;
      const end_ms = Math.min(windowEnd_ms, start_ms + PERFORMANCE_ACTIVITY_SLICE_MS);
      const width = Math.max(0, end_ms - start_ms);
      const candidates = tasks.map((span) => ({ span, overlap: overlap_ms(start_ms, end_ms, span.start_ms, span.end_ms) }))
        .filter((candidate) => candidate.overlap > 0)
        .sort((left, right) => evidencePriority[right.span.evidence] - evidencePriority[left.span.evidence]
          || right.overlap - left.overlap
          || left.span.id.localeCompare(right.span.id));
      const active_ms = Math.min(width, candidates.reduce((sum, candidate) => sum + candidate.overlap, 0));
      const observed_ms = Math.min(width, observations.reduce((sum, observation) =>
        sum + overlap_ms(start_ms, end_ms, observation.start_ms, observation.end_ms), 0));
      const dominant = candidates[0]?.span;
      const sliceEvents = rowEvents.filter((event) => event.at_ms >= start_ms
        && (event.at_ms < end_ms || index === count - 1 && event.at_ms === end_ms));
      const observedProgress = sliceEvents.find((event) => event.evidence === "measured")
        ?? sliceEvents.find((event) => event.evidence === "reconstructed");
      const taskId = dominant?.taskId ?? observedProgress?.taskId;
      const evidence: ActivityEvidence = dominant?.evidence
        ?? observedProgress?.evidence
        ?? (width > 0 && observed_ms >= width - 1e-9 ? "idle" : "unknown");
      slices.push({
        index,
        start_ms,
        end_ms,
        evidence,
        ...(taskId ? { taskId } : {}),
        active_ms,
        spanIds: candidates.map((candidate) => candidate.span.id),
        eventIds: sliceEvents.map((event) => event.id),
      });
    }
    return { resource, windowStart_ms, windowEnd_ms, slice_ms: PERFORMANCE_ACTIVITY_SLICE_MS, slices };
  });
}

export interface PerformanceActivityFrameAddition {
  resources?: readonly ActivityResource[];
  clocks?: readonly ActivityClockDescriptor[];
  tasks?: readonly ActivityTask[];
  spans?: readonly ActivitySpan[];
  events?: readonly ActivityEvent[];
  /** Explicit full row windows keep unobserved buckets unknown instead of cropping them away. */
  windows?: readonly ActivityRowWindow[];
  /** Capture verdict may arrive asynchronously with GPU readback evidence. */
  captureDiagnostics?: PerformanceActivityCaptureDiagnostics;
}

export type PerformanceActivityEvidenceIngestResult = "merged" | "buffered" | "rejected";

/**
 * Evidence can arrive before its base frame, so validate it against the
 * producer-supplied identity without requiring a PerformanceActivityFrame.
 */
export function performanceActivityAdditionMatchesIdentity(
  identity: ActivityFrameIdentity,
  addition: PerformanceActivityFrameAddition,
): boolean {
  if (!identity.frameId || !Number.isSafeInteger(identity.generation) || identity.generation < 0) return false;
  return [...addition.spans ?? [], ...addition.events ?? []].every((record) =>
    record.identity.frameId === identity.frameId
    && record.identity.generation === identity.generation);
}

/**
 * Merge detailed CPU/worker/GPU-heartbeat evidence into an existing frame.
 * Raw records remain authoritative and rows are rebuilt from their union.
 */
export function mergePerformanceActivityFrame(
  frame: PerformanceActivityFrame,
  addition: PerformanceActivityFrameAddition,
): PerformanceActivityFrame {
  if (!performanceActivityAdditionMatchesIdentity(frame.identity, addition)) {
    const record = [...addition.spans ?? [], ...addition.events ?? []].find((candidate) =>
      candidate.identity.frameId !== frame.identity.frameId
      || candidate.identity.generation !== frame.identity.generation);
    const source = record?.identity ?? { frameId: "invalid", generation: -1 };
    throw new Error(`Cannot merge activity for ${source.frameId}@${source.generation} into ${frame.identity.frameId}@${frame.identity.generation}`);
  }
  const byId = <T extends { id: string }>(existing: readonly T[], incoming: readonly T[] = []) =>
    [...new Map([...existing, ...incoming].map((value) => [value.id, value])).values()];
  const resources = byId(frame.resources, addition.resources);
  const clocks = byId(frame.clocks, addition.clocks);
  const tasks = byId(frame.tasks, addition.tasks);
  const spans = byId(frame.spans, addition.spans);
  const events = byId(frame.events, addition.events);
  const captureDiagnostics = mergePerformanceActivityCaptureDiagnostics(
    frame.captureDiagnostics,
    addition.captureDiagnostics,
  );
  const windows = new Map(frame.rows.map((row) => [row.resource.id, {
    resourceId: row.resource.id,
    start_ms: row.windowStart_ms,
    end_ms: row.windowEnd_ms,
  }]));
  const suppliedWindows = new Map((addition.windows ?? []).map((window) => [window.resourceId, window]));
  for (const resource of resources) {
    const existing = windows.get(resource.id);
    const supplied = suppliedWindows.get(resource.id);
    const resourceSpans = spans.filter((span) => span.resourceId === resource.id);
    const resourceEvents = events.filter((event) => event.resourceId === resource.id);
    const starts = [
      ...(existing ? [existing.start_ms] : []),
      ...(supplied ? [supplied.start_ms] : []),
      ...resourceSpans.map((span) => span.start_ms),
      ...resourceEvents.map((event) => event.at_ms),
    ];
    const ends = [
      ...(existing ? [existing.end_ms] : []),
      ...(supplied ? [supplied.end_ms] : []),
      ...resourceSpans.map((span) => span.end_ms),
      ...resourceEvents.map((event) => event.at_ms),
    ];
    const start_ms = starts.length > 0 ? Math.min(...starts) : 0;
    const observedEnd_ms = ends.length > 0 ? Math.max(...ends) : start_ms;
    windows.set(resource.id, { resourceId: resource.id, start_ms, end_ms: Math.max(start_ms + 1, observedEnd_ms) });
  }
  return {
    ...frame,
    resources,
    clocks,
    tasks,
    spans,
    events,
    rows: slicePerformanceActivityRows({ resources, spans, events, windows: [...windows.values()] }),
    ...(captureDiagnostics ? { captureDiagnostics } : {}),
  };
}

/**
 * Capture diagnostics are capture-level totals. Maxima make repeated evidence
 * ingestion idempotent while reason union preserves every failed invariant.
 */
export function mergePerformanceActivityCaptureDiagnostics(
  current?: PerformanceActivityCaptureDiagnostics,
  incoming?: PerformanceActivityCaptureDiagnostics,
): PerformanceActivityCaptureDiagnostics | undefined {
  if (!current) return incoming ? { ...incoming, reasons: [...new Set(incoming.reasons)] } : undefined;
  if (!incoming) return current;
  return {
    reasons: [...new Set([...current.reasons, ...incoming.reasons])],
    recorderOverflowed: Boolean(current.recorderOverflowed || incoming.recorderOverflowed),
    droppedEventCount: Math.max(current.droppedEventCount ?? 0, incoming.droppedEventCount ?? 0),
    droppedRowCount: Math.max(current.droppedRowCount ?? 0, incoming.droppedRowCount ?? 0),
    unprojectedEventCount: Math.max(current.unprojectedEventCount ?? 0, incoming.unprojectedEventCount ?? 0),
    unprofiledDispatchCount: Math.max(current.unprofiledDispatchCount ?? 0, incoming.unprofiledDispatchCount ?? 0),
    unprofiledPipelineLabels: [...new Set([
      ...current.unprofiledPipelineLabels ?? [],
      ...incoming.unprofiledPipelineLabels ?? [],
    ])],
  };
}

export interface SynthesizePerformanceActivityOptions {
  identity: ActivityFrameIdentity;
  context: string;
  cpu?: PerformanceTrace;
  physics?: PerformanceTrace;
  presentation?: PerformanceTrace;
  /** Defaults to the executing context's `performance.timeOrigin`. */
  cpuTimeOrigin_ms?: number;
  /** Optional explicit capture coordinate; defaults to the latest CPU-coordinate trace capture. */
  capturedAt_cpu_ms?: number;
  alignments?: readonly ActivityClockAlignment[];
  cpuSubmissionId?: string;
  physicsSubmissionId?: string;
  presentationSubmissionId?: string;
  physicsPublicationId?: string;
  presentationPublicationId?: string;
  /** Logical rows, not physical cores. Aggregate current traces use slot zero only. */
  gpuLogicalCapacityRows?: Partial<Record<"physics" | "presentation", number>>;
  workerRows?: readonly { id: string; label: string; timeOrigin_ms?: number }[];
}

const traceClock = (trace: PerformanceTrace): ActivityClockDomain => trace.domain === "cpu"
  ? "cpu-performance"
  : trace.lane === "physics" ? "gpu-physics-timestamp" : "gpu-presentation-timestamp";

const traceActivityLane = (trace: PerformanceTrace): ActivityLane => trace.domain === "cpu"
  ? "cpu-main"
  : trace.lane === "physics" ? "gpu-physics" : "gpu-presentation";

const traceResourceId = (trace: PerformanceTrace) => trace.domain === "cpu"
  ? "cpu.main"
  : trace.lane === "physics" ? "gpu.physics.aggregate" : "gpu.presentation.aggregate";

const phaseTaskId = (trace: PerformanceTrace, phase: PerformancePhaseSample) =>
  `${trace.domain === "cpu" ? "cpu.main" : `gpu.${trace.lane}`}.${phase.id}`;

const measurementLabel = (source: PerformanceMeasurementSource | undefined) => source ?? "unspecified";

function identityForTrace(trace: PerformanceTrace, options: SynthesizePerformanceActivityOptions): ActivityWorkIdentity {
  const lane = trace.domain === "cpu" ? "cpu" : trace.lane;
  const submissionId = lane === "cpu"
    ? options.cpuSubmissionId ?? `cpu:${trace.sampleId}`
    : lane === "physics"
      ? options.physicsSubmissionId ?? `physics:${trace.sampleId}`
      : options.presentationSubmissionId ?? `presentation:${trace.sampleId}`;
  const publicationId = lane === "physics" ? options.physicsPublicationId
    : lane === "presentation" ? options.presentationPublicationId : undefined;
  return { ...options.identity, submissionId, ...(publicationId ? { publicationId } : {}) };
}

function alignmentStatus(
  clock: ActivityClockDomain,
  alignments: readonly ActivityClockAlignment[],
): ActivityClockStatus {
  if (clock === "cpu-performance") return { state: "reference" };
  const alignment = alignments.find((candidate) => candidate.from === clock && candidate.to === "cpu-performance");
  return alignment
    ? { state: "synchronized", to: alignment.to, offset_ms: alignment.offset_ms,
      uncertainty_ms: Math.max(0, alignment.uncertainty_ms), source: alignment.source }
    : { state: "unsynchronized" };
}

function appendTraceEvidence(
  trace: PerformanceTrace,
  options: SynthesizePerformanceActivityOptions,
  tasks: Map<string, ActivityTask>,
  spans: ActivitySpan[],
  events: ActivityEvent[],
  windows: ActivityRowWindow[],
): void {
  const resourceId = traceResourceId(trace);
  const clockDomain = traceClock(trace);
  const lane = traceActivityLane(trace);
  const end_ms = trace.domain === "cpu" ? trace.capturedAt_ms : Math.max(0, trace.total_ms);
  const start_ms = trace.domain === "cpu" ? end_ms - Math.max(0, trace.total_ms) : 0;
  const identity = identityForTrace(trace, options);
  windows.push({ resourceId, start_ms, end_ms });
  spans.push({
    id: `${resourceId}.observation.${trace.sampleId}`,
    kind: "observation",
    resourceId,
    clockDomain,
    start_ms,
    end_ms,
    evidence: "measured",
    identity,
    metadata: { source: measurementLabel(trace.measurementSource), context: trace.context },
  });
  events.push({
    id: `${resourceId}.begin.${trace.sampleId}`,
    kind: "begin",
    resourceId,
    clockDomain,
    at_ms: start_ms,
    evidence: "measured",
    identity,
  });
  let cursor_ms = start_ms;
  trace.phases.forEach((phase, index) => {
    const duration_ms = Math.max(0, finiteTime(phase.duration_ms));
    const phaseEnd_ms = Math.min(end_ms, cursor_ms + duration_ms);
    const taskId = phaseTaskId(trace, phase);
    if (!tasks.has(taskId)) tasks.set(taskId, {
      id: taskId,
      label: phase.label,
      color: performanceActivityTaskColor(taskId),
      lane,
      stageId: phase.id,
    });
    const spanId = `${resourceId}.${trace.sampleId}.${index}.${phase.id}`;
    if (phaseEnd_ms > cursor_ms) spans.push({
      id: spanId,
      kind: "task",
      taskId,
      resourceId,
      clockDomain,
      start_ms: cursor_ms,
      end_ms: phaseEnd_ms,
      // PerformanceTrace retains measured duration but not raw boundaries.
      evidence: "reconstructed",
      identity,
      metadata: { source: measurementLabel(trace.measurementSource), phaseIndex: index, sourceSampleId: trace.sampleId },
    });
    events.push({
      id: `${spanId}.boundary`,
      kind: index === trace.phases.length - 1 ? "end" : "instant",
      taskId,
      resourceId,
      clockDomain,
      at_ms: phaseEnd_ms,
      evidence: "reconstructed",
      identity,
    });
    cursor_ms = phaseEnd_ms;
  });
  events.push({
    id: `${resourceId}.end.${trace.sampleId}`,
    kind: "end",
    resourceId,
    clockDomain,
    at_ms: end_ms,
    evidence: "measured",
    identity,
  });
  if (identity.publicationId) events.push({
    id: `${resourceId}.publish.${trace.sampleId}`,
    kind: "publish",
    resourceId,
    clockDomain,
    at_ms: end_ms,
    // The legacy aggregate does not retain the exact publication command.
    evidence: "reconstructed",
    identity,
  });
}

/**
 * Adapt the existing duration-only ledgers into the richer activity model.
 * GPU clocks remain separate and explicitly unsynchronized unless the caller
 * supplies a calibration. No physical GPU-core assignment is invented.
 */
export function synthesizePerformanceActivityFrame(
  options: SynthesizePerformanceActivityOptions,
): PerformanceActivityFrame {
  const traces = [options.cpu, options.physics, options.presentation]
    .filter((trace): trace is PerformanceTrace => trace !== undefined);
  const cpuTimeOrigin_ms = options.cpuTimeOrigin_ms ?? performance.timeOrigin;
  const capturedAt_cpu_ms = options.capturedAt_cpu_ms
    ?? traces.reduce((latest, trace) => Math.max(latest, trace.capturedAt_ms), 0);
  const resources: ActivityResource[] = [{
    id: "cpu.main", label: "CPU · main thread", kind: "cpu-main", lane: "cpu-main", clockDomain: "cpu-performance",
  }];
  const derivedAlignments: ActivityClockAlignment[] = [];
  const clockEpochOrigins = new Map<ActivityClockDomain, number>([["cpu-performance", cpuTimeOrigin_ms]]);
  for (const worker of options.workerRows ?? []) resources.push({
    id: `cpu.worker.${worker.id}`,
    label: worker.label,
    kind: "cpu-worker",
    lane: "cpu-worker",
    clockDomain: workerClockDomain(worker.id),
  });
  for (const worker of options.workerRows ?? []) {
    if (!Number.isFinite(worker.timeOrigin_ms)) continue;
    const clockDomain = workerClockDomain(worker.id);
    clockEpochOrigins.set(clockDomain, worker.timeOrigin_ms!);
    derivedAlignments.push({
      from: clockDomain,
      to: "cpu-performance",
      offset_ms: worker.timeOrigin_ms! - cpuTimeOrigin_ms,
      uncertainty_ms: 0,
      source: "shared-time-origin",
    });
  }
  if (options.physics) resources.push({
    id: "gpu.physics.aggregate", label: "GPU · physics queue aggregate", kind: "gpu-aggregate",
    lane: "gpu-physics", clockDomain: "gpu-physics-timestamp",
  });
  if (options.presentation) resources.push({
    id: "gpu.presentation.aggregate", label: "GPU · presentation queue aggregate", kind: "gpu-aggregate",
    lane: "gpu-presentation", clockDomain: "gpu-presentation-timestamp",
  });
  const gpuRows = options.gpuLogicalCapacityRows ?? {};
  const appendGPUResources = (lane: "physics" | "presentation", count: number) => {
    const activityLane: ActivityLane = lane === "physics" ? "gpu-physics" : "gpu-presentation";
    const clockDomain: ActivityClockDomain = lane === "physics" ? "gpu-physics-timestamp" : "gpu-presentation-timestamp";
    for (let slot = 0; slot < Math.max(0, Math.floor(count)); slot += 1) resources.push({
      id: `gpu.${lane}.logical.${slot}`,
      label: `GPU · ${lane} · logical capacity ${slot}`,
      kind: "gpu-logical-capacity",
      lane: activityLane,
      clockDomain,
      capacitySlot: slot,
    });
  };
  if (options.physics) appendGPUResources("physics", gpuRows.physics ?? 0);
  if (options.presentation) appendGPUResources("presentation", gpuRows.presentation ?? 0);

  const tasks = new Map<string, ActivityTask>();
  const spans: ActivitySpan[] = [];
  const events: ActivityEvent[] = [];
  const windows: ActivityRowWindow[] = [];
  traces.forEach((trace) => appendTraceEvidence(trace, options, tasks, spans, events, windows));
  // Rows without evidence are deliberately unknown over the aggregate lane's
  // window. They represent logical display capacity, never inferred activity.
  for (const resource of resources) {
    if (windows.some((window) => window.resourceId === resource.id)) continue;
    const peer = windows.find((window) => resource.lane === "cpu-worker"
      ? resource.clockDomain === windowClock(resources, window.resourceId)
      : resource.lane === resourceLane(resources, window.resourceId));
    windows.push({ resourceId: resource.id, start_ms: peer?.start_ms ?? 0, end_ms: peer?.end_ms ?? 1 });
  }

  const alignments = [...derivedAlignments, ...(options.alignments ?? [])];
  const clockIds = [...new Set(resources.map((resource) => resource.clockDomain))];
  const clocks = clockIds.map((id): ActivityClockDescriptor => ({
    id,
    label: id === "cpu-performance" ? "CPU performance clock"
      : id === "gpu-physics-timestamp" ? "GPU physics timestamp clock"
        : id === "gpu-presentation-timestamp" ? "GPU presentation timestamp clock" : id,
    ...(clockEpochOrigins.has(id) ? { epochOrigin_ms: clockEpochOrigins.get(id) } : {}),
    status: alignmentStatus(id, alignments),
  }));
  const timestampFallback = traces.some((trace) => trace.domain === "gpu"
    && trace.measurementSource === "gpu-queue-wall");
  return {
    identity: { ...options.identity },
    context: options.context,
    capturedAt_cpu_ms,
    capturedAt_epoch_ms: cpuTimeOrigin_ms + capturedAt_cpu_ms,
    cpuTimeOrigin_ms,
    clocks,
    tasks: [...tasks.values()],
    resources,
    spans,
    events,
    rows: slicePerformanceActivityRows({ resources, spans, events, windows }),
    ...(timestampFallback ? {
      captureDiagnostics: { reasons: ["timestamp-fallback"] },
    } : {}),
  };
}

function resourceLane(resources: readonly ActivityResource[], resourceId: string): ActivityLane | undefined {
  return resources.find((resource) => resource.id === resourceId)?.lane;
}

function windowClock(resources: readonly ActivityResource[], resourceId: string): ActivityClockDomain | undefined {
  return resources.find((resource) => resource.id === resourceId)?.clockDomain;
}

function workerClockDomain(workerId: string): ActivityClockDomain {
  return `cpu-worker:${workerId}:performance`;
}
