import {
  performanceActivityTaskColor,
  type ActivityClockAlignment,
  type ActivityClockDescriptor,
  type ActivityClockDomain,
  type ActivityEvent,
  type ActivityLane,
  type ActivityResource,
  type ActivitySpan,
  type ActivityTask,
  type ActivityWorkIdentity,
} from "./performance-activity";

export interface CPUPerformanceClock {
  readonly timeOrigin: number;
  now(): number;
}

export interface CPUActivityTaskDescriptor {
  id: string;
  label: string;
  parentId?: string;
  color?: string;
}

/** Canonical coarse seams ready for EulerianFluidSolver and its controller. */
export const CPU_PHYSICS_ACTIVITY_TASKS = {
  controller: { id: "cpu.physics.controller", label: "Physics controller" },
  couplingLoads: { id: "cpu.physics.coupling.loads", label: "Fluid loads", parentId: "cpu.physics.controller" },
  couplingReactions: { id: "cpu.physics.coupling.reactions", label: "Fluid reactions", parentId: "cpu.physics.controller" },
  rigidIntegration: { id: "cpu.physics.rigid.integrate", label: "Rigid integration and contacts", parentId: "cpu.physics.controller" },
  fluidStep: { id: "cpu.physics.fluid.step", label: "Eulerian fluid step", parentId: "cpu.physics.controller" },
  timestep: { id: "cpu.physics.fluid.timestep", label: "Timestep selection", parentId: "cpu.physics.fluid.step" },
  inflow: { id: "cpu.physics.fluid.inflow", label: "Inflow", parentId: "cpu.physics.fluid.step" },
  forces: { id: "cpu.physics.fluid.forces", label: "External forces", parentId: "cpu.physics.fluid.step" },
  velocityAdvection: { id: "cpu.physics.fluid.velocity-advection", label: "Velocity advection", parentId: "cpu.physics.fluid.step" },
  viscosity: { id: "cpu.physics.fluid.viscosity", label: "Viscosity", parentId: "cpu.physics.fluid.step" },
  divergenceBefore: { id: "cpu.physics.fluid.divergence-before", label: "Pre-projection divergence", parentId: "cpu.physics.fluid.step" },
  pressure: { id: "cpu.physics.fluid.pressure", label: "Pressure solve and projection", parentId: "cpu.physics.fluid.step" },
  divergenceAfter: { id: "cpu.physics.fluid.divergence-after", label: "Post-projection divergence", parentId: "cpu.physics.fluid.step" },
  markerAdvection: { id: "cpu.physics.fluid.marker-advection", label: "Marker advection", parentId: "cpu.physics.fluid.step" },
  diagnostics: { id: "cpu.physics.fluid.diagnostics", label: "Fluid diagnostics", parentId: "cpu.physics.fluid.step" },
  publication: { id: "cpu.physics.publication", label: "Physics publication", parentId: "cpu.physics.controller" },
} as const satisfies Readonly<Record<string, CPUActivityTaskDescriptor>>;

/** Canonical coarse seams ready for prepareQuadtreeProjectionCPU. */
export const CPU_QUADTREE_WORKER_ACTIVITY_TASKS = {
  request: { id: "cpu.worker.quadtree.request", label: "Quadtree topology request" },
  unpack: { id: "cpu.worker.quadtree.unpack", label: "Reconstruct packed quadtree", parentId: "cpu.worker.quadtree.request" },
  pressureGrid: { id: "cpu.worker.quadtree.pressure-grid", label: "Build tall pressure grid", parentId: "cpu.worker.quadtree.request" },
  topologyIdentity: { id: "cpu.worker.quadtree.topology-identity", label: "Topology identity", parentId: "cpu.worker.quadtree.request" },
  solidFields: { id: "cpu.worker.quadtree.solid-fields", label: "Solid fields", parentId: "cpu.worker.quadtree.request" },
  variationalSystem: { id: "cpu.worker.quadtree.variational-system", label: "Variational system", parentId: "cpu.worker.quadtree.request" },
  pack: { id: "cpu.worker.quadtree.pack", label: "Projection pack", parentId: "cpu.worker.quadtree.request" },
  response: { id: "cpu.worker.quadtree.response", label: "Quadtree topology response", parentId: "cpu.worker.quadtree.request" },
  receive: { id: "cpu.worker.transport.receive", label: "Worker request received" },
  send: { id: "cpu.worker.transport.send", label: "Worker response sent" },
} as const satisfies Readonly<Record<string, CPUActivityTaskDescriptor>>;

export interface CPUPerformanceActivityOutput {
  resource?: ActivityResource;
  clock?: ActivityClockDescriptor;
  alignments: ActivityClockAlignment[];
  tasks: ActivityTask[];
  spans: ActivitySpan[];
  events: ActivityEvent[];
}

export interface CPUPerformanceActivityProfiler {
  readonly enabled: boolean;
  measure<T>(task: CPUActivityTaskDescriptor, work: () => T): T;
  measureAsync<T>(task: CPUActivityTaskDescriptor, work: () => PromiseLike<T>): Promise<T>;
  event(kind: ActivityEvent["kind"], task: CPUActivityTaskDescriptor, metadata?: ActivityEvent["metadata"]): void;
  output(): CPUPerformanceActivityOutput;
}

export interface CPUPerformanceActivityProfilerOptions {
  enabled: boolean;
  identity: ActivityWorkIdentity;
  resourceId: string;
  resourceLabel: string;
  resourceKind: "cpu-main" | "cpu-worker";
  lane?: Extract<ActivityLane, "cpu-main" | "cpu-worker">;
  clockDomain?: ActivityClockDomain;
  clock?: CPUPerformanceClock;
  alignment?: ActivityClockAlignment;
}

const EMPTY_OUTPUT: CPUPerformanceActivityOutput = Object.freeze({
  alignments: Object.freeze([]) as unknown as ActivityClockAlignment[],
  tasks: Object.freeze([]) as unknown as ActivityTask[],
  spans: Object.freeze([]) as unknown as ActivitySpan[],
  events: Object.freeze([]) as unknown as ActivityEvent[],
});

/** Shared disabled path: no clock reads, allocations, task registration, or store access. */
export const NOOP_CPU_PERFORMANCE_ACTIVITY_PROFILER: CPUPerformanceActivityProfiler = Object.freeze({
  enabled: false,
  measure: <T>(_task: CPUActivityTaskDescriptor, work: () => T) => work(),
  measureAsync: async <T>(_task: CPUActivityTaskDescriptor, work: () => PromiseLike<T>) => work(),
  event: () => {},
  output: () => EMPTY_OUTPUT,
});

export function createCPUPerformanceActivityProfiler(
  options: CPUPerformanceActivityProfilerOptions,
): CPUPerformanceActivityProfiler {
  if (!options.enabled) return NOOP_CPU_PERFORMANCE_ACTIVITY_PROFILER;
  return new EnabledCPUPerformanceActivityProfiler(options);
}

class EnabledCPUPerformanceActivityProfiler implements CPUPerformanceActivityProfiler {
  readonly enabled = true;
  private readonly identity: ActivityWorkIdentity;
  private readonly resource: ActivityResource;
  private readonly clock: CPUPerformanceClock;
  private readonly clockDescriptor: ActivityClockDescriptor;
  private readonly alignments: ActivityClockAlignment[];
  private readonly tasks = new Map<string, ActivityTask>();
  private readonly spans: ActivitySpan[] = [];
  private readonly events: ActivityEvent[] = [];
  private sequence = 0;

  constructor(options: CPUPerformanceActivityProfilerOptions) {
    this.identity = { ...options.identity };
    this.clock = options.clock ?? performance;
    const lane = options.lane ?? (options.resourceKind === "cpu-worker" ? "cpu-worker" : "cpu-main");
    const clockDomain = options.clockDomain ?? "cpu-performance";
    this.resource = {
      id: options.resourceId,
      label: options.resourceLabel,
      kind: options.resourceKind,
      lane,
      clockDomain,
    };
    const status: ActivityClockDescriptor["status"] = clockDomain === "cpu-performance"
      ? { state: "reference" }
      : options.alignment
        ? { state: "synchronized", to: options.alignment.to, offset_ms: options.alignment.offset_ms,
          uncertainty_ms: options.alignment.uncertainty_ms, source: options.alignment.source }
        : { state: "unsynchronized" };
    this.clockDescriptor = {
      id: clockDomain,
      label: options.resourceLabel,
      epochOrigin_ms: this.clock.timeOrigin,
      status,
    };
    this.alignments = options.alignment ? [{ ...options.alignment }] : [];
  }

  measure<T>(task: CPUActivityTaskDescriptor, work: () => T): T {
    const startedAt_ms = this.clock.now();
    const occurrence = this.begin(task, startedAt_ms);
    try {
      return work();
    } finally {
      this.end(task, occurrence, startedAt_ms, this.clock.now());
    }
  }

  async measureAsync<T>(task: CPUActivityTaskDescriptor, work: () => PromiseLike<T>): Promise<T> {
    const startedAt_ms = this.clock.now();
    const occurrence = this.begin(task, startedAt_ms);
    try {
      return await work();
    } finally {
      this.end(task, occurrence, startedAt_ms, this.clock.now());
    }
  }

  event(kind: ActivityEvent["kind"], task: CPUActivityTaskDescriptor, metadata?: ActivityEvent["metadata"]): void {
    this.register(task);
    const occurrence = this.nextId(task.id);
    this.events.push({
      id: `${occurrence}.${kind}`,
      kind,
      taskId: task.id,
      resourceId: this.resource.id,
      clockDomain: this.resource.clockDomain,
      at_ms: this.clock.now(),
      evidence: "measured",
      identity: this.identity,
      ...(metadata ? { metadata } : {}),
    });
  }

  output(): CPUPerformanceActivityOutput {
    return {
      resource: { ...this.resource },
      clock: { ...this.clockDescriptor },
      alignments: this.alignments.map((alignment) => ({ ...alignment })),
      tasks: [...this.tasks.values()].map((task) => ({ ...task })),
      spans: this.spans.map((span) => ({ ...span })),
      events: this.events.map((event) => ({ ...event })),
    };
  }

  private register(task: CPUActivityTaskDescriptor): void {
    const existing = this.tasks.get(task.id);
    if (existing) {
      if (existing.label !== task.label || existing.parentId !== task.parentId) {
        throw new Error(`CPU performance task ${task.id} was registered with conflicting metadata`);
      }
      return;
    }
    this.tasks.set(task.id, {
      id: task.id,
      label: task.label,
      color: task.color ?? performanceActivityTaskColor(task.id),
      lane: this.resource.lane,
      ...(task.parentId ? { parentId: task.parentId } : {}),
    });
  }

  private begin(task: CPUActivityTaskDescriptor, at_ms: number): string {
    this.register(task);
    const occurrence = this.nextId(task.id);
    this.events.push({
      id: `${occurrence}.begin`,
      kind: "begin",
      taskId: task.id,
      resourceId: this.resource.id,
      clockDomain: this.resource.clockDomain,
      at_ms,
      evidence: "measured",
      identity: this.identity,
    });
    return occurrence;
  }

  private end(task: CPUActivityTaskDescriptor, occurrence: string, start_ms: number, end_ms: number): void {
    const closedAt_ms = Math.max(start_ms, end_ms);
    this.spans.push({
      id: occurrence,
      kind: "task",
      taskId: task.id,
      resourceId: this.resource.id,
      clockDomain: this.resource.clockDomain,
      start_ms,
      end_ms: closedAt_ms,
      evidence: "measured",
      identity: this.identity,
    });
    this.events.push({
      id: `${occurrence}.end`,
      kind: "end",
      taskId: task.id,
      resourceId: this.resource.id,
      clockDomain: this.resource.clockDomain,
      at_ms: closedAt_ms,
      evidence: "measured",
      identity: this.identity,
    });
  }

  private nextId(taskId: string): string {
    this.sequence += 1;
    return `${this.identity.frameId}.${this.identity.submissionId ?? "cpu"}.${this.resource.id}.${taskId}.${this.sequence}`;
  }
}

export interface CPUPerformanceActivityTransportContext {
  identity: ActivityWorkIdentity;
  sourceResourceId: string;
  sourceClockDomain: ActivityClockDomain;
  sourceTimeOrigin_ms: number;
  sentAt_ms: number;
  sentAt_epoch_ms: number;
}

/** Serializable context to include beside a worker request. */
export function createCPUPerformanceActivityTransportContext(input: {
  identity: ActivityWorkIdentity;
  sourceResourceId?: string;
  sourceClockDomain?: ActivityClockDomain;
  clock?: CPUPerformanceClock;
}): CPUPerformanceActivityTransportContext {
  const clock = input.clock ?? performance;
  const sentAt_ms = clock.now();
  return {
    identity: { ...input.identity },
    sourceResourceId: input.sourceResourceId ?? "cpu.main",
    sourceClockDomain: input.sourceClockDomain ?? "cpu-performance",
    sourceTimeOrigin_ms: clock.timeOrigin,
    sentAt_ms,
    sentAt_epoch_ms: clock.timeOrigin + sentAt_ms,
  };
}

export interface WorkerCPUPerformanceActivityProfiler {
  profiler: CPUPerformanceActivityProfiler;
  alignment?: ActivityClockAlignment;
  receivedAt_ms?: number;
  transportLatency_ms?: number;
}

/**
 * Rehydrate a worker recorder. Both browser and Node workers expose
 * performance.timeOrigin, allowing their raw local spans to remain untouched
 * while an explicit offset aligns them with the sender's CPU clock.
 */
export function createWorkerCPUPerformanceActivityProfiler(input: {
  enabled: boolean;
  workerId: string;
  workerLabel: string;
  transport?: CPUPerformanceActivityTransportContext;
  clock?: CPUPerformanceClock;
}): WorkerCPUPerformanceActivityProfiler {
  if (!input.enabled) return { profiler: NOOP_CPU_PERFORMANCE_ACTIVITY_PROFILER };
  const clock = input.clock ?? performance;
  const clockDomain: ActivityClockDomain = `cpu-worker:${input.workerId}:performance`;
  const alignment: ActivityClockAlignment | undefined = input.transport ? {
    from: clockDomain,
    to: input.transport.sourceClockDomain,
    offset_ms: clock.timeOrigin - input.transport.sourceTimeOrigin_ms,
    uncertainty_ms: 0,
    source: "shared-time-origin",
  } : undefined;
  const identity = input.transport?.identity ?? {
    frameId: `worker-unscoped:${input.workerId}`,
    generation: 0,
  };
  const profiler = createCPUPerformanceActivityProfiler({
    enabled: true,
    identity,
    resourceId: `cpu.worker.${input.workerId}`,
    resourceLabel: input.workerLabel,
    resourceKind: "cpu-worker",
    lane: "cpu-worker",
    clockDomain,
    clock,
    ...(alignment ? { alignment } : {}),
  });
  const receivedAt_ms = clock.now();
  const transportLatency_ms = input.transport
    ? Math.max(0, clock.timeOrigin + receivedAt_ms - input.transport.sentAt_epoch_ms)
    : undefined;
  profiler.event("instant", CPU_QUADTREE_WORKER_ACTIVITY_TASKS.receive, {
    ...(input.transport ? { sourceResourceId: input.transport.sourceResourceId,
      sourceSentAt_ms: input.transport.sentAt_ms } : {}),
    ...(transportLatency_ms !== undefined ? { transportLatency_ms } : {}),
  });
  return { profiler, alignment, receivedAt_ms, transportLatency_ms };
}
