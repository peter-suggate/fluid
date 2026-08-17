/**
 * The worker-owned authority for WGSL module and WebGPU pipeline compilation.
 *
 * Production resource owners consume `GPUCompilationService`; they never need
 * the raw `GPUDevice` methods that can compile synchronously.  The manager is
 * deliberately useful before that cut-over is complete, so existing pipeline
 * families can migrate one manifest at a time.
 */

export type GPUCompilationPriority = "critical" | "visible" | "background";

export interface GPUShaderModuleDefinition
  extends Omit<GPUShaderModuleDescriptor, "code"> {
  /** Evaluated by the manager in its worker realm, immediately before use. */
  readonly source: string | (() => string);
}

export interface GPUComputePipelineDefinition<ModuleName extends string>
  extends Omit<GPUComputePipelineDescriptor, "compute"> {
  readonly compute: Omit<GPUProgrammableStage, "module"> & {
    readonly module: ModuleName;
  };
}

export interface GPURenderPipelineDefinition<ModuleName extends string>
  extends Omit<GPURenderPipelineDescriptor, "vertex" | "fragment"> {
  readonly vertex: Omit<GPUVertexState, "module"> & {
    readonly module: ModuleName;
  };
  readonly fragment?: Omit<GPUFragmentState, "module"> & {
    readonly module: ModuleName;
  };
}

export interface GPUCompilationManifest<
  Modules extends Readonly<Record<string, GPUShaderModuleDefinition>> =
    Readonly<Record<string, GPUShaderModuleDefinition>>,
  Compute extends Readonly<Record<string, GPUComputePipelineDefinition<Extract<keyof Modules, string>>>> =
    Readonly<Record<string, GPUComputePipelineDefinition<Extract<keyof Modules, string>>>>,
  Render extends Readonly<Record<string, GPURenderPipelineDefinition<Extract<keyof Modules, string>>>> =
    Readonly<Record<string, GPURenderPipelineDefinition<Extract<keyof Modules, string>>>>,
> {
  /** Stable family identity. */
  readonly id: string;
  /** Must change whenever source, layout, constants, or fixed-function state changes. */
  readonly revision: string | number;
  readonly label?: string;
  readonly modules: Modules;
  readonly compute: Compute;
  readonly render: Render;
}

type AnyGPUCompilationManifest = GPUCompilationManifest<
  Readonly<Record<string, GPUShaderModuleDefinition>>,
  Readonly<Record<string, GPUComputePipelineDefinition<string>>>,
  Readonly<Record<string, GPURenderPipelineDefinition<string>>>
>;

export type CompiledGPUCompilationBundle<Manifest extends AnyGPUCompilationManifest> = Readonly<{
  key: string;
  manifest: Manifest;
  modules: Readonly<{ [Name in keyof Manifest["modules"]]: GPUShaderModule }>;
  compute: Readonly<{ [Name in keyof Manifest["compute"]]: GPUComputePipeline }>;
  render: Readonly<{ [Name in keyof Manifest["render"]]: GPURenderPipeline }>;
}>;

export interface GPUCompilationRequestOptions {
  readonly priority?: GPUCompilationPriority;
  /** Cancels this caller's interest; shared/in-flight compilation may continue. */
  readonly signal?: AbortSignal;
}

export interface GPUCompilationProgress {
  readonly key: string;
  readonly label: string;
  readonly completed: number;
  readonly total: number;
  readonly current?: string;
}

export interface GPUCompilationSnapshot {
  readonly state: "idle" | "compiling" | "invalidated";
  readonly generation: number;
  readonly queued: number;
  readonly active: number;
  readonly cached: number;
  readonly progress?: GPUCompilationProgress;
  readonly invalidationReason?: string;
}

export type GPUCompilationSnapshotListener = (snapshot: GPUCompilationSnapshot) => void;

export interface GPUCompilationService {
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;

  compileComputePipeline(
    descriptor: GPUComputePipelineDescriptor,
    options?: GPUCompilationRequestOptions,
  ): Promise<GPUComputePipeline>;

  compileRenderPipeline(
    descriptor: GPURenderPipelineDescriptor,
    options?: GPUCompilationRequestOptions,
  ): Promise<GPURenderPipeline>;

  acquire<Manifest extends AnyGPUCompilationManifest>(
    manifest: Manifest,
    options?: GPUCompilationRequestOptions,
  ): Promise<CompiledGPUCompilationBundle<Manifest>>;

  ready<Manifest extends AnyGPUCompilationManifest>(
    manifest: Manifest,
  ): CompiledGPUCompilationBundle<Manifest> | undefined;

  snapshot(): GPUCompilationSnapshot;
  /** Resolve after every queued/in-flight driver compilation has settled. */
  whenIdle(): Promise<void>;
  subscribe(listener: GPUCompilationSnapshotListener): () => void;
}

export interface GPUCompilationManagerOptions {
  /** Driver compilation is bounded; production uses two concurrent jobs. */
  readonly maximumConcurrentBundles?: number;
  /** Production defaults to fail-closed outside a WorkerGlobalScope. */
  readonly requireWorkerRealm?: boolean;
  /** Device lifecycle generation, useful in diagnostics after recovery. */
  readonly generation?: number;
}

export class GPUCompilationInvalidatedError extends Error {
  constructor(readonly reason: string) {
    super(`GPU compilation manager is invalidated: ${reason}`);
    this.name = "GPUCompilationInvalidatedError";
  }
}

interface CompilationJob {
  readonly kind: "manifest";
  readonly sequence: number;
  priority: GPUCompilationPriority;
  readonly manifest: AnyGPUCompilationManifest;
  resolve(bundle: CompiledGPUCompilationBundle<AnyGPUCompilationManifest>): void;
  reject(error: unknown): void;
}

interface DirectCompilationJob<Pipeline extends GPUComputePipeline | GPURenderPipeline> {
  readonly kind: "compute" | "render";
  readonly sequence: number;
  priority: GPUCompilationPriority;
  readonly key: string;
  readonly label: string;
  readonly descriptor: Pipeline extends GPUComputePipeline
    ? GPUComputePipelineDescriptor
    : GPURenderPipelineDescriptor;
  resolve(pipeline: Pipeline): void;
  reject(error: unknown): void;
}

type QueuedCompilationJob = CompilationJob
  | DirectCompilationJob<GPUComputePipeline>
  | DirectCompilationJob<GPURenderPipeline>;

const PRIORITY_ORDER: Readonly<Record<GPUCompilationPriority, number>> = {
  critical: 0,
  visible: 1,
  background: 2,
};

function manifestKey(manifest: Pick<GPUCompilationManifest, "id" | "revision">): string {
  return `${manifest.id}@${String(manifest.revision)}`;
}

function abortError(): DOMException {
  return new DOMException("GPU compilation request was aborted", "AbortError");
}

function isWorkerRealm(): boolean {
  const WorkerScope = (globalThis as unknown as {
    WorkerGlobalScope?: abstract new (...args: never[]) => object;
  }).WorkerGlobalScope;
  return typeof WorkerScope === "function" && globalThis instanceof WorkerScope;
}

/**
 * Validate and retain inference for a declarative pipeline family.
 *
 * The explicit revision is part of the cache contract. A caller changing any
 * descriptor or generated WGSL must also change that revision.
 */
export function defineGPUCompilationManifest<
  const Modules extends Readonly<Record<string, GPUShaderModuleDefinition>>,
  const Compute extends Readonly<Record<string, GPUComputePipelineDefinition<Extract<keyof Modules, string>>>>,
  const Render extends Readonly<Record<string, GPURenderPipelineDefinition<Extract<keyof Modules, string>>>>,
>(manifest: GPUCompilationManifest<Modules, Compute, Render>): GPUCompilationManifest<Modules, Compute, Render> {
  if (!manifest.id.trim()) throw new Error("GPU compilation manifest IDs must be non-empty");
  if (typeof manifest.revision === "number" && !Number.isFinite(manifest.revision)) {
    throw new Error(`GPU compilation manifest ${manifest.id} has a non-finite revision`);
  }
  const moduleNames = new Set(Object.keys(manifest.modules));
  for (const [name, definition] of Object.entries(manifest.compute)) {
    if (!moduleNames.has(definition.compute.module)) {
      throw new Error(`GPU compilation manifest ${manifest.id} compute pipeline ${name} references missing module ${definition.compute.module}`);
    }
  }
  for (const [name, definition] of Object.entries(manifest.render)) {
    if (!moduleNames.has(definition.vertex.module)) {
      throw new Error(`GPU compilation manifest ${manifest.id} render pipeline ${name} references missing vertex module ${definition.vertex.module}`);
    }
    if (definition.fragment && !moduleNames.has(definition.fragment.module)) {
      throw new Error(`GPU compilation manifest ${manifest.id} render pipeline ${name} references missing fragment module ${definition.fragment.module}`);
    }
  }
  return manifest;
}

/**
 * Device-scoped, worker-owned compilation scheduler and cache.
 *
 * `acquire()` never invokes a compilation API before returning its Promise.
 * Jobs begin in a microtask and all pipeline creation uses the asynchronous
 * WebGPU entry points. Complete bundles are published atomically.
 */
export class GPUCompilationManager implements GPUCompilationService {
  readonly #device: GPUDevice;
  private readonly maximumConcurrentBundles: number;
  private readonly generation: number;
  private readonly listeners = new Set<GPUCompilationSnapshotListener>();
  private readonly cache = new Map<string, CompiledGPUCompilationBundle<AnyGPUCompilationManifest>>();
  private readonly inFlight = new Map<string, Promise<CompiledGPUCompilationBundle<AnyGPUCompilationManifest>>>();
  private readonly queuedByKey = new Map<string, CompilationJob>();
  private readonly queue: QueuedCompilationJob[] = [];
  private active = 0;
  private sequence = 0;
  private pumpScheduled = false;
  private invalidationReason?: string;
  private progress?: GPUCompilationProgress;

  constructor(
    device: GPUDevice,
    options: GPUCompilationManagerOptions = {},
  ) {
    if (options.requireWorkerRealm !== false && !isWorkerRealm()) {
      throw new Error("GPUCompilationManager must be created in a worker realm");
    }
    this.#device = device;
    const maximumConcurrentBundles = options.maximumConcurrentBundles ?? 1;
    if (!Number.isSafeInteger(maximumConcurrentBundles)
      || maximumConcurrentBundles < 1 || maximumConcurrentBundles > 4) {
      throw new RangeError("GPU compilation concurrency must be a safe integer in [1, 4]");
    }
    this.maximumConcurrentBundles = maximumConcurrentBundles;
    this.generation = options.generation ?? 1;
  }

  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule {
    this.assertCurrentGeneration();
    return this.#device.createShaderModule(descriptor);
  }

  compileComputePipeline(
    descriptor: GPUComputePipelineDescriptor,
    options: GPUCompilationRequestOptions = {},
  ): Promise<GPUComputePipeline> {
    return this.enqueueDirect<GPUComputePipeline>("compute", descriptor, options);
  }

  compileRenderPipeline(
    descriptor: GPURenderPipelineDescriptor,
    options: GPUCompilationRequestOptions = {},
  ): Promise<GPURenderPipeline> {
    return this.enqueueDirect<GPURenderPipeline>("render", descriptor, options);
  }

  acquire<Manifest extends AnyGPUCompilationManifest>(
    manifest: Manifest,
    options: GPUCompilationRequestOptions = {},
  ): Promise<CompiledGPUCompilationBundle<Manifest>> {
    if (options.signal?.aborted) return Promise.reject(abortError());
    if (this.invalidationReason) {
      return Promise.reject(new GPUCompilationInvalidatedError(this.invalidationReason));
    }
    const key = manifestKey(manifest);
    const cached = this.cache.get(key);
    if (cached) {
      return this.forCaller(
        Promise.resolve(cached as CompiledGPUCompilationBundle<Manifest>),
        options.signal,
      );
    }
    let shared = this.inFlight.get(key);
    if (!shared) {
      shared = new Promise<CompiledGPUCompilationBundle<AnyGPUCompilationManifest>>((resolve, reject) => {
        const job: CompilationJob = {
          kind: "manifest",
          sequence: this.sequence++,
          priority: options.priority ?? "visible",
          manifest,
          resolve,
          reject,
        };
        this.queue.push(job);
        this.queuedByKey.set(key, job);
        this.sortQueue();
        this.schedulePump();
        this.emit();
      });
      this.inFlight.set(key, shared);
      void shared.then(
        () => { this.inFlight.delete(key); },
        () => { this.inFlight.delete(key); },
      );
    } else {
      const queued = this.queuedByKey.get(key);
      const requestedPriority = options.priority ?? "visible";
      if (queued && PRIORITY_ORDER[requestedPriority] < PRIORITY_ORDER[queued.priority]) {
        queued.priority = requestedPriority;
        this.sortQueue();
        this.emit();
      }
    }
    return this.forCaller(
      shared as Promise<CompiledGPUCompilationBundle<Manifest>>,
      options.signal,
    );
  }

  ready<Manifest extends AnyGPUCompilationManifest>(
    manifest: Manifest,
  ): CompiledGPUCompilationBundle<Manifest> | undefined {
    return this.cache.get(manifestKey(manifest)) as CompiledGPUCompilationBundle<Manifest> | undefined;
  }

  snapshot(): GPUCompilationSnapshot {
    return Object.freeze({
      state: this.invalidationReason
        ? "invalidated"
        : this.active > 0 || this.queue.length > 0 ? "compiling" : "idle",
      generation: this.generation,
      queued: this.queue.length,
      active: this.active,
      cached: this.cache.size,
      ...(this.progress ? { progress: this.progress } : {}),
      ...(this.invalidationReason ? { invalidationReason: this.invalidationReason } : {}),
    });
  }

  whenIdle(): Promise<void> {
    if (this.active === 0 && this.queue.length === 0) return Promise.resolve();
    return new Promise<void>((resolve) => {
      let unsubscribe = () => {};
      unsubscribe = this.subscribe((snapshot) => {
        if (snapshot.active !== 0 || snapshot.queued !== 0) return;
        unsubscribe();resolve();
      });
    });
  }

  subscribe(listener: GPUCompilationSnapshotListener): () => void {
    this.listeners.add(listener);
    listener(this.snapshot());
    return () => { this.listeners.delete(listener); };
  }

  /** Permanently retire this device generation. A replacement gets a new manager. */
  invalidate(reason = "GPU device retired"): void {
    if (this.invalidationReason) return;
    this.invalidationReason = reason;
    this.cache.clear();
    this.progress = undefined;
    for (const job of this.queue.splice(0)) {
      if (job.kind === "manifest") this.queuedByKey.delete(manifestKey(job.manifest));
      job.reject(new GPUCompilationInvalidatedError(reason));
    }
    this.emit();
  }

  private forCaller<T>(shared: Promise<T>, signal?: AbortSignal): Promise<T> {
    if (!signal) return shared;
    if (signal.aborted) return Promise.reject(abortError());
    return new Promise<T>((resolve, reject) => {
      const aborted = () => reject(abortError());
      signal.addEventListener("abort", aborted, { once: true });
      void shared.then(
        (value) => {
          signal.removeEventListener("abort", aborted);
          if (!signal.aborted) resolve(value);
        },
        (error) => {
          signal.removeEventListener("abort", aborted);
          if (!signal.aborted) reject(error);
        },
      );
    });
  }

  private sortQueue(): void {
    this.queue.sort((a, b) =>
      PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority]
      || a.sequence - b.sequence);
  }

  private schedulePump(): void {
    if (this.pumpScheduled) return;
    this.pumpScheduled = true;
    queueMicrotask(() => {
      this.pumpScheduled = false;
      this.pump();
    });
  }

  private pump(): void {
    if (this.invalidationReason) return;
    while (this.active < this.maximumConcurrentBundles && this.queue.length > 0) {
      const job = this.queue.shift()!;
      if (job.kind === "manifest") this.queuedByKey.delete(manifestKey(job.manifest));
      this.active += 1;
      void this.run(job);
    }
    this.emit();
  }

  private async run(job: QueuedCompilationJob): Promise<void> {
    const key = job.kind === "manifest" ? manifestKey(job.manifest) : job.key;
    try {
      if (job.kind === "manifest") {
        const bundle = await this.compile(job.manifest);
        if (this.invalidationReason) throw new GPUCompilationInvalidatedError(this.invalidationReason);
        const published = this.cache.get(key) ?? bundle;
        this.cache.set(key, published);
        job.resolve(published);
      } else {
        this.progress = Object.freeze({
          key,
          label: job.label,
          completed: 0,
          total: 1,
          current: `${job.kind === "compute" ? "Compile compute" : "Compile render"} pipeline`,
        });
        this.emit();
        const pipeline = job.kind === "compute"
          ? await this.#device.createComputePipelineAsync(
            job.descriptor as GPUComputePipelineDescriptor)
          : await this.#device.createRenderPipelineAsync(
            job.descriptor as GPURenderPipelineDescriptor);
        this.assertCurrentGeneration();
        job.resolve(pipeline as never);
      }
    } catch (error) {
      job.reject(error);
    } finally {
      this.active -= 1;
      if (this.progress?.key === key) this.progress = undefined;
      this.emit();
      this.pump();
    }
  }

  private enqueueDirect<Pipeline extends GPUComputePipeline | GPURenderPipeline>(
    kind: Pipeline extends GPUComputePipeline ? "compute" : "render",
    descriptor: Pipeline extends GPUComputePipeline
      ? GPUComputePipelineDescriptor
      : GPURenderPipelineDescriptor,
    options: GPUCompilationRequestOptions,
  ): Promise<Pipeline> {
    if (options.signal?.aborted) return Promise.reject(abortError());
    if (this.invalidationReason) {
      return Promise.reject(new GPUCompilationInvalidatedError(this.invalidationReason));
    }
    const key = `direct:${this.sequence}`;
    const shared = new Promise<Pipeline>((resolve, reject) => {
      this.queue.push({
        kind,
        sequence: this.sequence++,
        priority: options.priority ?? "visible",
        key,
        label: descriptor.label ?? `${kind} pipeline`,
        descriptor,
        resolve,
        reject,
      } as QueuedCompilationJob);
      this.sortQueue();
      this.schedulePump();
      this.emit();
    });
    return this.forCaller(shared, options.signal);
  }

  private async compile(
    manifest: AnyGPUCompilationManifest,
  ): Promise<CompiledGPUCompilationBundle<AnyGPUCompilationManifest>> {
    const key = manifestKey(manifest);
    const label = manifest.label ?? manifest.id;
    const moduleDefinitions = Object.entries(manifest.modules);
    const computeDefinitions = Object.entries(manifest.compute);
    const renderDefinitions = Object.entries(manifest.render);
    const total = moduleDefinitions.length + computeDefinitions.length + renderDefinitions.length;
    let completed = 0;
    const modules: Record<string, GPUShaderModule> = {};
    const compute: Record<string, GPUComputePipeline> = {};
    const render: Record<string, GPURenderPipeline> = {};
    const report = (current?: string) => {
      this.progress = Object.freeze({ key, label, completed, total, ...(current ? { current } : {}) });
      this.emit();
    };

    for (const [name, definition] of moduleDefinitions) {
      this.assertCurrentGeneration();
      report(`Create shader module: ${name}`);
      const { source, ...descriptor } = definition;
      modules[name] = this.#device.createShaderModule({
        ...descriptor,
        code: typeof source === "function" ? source() : source,
      });
      completed += 1;
      report();
    }
    for (const [name, definition] of computeDefinitions) {
      this.assertCurrentGeneration();
      report(`Compile compute pipeline: ${name}`);
      const { compute: stage, ...descriptor } = definition;
      const { module, ...programmableStage } = stage;
      compute[name] = await this.#device.createComputePipelineAsync({
        ...descriptor,
        compute: { ...programmableStage, module: modules[module]! },
      });
      completed += 1;
      report();
    }
    for (const [name, definition] of renderDefinitions) {
      this.assertCurrentGeneration();
      report(`Compile render pipeline: ${name}`);
      const { vertex, fragment, ...descriptor } = definition;
      const { module: vertexModule, ...vertexState } = vertex;
      const fragmentState = fragment
        ? (({ module, ...state }) => ({ ...state, module: modules[module]! }))(fragment)
        : undefined;
      render[name] = await this.#device.createRenderPipelineAsync({
        ...descriptor,
        vertex: { ...vertexState, module: modules[vertexModule]! },
        ...(fragmentState ? { fragment: fragmentState } : {}),
      });
      completed += 1;
      report();
    }
    this.assertCurrentGeneration();
    return Object.freeze({
      key,
      manifest,
      modules: Object.freeze(modules),
      compute: Object.freeze(compute),
      render: Object.freeze(render),
    });
  }

  private assertCurrentGeneration(): void {
    if (this.invalidationReason) throw new GPUCompilationInvalidatedError(this.invalidationReason);
  }

  private emit(): void {
    if (this.listeners.size === 0) return;
    const snapshot = this.snapshot();
    for (const listener of this.listeners) {
      try { listener(snapshot); } catch { /* Diagnostics must never break compilation. */ }
    }
  }
}

const managerByDevice = new WeakMap<GPUDevice, GPUCompilationManager>();
const managedDeviceByRawDevice = new WeakMap<GPUDevice, GPUDevice>();

/**
 * Resolve the one compilation authority for a device.
 *
 * Browser main-thread construction still fails closed. Node-based validation
 * and benchmark processes are permitted because they have no browser UI event
 * loop to protect; the browser runtime must be a WorkerGlobalScope.
 */
export function gpuCompilationManagerFor(device: GPUDevice): GPUCompilationManager {
  const existing = managerByDevice.get(device);
  if (existing) return existing;
  const requireWorkerRealm = typeof document !== "undefined" || isWorkerRealm();
  // This is intentionally not the old Promise-all fanout: the manager remains
  // the sole scheduler and never has more than this many driver compilation
  // calls active. Three separate B16/P16 Dawn processes completed cleanly at
  // width three (~2.58 s resident construction versus ~3.50 s at width two).
  // Width four was faster in isolated construction runs but repeated full
  // resident builds could leave Metal in a degraded state, so production
  // deliberately retains one slot of headroom below the hard cap.
  const requestedConcurrency = typeof process !== "undefined"
    ? Number(process.env?.FLUID_GPU_COMPILATION_CONCURRENCY ?? 3) : 3;
  const maximumConcurrentBundles = Number.isSafeInteger(requestedConcurrency)
    && requestedConcurrency >= 1 && requestedConcurrency <= 4
    ? requestedConcurrency : 1;
  const manager = new GPUCompilationManager(device, {
    requireWorkerRealm,
    maximumConcurrentBundles,
  });
  managerByDevice.set(device, manager);
  return manager;
}

/**
 * Give resource owners a GPUDevice-shaped capability whose compilation
 * methods are manager-owned. Immediate pipeline creation is intentionally
 * unavailable; any surviving call is a cut-over defect and fails loudly.
 */
export function managedGPUDevice(
  rawDevice: GPUDevice,
  options: GPUCompilationManagerOptions = {},
): GPUDevice {
  const existing = managedDeviceByRawDevice.get(rawDevice);
  if (existing) return existing;
  const manager = new GPUCompilationManager(rawDevice, options);
  const managed = new Proxy(rawDevice, {
    get(target, property) {
      if (property === "createShaderModule") {
        return (descriptor: GPUShaderModuleDescriptor) => manager.createShaderModule(descriptor);
      }
      if (property === "createComputePipelineAsync") {
        return (descriptor: GPUComputePipelineDescriptor) => manager.compileComputePipeline(descriptor);
      }
      if (property === "createRenderPipelineAsync") {
        return (descriptor: GPURenderPipelineDescriptor) => manager.compileRenderPipeline(descriptor);
      }
      if (property === "createComputePipeline" || property === "createRenderPipeline") {
        return () => {
          throw new Error(`${String(property)} is disabled; acquire the pipeline through GPUCompilationManager`);
        };
      }
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as GPUDevice;
  managerByDevice.set(rawDevice, manager);
  managerByDevice.set(managed, manager);
  managedDeviceByRawDevice.set(rawDevice, managed);
  return managed;
}

export function invalidateGPUCompilationManager(
  device: GPUDevice,
  reason = "GPU device retired",
): void {
  managerByDevice.get(device)?.invalidate(reason);
}
