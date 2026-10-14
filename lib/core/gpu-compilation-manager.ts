/**
 * The worker-owned authority for WGSL module and WebGPU pipeline compilation.
 *
 * Manifests and managed GPUDevice calls share the same descriptor caches and
 * bounded queue. Compilation resources live until device retirement, across
 * scene teardown; buffers, textures and bind groups remain scene-owned.
 */

import { GPUCompilationDescriptorKeys, normalizedBindGroupLayout, snapshotGPUDescriptor } from "./gpu-compilation-descriptors";

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
  /** Manifest bundles retained for compatibility with existing progress UI. */
  readonly cached: number;
  readonly cacheStats: GPUCompilationCacheStats;
  readonly progress?: GPUCompilationProgress;
  readonly invalidationReason?: string;
}

export interface GPUCompilationCacheStats {
  readonly modules: number;
  readonly bindGroupLayouts: number;
  readonly pipelineLayouts: number;
  readonly pipelines: number;
  readonly moduleHits: number;
  readonly layoutHits: number;
  readonly pipelineHits: number;
  readonly inFlightHits: number;
  readonly compilationAttempts: number;
  /** Sum of native pipeline call durations, excluding queue wait. Concurrent
   * compilations overlap, so this is not elapsed scene startup time. */
  readonly compilationMilliseconds: number;
}

export type GPUCompilationSnapshotListener = (snapshot: GPUCompilationSnapshot) => void;

export interface GPUCompilationService {
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;

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
  /** Driver compilation is bounded; production uses three concurrent jobs. */
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

type Pipeline = GPUComputePipeline | GPURenderPipeline;
type PipelineDescriptor = GPUComputePipelineDescriptor | GPURenderPipelineDescriptor;
interface CompilationJob {
  readonly kind: "compute" | "render";
  readonly sequence: number;
  priority: GPUCompilationPriority;
  readonly key: string;
  readonly label: string;
  readonly descriptor: PipelineDescriptor;
  readonly promise: Promise<Pipeline>;
  consumers: number;
  started: boolean;
  resolve(pipeline: Pipeline): void;
  reject(error: unknown): void;
}
interface BundleRequest {
  priority: GPUCompilationPriority;
  readonly jobs: Set<CompilationJob>;
}

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
 * Jobs begin in a microtask and pipeline creation uses asynchronous WebGPU
 * entry points. Complete bundles are published atomically.
 */
export class GPUCompilationManager implements GPUCompilationService {
  readonly #device: GPUDevice;
  private readonly maximumConcurrentBundles: number;
  private readonly generation: number;
  private readonly listeners = new Set<GPUCompilationSnapshotListener>();
  private readonly cache = new Map<string, CompiledGPUCompilationBundle<AnyGPUCompilationManifest>>();
  private readonly inFlight = new Map<string, Promise<CompiledGPUCompilationBundle<AnyGPUCompilationManifest>>>();
  private readonly bundleRequests = new Map<string, BundleRequest>();
  private readonly queue: CompilationJob[] = [];
  private readonly keys = new GPUCompilationDescriptorKeys();
  private readonly modules = new Map<string, GPUShaderModule>();
  private readonly bindGroupLayouts = new Map<string, GPUBindGroupLayout>();
  private readonly pipelineLayouts = new Map<string, GPUPipelineLayout>();
  private readonly pipelines = new Map<string, Pipeline>();
  private readonly pipelineJobs = new Map<string, CompilationJob>();
  private moduleHits = 0;
  private layoutHits = 0;
  private pipelineHits = 0;
  private inFlightHits = 0;
  private compilationAttempts = 0;
  private compilationMilliseconds = 0;
  private active = 0;
  private sequence = 0;
  private pumpScheduled = false;
  private invalidationReason?: string;
  private progress?: GPUCompilationProgress;
  private readonly abortListeners = new Map<AbortSignal, { callbacks: Set<() => void>; listener: () => void }>();

  private onAbort(signal: AbortSignal, callback: () => void): () => void {
    let entry = this.abortListeners.get(signal);
    if (!entry) {
      const callbacks = new Set<() => void>();
      const listener = () => { for (const cancel of callbacks) cancel(); };
      entry = { callbacks, listener };
      this.abortListeners.set(signal, entry);
      signal.addEventListener("abort", listener, { once: true });
    }
    entry.callbacks.add(callback);
    const current = entry;
    return () => {
      current.callbacks.delete(callback);
      if (current.callbacks.size === 0) {
        signal.removeEventListener("abort", current.listener);
        this.abortListeners.delete(signal);
      }
    };
  }

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
    void device.lost?.then(info => this.invalidate(info.message || info.reason));
  }

  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule {
    this.assertCurrentGeneration();
    const saved = snapshotGPUDescriptor(descriptor);
    const key = this.keys.key(saved);
    const cached = this.modules.get(key);
    if (cached) { this.moduleHits++; return cached; }
    const module = this.#device.createShaderModule(saved);
    this.modules.set(key, module);
    return module;
  }

  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
    this.assertCurrentGeneration();
    const saved = normalizedBindGroupLayout(snapshotGPUDescriptor(descriptor));
    const key = this.keys.key(saved);
    const cached = this.bindGroupLayouts.get(key);
    if (cached) { this.layoutHits++; return cached; }
    const layout = this.#device.createBindGroupLayout(saved);
    this.bindGroupLayouts.set(key, layout);
    return layout;
  }

  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
    this.assertCurrentGeneration();
    const saved = snapshotGPUDescriptor({ immediateSize: 0, ...descriptor });
    const key = this.keys.key(saved);
    const cached = this.pipelineLayouts.get(key);
    if (cached) { this.layoutHits++; return cached; }
    const layout = this.#device.createPipelineLayout(saved);
    this.pipelineLayouts.set(key, layout);
    return layout;
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
      const request: BundleRequest = { priority: options.priority ?? "visible", jobs: new Set() };
      this.bundleRequests.set(key, request);
      // Bundles compose ordinary cached jobs; they never occupy a scheduler
      // slot while waiting for one of those jobs (including another bundle's).
      shared = Promise.resolve().then(() => this.compile(manifest, request)).then(bundle => {
        this.assertCurrentGeneration();
        this.cache.set(key, bundle);
        return bundle;
      });
      this.inFlight.set(key, shared);
      const finish = () => {
        this.inFlight.delete(key);
        this.bundleRequests.delete(key);
        if (this.progress?.key === key) this.progress = undefined;
        this.emit();
      };
      void shared.then(finish, finish);
      this.emit();
    } else {
      const request = this.bundleRequests.get(key)!;
      const priority = options.priority ?? "visible";
      if (PRIORITY_ORDER[priority] < PRIORITY_ORDER[request.priority]) {
        request.priority = priority;
        for (const job of request.jobs) this.promote(job, priority);
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
        : this.active > 0 || this.queue.length > 0 || this.inFlight.size > 0 ? "compiling" : "idle",
      generation: this.generation,
      queued: this.queue.length,
      active: this.active,
      cached: this.cache.size,
      cacheStats: Object.freeze({
        modules: this.modules.size, bindGroupLayouts: this.bindGroupLayouts.size,
        pipelineLayouts: this.pipelineLayouts.size, pipelines: this.pipelines.size,
        moduleHits: this.moduleHits, layoutHits: this.layoutHits,
        pipelineHits: this.pipelineHits, inFlightHits: this.inFlightHits,
        compilationAttempts: this.compilationAttempts,
        compilationMilliseconds: this.compilationMilliseconds,
      }),
      ...(this.progress ? { progress: this.progress } : {}),
      ...(this.invalidationReason ? { invalidationReason: this.invalidationReason } : {}),
    });
  }

  whenIdle(): Promise<void> {
    if (this.active === 0 && this.queue.length === 0 && this.inFlight.size === 0) return Promise.resolve();
    return new Promise<void>((resolve) => {
      let unsubscribe = () => {};
      unsubscribe = this.subscribe((snapshot) => {
        if (snapshot.active !== 0 || snapshot.queued !== 0 || this.inFlight.size !== 0) return;
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
    this.modules.clear();
    this.bindGroupLayouts.clear();
    this.pipelineLayouts.clear();
    this.pipelines.clear();
    this.progress = undefined;
    for (const job of this.queue.splice(0)) {
      this.pipelineJobs.delete(job.key);
      job.reject(new GPUCompilationInvalidatedError(reason));
    }
    this.emit();
  }

  private forCaller<T>(shared: Promise<T>, signal?: AbortSignal): Promise<T> {
    if (!signal) return shared;
    if (signal.aborted) return Promise.reject(abortError());
    return new Promise<T>((resolve, reject) => {
      const aborted = () => reject(abortError());
      const unsubscribe = this.onAbort(signal, aborted);
      void shared.then(
        (value) => {
          unsubscribe();
          if (!signal.aborted) resolve(value);
        },
        (error) => {
          unsubscribe();
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
      job.started = true;
      this.active += 1;
      void this.run(job);
    }
    this.emit();
  }

  private async run(job: CompilationJob): Promise<void> {
    const started = performance.now();
    this.compilationAttempts++;
    try {
      this.progress = Object.freeze({ key: job.key, label: job.label,
        completed: 0, total: 1, current: `Compile ${job.kind} pipeline` });
      this.emit();
      this.assertCurrentGeneration();
      const pipeline = job.kind === "compute"
        ? await this.#device.createComputePipelineAsync(job.descriptor as GPUComputePipelineDescriptor)
        : await this.#device.createRenderPipelineAsync(job.descriptor as GPURenderPipelineDescriptor);
      this.assertCurrentGeneration();
      this.pipelines.set(job.key, pipeline);
      job.resolve(pipeline);
    } catch (error) {
      job.reject(error);
    } finally {
      this.compilationMilliseconds += performance.now() - started;
      this.pipelineJobs.delete(job.key);
      this.active -= 1;
      if (this.progress?.key === job.key) this.progress = undefined;
      this.emit();
      this.pump();
    }
  }

  private promote(job: CompilationJob, priority: GPUCompilationPriority): void {
    if (PRIORITY_ORDER[priority] >= PRIORITY_ORDER[job.priority]) return;
    job.priority = priority;
    this.sortQueue();
    this.emit();
  }

  private enqueueDirect<T extends Pipeline>(
    kind: "compute" | "render",
    descriptor: PipelineDescriptor,
    options: GPUCompilationRequestOptions,
    bundle?: BundleRequest,
  ): Promise<T> {
    if (options.signal?.aborted) return Promise.reject(abortError());
    if (this.invalidationReason) return Promise.reject(new GPUCompilationInvalidatedError(this.invalidationReason));
    // Consume iterables and copy mutable dictionaries now, as WebGPU does at
    // the API boundary, rather than when a scheduler slot becomes available.
    const saved = snapshotGPUDescriptor(descriptor);
    const key = `${kind}:${this.keys.key(saved)}`;
    const cached = this.pipelines.get(key);
    if (cached) {
      this.pipelineHits++;
      return this.forCaller(Promise.resolve(cached as T), options.signal);
    }
    let job = this.pipelineJobs.get(key);
    if (job) {
      this.inFlightHits++;
      this.promote(job, options.priority ?? "visible");
    } else {
      let resolve!: (pipeline: Pipeline) => void;
      let reject!: (error: unknown) => void;
      const promise = new Promise<Pipeline>((accept, fail) => { resolve = accept; reject = fail; });
      job = { kind, sequence: this.sequence++, priority: options.priority ?? "visible", key,
        label: saved.label ?? `${kind} pipeline`, descriptor: saved, promise,
        consumers: 0, started: false, resolve, reject };
      this.pipelineJobs.set(key, job);
      this.queue.push(job);
      this.sortQueue();
      this.schedulePump();
    }
    bundle?.jobs.add(job);
    const result = this.consume(job, options.signal) as Promise<T>;
    this.emit();
    return result;
  }

  private consume(job: CompilationJob, signal?: AbortSignal): Promise<Pipeline> {
    if (signal?.aborted) return Promise.reject(abortError());
    job.consumers++;
    return new Promise<Pipeline>((resolve, reject) => {
      let cancelled = false;
      const unsubscribe = signal ? this.onAbort(signal, () => {
        cancelled = true;
        job.consumers--;
        if (job.started) return; // Drain native work before this caller cleans up.
        reject(abortError());
        if (job.consumers !== 0) return;
        const index = this.queue.indexOf(job);
        if (index < 0) return;
        this.queue.splice(index, 1);
        this.pipelineJobs.delete(job.key);
        job.reject(abortError());
        this.emit();
      }) : undefined;
      void job.promise.then(
        pipeline => { if (cancelled) reject(abortError()); else resolve(pipeline); },
        error => reject(cancelled ? abortError() : error),
      ).finally(() => { unsubscribe?.(); });
    });
  }

  private async compile(
    manifest: AnyGPUCompilationManifest,
    request: BundleRequest,
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
      modules[name] = this.createShaderModule({
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
      compute[name] = await this.enqueueDirect<GPUComputePipeline>("compute", {
        ...descriptor,
        compute: { ...programmableStage, module: modules[module]! },
      }, { priority: request.priority }, request);
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
      render[name] = await this.enqueueDirect<GPURenderPipeline>("render", {
        ...descriptor,
        vertex: { ...vertexState, module: modules[vertexModule]! },
        ...(fragmentState ? { fragment: fragmentState } : {}),
      }, { priority: request.priority }, request);
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

/** Scoped allocation capabilities still share the device's single scheduler. */
export function aliasGPUCompilationDevice(device: GPUDevice, alias: GPUDevice): void {
  managerByDevice.set(alias, gpuCompilationManagerFor(device));
}

function defaultGPUCompilationConcurrency():number{
  // This is intentionally not the old Promise-all fanout: the manager remains
  // the sole scheduler and never has more than this many driver compilation
  // calls active. Three separate B16/P16 Dawn processes completed cleanly at
  // width three (~2.58 s resident construction versus ~3.50 s at width two).
  // Width four was faster in isolated construction runs but repeated full
  // resident builds could leave Metal in a degraded state, so production
  // deliberately retains one slot of headroom below the hard cap.
  const requestedConcurrency = typeof process !== "undefined"
    ? Number(process.env?.FLUID_GPU_COMPILATION_CONCURRENCY ?? 3) : 3;
  return Number.isSafeInteger(requestedConcurrency)
    && requestedConcurrency >= 1 && requestedConcurrency <= 4
    ? requestedConcurrency : 1;
}

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
  const maximumConcurrentBundles = defaultGPUCompilationConcurrency();
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
  const manager = managerByDevice.get(rawDevice) ?? new GPUCompilationManager(rawDevice,
    { ...options, maximumConcurrentBundles: options.maximumConcurrentBundles ?? defaultGPUCompilationConcurrency() });
  // One bound method per property for the device's life: the frame path reads
  // createCommandEncoder and the like every frame and must not allocate.
  const methods = new Map<string | symbol, unknown>();
  const disabled = (property: string) => () => {
    throw new Error(`${property} is disabled; acquire the pipeline through GPUCompilationManager`);
  };
  methods.set("createShaderModule", (descriptor: GPUShaderModuleDescriptor) => manager.createShaderModule(descriptor));
  methods.set("createBindGroupLayout", (descriptor: GPUBindGroupLayoutDescriptor) => manager.createBindGroupLayout(descriptor));
  methods.set("createPipelineLayout", (descriptor: GPUPipelineLayoutDescriptor) => manager.createPipelineLayout(descriptor));
  methods.set("destroy", () => { manager.invalidate("device destroyed"); rawDevice.destroy(); });
  methods.set("createComputePipelineAsync", (descriptor: GPUComputePipelineDescriptor) => manager.compileComputePipeline(descriptor));
  methods.set("createRenderPipelineAsync", (descriptor: GPURenderPipelineDescriptor) => manager.compileRenderPipeline(descriptor));
  methods.set("createComputePipeline", disabled("createComputePipeline"));
  methods.set("createRenderPipeline", disabled("createRenderPipeline"));
  const managed = new Proxy(rawDevice, {
    // Harness instrumentation can decorate this capability without replacing
    // the native method the manager calls (which would recurse into itself).
    set(target, property, value) {
      if (typeof value === "function") { methods.set(property, value); return true; }
      methods.delete(property);
      return Reflect.set(target, property, value, target);
    },
    get(target, property) {
      const method = methods.get(property);
      if (method) return method;
      const value = Reflect.get(target, property, target);
      if (typeof value !== "function") return value;
      const bound = value.bind(target);
      methods.set(property, bound);
      return bound;
    },
  }) as GPUDevice;
  managerByDevice.set(rawDevice, manager);
  managerByDevice.set(managed, manager);
  managedDeviceByRawDevice.set(rawDevice, managed);
  managedDeviceByRawDevice.set(managed, managed);
  return managed;
}

export function invalidateGPUCompilationManager(
  device: GPUDevice,
  reason = "GPU device retired",
): void {
  managerByDevice.get(device)?.invalidate(reason);
}
