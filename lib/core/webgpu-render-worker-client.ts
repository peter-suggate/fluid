import type { GPUEulerianInfo, GPURigidLoad } from "./webgpu-eulerian";
import type { PressureJournal } from "./pressure-journal";
import type { StageLensReceipt } from "./stage-lens";
import type { StageLensLayerReport } from "./webgpu-stage-lens-overlay";
import type { InjectedLiquidBall } from "./method-contract";
import type { SceneDescription } from "./model";
import { terrainContentStamp, type TerrainDescription } from "./terrain";
import type { GPUStatus } from "./gpu-status";
import type { EffectiveRendererStatus } from "./renderer-status";
import type { FluidLabRenderer, RendererFrameMetrics } from "./webgpu-renderer";
import { webGPUPlatformResourcePlugin } from "./webgpu-platform-resource";
import { usePerformanceInstrumentationStore, type PerformanceInstrumentationMode } from "./stores/performance-instrumentation-store";

type DrawArguments = Parameters<FluidLabRenderer["draw"]>;
type DrawArgumentsWithoutScene = DrawArguments extends [infer Time, unknown, ...infer Rest]
  ? [Time, ...Rest]
  : never;
type PickArguments = Parameters<FluidLabRenderer["pickRigidBody"]>;
type PickResult = Awaited<ReturnType<FluidLabRenderer["pickRigidBody"]>>;

export interface WebGPURenderWorkerSnapshot {
  presentationRevision: number;
  presentationResolution: string;
  completedPresentationCount: number;
  latestPixelTrace: FluidLabRenderer["latestPixelTrace"];
  pixelTraceRevision: number;
  pixelTraceAvailable: boolean;
  pixelTraceStatus: FluidLabRenderer["pixelTraceStatus"];
  pixelTraceAnswersRequest: boolean;
  pixelTraceStale: boolean;
  latestFluidCellTrace: FluidLabRenderer["latestFluidCellTrace"];
  fluidCellTraceRevision: number;
  fluidCellTraceReady: boolean;
  fluidCellTraceFineBand: FluidLabRenderer["fluidCellTraceFineBand"];
  /** Absent means the tank is the domain; the host resolves it against the scene. */
  fluidCellTraceDomain: FluidLabRenderer["fluidCellTraceDomain"];
  /** Poses the drawn frame used, in roster order. See FluidLabRenderer. */
  rigidBodyPoses: FluidLabRenderer["rigidBodyPoses"];
  rigidBodyPoseRevision: number;
}

export type WebGPURenderWorkerRequest =
  | { type: "attach"; canvas: OffscreenCanvas }
  | { type: "initialize"; requestId: number }
  | { type: "set-render-scene"; revision: number; scene: SceneDescription; terrainContentStamp: string }
  | { type: "draw"; frameId: number; sceneRevision: number; args: DrawArgumentsWithoutScene; viewport: { width: number; height: number; devicePixelRatio: number }; instrumentationMode: PerformanceInstrumentationMode }
  | { type: "set-simulation-scene"; scene: DrawArguments[1] | undefined }
  | { type: "set-hover-highlight"; range: { first: number; last: number } | undefined }
  | { type: "set-simulation-running"; requestId: number; running: boolean }
  | { type: "reset-simulation-timeline" }
  | { type: "inject-liquid-ball"; requestId: number; ball: InjectedLiquidBall }
  | { type: "pick-rigid-body"; requestId: number; args: PickArguments }
  | { type: "shutdown"; requestId: number };

export type WebGPURenderWorkerResponse =
  | { type: "attached" }
  | { type: "status"; status: GPUStatus; workerNow_ms: number }
  | { type: "gpu-info"; info: GPUEulerianInfo }
  | { type: "pressure-journal"; journal: PressureJournal | undefined }
  | { type: "stage-lens"; receipt?: StageLensReceipt; layers: readonly StageLensLayerReport[] }
  | { type: "rigid-loads"; loads: GPURigidLoad[] }
  | { type: "advance-completed"; time_s: number }
  | { type: "effective-renderer-status"; status: EffectiveRendererStatus }
  | { type: "initialized"; requestId: number }
  | { type: "simulation-running-set"; requestId: number; submittedTime_s: number | undefined }
  | { type: "frame"; frameId: number; metrics: RendererFrameMetrics; snapshot: WebGPURenderWorkerSnapshot }
  | { type: "pick-result"; requestId: number; result: PickResult }
  | { type: "inject-result"; requestId: number; taken: boolean }
  | { type: "shutdown-complete"; requestId: number }
  | { type: "request-failed"; requestId: number; message: string };

interface WorkerClientCallbacks {
  onStatus(status: GPUStatus): void;
  onGPUInfo?(info: GPUEulerianInfo): void;
  onGPUPressureJournal?(journal: PressureJournal | undefined): void;
  onGPUStageLens?(
    receipt: StageLensReceipt | undefined,
    layers: readonly StageLensLayerReport[],
  ): void;
  onGPURigidLoads?(loads: GPURigidLoad[]): void;
  onGPUAdvanceCompleted?(time_s: number): void;
  onEffectiveRendererStatus?(status: EffectiveRendererStatus): void;
}

const EMPTY_SNAPSHOT: WebGPURenderWorkerSnapshot = {
  presentationRevision: 0,
  presentationResolution: "worker starting",
  completedPresentationCount: 0,
  latestPixelTrace: undefined,
  pixelTraceRevision: 0,
  pixelTraceAvailable: false,
  pixelTraceStatus: "path-inactive",
  pixelTraceAnswersRequest: false,
  pixelTraceStale: false,
  latestFluidCellTrace: undefined,
  fluidCellTraceRevision: 0,
  fluidCellTraceReady: false,
  fluidCellTraceFineBand: undefined,
  fluidCellTraceDomain: undefined,
  rigidBodyPoses: [],
  rigidBodyPoseRevision: 0,
};

/**
 * Main-thread façade for the worker-owned renderer.
 *
 * Every public method is non-blocking. Frame requests are latest-wins: while
 * the worker is allocating or compiling, the UI can keep painting and only
 * the newest pending camera/scene snapshot is retained.
 */
export class WebGPURenderWorkerClient {
  private readonly worker: Worker;
  private requestId = 0;
  private frameId = 0;
  private nextRenderSceneRevision = 0;
  private publishedRenderSceneRevision = 0;
  private readonly renderSceneRevisionByDocument = new WeakMap<SceneDescription, number>();
  private readonly terrainStampByDocument = new WeakMap<TerrainDescription, string>();
  private simulationScene?: SceneDescription;
  private frameInFlight = false;
  /** Control edges fence frame dispatch until the worker has observed them. */
  private simulationControlRevision = 0;
  private frameDispatchSuspended = false;
  private queuedFrame?: Extract<WebGPURenderWorkerRequest, { type: "draw" }>;
  private completedMetrics?: RendererFrameMetrics;
  private snapshot: WebGPURenderWorkerSnapshot = EMPTY_SNAPSHOT;
  private stopped = false;
  private failed = false;
  private readonly requests = new Map<number, { resolve(value: unknown): void; reject(error: Error): void }>();

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly callbacks: WorkerClientCallbacks,
  ) {
    if (typeof canvas.transferControlToOffscreen !== "function") {
      throw new Error("This browser cannot move WebGPU rendering off the main thread");
    }
    this.worker = new Worker(new URL("./webgpu-render-worker.ts", import.meta.url), {
      type: "module",
      name: "fluid-lab-webgpu-runtime",
    });
    this.worker.addEventListener("message", (event: MessageEvent<WebGPURenderWorkerResponse>) => this.receive(event.data));
    this.worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "WebGPU runtime worker failed");
      this.failed = true;
      this.rejectAll(error);
      this.callbacks.onStatus({ state: "unavailable", label: error.message, resource: webGPUPlatformResourcePlugin });
    });
    this.worker.addEventListener("messageerror", () => {
      const error = new Error("WebGPU runtime worker returned an unserializable result");
      this.failed = true;
      this.rejectAll(error);
      this.callbacks.onStatus({ state: "unavailable", label: error.message, resource: webGPUPlatformResourcePlugin });
    });
    const offscreen = canvas.transferControlToOffscreen();
    this.worker.postMessage({ type: "attach", canvas: offscreen } satisfies WebGPURenderWorkerRequest, [offscreen]);
  }

  get presentationRevision() { return this.snapshot.presentationRevision; }
  get presentationResolution() { return this.snapshot.presentationResolution; }
  get completedPresentationCount() { return this.snapshot.completedPresentationCount; }
  get latestPixelTrace() { return this.snapshot.latestPixelTrace; }
  get pixelTraceRevision() { return this.snapshot.pixelTraceRevision; }
  get pixelTraceAvailable() { return this.snapshot.pixelTraceAvailable; }
  get pixelTraceStatus() { return this.snapshot.pixelTraceStatus; }
  get pixelTraceAnswersRequest() { return this.snapshot.pixelTraceAnswersRequest; }
  get pixelTraceStale() { return this.snapshot.pixelTraceStale; }
  get latestFluidCellTrace() { return this.snapshot.latestFluidCellTrace; }
  get fluidCellTraceRevision() { return this.snapshot.fluidCellTraceRevision; }
  get fluidCellTraceReady() { return this.snapshot.fluidCellTraceReady; }
  get fluidCellTraceFineBand() { return this.snapshot.fluidCellTraceFineBand; }
  get fluidCellTraceDomain() { return this.snapshot.fluidCellTraceDomain; }
  get rigidBodyPoses() { return this.snapshot.rigidBodyPoses; }
  get rigidBodyPoseRevision() { return this.snapshot.rigidBodyPoseRevision; }

  initialize(): Promise<void> {
    return this.request<void>({ type: "initialize", requestId: this.nextRequestId() });
  }

  draw(...args: DrawArguments): RendererFrameMetrics {
    const metrics = this.completedMetrics ?? {
      context: "webgpu-worker-pending",
      methodId: args[5].methodId,
      presentationSubmitted: false,
    };
    this.completedMetrics = undefined;
    const [time_s, scene, ...remainingArgs] = args;
    const sceneRevision = this.publishRenderScene(scene);
    const rect = this.canvas.getBoundingClientRect();
    const message: Extract<WebGPURenderWorkerRequest, { type: "draw" }> = {
      type: "draw",
      frameId: ++this.frameId,
      sceneRevision,
      args: [time_s, ...remainingArgs] as DrawArgumentsWithoutScene,
      // Zustand stores are realm-local. The render panel changes the window's
      // store, while FluidLabRenderer reads the worker's copy; carrying the
      // mode on every latest-wins frame keeps those two copies one contract.
      instrumentationMode: usePerformanceInstrumentationStore.getState().mode,
      viewport: {
        width: Math.max(1, rect.width),
        height: Math.max(1, rect.height),
        devicePixelRatio: Math.min(globalThis.devicePixelRatio || 1, 2),
      },
    };
    if (this.frameInFlight || this.frameDispatchSuspended) this.queuedFrame = message;
    else this.sendFrame(message);
    return metrics;
  }

  setSimulationScene(scene: DrawArguments[1] | undefined): void {
    if (scene === this.simulationScene) return;
    this.simulationScene = scene;
    this.post({ type: "set-simulation-scene", scene });
  }

  /**
   * The described object the cursor is over, for the hover rim.
   *
   * Posted rather than carried on the frame because hover changes at pointer
   * rate, which has nothing to do with the frame loop, and because a rim is not
   * part of what the frame is *of* — it is an annotation on it.
   */
  setHoverHighlight(range: { first: number; last: number } | undefined): void {
    this.post({ type: "set-hover-highlight", range });
  }

  setSimulationRunning(running: boolean): Promise<number | undefined> {
    const revision = ++this.simulationControlRevision;
    // A frame captured before this edge still carries the old admission
    // intent. Do not send it after pause/play; wait for the worker's exact
    // submitted-time acknowledgement and let the next rAF provide a fresh
    // clock snapshot.
    this.frameDispatchSuspended = true;
    this.queuedFrame = undefined;
    return this.request<number | undefined>({
      type: "set-simulation-running",
      requestId: this.nextRequestId(),
      running,
    }).finally(() => {
      if (revision !== this.simulationControlRevision) return;
      this.queuedFrame = undefined;
      this.frameDispatchSuspended = false;
    });
  }

  resetSimulationTimeline(): void {
    this.post({ type: "reset-simulation-timeline" });
  }

  /**
   * Hand a dropped ball to the running solve, and say whether it was taken.
   *
   * The answer is the whole point of the round trip: not every method has an
   * injection, and a caller that assumed one would silently drop the user's
   * water on the floor. False means "author it into the document and pay the
   * re-seed instead", which is a worse outcome than a live drop but a far
   * better one than nothing happening.
   */
  injectLiquidBall(ball: InjectedLiquidBall): Promise<boolean> {
    return this.request<boolean>({ type: "inject-liquid-ball", requestId: this.nextRequestId(), ball });
  }

  pickRigidBody(...args: PickArguments): Promise<PickResult> {
    return this.request<PickResult>({ type: "pick-rigid-body", requestId: this.nextRequestId(), args });
  }

  shutdown(): Promise<void> {
    if (this.stopped) return Promise.resolve();
    this.stopped = true;
    if (this.failed) {
      this.worker.terminate();
      return Promise.resolve();
    }
    return this.request<void>({ type: "shutdown", requestId: this.nextRequestId() })
      .finally(() => { this.worker.terminate(); });
  }

  private nextRequestId() { return ++this.requestId; }

  private publishRenderScene(scene: SceneDescription): number {
    let revision = this.renderSceneRevisionByDocument.get(scene);
    if (revision === undefined) {
      this.nextRenderSceneRevision = this.nextRenderSceneRevision >= Number.MAX_SAFE_INTEGER
        ? 1 : this.nextRenderSceneRevision + 1;
      revision = this.nextRenderSceneRevision;
      this.renderSceneRevisionByDocument.set(scene, revision);
    }
    if (revision === this.publishedRenderSceneRevision) return revision;
    this.publishedRenderSceneRevision = revision;
    const terrain = scene.terrain;
    let stamp = terrain && this.terrainStampByDocument.get(terrain);
    if (stamp === undefined) {
      stamp = terrainContentStamp(terrain);
      if (terrain) this.terrainStampByDocument.set(terrain, stamp);
    }
    this.post({ type: "set-render-scene", revision, scene, terrainContentStamp: stamp });
    return revision;
  }

  private request<T>(message: Extract<WebGPURenderWorkerRequest, { requestId: number }>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.requests.set(message.requestId, { resolve: (value) => resolve(value as T), reject });
      this.worker.postMessage(message);
    });
  }

  private post(message: WebGPURenderWorkerRequest): void {
    if (!this.stopped) this.worker.postMessage(message);
  }

  private sendFrame(message: Extract<WebGPURenderWorkerRequest, { type: "draw" }>): void {
    if (this.stopped) return;
    this.frameInFlight = true;
    this.worker.postMessage(message);
  }

  private settle(requestId: number, value?: unknown, error?: string): void {
    const request = this.requests.get(requestId);
    if (!request) return;
    this.requests.delete(requestId);
    if (error) request.reject(new Error(error));
    else request.resolve(value);
  }

  private receive(message: WebGPURenderWorkerResponse): void {
    if (message.type === "status") {
      const status = message.status.state === "initializing" && message.status.startedAt_ms !== undefined
        ? { ...message.status,
          // `performance.now()` is realm-relative. Preserve elapsed worker
          // time while translating its origin into the document's clock.
          startedAt_ms: performance.now() - Math.max(0, message.workerNow_ms - message.status.startedAt_ms) }
        : message.status;
      this.callbacks.onStatus(status);
    }
    else if (message.type === "gpu-info") this.callbacks.onGPUInfo?.(message.info);
    else if (message.type === "pressure-journal") {
      this.callbacks.onGPUPressureJournal?.(message.journal);
    }
    else if (message.type === "stage-lens") {
      this.callbacks.onGPUStageLens?.(message.receipt, message.layers);
    }
    else if (message.type === "rigid-loads") this.callbacks.onGPURigidLoads?.(message.loads);
    else if (message.type === "advance-completed") this.callbacks.onGPUAdvanceCompleted?.(message.time_s);
    else if (message.type === "effective-renderer-status") this.callbacks.onEffectiveRendererStatus?.(message.status);
    else if (message.type === "initialized" || message.type === "shutdown-complete") this.settle(message.requestId);
    else if (message.type === "simulation-running-set") this.settle(message.requestId, message.submittedTime_s);
    else if (message.type === "pick-result") this.settle(message.requestId, message.result);
    else if (message.type === "inject-result") this.settle(message.requestId, message.taken);
    else if (message.type === "request-failed") this.settle(message.requestId, undefined, message.message);
    else if (message.type === "frame") {
      this.snapshot = message.snapshot;
      this.completedMetrics = message.metrics;
      this.frameInFlight = false;
      const queued = this.queuedFrame;
      this.queuedFrame = undefined;
      if (queued && !this.frameDispatchSuspended) this.sendFrame(queued);
    }
  }

  private rejectAll(error: Error): void {
    for (const request of this.requests.values()) request.reject(error);
    this.requests.clear();
  }
}

export type FluidLabRendererHandle = WebGPURenderWorkerClient;
