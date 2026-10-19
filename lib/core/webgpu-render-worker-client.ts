import type { FluidStepRequest } from "../interaction/runtime/physics";
import type { FluidExchange } from "../interaction/contracts";
import { registerMeshAssetDistanceField } from "./imported-assets/asset-library";
import type { MeshVolumeDistanceField } from "./imported-assets/mesh-volume";
import { adoptEnvironmentProxyCatalog, useRemoteEnvironmentCatalogs, cachedEnvironmentProxyCatalog, reuseEnvironmentProxyCatalog, markEnvironmentCatalogRemote, type EnvironmentProxyCatalog } from "./voxel-environments";
import type { LiveFluidEdit, LiveFluidEditResult } from "./live-fluid-edit";
import type { GPUEulerianInfo, GPURigidLoad } from "./webgpu-eulerian";
import type { InjectedLiquidBall } from "./method-contract";
import type { SceneDescription } from "./model";
import { applyDocumentDelta, documentDelta, DocumentDeltaError, type DocumentDeltaEntry } from "./document-delta";
import { appendedSolidPatches, solidPatchListExtends } from "./solid-patch-list";
import { solidWorldContentStamp, type SolidWorldVoxelPatch } from "./solid-world";
import type { SolidEditBaseIdentity } from "./voxel-editor/worker-solid-edit-acceptance";
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
  /** Absent means the tank is the domain; the host resolves it against the scene. */
  /** Poses the drawn frame used, in roster order. See FluidLabRenderer. */
  rigidBodyPoses: FluidLabRenderer["rigidBodyPoses"];
  rigidBodyPoseRevision: number;
}

export type { SolidEditBaseIdentity };

/** Whether the patch list is all that separates the two documents: every other field is the same object. */
function differInSolidPatchesAlone(base: SceneDescription, next: SceneDescription): boolean {
  const fields = Object.keys(next) as (keyof SceneDescription)[];
  if (fields.length !== Object.keys(base).length) return false;
  for (const field of fields) {
    if (field !== "solidVoxels" && next[field] !== base[field]) return false;
  }
  return true;
}

/**
 * What separates `next` from `base`, in the smallest form that names it: the
 * patches a stroke or a redo appends, or the patch count an undo goes back to.
 * Undefined for anything else — a clone shares no fields, and a branch rewrites
 * the list — which is a document that has to cross whole.
 */
function solidEditDelta(base: SceneDescription, next: SceneDescription):
  { patches: readonly SolidWorldVoxelPatch[] } | { patchCount: number } | undefined {
  if (!differInSolidPatchesAlone(base, next)) return undefined;
  const patches = appendedSolidPatches(base.solidVoxels, next.solidVoxels);
  if (patches) return { patches };
  return solidPatchListExtends(next.solidVoxels, base.solidVoxels) ? { patchCount: next.solidVoxels.length } : undefined;
}

export type WebGPURenderWorkerRequest =
  | { type: "attach"; canvas: OffscreenCanvas }
  | { type: "initialize"; requestId: number }
  | { type: "set-render-scene"; revision: number; scene: SceneDescription; needsSceneryCatalog?: boolean; terrainContentStamp: string }
  /**
   * The next document over the one the worker holds: the parts that changed,
   * with the identity of the document they changed from. The worker refuses
   * them unless that is the document it holds, and keeps every part they do
   * not name the object it already has. No entries is the same document under
   * a new revision. `terrainContentStamp` crosses only when the ground changed.
   */
  | { type: "update-render-scene"; revision: number; base: SolidEditBaseIdentity; entries: DocumentDeltaEntry[]; needsSceneryCatalog?: boolean; terrainContentStamp?: string }
  /** The worker already holds this document: it accepted it as a solid edit. Only the revision crosses. */
  | { type: "adopt-render-scene"; revision: number }
  | { type: "draw"; frameId: number; sceneRevision: number; args: DrawArgumentsWithoutScene; viewport: { width: number; height: number; devicePixelRatio: number }; instrumentationMode: PerformanceInstrumentationMode }
  | { type: "set-simulation-scene"; scene: DrawArguments[1] | undefined }
  | { type: "set-hover-highlight"; range: { first: number; last: number } | undefined }
  | { type: "set-simulation-running"; requestId: number; running: boolean }
  | { type: "reset-simulation-timeline" }
  /**
   * A stroke or a redo over the document the worker holds: only its patches
   * cross, with the identity of the document they were authored against. The
   * worker refuses them unless that is the document it holds.
   *
   * Every solid edit names the `chain` it was sent on. Edits are sent without
   * waiting for the one before; a refusal ends its chain, and the worker
   * answers the edits already sent behind it as cancelled.
   */
  | { type: "extend-solid-edit"; requestId: number; chain: number; patches: readonly SolidWorldVoxelPatch[]; base: SolidEditBaseIdentity }
  /** An undo over the document the worker holds: the patch count it goes back to, and the stamp of the document that is. */
  | { type: "truncate-solid-edit"; requestId: number; chain: number; base: SolidEditBaseIdentity; target: { patchCount: number; stamp: string } }
  /** Any other solid edit — a loaded or mirrored document — crosses whole. */
  | { type: "validate-solid-edit"; requestId: number; chain: number; scene: SceneDescription }
  /**
   * A live gesture's proposal, for the display alone: its patches over the
   * document the worker holds, or none to put the display back. Not an edit —
   * nothing is accepted, and the simulation stays on the resident document.
   */
  | { type: "preview-solid-edit"; previewId: number; base: SolidEditBaseIdentity; patches?: readonly SolidWorldVoxelPatch[] }
  | { type: "interaction-fluid-step"; requestId: number; step: FluidStepRequest }
  | { type: "edit-fluid"; requestId: number; edit: LiveFluidEdit }
  | { type: "inject-liquid-ball"; requestId: number; ball: InjectedLiquidBall }
  | { type: "pick-rigid-body"; requestId: number; args: PickArguments }
  | { type: "shutdown"; requestId: number };

export type WebGPURenderWorkerResponse =
  | { type: "mesh-asset-distance-field"; id: string; field: MeshVolumeDistanceField }
  | { type: "scenery-catalog"; sceneRevision: number; catalog: EnvironmentProxyCatalog }
  /** The catalog of this revision over the last one sent: the parts that differ, or none when it is that catalog still. */
  | { type: "scenery-catalog-changes"; sceneRevision: number; entries: DocumentDeltaEntry[] }
  | { type: "attached"; timeOrigin_ms: number }
  | { type: "status"; status: GPUStatus; workerNow_ms: number }
  | { type: "gpu-info"; info: GPUEulerianInfo }
  | { type: "rigid-loads"; loads: GPURigidLoad[] }
  | { type: "advance-completed"; time_s: number }
  | { type: "effective-renderer-status"; status: EffectiveRendererStatus }
  | { type: "initialized"; requestId: number }
  | { type: "simulation-running-set"; requestId: number; submittedTime_s: number | undefined }
  | { type: "frame"; frameId: number; metrics: RendererFrameMetrics; snapshot: WebGPURenderWorkerSnapshot }
  | { type: "pick-result"; requestId: number; result: PickResult }
  | { type: "solid-edit-validated"; requestId: number }
  /** `staged` is whether the display now shows the proposal; false leaves it on the resident document. */
  | { type: "solid-edit-previewed"; previewId: number; staged: boolean }
  | { type: "interaction-fluid-result"; requestId: number; result: readonly FluidExchange[] }
  | { type: "fluid-edit-result"; requestId: number; result: LiveFluidEditResult }
  | { type: "inject-result"; requestId: number; taken: boolean }
  | { type: "shutdown-complete"; requestId: number }
  | { type: "request-failed"; requestId: number; message: string };

interface WorkerClientCallbacks {
  onStatus(status: GPUStatus): void;
  onGPUInfo?(info: GPUEulerianInfo): void;
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
  private publishedRenderDocument?: SceneDescription;
  private readonly renderSceneRevisionByDocument = new WeakMap<SceneDescription, number>();
  private readonly terrainStampByDocument = new WeakMap<TerrainDescription, string>();
  /** The last catalog the worker sent, which its next changes are over. */
  private sceneryCatalog?: EnvironmentProxyCatalog;
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
  private runtimeUnavailable?: string;
  private lastRuntimeStatus?: GPUStatus;
  /** One fixed translation keeps an activity's start time stable across progress messages. */
  private workerClockOffset_ms?: number;
  private readonly requests = new Map<number, { kind: WebGPURenderWorkerRequest["type"]; resolve(value: unknown): void; reject(error: Error): void }>();

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
    useRemoteEnvironmentCatalogs();
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
  /** The document the worker last accepted as a solid edit, until anything else is published. */
  private acceptedSolidEditDocument?: SceneDescription;
  /** Whole documents sent so far: an acceptance one of them overtook is not what the worker holds. */
  private renderScenePublications = 0;
  private solidEditsInFlight = 0;
  /** The last document proposed, while edits are in flight: what the worker holds once it has answered them. */
  private solidEditHead?: SceneDescription;
  /** Stamped on every solid edit; moved on by the first refusal, which cancels the edits sent behind it. */
  private solidEditChain = 1;
  /** The document the worker holds, or will once it has answered the edits sent so far. */
  private heldSolidEditDocument(): SceneDescription | undefined {
    return this.solidEditHead ?? this.acceptedSolidEditDocument ?? this.publishedRenderDocument;
  }
  private solidEditBase(base: SceneDescription): SolidEditBaseIdentity {
    return { revision: this.publishedRenderSceneRevision, patchCount: base.solidVoxels.length, stamp: solidWorldContentStamp(base) };
  }
  /**
   * Ask the worker to accept `scene` in place of `base`.
   *
   * Edits are sent as they are made, without waiting for the one before: the
   * worker answers them in order, each over the document the last one left. An
   * edit over the document the worker holds sends only what separates the two
   * — a stroke's or a redo's patches, an undo's patch count — with that
   * document's identity, so its cost is the edit's however long the document
   * has grown. Everything else sends `scene` whole: a mirrored or loaded
   * document, and any edit whose base the worker was never given.
   *
   * A refusal rejects its own edit with the reason and every edit already sent
   * behind it as cancelled; the worker is left holding the last accepted
   * document.
   */
  validateLiveSolidEdit(scene: SceneDescription, base: SceneDescription): Promise<void> {
    const publications = this.renderScenePublications;
    const delta = this.heldSolidEditDocument() === base ? solidEditDelta(base, scene) : undefined;
    const requestId = this.nextRequestId(), chain = this.solidEditChain;
    // The edit decides what the display shows from here: accepted, it is what a
    // preview showed or replaces it; refused, the worker puts the display back.
    this.queuedPreview = undefined;
    this.previewBase = undefined;
    const accepted = !delta ? this.request<void>({ type: "validate-solid-edit", requestId, chain, scene })
      : "patches" in delta
        ? this.request<void>({ type: "extend-solid-edit", requestId, chain, patches: delta.patches, base: this.solidEditBase(base) })
        : this.request<void>({ type: "truncate-solid-edit", requestId, chain, base: this.solidEditBase(base),
          target: { patchCount: delta.patchCount, stamp: solidWorldContentStamp(scene) } });
    this.solidEditsInFlight += 1;
    this.solidEditHead = scene;
    return accepted.then(() => {
      if (publications === this.renderScenePublications) this.acceptedSolidEditDocument = scene;
    }, (error: unknown) => {
      // The first refusal of a chain: what was sent behind it is cancelled, and the next edit starts another.
      if (chain === this.solidEditChain) { this.solidEditChain += 1; this.solidEditHead = undefined; }
      throw error;
    }).finally(() => {
      this.solidEditsInFlight -= 1;
      if (this.solidEditsInFlight === 0) this.solidEditHead = undefined;
    });
  }

  private previewId = 0;
  /** The preview the worker has not answered yet. */
  private previewInFlight?: number;
  /** The newest sample, waiting its turn; a newer one replaces it. */
  private queuedPreview?: { patches: readonly SolidWorldVoxelPatch[] | undefined; base: SceneDescription };
  /** The worker answered a preview and has not drawn a frame since. */
  private previewAwaitsFrame = false;
  /** The document the display is showing a proposal over, from the first sample sent until it is put back or accepted. */
  private previewBase?: SceneDescription;
  /**
   * Show a live gesture's patches over `base` through the display alone, or
   * with none put the display back on `base`.
   *
   * Latest wins: one preview is with the worker at a time, a newer sample
   * replaces the one waiting, and the next is not sent until the worker has
   * drawn a frame since the last, so the display is never asked for more than
   * one a frame. Only the patches cross. A proposal over a document the worker
   * does not hold is not sent; the gesture keeps its outline.
   */
  previewLiveSolidEdit(patches: readonly SolidWorldVoxelPatch[] | undefined, base: SceneDescription): void {
    if (this.stopped || this.failed || this.runtimeUnavailable) return;
    if (!patches?.length) {
      if (this.queuedPreview?.base === base) this.queuedPreview = undefined;
      // Only the gesture whose proposal is up can take it down.
      if (this.previewBase !== base) return;
      this.queuedPreview = { patches: undefined, base };
    } else {
      if (this.heldSolidEditDocument() !== base) return;
      this.queuedPreview = { patches, base };
    }
    this.sendQueuedPreview();
  }

  private sendQueuedPreview(): void {
    const queued = this.queuedPreview;
    if (!queued || this.previewInFlight !== undefined) return;
    // Putting the display back is not held for a frame: nothing new is being shown.
    if (queued.patches && this.previewAwaitsFrame && this.frameInFlight) return;
    this.queuedPreview = undefined;
    // An edit sent since the sample was taken moved the document it was over.
    if (queued.patches && this.heldSolidEditDocument() !== queued.base) return;
    this.previewBase = queued.patches ? queued.base : undefined;
    this.previewInFlight = ++this.previewId;
    this.previewAwaitsFrame = false;
    this.post({ type: "preview-solid-edit", previewId: this.previewInFlight, base: this.solidEditBase(queued.base),
      ...(queued.patches && { patches: queued.patches }) });
  }

  advanceInteractionFluid(step: FluidStepRequest): Promise<readonly FluidExchange[]> {
    return this.request({ type: "interaction-fluid-step", requestId: this.nextRequestId(), step });
  }

  editFluid(edit: LiveFluidEdit): Promise<LiveFluidEditResult> {
    return this.request<LiveFluidEditResult>({ type: "edit-fluid", requestId: this.nextRequestId(), edit });
  }

  injectLiquidBall(ball: InjectedLiquidBall): Promise<boolean> {
    return this.request<boolean>({ type: "inject-liquid-ball", requestId: this.nextRequestId(), ball });
  }

  pickRigidBody(...args: PickArguments): Promise<PickResult> {
    return this.request<PickResult>({ type: "pick-rigid-body", requestId: this.nextRequestId(), args });
  }

  shutdown(): Promise<void> {
    if (this.stopped) return Promise.resolve();
    this.stopped = true;
    this.rejectAll(new Error("WebGPU runtime stopped; the last accepted scene is retained."));
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
    // The document the worker holds, known only once it has answered every
    // edit sent to it: the last it accepted, else the last published; and the
    // revision it holds it under, before this publication moves that on.
    const held = this.solidEditsInFlight === 0 ? this.acceptedSolidEditDocument ?? this.publishedRenderDocument : undefined;
    const heldRevision = this.publishedRenderSceneRevision;
    this.publishedRenderSceneRevision = revision;
    // Only scenery this realm can tell is the same by identity: comparing two
    // sceneries by content serializes both, and a document that brings its own
    // is answered with the worker's catalog below anyway.
    const published = this.publishedRenderDocument;
    if (published && published.scenery === scene.scenery) reuseEnvironmentProxyCatalog(published, scene);
    this.publishedRenderDocument = scene;
    markEnvironmentCatalogRemote(scene);
    // Acceptance already cloned this exact document into the worker and both
    // GPU consumers adopted it there. Sending it again would clone the scene a
    // second time and republish it under a second worker-side identity.
    const accepted = this.acceptedSolidEditDocument === scene;
    this.acceptedSolidEditDocument = undefined;
    if (accepted) { this.post({ type: "adopt-render-scene", revision }); return revision; }
    this.renderScenePublications += 1;
    // The worker holds this document from here, whatever edits it is still to answer.
    this.solidEditHead = undefined;
    this.previewBase = undefined;
    // Over the document the worker holds, only what changed crosses, and the
    // worker keeps every other part the object it already has. The first
    // document, another document altogether, and one published while edits are
    // unanswered cross whole.
    const changes = held && documentDelta(held, scene);
    if (held && changes) {
      const touches = (part: string) => changes.some((entry) => entry.path[0] === part);
      let stamp: string | undefined;
      if (touches("terrain")) stamp = this.terrainStamp(scene.terrain);
      else if (held.terrain && scene.terrain && this.terrainStampByDocument.has(held.terrain)) {
        this.terrainStampByDocument.set(scene.terrain, this.terrainStampByDocument.get(held.terrain)!);
      }
      // Scenery that did not change is the scenery the held document's catalog
      // was expanded from, whichever object carries it; the catalog is then
      // this document's too wherever the two agree on the frame it is placed in.
      const catalog = scene.scenery && held.scenery !== scene.scenery && !touches("scenery")
        ? cachedEnvironmentProxyCatalog(held) : undefined;
      if (catalog) adoptEnvironmentProxyCatalog({ ...held, scenery: scene.scenery }, catalog);
      // Named the way a solid edit names its base.
      const base = { revision: heldRevision, patchCount: held.solidVoxels.length, stamp: solidWorldContentStamp(held) };
      this.post({ type: "update-render-scene", revision, base, entries: changes, terrainContentStamp: stamp,
        needsSceneryCatalog: !cachedEnvironmentProxyCatalog(scene) });
      return revision;
    }
    this.post({ type: "set-render-scene", revision, scene, terrainContentStamp: this.terrainStamp(scene.terrain),
      needsSceneryCatalog: !cachedEnvironmentProxyCatalog(scene) });
    return revision;
  }

  private terrainStamp(terrain: TerrainDescription | undefined): string {
    let stamp = terrain && this.terrainStampByDocument.get(terrain);
    if (stamp === undefined) {
      stamp = terrainContentStamp(terrain);
      if (terrain) this.terrainStampByDocument.set(terrain, stamp);
    }
    return stamp;
  }

  private request<T>(message: Extract<WebGPURenderWorkerRequest, { requestId: number }>): Promise<T> {
    if (this.failed || (message.type !== "shutdown" && (this.stopped || this.runtimeUnavailable))) {
      return Promise.reject(new Error(this.runtimeUnavailable ?? "WebGPU runtime is no longer available."));
    }
    return new Promise<T>((resolve, reject) => {
      this.requests.set(message.requestId, { kind: message.type, resolve: (value) => resolve(value as T), reject });
      try { this.worker.postMessage(message); }
      catch (error) {
        this.requests.delete(message.requestId);
        reject(error);
      }
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
    if (message.type === "attached") {
      // Translate clocks using their actual origins, not message delivery time:
      // a busy worker/window adds latency, not a change of clock origin.
      this.workerClockOffset_ms = message.timeOrigin_ms - performance.timeOrigin;
    }
    else if (message.type === "scenery-catalog" || message.type === "scenery-catalog-changes") {
      // Kept whether or not its document is still the published one: the next changes are over this catalog.
      if (message.type === "scenery-catalog") this.sceneryCatalog = message.catalog;
      else {
        try {
          if (!this.sceneryCatalog) throw new DocumentDeltaError("Scenery catalog changes arrived before any catalog.");
          this.sceneryCatalog = applyDocumentDelta(this.sceneryCatalog, message.entries);
        } catch (error) {
          // Every later catalog would be changes over the same absent one.
          const label = `GPU runtime stopped: ${error instanceof Error ? error.message : String(error)}`;
          this.runtimeUnavailable = label;
          this.rejectAll(new Error(label), true);
          this.callbacks.onStatus({ state: "unavailable", label, resource: webGPUPlatformResourcePlugin });
          return;
        }
      }
      if (message.sceneRevision === this.publishedRenderSceneRevision && this.publishedRenderDocument) {
        adoptEnvironmentProxyCatalog(this.publishedRenderDocument, this.sceneryCatalog);
      }
    }
    else if (message.type === "mesh-asset-distance-field") registerMeshAssetDistanceField(message.id, message.field);
    else if (message.type === "status") {
      const status = message.status.state === "initializing" && message.status.startedAt_ms !== undefined
        ? { ...message.status,
          // `performance.now()` is realm-relative. Calibrate once: deriving a
          // fresh offset from each delivery gives the same activity a slightly
          // different start time on every update, restarting progress effects.
          startedAt_ms: message.status.startedAt_ms
            + (this.workerClockOffset_ms ??= performance.now() - message.workerNow_ms) }
        : message.status;
      if (status.state === "unavailable" || status.state === "lost" || status.state === "blocked") {
        // Worker console entries disappear when terminal cleanup terminates
        // it. Keep the failure and its last progress milestone in the page.
        console.error("WebGPU runtime failed", JSON.stringify({
          state: status.state, label: status.label,
          previousState: this.lastRuntimeStatus?.state,
          previousLabel: this.lastRuntimeStatus?.label,
          completedPresentations: this.snapshot.completedPresentationCount,
          presentationResolution: this.snapshot.presentationResolution,
        }));
        this.runtimeUnavailable = status.label;
        // The worker may still need to drain/destroy GPU resources. Reject
        // authoring requests now without interrupting that shutdown handshake.
        this.rejectAll(new Error(status.label), true);
      }
      this.lastRuntimeStatus = status;
      this.callbacks.onStatus(status);
    }
    else if (message.type === "gpu-info") {
      const offset = this.workerClockOffset_ms ?? 0;
      if (message.info.physicsTrace) message.info.physicsTrace.capturedAt_ms += offset;
      if (message.info.physicsCPUTrace && message.info.physicsCPUTrace !== message.info.physicsTrace) {
        message.info.physicsCPUTrace.capturedAt_ms += offset;
      }
      this.callbacks.onGPUInfo?.(message.info);
    }
    else if (message.type === "rigid-loads") this.callbacks.onGPURigidLoads?.(message.loads);
    else if (message.type === "advance-completed") this.callbacks.onGPUAdvanceCompleted?.(message.time_s);
    else if (message.type === "effective-renderer-status") this.callbacks.onEffectiveRendererStatus?.(message.status);
    else if (message.type === "initialized" || message.type === "shutdown-complete" || message.type === "solid-edit-validated") this.settle(message.requestId);
    else if (message.type === "simulation-running-set") this.settle(message.requestId, message.submittedTime_s);
    else if (message.type === "pick-result") this.settle(message.requestId, message.result);
    else if (message.type === "interaction-fluid-result") this.settle(message.requestId, message.result);
    else if (message.type === "fluid-edit-result") this.settle(message.requestId, message.result);
    else if (message.type === "inject-result") this.settle(message.requestId, message.taken);
    else if (message.type === "request-failed") this.settle(message.requestId, undefined, message.message);
    else if (message.type === "solid-edit-previewed") {
      if (message.previewId !== this.previewInFlight) return;
      this.previewInFlight = undefined;
      // A sample that was not shown costs the display nothing, so the next need not wait for a frame.
      this.previewAwaitsFrame = message.staged;
      this.sendQueuedPreview();
    }
    else if (message.type === "frame") {
      // postMessage gives this realm ownership of these cloned samples. Convert
      // each distinct trace once; presentation and stages may alias one object.
      const { cpu, presentation, presentationStages, presentationQueue } = message.metrics;
      const offset = this.workerClockOffset_ms ?? 0;
      if (cpu) cpu.capturedAt_ms += offset;
      if (presentation && presentation !== cpu) presentation.capturedAt_ms += offset;
      if (presentationStages && presentationStages !== cpu && presentationStages !== presentation) presentationStages.capturedAt_ms += offset;
      if (presentationQueue && presentationQueue !== cpu && presentationQueue !== presentation && presentationQueue !== presentationStages) presentationQueue.capturedAt_ms += offset;
      this.snapshot = message.snapshot;
      this.completedMetrics = message.metrics;
      this.frameInFlight = false;
      const queued = this.queuedFrame;
      this.queuedFrame = undefined;
      if (queued && !this.frameDispatchSuspended) this.sendFrame(queued);
      this.previewAwaitsFrame = false;
      this.sendQueuedPreview();
    }
  }

  private rejectAll(error: Error, preserveShutdown = false): void {
    for (const [id, request] of this.requests) {
      if (preserveShutdown && request.kind === "shutdown") continue;
      this.requests.delete(id);
      request.reject(error);
    }
  }
}

export type FluidLabRendererHandle = WebGPURenderWorkerClient;
