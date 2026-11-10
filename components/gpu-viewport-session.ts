import { createBrowserFrameBenchmark } from "../lib/core/browser-frame-benchmark";
import { paneFrameCapture, type FrameInstrumentationInput } from "../lib/core/frame-instrumentation";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import type { PixelTraceConfig } from "../lib/core/webgpu-renderer";
import { webGPUPlatformResourcePlugin } from "../lib/core/webgpu-platform-resource";
import { WebGPURenderWorkerClient, type FluidLabRendererHandle } from "../lib/core/webgpu-render-worker-client";
import { defaultMethodId, getMethod } from "../lib/core/method-registry";
import { canonicalScene, type RunState } from "../lib/core/model";
import type { GPUStatus } from "../lib/core/gpu-status";
import { simulation } from "../lib/core/simulation/controller";
import { createInteractionFluidDriver } from "../lib/core/simulation/interaction-fluid-driver";
import { simulationRecording } from "../lib/core/simulation/recording";
import { applySceneDraft, type SceneDraftSubject } from "../lib/core/stores/scene-draft-store";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import { createDetailFocusCoalescer } from "../lib/core/solver-detail";
import type { UIStoreHook } from "../lib/core/stores/ui-store";
import type { PaneSession } from "../lib/core/session/session";
import { SmoothedFrameRate } from "../lib/core/frame-rate-meter";
import { getScenePreset } from "../lib/core/scenes";
import { RunScope } from "../lib/core/run-scope";
import {
  GPU_MANUAL_START_EVENT,
  GPU_MANUAL_STOP_EVENT,
  manualGPUControlTargetsPane,
  resolveGPUStartupMode,
  safeBrowserGPUBringupEnabled,
  safeBrowserGPUBringupViolations,
  safeBrowserSimulationEpochChanged,
  shutdownBrowserGPUSession,
  type PaneId,
} from "../lib/core/gpu-startup";
import { acquirePaneGPULease } from "../lib/core/session/pane-lease";

/**
 * The UI state a draw reads, named rather than derived from a store
 * singleton: the shape is the same in every pane.
 */
type UIViewState = ReturnType<UIStoreHook["getState"]>;

/** What a draw hands back to the component that currently owns the canvas. */
export interface GPUViewportRenderBinding {
  readonly publishFrameRate: (fps: number | undefined) => void;
  readonly pixelTraceDrawConfig: (
    ui: UIViewState,
    renderer: FluidLabRendererHandle,
  ) => PixelTraceConfig | undefined;
  readonly publishPixelTrace: (renderer: FluidLabRendererHandle) => void;
  readonly publishBodyPoses: (renderer: FluidLabRendererHandle) => void;
}

/**
 * The surface a retained session offers a replacement component. A session
 * created by an earlier module version is reached only through this, so its
 * shape changes only with `GPU_VIEWPORT_RUNTIME_ABI`.
 */
export interface GPUViewportLifecycle {
  /**
   * Compatibility of the retained renderer, its worker messages and every GPU
   * resource ABI it compiled. Fast Refresh may reclaim a session only when the
   * module that created it and the module doing the reclaim agree exactly.
   */
  readonly runtimeAbi: string;
  readonly canvas: HTMLCanvasElement;
  readonly renderer: FluidLabRendererHandle;
  /** Point the retained render loop at the current component's refs/state. */
  readonly rebind: (binding: GPUViewportRenderBinding) => void;
  /** Cancel the teardown queued by React's development effect replay. */
  readonly cancelDeferredCleanup: () => boolean;
  /** Give an HMR/RSC replacement time to reclaim this GPU session. */
  readonly deferCleanup: () => void;
  /** Tear down immediately when the retained canvas really changes. */
  readonly cleanupImmediately: () => Promise<void>;
}

export type GPUViewportWindow = Window & {
  /**
   * Survives Vinext RSC program reloads, which can replace the React ref.
   *
   * Keyed by pane, because compare mode mounts two viewports in one page and a
   * single slot made the second mount reclaim — or tear down — the first one's
   * device. The retain-on-canvas-identity rule is per pane and unchanged.
   */
  __fluidLabGPUViewportLifecycle?: Map<PaneId, GPUViewportLifecycle>;
};

/**
 * The pane ledger, tolerating a slot written by a module version that predates
 * pane keying (a single lifecycle object rather than a map).
 */
export function gpuViewportLifecycles(host: GPUViewportWindow): Map<PaneId, GPUViewportLifecycle> {
  const retained = host.__fluidLabGPUViewportLifecycle;
  if (retained instanceof Map) return retained;
  const lifecycles = new Map<PaneId, GPUViewportLifecycle>();
  host.__fluidLabGPUViewportLifecycle = lifecycles;
  return lifecycles;
}

const GPU_DEVELOPMENT_REBIND_GRACE_MS = 1_000;
// Bump whenever a hot update changes a scene field consumed by the renderer,
// worker message arguments, pipeline bindings, or an encode method signature.
// Schema 2's authoritative wall field landed alongside new terrain/water and
// render-stage ABIs; retaining a pre-change renderer made the next frame read
// the new document through stale code and fail with an unrelated `height_m`
// access. A cold page was sound because it never crossed that ABI boundary.
export const GPU_VIEWPORT_RUNTIME_ABI = "scene-v3-solid-world-render-stages-v1";

/**
 * Which open gestures the renderer is allowed to draw.
 *
 * A terrain proposal redraws the ground immediately; it cannot move the lattice,
 * so presenting it against the committed solver is safe. A prop is safer still —
 * it is render-only and outside the solver's keys entirely, and so is the sun. Every other draft is
 * overlay-only: reshaping the tank or the water does change the geometry the
 * solver owns, and drawing the fluid at a size it was not allocated for would
 * tear. Those wait for the release, and preview as the wireframe box instead.
 *
 * Declared as a list rather than a condition because the answer is a property of
 * each subject, and the next entity has to state its own rather than being
 * folded into somebody else's boolean.
 */
const PRESENTED_DRAFT_SUBJECTS: ReadonlySet<SceneDraftSubject> = new Set<SceneDraftSubject>(["terrain", "scenery", "lighting"]);

/**
 * Where the session is:
 *
 *   waiting ─start→ starting ─renderer ready→ rendering
 *      ↑               │ lease refused
 *      └───────────────┘
 *   any ─stop→ stopping ─device released→ stopped
 *
 * `stopped` is final: a pane gets a GPU again only through a new session.
 */
export type GPUViewportSessionPhase = "waiting" | "starting" | "rendering" | "stopping" | "stopped";

export interface GPUViewportSessionOptions {
  readonly canvas: HTMLCanvasElement;
  readonly paneId: PaneId;
  readonly session: PaneSession;
  readonly binding: GPUViewportRenderBinding;
  /** The predecessor's teardown; this session holds no device until it settles. */
  readonly previousShutdown?: Promise<void>;
  /** The component lets go of this session's renderer and lifecycle refs. */
  readonly onReleased: (session: GPUViewportSession) => void;
  /** The pane ledger this session registers in. */
  readonly lifecycles: Map<PaneId, GPUViewportLifecycle>;
  /** The renderer worker; replaced only by tests, which have no worker. */
  readonly createRenderer?: (canvas: HTMLCanvasElement, callbacks: RendererCallbacks) => WebGPURenderWorkerClient;
}

type RendererCallbacks = ConstructorParameters<typeof WebGPURenderWorkerClient>[1];

/**
 * One pane's WebGPU session: the renderer worker, its device lease, the
 * render loop and the status the pane shows for all of them.
 *
 * It replaces a React effect's nine closure flags (`running`, `alive`,
 * `initializationStarted`, `stopping`, `stopped`, `cleanupCompleted`, …) with
 * one `phase`, and every status it publishes passes `publish`, which refuses
 * anything once the session is stopping: a stopped GPU never reads as
 * initializing again.
 */
export class GPUViewportSession implements GPUViewportLifecycle {
  readonly runtimeAbi = GPU_VIEWPORT_RUNTIME_ABI;
  readonly canvas: HTMLCanvasElement;
  readonly renderer: WebGPURenderWorkerClient;
  private currentPhase: GPUViewportSessionPhase = "waiting";
  private binding: GPUViewportRenderBinding;
  /** The start attempt in flight; a stop or a refused lease ends it. */
  private readonly starts = new RunScope();
  /** The run-state request in flight; a newer one supersedes it. */
  private readonly runStateSyncs = new RunScope();
  private stopPromise?: Promise<void>;
  private releaseLease?: () => void;
  private frame = 0;
  private cleanupTimer?: number;
  private released = false;
  private safeSimulationEpoch?: number;
  private readonly safeBringup = safeBrowserGPUBringupEnabled(window.location.search);
  private readonly frameBenchmark = createBrowserFrameBenchmark();
  private readonly frameRate = new SmoothedFrameRate(5);
  /** This pane's orbit target as a transient detail focus, never written to the document. */
  private readonly detailFocus = createDetailFocusCoalescer();
  private readonly detach: (() => void)[] = [];
  private stopAutomaticStart: () => void = () => {};

  /** Undefined when the address turned WebGPU off for this page. */
  static start(options: GPUViewportSessionOptions): GPUViewportSession | undefined {
    const startupMode = resolveGPUStartupMode(window.location.search, {
      presetId: options.session.scene.getState().presetId,
      methodId: options.session.method.getState().methodId,
    });
    if (startupMode === "off") {
      options.session.diagnostics.getState().set({ gpuStatus: platformStatus("unavailable", "WebGPU disabled by gpu=off (UI-only mode)") });
      return undefined;
    }
    return new GPUViewportSession(options);
  }

  private constructor(private readonly options: GPUViewportSessionOptions) {
    const { canvas, paneId, session } = options;
    this.canvas = canvas;
    this.binding = options.binding;
    const createRenderer = options.createRenderer ?? ((target, callbacks) => new WebGPURenderWorkerClient(target, callbacks));
    this.renderer = createRenderer(canvas, {
      onStatus: (status) => this.rendererStatus(status),
      onGPUInfo: (info) => session.diagnostics.getState().set({ gpuInfo: info }),
      // Named, because the host's completed time is the *minimum* over the
      // panes: an unattributed completion would let one pane's progress
      // stand for the other's and break the barrier it exists to keep.
      onGPUAdvanceCompleted: (time_s) => simulation.gpuAdvanceCompleted(time_s, paneId),
      onEffectiveRendererStatus: (effectiveRendererStatus) => session.diagnostics.getState().set({ effectiveRendererStatus }),
      onLoad: (sample) => session.diagnostics.getState().pushWorkerLoad(sample),
    });
    this.detach.push(simulation.attachInteractionFluid(paneId,
      createInteractionFluidDriver(session.diagnostics, request => this.renderer.advanceInteractionFluid(request))));
    this.syncRunState(session.runtime.getState().runState);
    this.detach.push(session.runtime.subscribe((state, previous) => {
      if (state.simulationEpoch !== previous.simulationEpoch) {
        if (safeBrowserSimulationEpochChanged(this.safeBringup, this.currentPhase !== "waiting", this.safeSimulationEpoch, state.simulationEpoch)) {
          void this.stop("Safe WebGPU session stopped after a reset/rebuild attempt");
          return;
        }
        this.renderer.resetSimulationTimeline();
      }
      if (state.runState !== previous.runState) this.syncRunState(state.runState);
    }));
    // A document that turns automatic startup on starts the session; only the
    // first start is automatic, so a refused lease waits for an explicit one.
    const startAutomatically = () => {
      if (this.startupMode() === "automatic") void this.begin();
    };
    const unsubscribeScene = session.scene.subscribe(startAutomatically);
    const unsubscribeMethod = session.method.subscribe(startAutomatically);
    this.stopAutomaticStart = () => { unsubscribeScene(); unsubscribeMethod(); };
    this.detach.push(() => this.stopAutomaticStart());
    const enforceSafeConfiguration = () => {
      if (!this.safeBringup || this.currentPhase === "waiting") return;
      const violations = this.safeViolations();
      if (violations.length > 0) void this.stop(`Safe WebGPU session stopped after configuration drift: ${violations.join("; ")}`);
    };
    this.detach.push(session.scene.subscribe(enforceSafeConfiguration), session.method.subscribe(enforceSafeConfiguration),
      session.ui.subscribe(enforceSafeConfiguration));
    // A START request that names no pane starts every pane; one that names a
    // pane is that pane's alone.
    const manualStart = (event: Event) => { if (manualGPUControlTargetsPane(event, paneId)) void this.begin(); };
    const manualStop = (event: Event) => { if (manualGPUControlTargetsPane(event, paneId)) void this.stop(); };
    const pageHide = () => { void this.stop("WebGPU stopped during page close", false); };
    window.addEventListener(GPU_MANUAL_START_EVENT, manualStart);
    window.addEventListener(GPU_MANUAL_STOP_EVENT, manualStop);
    window.addEventListener("pagehide", pageHide, { once: true });
    this.detach.push(
      () => window.removeEventListener(GPU_MANUAL_START_EVENT, manualStart),
      () => window.removeEventListener(GPU_MANUAL_STOP_EVENT, manualStop),
      () => window.removeEventListener("pagehide", pageHide),
    );
    options.lifecycles.set(paneId, this);
    const mode = this.startupMode();
    if (mode === "manual" || mode === "safe") this.publish(platformStatus("manual", "WebGPU is waiting for explicit startup"));
    else void this.begin();
  }

  get phase(): GPUViewportSessionPhase { return this.currentPhase; }

  private get session(): PaneSession { return this.options.session; }

  private startupMode() {
    return resolveGPUStartupMode(window.location.search, {
      presetId: this.session.scene.getState().presetId,
      methodId: this.session.method.getState().methodId,
    });
  }

  /** The one status writer, closed once the session is stopping. */
  private publish(status: GPUStatus) {
    if (this.currentPhase === "stopping" || this.currentPhase === "stopped") return;
    this.session.diagnostics.getState().set({ gpuStatus: status });
  }

  private rendererStatus(status: GPUStatus) {
    // A terminal status stops the session whatever phase it is in: one that
    // arrived before the start used to leave the pane neither running nor
    // stopped, with its reason never shown. Deferred, because the client is
    // still dispatching the worker's message when it reports.
    if (status.state === "lost" || status.state === "unavailable" || status.state === "blocked") {
      // Published now so the stop, whoever's it is, carries the failure.
      if (status.state === "unavailable" && status.failure) this.publish(status);
      queueMicrotask(() => { void this.stop(status.label); });
      return;
    }
    const current = this.session.diagnostics.getState().gpuStatus;
    // The controller publishes the user's intent before the next render
    // can start expensive work. Preserve that context as detailed task
    // progress arrives from the renderer.
    const rendererOnlyReady = status.state === "ready" && status.label === "WebGPU renderer ready"
      && getMethod(this.session.method.getState().methodId).backend === "webgpu";
    // Close the platform plugin before opening the fluid plugin. Replacing
    // this event used to leave the completed 4/4 renderer task permanently
    // active while its label claimed the solver was being prepared.
    if (rendererOnlyReady) this.publish(status);
    const reportedStatus = rendererOnlyReady
      ? { state: "initializing" as const, label: "Renderer ready; preparing fenced t=0 solver authority", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup" as const, resource: getMethod(this.session.method.getState().methodId).resource }
      : status;
    this.publish(reportedStatus.state === "initializing" && current.state === "initializing" && current.operation
      ? { ...reportedStatus, operation: current.operation, kind: reportedStatus.kind ?? current.kind, retainingPrevious: reportedStatus.retainingPrevious ?? current.retainingPrevious }
      : reportedStatus);
  }

  private syncRunState(runState: RunState) {
    const sync = this.runStateSyncs.start();
    void this.renderer.setSimulationRunning(runState === "running").then((submittedTime_s) => {
      if (!sync.current() || runState !== "paused" || this.session.runtime.getState().runState !== "paused") return;
      simulation.gpuSchedulingPaused(submittedTime_s, this.options.paneId);
    }).catch(() => {
      // Worker failure is published through the renderer status callback.
    });
  }

  private safeViolations() {
    const sceneState = this.session.scene.getState(), methodState = this.session.method.getState(), ui = this.session.ui.getState();
    return safeBrowserGPUBringupViolations({
      presetId: sceneState.presetId,
      methodId: methodState.methodId,
      quality: methodState.quality,
      methodValues: resolvedMethodValues(methodState),
      canonicalMethodValues: resolvedMethodValues({ methodId: defaultMethodId(), quality: "balanced", overrides: {} }),
      exactScene: canonicalScene(sceneState.scene) === canonicalScene(getScenePreset("water-box-dam-break").create()),
      gridOverlayAxis: ui.gridOverlayAxis,
      armedGesture: ui.armedGesture,
      search: window.location.search,
    });
  }

  /** waiting → starting → rendering; a refused lease returns to waiting. */
  private async begin() {
    if (this.currentPhase !== "waiting") return;
    if (this.safeBringup) {
      const violations = this.safeViolations();
      if (violations.length > 0) {
        this.publish(platformStatus("manual", `Safe WebGPU start refused: ${violations.join("; ")}`));
        return;
      }
      simulation.setRunState("paused");
      this.safeSimulationEpoch = this.session.runtime.getState().simulationEpoch;
    }
    this.currentPhase = "starting";
    this.stopAutomaticStart();
    const start = this.starts.start();
    const { previousShutdown, paneId } = this.options;
    if (previousShutdown) {
      this.publish(startupStatus("Waiting for the previous WebGPU session to finish"));
      await previousShutdown;
      if (!start.current()) return;
    }
    const lease = acquirePaneGPULease(paneId);
    if (lease.status !== "acquired") {
      this.currentPhase = "waiting";
      this.publish(platformStatus("manual", `WebGPU start refused: ${lease.message}`));
      return;
    }
    this.releaseLease = lease.release;
    this.publish(startupStatus("Initializing WebGPU"));
    try {
      await this.renderer.initialize();
    } catch (error: unknown) {
      if (start.current()) void this.stop(error instanceof Error ? error.message : "WebGPU initialization failed");
      return;
    }
    if (!start.current()) return;
    this.currentPhase = "rendering";
    this.frame = requestAnimationFrame(this.render);
  }

  /** Any phase → stopping → stopped. Idempotent; the first label wins. */
  stop(label = "WebGPU stopped; device released — safe to close this tab", publishStatus = true): Promise<void> {
    if (this.stopPromise) return this.stopPromise;
    const diagnostics = this.session.diagnostics.getState();
    const failure = diagnostics.gpuStatus.state === "unavailable" ? diagnostics.gpuStatus.failure
      : diagnostics.gpuInfo?.simulationFailure;
    if (publishStatus) this.publish(platformStatus("stopping", `${label} — stopping WebGPU and draining pending work`));
    this.currentPhase = "stopping";
    this.starts.dispose();
    simulation.setRunState("paused");
    cancelAnimationFrame(this.frame);
    const releasedLabel = label.includes("device released") ? label : `${label}; device released — safe to close this tab`;
    this.stopPromise = (async () => {
      await shutdownBrowserGPUSession(this.renderer, this.releaseLease, this.options.previousShutdown);
      this.releaseLease = undefined;
      this.currentPhase = "stopped";
      if (publishStatus) {
        this.session.diagnostics.getState().set({ gpuStatus: { ...platformStatus("unavailable", releasedLabel), failure } });
      }
    })();
    return this.stopPromise;
  }

  private readonly render = (raf_ms?: number) => {
    if (this.currentPhase !== "rendering") return;
    this.frame = requestAnimationFrame(this.render);
    const { session, paneId } = this.options;
    const renderer = this.renderer;
    const sceneState = session.scene.getState();
    const scene = sceneState.scene;
    const draft = session.sceneDraft.getState().draft;
    const presentationScene = PRESENTED_DRAFT_SUBJECTS.has(draft?.subject as SceneDraftSubject)
      ? applySceneDraft(scene, draft)
      : scene;
    renderer.setSimulationScene(presentationScene === scene ? undefined : scene);
    const ui = session.ui.getState();
    const method = session.method.getState();
    const state = session.diagnostics.getState();
    const runtime = session.runtime.getState();
    const scenePreset = getScenePreset(sceneState.presetId);
    // Pausing freezes simulation time, not presentation. Attempt every
    // browser animation frame; the renderer's double buffer bounds latency
    // without changing cadence for camera motion or simulation state.
    let metrics;
    const frameBenchmark = this.frameBenchmark;
    const drawStarted = frameBenchmark ? performance.now() : 0;
    try {
      metrics = renderer.draw(
        simulation.time(), presentationScene, frameBenchmark?.camera(ui.camera) ?? ui.camera, state.bodies, ui.selectedBodyId,
        {
          methodId: method.methodId,
          quality: method.quality,
          values: resolvedMethodValues(method),
          simulationEpoch: runtime.simulationEpoch,
          // Read every frame rather than captured: entering compare mode
          // pins the window to one advance, and a depth captured at mount
          // would leave this pane two deep inside the barrier.
          inFlightDepth: simulation.inFlightDepth(),
          detailFocus: this.detailFocus(ui.camera.target_m),
        },
        {
          axis: ui.gridOverlayAxis, position: ui.gridOverlaySlice, mode: ui.gridOverlayMode, layers: ui.visualLayers, particles: ui.fluidParticleDisplay,
          particleSurface: {
            greed: ui.fineSurfaceGreed, smoothing: ui.fineSurfaceSmoothing, thickness: ui.fineSurfaceThickness,
            spraySize: ui.fineSpraySize, sprayOpacity: ui.fineSprayOpacity, merge: ui.fineSurfaceMerge,
          },
        },
        scenePreset.background,
        scenePreset.id === sceneState.presetId ? scenePreset.presentationMode : "full-scene",
        ui.fluidSurfaceRenderMode,
        {
          surfaceTexturesEnabled: ui.svoSurfaceTexturesEnabled,
          nativeAmbientOcclusionEnabled: ui.svoNativeAmbientOcclusionEnabled,
          shadowsEnabled: ui.svoShadowsEnabled,
          waterReflectionMode: ui.waterReflectionMode,
          waterWorldRefractionEnabled: ui.waterWorldRefractionEnabled,
          lensBloom: { enabled: ui.lensBloomEnabled, strength: ui.lensBloomStrength,
            threshold: ui.lensBloomThreshold, radius: ui.lensBloomRadius },
          ambientOcclusionEnabled: ui.svoAmbientOcclusionEnabled,
          silhouetteRefinementEnabled: ui.silhouetteRefinementEnabled,
          coneTracingMode: ui.svoConeTracingMode,
          globalIlluminationEnabled: ui.svoGlobalIlluminationEnabled,
          voxelSpecularReflectionsEnabled: ui.svoVoxelSpecularReflectionsEnabled,
          voxelSpecularSteps: ui.svoVoxelSpecularSteps,
          worldGiCacheEnabled: ui.svoWorldGiCacheEnabled,
          latticeVisibilityEnabled: ui.svoLatticeVisibilityEnabled,
          primaryTraversal: ui.svoPrimaryTraversal,
          disabledStages: ui.disabledRenderStages,
        },
        {
          stageView: ui.svoStageView,
          lightSlot: ui.svoStageLightSlot,
          maximumTraversalDepth: ui.svoMaximumTraversalDepth,
          maximumNodeVisits: ui.svoMaximumNodeVisits,
        },
        ui.svoRenderTuning,
        this.binding.pixelTraceDrawConfig(ui, renderer),
      );
    } catch (error: unknown) {
      void this.stop(error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped");
      return;
    }
    this.binding.publishBodyPoses(renderer);
    frameBenchmark?.sample(renderer.completedPresentationCount, performance.now() - drawStarted, metrics, renderer.presentationResolution, {
      scene: sceneState.presetId, runState: runtime.runState, camera: ui.camera,
      shadows: ui.svoShadowsEnabled, ao: ui.svoAmbientOcclusionEnabled, gi: ui.svoGlobalIlluminationEnabled,
      lighting: ui.svoConeTracingMode, primary: ui.svoPrimaryTraversal, water: ui.fluidSurfaceRenderMode,
      disabledStages: ui.disabledRenderStages, tuning: ui.svoRenderTuning,
      effectiveRenderer: state.effectiveRendererStatus,
      readiness: { gpu: state.gpuStatus, svo: state.resourceReadiness.svo, fluid: state.resourceReadiness.fluid },
    }, state.effectiveRendererStatus.state === "active"
      && (ui.svoPrimaryTraversal !== "mesh" || state.effectiveRendererStatus.surfaceMesh?.state === "ready")
      && state.effectiveRendererStatus.lightingVisibility?.state === ui.svoConeTracingMode
      && state.resourceReadiness.platform.state === "ready"
      && state.resourceReadiness.svo.state === "ready"
      && state.resourceReadiness.fluid.state === "ready");
    this.binding.publishPixelTrace(renderer);
    simulation.recordFrame(metrics, renderer.presentationResolution, paneId);
    // Phase 0 frame capture: built only while instrumentation is enabled.
    paneFrameCapture(paneId, usePerformanceInstrumentationStore.getState().enabled)?.record({
      raf_ms, cpu: metrics.cpu, presentation: metrics.presentation, dispatches: metrics.dispatches,
      presentationSubmitted: metrics.presentationSubmitted, solver: state.gpuInfo as FrameInstrumentationInput["solver"],
    });
    this.binding.publishFrameRate(this.frameRate.sampleCompleted(renderer.completedPresentationCount, performance.now()));
    if (metrics.presentationSubmitted) {
      simulationRecording.capturePresentedState(this.canvas, runtime.simulationTime);
    }
  };

  readonly rebind = (binding: GPUViewportRenderBinding) => { this.binding = binding; };

  readonly cancelDeferredCleanup = () => {
    if (this.released) return false;
    if (this.cleanupTimer !== undefined) window.clearTimeout(this.cleanupTimer);
    this.cleanupTimer = undefined;
    return true;
  };

  readonly deferCleanup = () => {
    if (this.released || this.cleanupTimer !== undefined) return;
    this.cleanupTimer = window.setTimeout(
      this.cleanupImmediately,
      process.env.NODE_ENV === "development" ? GPU_DEVELOPMENT_REBIND_GRACE_MS : 0,
    );
  };

  readonly cleanupImmediately = (): Promise<void> => {
    if (this.released) return this.stopPromise!;
    this.released = true;
    this.frameBenchmark?.dispose();
    if (this.cleanupTimer !== undefined) window.clearTimeout(this.cleanupTimer);
    this.cleanupTimer = undefined;
    this.runStateSyncs.dispose();
    for (const detach of this.detach.splice(0)) detach();
    this.options.onReleased(this);
    // Keep the draining lifecycle discoverable by a replacement canvas. Its
    // lease remains held until outstanding compilation and teardown finish.
    const shutdown = this.stop("WebGPU stopped during component cleanup", false);
    const { lifecycles, paneId } = this.options;
    void shutdown.then(() => {
      if (lifecycles.get(paneId) === this) lifecycles.delete(paneId);
    });
    return shutdown;
  };
}

function platformStatus<State extends "manual" | "stopping" | "unavailable">(state: State, label: string) {
  return { state, label, resource: webGPUPlatformResourcePlugin };
}

function startupStatus(label: string): GPUStatus {
  return { state: "initializing", label, phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup", resource: webGPUPlatformResourcePlugin };
}
