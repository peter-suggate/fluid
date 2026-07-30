"use client";

import { useEffect, useRef, useState } from "react";
import { FluidLabRenderer, type PixelTraceConfig, type PixelTraceStatus } from "@/lib/webgpu-renderer";
import {
  resolveSvoPixelTracePin, resolveSvoPixelTracePinnedFrame, svoPixelTracePinClick,
  type SvoPixelTrace, type SvoPixelTracePinRequest,
} from "@/lib/svo-pixel-trace";
import { PixelTraceHud } from "./PixelTraceHud";
import { getMethod } from "@/lib/methods";
import { canonicalScene, type CameraState } from "@/lib/model";
import { add, cameraBasis, dot, length, orbit, pan, scale, sub, zoom } from "@/lib/math";
import { boundingRadius, createBodyDescription, type RigidBodyState } from "@/lib/rigid-body";
import { simulation } from "@/lib/simulation/controller";
import { simulationRecording } from "@/lib/simulation/recording";
import { projectToViewport, viewportRayForPointer } from "@/lib/webgpu-camera";
import {
  closestPointOnAxis,
  gizmoAxisDragPosition,
  gizmoHandleAtPointer,
  projectGizmo,
  GIZMO_AXIS_DIRECTIONS,
  type GizmoAxis,
} from "@/lib/editor-gizmo";
import { CLICK_SLOP_PX, emptySpaceClickDeselects } from "@/lib/editor-tools";
import { hoverSceneAt, restOnHover, type EditorHover } from "@/lib/editor-hover";
import {
  aimInflow,
  createInflowAt,
  inflowHandles,
  moveInflow,
  INFLOW_SELECTION_ID,
  type InflowHandleKind,
} from "@/lib/editor-inflow";
import {
  editorFluidLattice,
  eraseFluidBrick,
  fillFractionForHeight,
  fillLevelHandlePosition,
  fluidBrickCenter,
  fluidBrickIndexAt,
  fluidBrickKey,
  fluidPaintPatch,
  paintFluidBrick,
} from "@/lib/editor-fluid";
import {
  applyTerrainFeatureDrag,
  terrainFeatureAt,
  terrainFeatureHandles,
  terrainFeatureIndex,
  terrainFeatureSelectionId,
  type TerrainHandleKind,
} from "@/lib/editor-terrain";
import { SelectionFlyout } from "./SelectionFlyout";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useMethodStore, resolvedMethodValues } from "@/lib/stores/method-store";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { SmoothedFrameRate } from "@/lib/frame-rate-meter";
import { getScenePreset } from "@/lib/scenes";
import { SVO_COST_OVERLAY_DEFINITIONS } from "@/lib/svo-render-diagnostics";
import { projectViewportFailure, viewportFailureIndicator } from "@/lib/viewport-failure-diagnostics";
import { dawnReproductionForGPUFailure } from "@/lib/webgpu-failure-reproduction";
import {
  acquireBrowserGPULease,
  GPU_MANUAL_START_EVENT,
  GPU_MANUAL_STOP_EVENT,
  resolveGPUStartupMode,
  safeBrowserGPUBringupEnabled,
  safeBrowserGPUBringupViolations,
  safeBrowserSimulationEpochChanged,
  shutdownBrowserGPUSession,
} from "@/lib/gpu-startup";

type Vec3 = RigidBodyState["position_m"];

/** Duration of the pinned trace's self-drawing sweep. */
const PIXEL_TRACE_REVEAL_MS = 1100;
/** The HUD's readout cadence; the 3D overlay itself stays per-frame. */
const PIXEL_TRACE_HUD_INTERVAL_MS = 110;

export function WebGPUViewport() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fpsRef = useRef<HTMLOutputElement>(null);
  const rendererRef = useRef<FluidLabRenderer | null>(null);
  const camera = useUIStore((state) => state.camera);
  const setCamera = useUIStore((state) => state.setCamera);
  const setDiagnosticsOpen = useUIStore((state) => state.setDiagnosticsOpen);
  const scene = useSceneStore((state) => state.scene);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const waterSurfacePresentation = useDiagnosticsStore((state) => state.waterSurfacePresentation);
  const [viewportSize, setViewportSize] = useState({ width: 1, height: 1 });
  const svoCostOverlay = useUIStore((state) => state.svoCostOverlay);
  const svoRenderMode = useUIStore((state) => state.svoRenderMode);
  const voxelRenderMode = useUIStore((state) => state.voxelRenderMode);
  const svoMaximumTraversalDepth = useUIStore((state) => state.svoMaximumTraversalDepth);
  const svoMaximumNodeVisits = useUIStore((state) => state.svoMaximumNodeVisits);
  const svoOverlayDefinition = SVO_COST_OVERLAY_DEFINITIONS[svoCostOverlay];
  const svoOverlayRamp = `linear-gradient(90deg,${svoOverlayDefinition.legend
    .map((stop) => `${stop.color} ${Math.round(stop.at * 100)}%`).join(",")})`;
  const activeTool = useUIStore((state) => state.activeTool);
  const selection = useUIStore((state) => state.selection);
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const [hover, setHover] = useState<EditorHover | null>(null);
  const pixelTraceEnabled = useUIStore((state) => state.pixelTraceEnabled);
  const pixelTracePinned = useUIStore((state) => state.pixelTracePinned);
  const pixelTraceLayers = useUIStore((state) => state.pixelTraceLayers);
  const setPixelTraceEnabled = useUIStore((state) => state.setPixelTraceEnabled);
  const setPixelTracePinned = useUIStore((state) => state.setPixelTracePinned);
  const requestPixelTracePin = useUIStore((state) => state.requestPixelTracePin);
  const togglePixelTraceLayer = useUIStore((state) => state.togglePixelTraceLayer);
  const [pixelTraceState, setPixelTraceState] = useState<{
    trace: SvoPixelTrace | undefined;
    status: PixelTraceStatus;
    pointerSeen: boolean;
    /** A pinned trace whose scene has changed and whose aim can no longer refresh. */
    stale: boolean;
  }>({ trace: undefined, status: "path-inactive", pointerSeen: false, stale: false });
  const pixelTrace = pixelTraceState.trace;
  /** Latest pointer position in viewport fractions; read by the render loop. */
  const tracePointerRef = useRef<{ normalizedX: number; normalizedY: number } | null>(null);
  /**
   * A click waiting for the probe to answer its own pixel. While one is held the
   * probe traces the clicked position rather than the live pointer, so the ray
   * that ends up frozen is the one the user aimed at.
   */
  const tracePinRequestRef = useRef<SvoPixelTracePinRequest | null>(null);
  /**
   * Where the pinned ray was aimed, and from which view. The position is what the
   * probe must re-trace when the scene changes under a pinned ray; the camera key
   * is what says whether re-tracing it would still be the same ray.
   */
  const tracePinnedRef = useRef<{ normalizedX: number; normalizedY: number; cameraKey: string } | null>(null);
  /** Press origin of a gesture that could still turn out to be a pinning click. */
  const tracePinGestureRef = useRef<{ id: number; downX: number; downY: number } | null>(null);
  /** Pin transitions restart the reveal sweep; live hover always shows it whole. */
  const traceRevealRef = useRef({ pinned: false, startedAt_ms: 0 });
  const tracePublishRef = useRef<{
    revision: number;
    at_ms: number;
    status: PixelTraceStatus;
    pointerSeen: boolean;
    pinned: boolean;
    stale: boolean;
  }>({ revision: -1, at_ms: 0, status: "path-inactive", pointerSeen: false, pinned: false, stale: false });

  /**
   * Identity of the view a pin request was aimed from. Any change means the
   * clicked pixel names a different ray, so the request no longer describes what
   * the user pointed at.
   */
  const pixelTraceCameraKey = (view: CameraState): string =>
    `${view.azimuth_rad}|${view.elevation_rad}|${view.distance_m}|${view.target_m.x},${view.target_m.y},${view.target_m.z}`;

  const pixelTraceDrawConfig = (
    ui: ReturnType<typeof useUIStore.getState>,
    renderer: FluidLabRenderer,
  ): PixelTraceConfig | undefined => {
    if (!ui.pixelTraceEnabled) { tracePinRequestRef.current = null; tracePinnedRef.current = null; return undefined; }
    const pinnedAt = ui.pixelTracePinned ? tracePinnedRef.current : null;
    // A pinned ray traces its own pixel, not the pointer's, so a refresh answers
    // the ray that is frozen rather than wherever the mouse happens to be. A
    // pending click outranks the live pointer for the same reason: the probe must
    // stay on the clicked pixel until that pixel comes back.
    const pointer = pinnedAt ?? tracePinRequestRef.current ?? tracePointerRef.current;
    if (!pointer) return undefined;
    const now_ms = performance.now();
    if (ui.pixelTracePinned !== traceRevealRef.current.pinned) {
      traceRevealRef.current = { pinned: ui.pixelTracePinned, startedAt_ms: now_ms };
    }
    const { refresh } = resolveSvoPixelTracePinnedFrame({
      pinned: ui.pixelTracePinned,
      sceneChanged: renderer.pixelTraceStale,
      aimCameraKey: pinnedAt?.cameraKey,
      cameraKey: pixelTraceCameraKey(ui.camera),
    });
    return {
      normalizedX: pointer.normalizedX,
      normalizedY: pointer.normalizedY,
      layers: ui.pixelTraceLayers,
      // Live hover changes the ray every frame, so an animated sweep would only
      // strobe. Pinning is the moment worth animating: the frozen ray then draws
      // its own work in the order the shader did it. A refresh deliberately does
      // not replay it — the ray did not change, only the work it found.
      reveal: ui.pixelTracePinned
        ? Math.min(1, (now_ms - traceRevealRef.current.startedAt_ms) / PIXEL_TRACE_REVEAL_MS)
        : 1,
      pinned: ui.pixelTracePinned,
      refresh,
    };
  };

  const publishPixelTrace = (renderer: FluidLabRenderer) => {
    const status = renderer.pixelTraceStatus;
    const pointerSeen = tracePointerRef.current !== null;
    const revision = renderer.pixelTraceRevision;
    // A pin asked for from the panel or the HUD button becomes the same request a
    // click makes, so both record an exact aim and neither can re-aim later.
    const ui = useUIStore.getState();
    if (ui.pixelTracePinRequested && !ui.pixelTracePinned && !tracePinRequestRef.current && tracePointerRef.current) {
      tracePinRequestRef.current = svoPixelTracePinClick({
        pinned: false, pending: false,
        ...tracePointerRef.current,
        cameraKey: pixelTraceCameraKey(ui.camera),
        revision,
      }).request ?? null;
      useUIStore.setState({ pixelTracePinRequested: false });
    }
    const pinRequest = tracePinRequestRef.current;
    if (pinRequest) {
      // The probe has been tracing the clicked pixel since the click; freeze it
      // only once that pixel is what came back.
      const resolution = resolveSvoPixelTracePin(pinRequest, {
        answered: renderer.pixelTraceAnswersRequest,
        cameraKey: pixelTraceCameraKey(useUIStore.getState().camera),
        probeCanAnswer: status !== "unsupported" && status !== "path-inactive",
        revision,
      });
      if (resolution !== "wait") {
        tracePinRequestRef.current = null;
        if (resolution === "pin") {
          // Keep the aim: a pinned ray has to be able to re-trace its own pixel
          // when the scene changes, and to know which view that pixel meant.
          tracePinnedRef.current = {
            normalizedX: pinRequest.normalizedX,
            normalizedY: pinRequest.normalizedY,
            cameraKey: pinRequest.cameraKey,
          };
          setPixelTracePinned(true);
        }
      }
    }
    const pinnedNow = useUIStore.getState().pixelTracePinned;
    // A pin with no aim on record can never be refreshed, so it must never be
    // possible: an aim invented from the current pointer and camera would let a
    // later refresh answer a ray the user never pinned. Controls outside the
    // viewport therefore ask, and the ask is honoured through the same handshake
    // a click uses.
    if (!pinnedNow) tracePinnedRef.current = null;
    // Staleness worth reporting is the kind no refresh can fix; everything else
    // resolves itself on the next frame.
    const { stale } = resolveSvoPixelTracePinnedFrame({
      pinned: pinnedNow,
      sceneChanged: renderer.pixelTraceStale,
      aimCameraKey: tracePinnedRef.current?.cameraKey,
      cameraKey: pixelTraceCameraKey(useUIStore.getState().camera),
    });
    const published = tracePublishRef.current;
    // Why nothing is showing matters as much as what is showing, and a frozen
    // answer about a scene that has since changed is worth saying out loud, so
    // both publish even on frames where no new trace arrived.
    if (status !== published.status || pointerSeen !== published.pointerSeen || stale !== published.stale) {
      tracePublishRef.current = { ...published, status, pointerSeen, stale };
      setPixelTraceState((current) => ({ ...current, status, pointerSeen, stale }));
    }
    const pinnedChanged = pinnedNow !== tracePublishRef.current.pinned;
    if (pinnedChanged) tracePublishRef.current = { ...tracePublishRef.current, pinned: pinnedNow };
    if (revision === published.revision) return;
    const now_ms = performance.now();
    // The overlay is already live on the GPU; the HUD's numbers only need to keep
    // up with reading, not with the pointer. Pinning is the exception: it stops
    // the readback, so the frozen trace is the last one there will ever be and
    // skipping it would leave the HUD describing a different pixel.
    if (!pinnedChanged && now_ms - published.at_ms < PIXEL_TRACE_HUD_INTERVAL_MS) return;
    tracePublishRef.current = { ...tracePublishRef.current, revision, at_ms: now_ms };
    setPixelTraceState((current) => ({ ...current, trace: renderer.latestPixelTrace }));
  };

  /**
   * Fold a completed press into the ray diagnostic.
   *
   * Pinning is orthogonal to whatever else the click did — select a body, clear
   * the selection — so it neither consumes the gesture nor has to win a priority
   * fight with the pointer machine. It only declines when an authoring tool is
   * armed, because a paint dab is a click too and flip-flopping the pin under
   * every dab would be nothing but noise.
   */
  const resolvePixelTracePinGesture = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const gesture = tracePinGestureRef.current;
    tracePinGestureRef.current = null;
    if (!gesture || gesture.id !== event.pointerId || event.type === "pointercancel") return;
    if (Math.hypot(event.clientX - gesture.downX, event.clientY - gesture.downY) > CLICK_SLOP_PX) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const pointer = {
      normalizedX: (event.clientX - rect.left) / Math.max(rect.width, 1),
      normalizedY: (event.clientY - rect.top) / Math.max(rect.height, 1),
    };
    // A click is a pointer observation in its own right: without this a click
    // that never moved first would have nowhere to trace.
    tracePointerRef.current = pointer;
    const ui = useUIStore.getState();
    const { request } = svoPixelTracePinClick({
      pinned: ui.pixelTracePinned,
      pending: tracePinRequestRef.current !== null,
      ...pointer,
      cameraKey: pixelTraceCameraKey(ui.camera),
      revision: rendererRef.current?.pixelTraceRevision ?? 0,
    });
    tracePinRequestRef.current = request ?? null;
    // Releasing needs no handshake: there is nothing to wait for, and the frozen
    // work should give way to the pointer on the very next frame.
    if (!request && ui.pixelTracePinned) ui.setPixelTracePinned(false);
  };
  const pointerRef = useRef<
    // `x`/`y` track the last move; `downX`/`downY` stay at the press origin.
    // The distance between them is what separates a background click, which
    // deselects, from a camera drag, which keeps the selection.
    | { id: number; x: number; y: number; downX: number; downY: number; action: "orbit" | "pan" }
    // `released` records a pointerup that arrived while the GPU pick readback
    // was still in flight, so a fast click still resolves instead of being
    // dropped along with the gesture.
    | { id: number; x: number; y: number; downX: number; downY: number; action: "pick"; released?: boolean }
    | { id: number; action: "body"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3; lastTime: number }
    // Editor gizmo drags preview on runtime state only and commit on release.
    | { id: number; action: "gizmo-axis"; bodyId: string; axis: GizmoAxis; axisOrigin: Vec3; grabOffset: Vec3; lastPosition: Vec3 }
    | { id: number; action: "gizmo-free"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3 }
    | { id: number; action: "terrain-handle"; index: number; kind: TerrainHandleKind; anchor: Vec3 }
    | { id: number; action: "fluid-paint"; erase: boolean; lastBrickKey?: string }
    | { id: number; action: "fill-level" }
    | { id: number; action: "inflow-handle"; kind: InflowHandleKind; anchor: Vec3 }
    | { id: number; action: "slice"; axis: "x" | "y" | "z"; grabY: number; startClientY: number; startSlice: number }
    | null
  >(null);
  const selectedBody = selection?.kind === "body"
    ? bodies.find((body) => body.description.id === selection.id)
    : undefined;
  const gizmo = selectedBody && activeTool === "select"
    ? projectGizmo(selectedBody.position_m, camera, viewportSize.width, viewportSize.height)
    : undefined;
  const fluidToolArmed = activeTool === "fluid-paint" || activeTool === "fluid-erase";
  const fillHandle = fluidToolArmed && scene.fluid.initialCondition === "tank-fill"
    ? projectToViewport(fillLevelHandlePosition(scene), camera, viewportSize.width, viewportSize.height)
    : undefined;
  const inflowGizmo = scene.fluid.inflow && selection?.kind === "inflow" && (activeTool === "select" || activeTool === "inflow")
    ? inflowHandles(scene.fluid.inflow).map((handle) => ({
      ...handle,
      projection: projectToViewport(handle.position_m, camera, viewportSize.width, viewportSize.height),
    }))
    : undefined;
  const selectedTerrainFeature = selection?.kind === "terrain-feature" && activeTool === "select"
    ? terrainFeatureIndex(selection.id, scene.terrain)
    : undefined;
  const terrainHandles = selectedTerrainFeature !== undefined && scene.terrain
    ? terrainFeatureHandles(scene.terrain, selectedTerrainFeature).map((handle) => ({
      ...handle,
      projection: projectToViewport(handle.position_m, camera, viewportSize.width, viewportSize.height),
    }))
    : undefined;
  const hoverProjection = hover ? projectToViewport(hover.position_m, camera, viewportSize.width, viewportSize.height) : undefined;
  // Where the traced ray currently appears on screen. Projecting a point on the
  // ray rather than reusing the pointer is what keeps the marker on a pinned ray
  // while the camera orbits away from the pixel that produced it.
  const traceReticle = pixelTraceEnabled && pixelTrace
    ? projectToViewport(
      {
        x: pixelTrace.ray.origin_m[0] + pixelTrace.ray.direction[0] * (pixelTrace.hit?.distance_m ?? 1),
        y: pixelTrace.ray.origin_m[1] + pixelTrace.ray.direction[1] * (pixelTrace.hit?.distance_m ?? 1),
        z: pixelTrace.ray.origin_m[2] + pixelTrace.ray.direction[2] * (pixelTrace.hit?.distance_m ?? 1),
      },
      camera, viewportSize.width, viewportSize.height,
    )
    : undefined;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const update = () => {
      const bounds = canvas.getBoundingClientRect();
      setViewportSize({ width: Math.max(1, bounds.width), height: Math.max(1, bounds.height) });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  const failure = viewportFailureIndicator(gpuInfo, waterSurfacePresentation, scene);
  const failureProjection = failure?.location_m
    ? projectViewportFailure(failure.location_m, camera, viewportSize.width, viewportSize.height)
    : undefined;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const diagnostics = useDiagnosticsStore.getState();
    const safeBringup = safeBrowserGPUBringupEnabled(window.location.search);
    const canonicalSafeMethodValues = resolvedMethodValues({ methodId: "octree", quality: "balanced", overrides: {} });
    const startupMode = () => resolveGPUStartupMode(window.location.search, {
      presetId: useSceneStore.getState().presetId,
      methodId: useMethodStore.getState().methodId,
    });
    if (startupMode() === "off") {
      diagnostics.set({ gpuStatus: { state: "unavailable", label: "WebGPU disabled by gpu=off (UI-only mode)" } });
      return;
    }
    let running = true;
    let releaseGPULease: (() => void) | undefined;
    const renderer = new FluidLabRenderer(
      canvas,
      (status) => {
        if (status.state === "lost" || status.state === "unavailable") {
          running = false;
          queueMicrotask(() => { if (initializationStarted && !stopping && !stopped) void stopGPU(status.label); });
          return;
        }
        const current = useDiagnosticsStore.getState().gpuStatus;
        // The controller publishes the user's intent before the next render
        // can start expensive work. Preserve that context as detailed task
        // progress arrives from the renderer.
        const rendererOnlyReady = status.state === "ready" && status.label === "WebGPU renderer ready"
          && getMethod(useMethodStore.getState().methodId).backend === "webgpu";
        const reportedStatus = rendererOnlyReady
          ? { state: "initializing" as const, label: "Renderer ready; preparing fenced t=0 solver authority", phase: "warmup", completed: 0, total: 1, startedAt_ms: performance.now(), kind: "startup" as const }
          : status;
        const gpuStatus = reportedStatus.state === "initializing" && current.state === "initializing" && current.operation
          ? { ...reportedStatus, operation: current.operation, kind: reportedStatus.kind ?? current.kind, retainingPrevious: reportedStatus.retainingPrevious ?? current.retainingPrevious }
          : reportedStatus;
        useDiagnosticsStore.getState().set({ gpuStatus });
      },
      (info) => useDiagnosticsStore.getState().set({ gpuInfo: info }),
      undefined,
      (time_s) => simulation.gpuAdvanceCompleted(time_s),
      (effectiveRendererStatus) => useDiagnosticsStore.getState().set({ effectiveRendererStatus })
    );
    let safeSimulationEpoch: number | undefined;
    const syncRunState = (runState: ReturnType<typeof useRuntimeStore.getState>["runState"]) => {
      const submittedTime_s = renderer.setSimulationRunning(runState === "running");
      if (runState === "paused") simulation.gpuSchedulingPaused(submittedTime_s);
    };
    syncRunState(useRuntimeStore.getState().runState);
    const unsubscribeRunState = useRuntimeStore.subscribe((state, previous) => {
      if (state.simulationEpoch !== previous.simulationEpoch) {
        if (safeBrowserSimulationEpochChanged(safeBringup, initializationStarted, safeSimulationEpoch, state.simulationEpoch)) {
          void stopGPU("Safe WebGPU session stopped after a reset/rebuild attempt");
          return;
        }
        renderer.resetSimulationTimeline();
      }
      if (state.runState !== previous.runState) syncRunState(state.runState);
    });
    rendererRef.current = renderer;
    let frame = 0;
    let alive = true;
    const frameRate = new SmoothedFrameRate(5);
    let lastSubmittedAt_ms: number | undefined;
    let measuredRunState = useRuntimeStore.getState().runState;
    const publishFrameRate = (fps: number | undefined) => {
      if (fpsRef.current) fpsRef.current.textContent = fps === undefined ? "— FPS" : `${fps.toFixed(1)} FPS`;
    };
    let initializationStarted = false;
    let stopping = false;
    let stopped = false;
    let leaseAcquisition: ReturnType<typeof acquireBrowserGPULease> | undefined;
    let stopPromise: Promise<void> | undefined;
    const safeViolations = () => {
      const sceneState = useSceneStore.getState(), methodState = useMethodStore.getState(), ui = useUIStore.getState();
      return safeBrowserGPUBringupViolations({
        presetId: sceneState.presetId,
        methodId: methodState.methodId,
        quality: methodState.quality,
        methodValues: resolvedMethodValues(methodState),
        canonicalMethodValues: canonicalSafeMethodValues,
        exactScene: canonicalScene(sceneState.scene) === canonicalScene(getScenePreset("water-box-dam-break").create()),
        voxelRenderMode: ui.voxelRenderMode,
        svoRenderMode: ui.svoRenderMode,
        diagnosticsOpen: ui.diagnosticsOpen,
        rightPanel: ui.rightPanel,
        gridOverlayAxis: ui.gridOverlayAxis,
        activeTool: ui.activeTool,
        search: window.location.search,
      });
    };
    function stopGPU(label = "WebGPU stopped; device released — safe to close this tab", publishStatus = true): Promise<void> {
      if (stopPromise) return stopPromise;
      stopping = true;
      running = false;
      useRuntimeStore.getState().setRunState("paused");
      cancelAnimationFrame(frame);
      if (publishStatus) diagnostics.set({ gpuStatus: { state: "stopping", label: "Stopping WebGPU; waiting for initialization and solver tasks to drain" } });
      const pendingLease = leaseAcquisition;
      const releasedLabel = label.includes("device released") ? label : `${label}; device released — safe to close this tab`;
      const reproduction = dawnReproductionForGPUFailure(label);
      stopPromise = (async () => {
        await shutdownBrowserGPUSession(renderer, pendingLease, releaseGPULease);
        releaseGPULease = undefined;
        stopping = false;
        stopped = true;
        if (publishStatus) diagnostics.set({ gpuStatus: { state: "unavailable", label: releasedLabel, reproduction } });
      })();
      return stopPromise;
    }
    const beginInitialization = async () => {
      if (initializationStarted || !alive || stopping || stopped) return;
      if (safeBringup) {
        const violations = safeViolations();
        if (violations.length > 0) {
          diagnostics.set({ gpuStatus: { state: "manual", label: `Safe WebGPU start refused: ${violations.join("; ")}` } });
          return;
        }
        useRuntimeStore.getState().setRunState("paused");
        safeSimulationEpoch = useRuntimeStore.getState().simulationEpoch;
      }
      initializationStarted = true;
      window.removeEventListener(GPU_MANUAL_START_EVENT, beginInitialization);
      unsubscribeAutomaticStart();
      diagnostics.set({ gpuStatus: { state: "initializing", label: "Acquiring exclusive browser WebGPU lease", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup" } });
      const lockManager = "locks" in navigator
        ? navigator.locks as Parameters<typeof acquireBrowserGPULease>[0]
        : undefined;
      const acquisition = acquireBrowserGPULease(lockManager);
      leaseAcquisition = acquisition;
      const lease = await acquisition;
      if (leaseAcquisition === acquisition) leaseAcquisition = undefined;
      if (!alive || stopping || stopped) { if (lease.status === "acquired") lease.release(); return; }
      if (lease.status !== "acquired") {
        initializationStarted = false;
        if (safeBringup || lease.status !== "unsupported") {
          diagnostics.set({ gpuStatus: { state: "manual", label: `WebGPU start refused: ${lease.message}` } });
          window.addEventListener(GPU_MANUAL_START_EVENT, beginInitialization);
          return;
        }
      } else releaseGPULease = lease.release;
      diagnostics.set({ gpuStatus: { state: "initializing", label: "Initializing WebGPU", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup" } });
      void renderer.initialize().then(async () => {
      if (!alive || stopping || stopped) return;
      const status = useDiagnosticsStore.getState().gpuStatus;
      if (status.state === "lost" || status.state === "unavailable") {
        await stopGPU(status.label);
        return;
      }
      const render = () => {
        if (!alive || !running) return;
        frame = requestAnimationFrame(render);
        const sceneState = useSceneStore.getState();
        const scene = sceneState.scene;
        const ui = useUIStore.getState();
        const method = useMethodStore.getState();
        const state = useDiagnosticsStore.getState();
        const runtime = useRuntimeStore.getState();
        if (runtime.runState !== measuredRunState) {
          measuredRunState = runtime.runState;
          frameRate.reset();
          lastSubmittedAt_ms = undefined;
          publishFrameRate(undefined);
        } else if (lastSubmittedAt_ms !== undefined && performance.now() - lastSubmittedAt_ms >= 1_000) {
          publishFrameRate(0);
        }
        // Pausing freezes simulation time, not presentation. Attempt every
        // browser animation frame; FluidLabRenderer's in-flight guard is the
        // only backpressure on rendering.
        let metrics;
        try {
          metrics = renderer.draw(
            simulation.time(), scene, ui.camera, state.bodies, ui.selectedBodyId,
            state.fluidRenderState ?? undefined, simulation.backend,
            { methodId: method.methodId, quality: method.quality, values: resolvedMethodValues(method), simulationEpoch: runtime.simulationEpoch },
            { axis: ui.gridOverlayAxis, position: ui.gridOverlaySlice, mode: ui.gridOverlayMode },
            getScenePreset(sceneState.presetId).background,
            ui.voxelRenderMode,
            ui.svoRenderMode,
            ui.svoLightingMode,
            {
              shadowsEnabled: ui.svoShadowsEnabled,
              ambientOcclusionEnabled: ui.svoAmbientOcclusionEnabled,
            },
            {
              overlay: ui.svoCostOverlay,
              maximumTraversalDepth: ui.svoMaximumTraversalDepth,
              maximumNodeVisits: ui.svoMaximumNodeVisits,
            },
            ui.svoRenderTuning,
            pixelTraceDrawConfig(ui, renderer),
          );
        } catch (error: unknown) {
          void stopGPU(error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped");
          return;
        }
        publishPixelTrace(renderer);
        simulation.recordFrame(metrics, renderer.presentationResolution);
        if (metrics.presentationSubmitted) {
          const submittedAt_ms = performance.now();
          lastSubmittedAt_ms = submittedAt_ms;
          publishFrameRate(frameRate.sample(submittedAt_ms));
          simulationRecording.capturePresentedState(canvas, runtime.simulationTime);
        }
      };
      frame = requestAnimationFrame(render);
      }).catch((error: unknown) => {
      if (!stopping && !stopped) void stopGPU(error instanceof Error ? error.message : "WebGPU initialization failed");
      });
    };
    const maybeStartAutomatically = () => {
      if (startupMode() === "automatic") beginInitialization();
    };
    const unsubscribeScene = useSceneStore.subscribe(maybeStartAutomatically);
    const unsubscribeMethod = useMethodStore.subscribe(maybeStartAutomatically);
    const unsubscribeAutomaticStart = () => { unsubscribeScene(); unsubscribeMethod(); };
    const enforceSafeConfiguration = () => {
      if (!safeBringup || !initializationStarted || stopped) return;
      const violations = safeViolations();
      if (violations.length > 0) stopGPU(`Safe WebGPU session stopped after configuration drift: ${violations.join("; ")}`);
    };
    const unsubscribeSafeScene = useSceneStore.subscribe(enforceSafeConfiguration);
    const unsubscribeSafeMethod = useMethodStore.subscribe(enforceSafeConfiguration);
    const unsubscribeSafeUI = useUIStore.subscribe(enforceSafeConfiguration);
    const manualStop = () => { void stopGPU(); };
    window.addEventListener(GPU_MANUAL_STOP_EVENT, manualStop);
    const pageHide = () => { void stopGPU("WebGPU stopped during page close", false); };
    window.addEventListener("pagehide", pageHide, { once: true });
    if (startupMode() === "manual" || startupMode() === "safe") {
      diagnostics.set({ gpuStatus: { state: "manual", label: "WebGPU is waiting for explicit startup" } });
      window.addEventListener(GPU_MANUAL_START_EVENT, beginInitialization);
    } else beginInitialization();
    return () => { alive = false; running = false; window.removeEventListener(GPU_MANUAL_START_EVENT, beginInitialization); window.removeEventListener(GPU_MANUAL_STOP_EVENT, manualStop); window.removeEventListener("pagehide", pageHide); unsubscribeAutomaticStart(); unsubscribeSafeScene(); unsubscribeSafeMethod(); unsubscribeSafeUI(); unsubscribeRunState(); cancelAnimationFrame(frame); if(rendererRef.current===renderer)rendererRef.current=null; void stopGPU("WebGPU stopped during component cleanup", false); };
  }, []);

  const pointerRay = (event: React.PointerEvent<HTMLCanvasElement>) =>
    viewportRayForPointer(useUIStore.getState().camera, event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());
  const planeHit = (origin: Vec3, direction: Vec3, point: Vec3, normal: Vec3) => {
    const denominator = dot(direction, normal); if (Math.abs(denominator) < 1e-6) return point;
    return add(origin, scale(direction, dot(sub(point, origin), normal) / denominator));
  };

  // Hit test for the slice gripper: vertical planes use their top edge, while
  // the horizontal Y plane uses its perimeter. Grabbing either sweeps the
  // slice through the volume.
  const sliceGrabHit = (origin: Vec3, direction: Vec3) => {
    const ui = useUIStore.getState();
    if (ui.gridOverlayAxis === "off" || ui.gridOverlayAxis === "volume") return undefined;
    const axis = ui.gridOverlayAxis;
    const c = useSceneStore.getState().scene.container;
    const planeCoordinate = axis === "z" ? -c.depth_m / 2 + ui.gridOverlaySlice * c.depth_m : axis === "x" ? -c.width_m / 2 + ui.gridOverlaySlice * c.width_m : ui.gridOverlaySlice * c.height_m;
    const denominator = axis === "z" ? direction.z : axis === "x" ? direction.x : direction.y;
    if (Math.abs(denominator) < 1e-5) return undefined;
    const rayOrigin = axis === "z" ? origin.z : axis === "x" ? origin.x : origin.y;
    const t = (planeCoordinate - rayOrigin) / denominator;
    if (t <= 0) return undefined;
    const point = add(origin, scale(direction, t));
    const inFootprint = Math.abs(point.x) <= c.width_m / 2 && Math.abs(point.z) <= c.depth_m / 2;
    const nearTop = point.y >= c.height_m * 0.94 && point.y <= c.height_m * 1.02;
    const horizontalEdgeDistance = Math.min(point.x + c.width_m / 2, c.width_m / 2 - point.x, point.z + c.depth_m / 2, c.depth_m / 2 - point.z);
    const nearHorizontalEdge = horizontalEdgeDistance >= 0 && horizontalEdgeDistance <= 0.035 * Math.min(c.width_m, c.depth_m);
    return inFootprint && (axis === "y" ? nearHorizontalEdge : nearTop) ? { axis, grabY: Math.min(point.y, c.height_m) } : undefined;
  };

  const beginBodyDrag = (pointerId: number, timeStamp: number, ray: { origin: Vec3; direction: Vec3 }, body: RigidBodyState, position: Vec3, orientation?: RigidBodyState["orientation"], surfacePosition = position) => {
    const basis = cameraBasis(useUIStore.getState().camera);
    const dragPoint = planeHit(ray.origin, ray.direction, surfacePosition, basis.forward), grabOffset = sub(position, dragPoint);
    pointerRef.current = { id: pointerId, action: "body", bodyId: body.description.id, planePoint: surfacePosition, planeNormal: basis.forward, grabOffset, lastPosition: position, lastTime: timeStamp };
    useUIStore.getState().selectBody(body.description.id);
    simulation.dragBody(body.description.id, position, { x: 0, y: 0, z: 0 }, "start", orientation);
  };

  /**
   * Gizmo handles own the click before anything else so that dragging an axis
   * next to the body cannot be misread as a throw. Returns true when the
   * gesture was claimed.
   */
  const beginGizmoDrag = (event: React.PointerEvent<HTMLCanvasElement>, ray: { origin: Vec3; direction: Vec3 }) => {
    const ui = useUIStore.getState();
    if (ui.activeTool !== "select" || ui.selection?.kind !== "body") return false;
    const body = useDiagnosticsStore.getState().bodies.find((candidate) => candidate.description.id === ui.selection?.id);
    if (!body) return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const projected = projectGizmo(body.position_m, ui.camera, rect.width, rect.height);
    const handle = gizmoHandleAtPointer(projected, { x: event.clientX - rect.left, y: event.clientY - rect.top }, rect.width, rect.height);
    if (!handle) return false;
    if (handle === "free") {
      const basis = cameraBasis(ui.camera);
      const dragPoint = planeHit(ray.origin, ray.direction, body.position_m, basis.forward);
      pointerRef.current = {
        id: event.pointerId, action: "gizmo-free", bodyId: body.description.id,
        planePoint: body.position_m, planeNormal: basis.forward,
        grabOffset: sub(body.position_m, dragPoint), lastPosition: body.position_m,
      };
    } else {
      const grabPoint = closestPointOnAxis(ray.origin, ray.direction, body.position_m, GIZMO_AXIS_DIRECTIONS[handle]);
      if (!grabPoint) return false;
      pointerRef.current = {
        id: event.pointerId, action: "gizmo-axis", bodyId: body.description.id, axis: handle,
        axisOrigin: body.position_m, grabOffset: sub(body.position_m, grabPoint), lastPosition: body.position_m,
      };
    }
    simulation.beginEdit(`Moved ${body.description.name}`);
    simulation.manipulateBody(body.description.id, body.position_m, "start", body.orientation);
    return true;
  };

  const TERRAIN_HANDLE_TOLERANCE_PX = 12;
  const FILL_HANDLE_TOLERANCE_PX = 12;
  const INFLOW_HANDLE_TOLERANCE_PX = 12;

  /** Grab the hose body or its aim arrow. */
  const beginInflowHandleDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = useUIStore.getState();
    const inflow = useSceneStore.getState().scene.fluid.inflow;
    if (!inflow || ui.selection?.kind !== "inflow") return false;
    if (ui.activeTool !== "select" && ui.activeTool !== "inflow") return false;
    const rect = event.currentTarget.getBoundingClientRect();
    let best: { kind: InflowHandleKind; anchor: Vec3; distance_px: number } | undefined;
    for (const handle of inflowHandles(inflow)) {
      const projection = projectToViewport(handle.position_m, ui.camera, rect.width, rect.height);
      if (!(projection.depth_m > 1e-6)) continue;
      const distance_px = Math.hypot(
        event.clientX - rect.left - projection.leftFraction * rect.width,
        event.clientY - rect.top - projection.topFraction * rect.height,
      );
      if (distance_px <= INFLOW_HANDLE_TOLERANCE_PX && (!best || distance_px < best.distance_px)) {
        best = { kind: handle.kind, anchor: handle.position_m, distance_px };
      }
    }
    if (!best) return false;
    pointerRef.current = { id: event.pointerId, action: "inflow-handle", kind: best.kind, anchor: best.anchor };
    simulation.beginEdit(best.kind === "center" ? "Moved the hose" : "Aimed the hose");
    return true;
  };

  /** Place (or re-place) the single nozzle on the surface under the cursor. */
  const placeInflowAt = (ray: { origin: Vec3; direction: Vec3 }) => {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    const hit = hoverSceneAt(scene, useDiagnosticsStore.getState().bodies, ray);
    if (!hit) return;
    simulation.beginEdit(scene.fluid.inflow ? "Moved the hose" : "Placed a hose");
    sceneStore.patchFluid({ inflow: createInflowAt(hit.position_m, hit.normal, scene) });
    useUIStore.getState().select({ kind: "inflow", id: INFLOW_SELECTION_ID });
    simulation.commitEdit(undefined, { reseed: true });
  };

  /** The tank-fill surface rides a corner post, clear of painting targets. */
  const beginFillLevelDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = useUIStore.getState();
    const scene = useSceneStore.getState().scene;
    if (ui.activeTool !== "fluid-paint" && ui.activeTool !== "fluid-erase") return false;
    if (scene.fluid.initialCondition !== "tank-fill") return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const projection = projectToViewport(fillLevelHandlePosition(scene), ui.camera, rect.width, rect.height);
    if (!(projection.depth_m > 1e-6)) return false;
    const distance_px = Math.hypot(
      event.clientX - rect.left - projection.leftFraction * rect.width,
      event.clientY - rect.top - projection.topFraction * rect.height,
    );
    if (distance_px > FILL_HANDLE_TOLERANCE_PX) return false;
    pointerRef.current = { id: event.pointerId, action: "fill-level" };
    simulation.beginEdit("Set fill level");
    return true;
  };

  /** Terrain feature handles are grabbed in screen space, like the body gizmo. */
  const beginTerrainHandleDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = useUIStore.getState();
    const terrain = useSceneStore.getState().scene.terrain;
    if (ui.activeTool !== "select" || ui.selection?.kind !== "terrain-feature" || !terrain) return false;
    const index = terrainFeatureIndex(ui.selection.id, terrain);
    if (index === undefined) return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    let best: { kind: TerrainHandleKind; anchor: Vec3; distance_px: number } | undefined;
    for (const handle of terrainFeatureHandles(terrain, index)) {
      const projection = projectToViewport(handle.position_m, ui.camera, rect.width, rect.height);
      if (!(projection.depth_m > 1e-6)) continue;
      const distance_px = Math.hypot(pointer.x - projection.leftFraction * rect.width, pointer.y - projection.topFraction * rect.height);
      if (distance_px <= TERRAIN_HANDLE_TOLERANCE_PX && (!best || distance_px < best.distance_px)) {
        best = { kind: handle.kind, anchor: handle.position_m, distance_px };
      }
    }
    if (!best) return false;
    pointerRef.current = { id: event.pointerId, action: "terrain-handle", index, kind: best.kind, anchor: best.anchor };
    simulation.beginEdit(`Shaped terrain ${terrain.features[index]?.kind ?? "feature"}`);
    return true;
  };

  /**
   * Constrain the pointer to the handle's own surface: the ground plane for
   * centre and radius, the vertical axis for amount.
   */
  const terrainHandlePoint = (
    active: { kind: TerrainHandleKind; anchor: Vec3 },
    ray: { origin: Vec3; direction: Vec3 },
  ) => active.kind === "amount"
    ? closestPointOnAxis(ray.origin, ray.direction, active.anchor, GIZMO_AXIS_DIRECTIONS.y)
    : planeHit(ray.origin, ray.direction, active.anchor, { x: 0, y: 1, z: 0 });

  /**
   * Paint or erase the brick under the pointer. Returns the brick key so a
   * drag only touches the document when it crosses into a new brick.
   */
  const paintFluidAt = (ray: { origin: Vec3; direction: Vec3 }, erase: boolean, lastBrickKey?: string) => {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    const hit = hoverSceneAt(scene, useDiagnosticsStore.getState().bodies, ray);
    // Paint onto whatever surface is under the cursor; with nothing there,
    // fall back to the fill-level plane so open air is still paintable.
    const point = hit?.position_m
      ?? planeHit(ray.origin, ray.direction, { x: 0, y: fillLevelHandlePosition(scene).y, z: 0 }, { x: 0, y: 1, z: 0 });
    const lattice = editorFluidLattice(scene);
    const index = fluidBrickIndexAt(lattice, point);
    if (!index) return lastBrickKey;
    const key = fluidBrickKey(index);
    if (key === lastBrickKey) return key;
    // A surface hit sits on the boundary of the brick behind it; nudging into
    // the brick centre keeps a stroke on the surface from seeding solids.
    const target = erase ? point : fluidBrickCenter(lattice, index);
    const result = erase ? eraseFluidBrick(scene, target) : paintFluidBrick(scene, target);
    if (result) sceneStore.patchScene({ fluid: fluidPaintPatch(scene, result) });
    return key;
  };

  /** Drop the armed placement shape onto whatever the cursor is over. */
  const placeBodyAt = (ray: { origin: Vec3; direction: Vec3 }) => {
    const ui = useUIStore.getState();
    const scene = useSceneStore.getState().scene;
    const shape = ui.placementShape;
    const template = createBodyDescription(shape, 1, scene.container.height_m);
    const radius_m = boundingRadius(template);
    const target = hoverSceneAt(scene, useDiagnosticsStore.getState().bodies, ray);
    // Without a surface under the cursor, fall back to the camera-facing plane
    // through the container centre — the same target the tray drop uses.
    const position = target
      ? restOnHover(target, radius_m, scene)
      : planeHit(ray.origin, ray.direction, { x: 0, y: scene.container.height_m / 2, z: 0 }, cameraBasis(ui.camera).forward);
    simulation.addBodyAt(shape, position, { autoRun: false });
  };

  const pointerDown = async (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    // Arm before any of the early returns below claim the press: the release, not
    // the press, is what decides whether this was a click.
    const traceUI = useUIStore.getState();
    tracePinGestureRef.current = traceUI.pixelTraceEnabled && traceUI.activeTool === "select"
      && event.button === 0 && !event.shiftKey
      ? { id: event.pointerId, downX: event.clientX, downY: event.clientY }
      : null;
    // pointerRef is a ref, so clearing hover here is what actually re-renders
    // the chip away for the duration of the gesture.
    setHover(null);
    if (event.button === 0 && !event.shiftKey) {
      const ray = pointerRay(event);
      if (beginTerrainHandleDrag(event)) return;
      if (beginInflowHandleDrag(event)) return;
      if (beginGizmoDrag(event, ray)) return;
      if (useUIStore.getState().activeTool === "inflow") { placeInflowAt(ray); return; }
      // Armed tools claim the click before the slice/pick/orbit fallback so a
      // placement never orbits the camera instead.
      if (useUIStore.getState().activeTool === "body-place") { placeBodyAt(ray); return; }
      if (useUIStore.getState().activeTool === "prop-place") {
        const target = hoverSceneAt(useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray);
        if (target) simulation.addProp(useUIStore.getState().propShape, target.position_m, target.normal);
        return;
      }
      if (beginFillLevelDrag(event)) return;
      const paintTool = useUIStore.getState().activeTool;
      if (paintTool === "fluid-paint" || paintTool === "fluid-erase") {
        const erase = paintTool === "fluid-erase";
        simulation.beginEdit(erase ? "Erased water" : "Painted water");
        pointerRef.current = { id: event.pointerId, action: "fluid-paint", erase, lastBrickKey: paintFluidAt(ray, erase) };
        return;
      }
      // Clicking the ground selects the terrain feature under the cursor, so
      // basins and mounds are addressable without a roster.
      if (useUIStore.getState().activeTool === "select") {
        const ground = hoverSceneAt(useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray);
        if (ground?.kind === "terrain") {
          const feature = terrainFeatureAt(useSceneStore.getState().scene.terrain, ground.position_m.x, ground.position_m.z);
          if (feature !== undefined) {
            useUIStore.getState().select({ kind: "terrain-feature", id: terrainFeatureSelectionId(feature) });
            return;
          }
        }
      }
      const grab = sliceGrabHit(ray.origin, ray.direction);
      if (grab) { pointerRef.current = { id: event.pointerId, action: "slice", ...grab, startClientY: event.clientY, startSlice: useUIStore.getState().gridOverlaySlice }; return; }
      if (simulation.backend === "webgpu" && rendererRef.current) {
        const pointerId=event.pointerId,timeStamp=event.timeStamp,x=event.clientX,y=event.clientY;
        pointerRef.current={id:pointerId,x,y,downX:x,downY:y,action:"pick"};
        const rect=event.currentTarget.getBoundingClientRect();
        const picked=await rendererRef.current.pickRigidBody(ray.origin,ray.direction,{
          normalizedX:(event.clientX-rect.left)/Math.max(rect.width,1),
          normalizedY:(event.clientY-rect.top)/Math.max(rect.height,1),
        });
        const active=pointerRef.current;
        if(!active||active.id!==pointerId||active.action!=="pick")return;
        const body=picked?useDiagnosticsStore.getState().bodies[picked.bodyIndex]:undefined;
        if(body&&picked){
          // A pointer already released cannot be dragged, so a fast click on a
          // body selects it without opening a throw that never ends.
          if(active.released){pointerRef.current=null;useUIStore.getState().selectBody(body.description.id);return;}
          beginBodyDrag(pointerId,timeStamp,ray,body,picked.position_m,picked.orientation,"surfacePosition_m" in picked?picked.surfacePosition_m:picked.position_m);return;
        }
        // Nothing under the cursor: a released pointer was a background click,
        // a held one becomes the orbit fallback.
        if(active.released){pointerRef.current=null;useUIStore.getState().select(undefined);return;}
        pointerRef.current={...active,action:"orbit"};
        return;
      }
      let nearest: { body: RigidBodyState; t: number } | undefined;
      for (const body of useDiagnosticsStore.getState().bodies) {
        const oc = sub(ray.origin, body.position_m), radius = boundingRadius(body), b = dot(oc, ray.direction), c = dot(oc, oc) - radius * radius, discriminant = b * b - c;
        if (discriminant < 0) continue; const t = -b - Math.sqrt(discriminant);
        if (t > 0 && (!nearest || t < nearest.t)) nearest = { body, t };
      }
      if (nearest) {
        beginBodyDrag(event.pointerId,event.timeStamp,ray,nearest.body,nearest.body.position_m,nearest.body.orientation);
        return;
      }
    }
    pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, downX: event.clientX, downY: event.clientY, action: event.shiftKey || event.button === 1 ? "pan" : "orbit" };
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const active = pointerRef.current;
    // The pixel-trace probe follows the pointer whatever else the gesture is
    // doing: a ref, so tracking it costs no React render per move.
    const rect = event.currentTarget.getBoundingClientRect();
    tracePointerRef.current = {
      normalizedX: (event.clientX - rect.left) / Math.max(rect.width, 1),
      normalizedY: (event.clientY - rect.top) / Math.max(rect.height, 1),
    };
    if (!active) {
      // Analytic hover: no GPU readback, so it is safe at pointer-move rate.
      const ray = pointerRay(event);
      setHover(hoverSceneAt(useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray) ?? null);
      return;
    }
    if (active.id !== event.pointerId) return;
    if (active.action === "pick") return;
    if (active.action === "inflow-handle") {
      const ray = pointerRay(event);
      const sceneStore = useSceneStore.getState();
      const inflow = sceneStore.scene.fluid.inflow;
      if (!inflow) return;
      // Both handles drag in the camera plane through their own anchor, so a
      // hose can be aimed in any direction rather than only along an axis.
      const point = planeHit(ray.origin, ray.direction, active.anchor, cameraBasis(useUIStore.getState().camera).forward);
      sceneStore.patchFluid({ inflow: active.kind === "center"
        ? moveInflow(inflow, point, sceneStore.scene.container)
        : aimInflow(inflow, point) });
      return;
    }
    if (active.action === "fluid-paint") {
      pointerRef.current = { ...active, lastBrickKey: paintFluidAt(pointerRay(event), active.erase, active.lastBrickKey) };
      return;
    }
    if (active.action === "fill-level") {
      const ray = pointerRay(event);
      const sceneStore = useSceneStore.getState();
      const corner = fillLevelHandlePosition(sceneStore.scene);
      const point = closestPointOnAxis(ray.origin, ray.direction, corner, GIZMO_AXIS_DIRECTIONS.y);
      if (!point) return;
      sceneStore.patchContainer({ fillFraction: fillFractionForHeight(sceneStore.scene, point.y) });
      return;
    }
    if (active.action === "terrain-handle") {
      const point = terrainHandlePoint(active, pointerRay(event));
      if (!point) return;
      const sceneStore = useSceneStore.getState();
      const terrain = sceneStore.scene.terrain;
      if (!terrain) return;
      // Terrain is absent from gpuSceneSolverKey, so patching the document
      // mid-drag repaints the ground without churning the solver. The commit
      // is what re-seeds it.
      sceneStore.patchScene({ terrain: applyTerrainFeatureDrag(terrain, active.index, active.kind, point, sceneStore.scene.container) });
      return;
    }
    if (active.action === "gizmo-axis" || active.action === "gizmo-free") {
      const ray = pointerRay(event);
      const position = active.action === "gizmo-axis"
        ? gizmoAxisDragPosition(ray.origin, ray.direction, active.axis, active.axisOrigin, active.grabOffset)
        : add(planeHit(ray.origin, ray.direction, active.planePoint, active.planeNormal), active.grabOffset);
      // A ray nearly parallel to the constrained axis has no stable solution;
      // holding the last pose beats letting the body shoot to infinity.
      if (!position) return;
      pointerRef.current = { ...active, lastPosition: position };
      simulation.manipulateBody(active.bodyId, position, "move");
      return;
    }
    if (active.action === "slice") {
      if (active.axis === "y") {
        const rect = event.currentTarget.getBoundingClientRect();
        useUIStore.getState().setGridOverlaySlice(active.startSlice + (active.startClientY - event.clientY) / Math.max(rect.height, 1));
        return;
      }
      // Keep the grab height fixed and slide the plane along its normal.
      const ray = pointerRay(event);
      if (Math.abs(ray.direction.y) < 1e-4) return;
      const t = (active.grabY - ray.origin.y) / ray.direction.y;
      if (t <= 0) return;
      const point = add(ray.origin, scale(ray.direction, t));
      const c = useSceneStore.getState().scene.container;
      const fraction = active.axis === "z" ? (point.z + c.depth_m / 2) / c.depth_m : (point.x + c.width_m / 2) / c.width_m;
      useUIStore.getState().setGridOverlaySlice(fraction);
      return;
    }
    if (active.action === "body") {
      const ray = pointerRay(event), position = add(planeHit(ray.origin, ray.direction, active.planePoint, active.planeNormal), active.grabOffset);
      const dt = Math.max((event.timeStamp - active.lastTime) / 1000, 1 / 240), rawVelocity = scale(sub(position, active.lastPosition), 1 / dt), speed = length(rawVelocity), velocity = speed > 6 ? scale(rawVelocity, 6 / speed) : rawVelocity;
      pointerRef.current = { ...active, lastPosition: position, lastTime: event.timeStamp };
      simulation.dragBody(active.bodyId, position, velocity, "move"); return;
    }
    const dx = event.clientX - active.x;
    const dy = event.clientY - active.y;
    pointerRef.current = { ...active, x: event.clientX, y: event.clientY };
    setCamera((current) => active.action === "pan" ? pan(current, dx, dy) : orbit(current, dx, dy));
  };
  const pointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    // Ahead of the pointer-id guard below: a pin gesture is tracked separately,
    // so it must resolve even on the releases the pointer machine ignores.
    resolvePixelTracePinGesture(event);
    const active = pointerRef.current;
    if (active?.id !== event.pointerId) return;
    // An interrupted gesture is not a click, so it must not move the selection.
    const cancelled = event.type === "pointercancel";
    // A pick whose readback has not landed yet outlives the gesture: clearing
    // it here would silently swallow every click faster than one GPU frame.
    if (active.action === "pick") {
      const travelled = Math.hypot(event.clientX - active.downX, event.clientY - active.downY);
      if (!cancelled && travelled <= CLICK_SLOP_PX) {
        pointerRef.current = { ...active, released: true };
        return;
      }
    }
    pointerRef.current = null;
    if (active.action === "body") { simulation.dragBody(active.bodyId, active.lastPosition, { x: 0, y: 0, z: 0 }, "end"); return; }
    // Terrain reaches the solver only through a re-seed.
    if (active.action === "terrain-handle") { simulation.commitEdit(undefined, { reseed: true }); return; }
    // Brick seeds and fill fraction are already in the solver key, so the
    // document write alone invalidates it; the reseed makes the edit start
    // from a defined t=0 instead of mid-flight.
    if (active.action === "fluid-paint" || active.action === "fill-level" || active.action === "inflow-handle") {
      simulation.commitEdit(undefined, { reseed: true });
      return;
    }
    if (active.action === "gizmo-axis" || active.action === "gizmo-free") {
      // Commit-on-release: one document write and one undo entry per gesture.
      simulation.manipulateBody(active.bodyId, active.lastPosition, "end");
      simulation.updateBody(active.bodyId, { position_m: active.lastPosition, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
      simulation.commitEdit();
      return;
    }
    // Nothing claimed the press, so a click that never became a drag is the
    // user clicking through to the background. Deselect, whatever was selected.
    if (!cancelled && (active.action === "orbit" || active.action === "pan")
      && emptySpaceClickDeselects(active.action, event.clientX - active.downX, event.clientY - active.downY)) {
      useUIStore.getState().select(undefined);
    }
  };

  return <>
    <canvas
      ref={canvasRef}
      className="gpu-canvas"
      aria-label="Interactive three-dimensional fluid laboratory viewport"
      data-testid="gpu-viewport"
      data-camera-azimuth={camera.azimuth_rad.toFixed(6)}
      data-camera-elevation={camera.elevation_rad.toFixed(6)}
      data-pixel-trace={pixelTraceEnabled && activeTool === "select" && !pixelTracePinned ? "live" : undefined}
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={pointerUp}
      onPointerLeave={() => setHover(null)}
      onWheel={(event) => { event.preventDefault(); setCamera((current) => zoom(current, event.deltaY)); }}
      onContextMenu={(event) => event.preventDefault()}
    />
    <output
      ref={fpsRef}
      className="fps-meter"
      data-testid="fps-meter"
      aria-label="Presentation frame rate"
      title="WebGPU presentations per second · rolling mean of the latest 5 frame intervals"
    >— FPS</output>
    {gizmo && gizmo.origin.depth_m > 1e-6 && <svg
      className="editor-gizmo"
      data-testid="editor-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      {gizmo.handles.filter((handle) => handle.tip.depth_m > 1e-6).map((handle) => (
        <g key={handle.axis} className={`gizmo-axis axis-${handle.axis}`}>
          <line
            x1={gizmo.origin.leftFraction * viewportSize.width}
            y1={gizmo.origin.topFraction * viewportSize.height}
            x2={handle.tip.leftFraction * viewportSize.width}
            y2={handle.tip.topFraction * viewportSize.height}
          />
          <circle cx={handle.tip.leftFraction * viewportSize.width} cy={handle.tip.topFraction * viewportSize.height} r={4} />
        </g>
      ))}
      <circle
        className="gizmo-center"
        cx={gizmo.origin.leftFraction * viewportSize.width}
        cy={gizmo.origin.topFraction * viewportSize.height}
        r={5}
      />
    </svg>}
    {fillHandle?.visible && <div
      className="editor-fill-handle"
      data-testid="editor-fill-handle"
      style={{ left: `${fillHandle.leftFraction * 100}%`, top: `${fillHandle.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>FILL {(scene.container.fillFraction * 100).toFixed(0)}%</span>
    </div>}
    {inflowGizmo && inflowGizmo.every((handle) => handle.projection.depth_m > 1e-6) && <svg
      className="editor-gizmo editor-inflow-gizmo"
      data-testid="editor-inflow-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      <line
        x1={inflowGizmo[0]!.projection.leftFraction * viewportSize.width}
        y1={inflowGizmo[0]!.projection.topFraction * viewportSize.height}
        x2={inflowGizmo[1]!.projection.leftFraction * viewportSize.width}
        y2={inflowGizmo[1]!.projection.topFraction * viewportSize.height}
      />
      <circle className="inflow-center" cx={inflowGizmo[0]!.projection.leftFraction * viewportSize.width} cy={inflowGizmo[0]!.projection.topFraction * viewportSize.height} r={6} />
      <circle className="inflow-nozzle" cx={inflowGizmo[1]!.projection.leftFraction * viewportSize.width} cy={inflowGizmo[1]!.projection.topFraction * viewportSize.height} r={5} />
    </svg>}
    {terrainHandles && <svg
      className="editor-gizmo editor-terrain-gizmo"
      data-testid="editor-terrain-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      {terrainHandles.filter((handle) => handle.projection.depth_m > 1e-6).map((handle) => (
        <circle
          key={handle.kind}
          className={`terrain-handle handle-${handle.kind}`}
          cx={handle.projection.leftFraction * viewportSize.width}
          cy={handle.projection.topFraction * viewportSize.height}
          r={handle.kind === "center" ? 6 : 5}
        />
      ))}
    </svg>}
    {selectedBody && gizmo?.origin.visible && <SelectionFlyout
      body={selectedBody}
      leftFraction={gizmo.origin.leftFraction}
      topFraction={gizmo.origin.topFraction}
    />}
    {hover && hoverProjection?.visible && !pointerRef.current && <div
      className={`editor-hover-chip kind-${hover.kind}`}
      data-testid="editor-hover-chip"
      style={{ left: `${hoverProjection.leftFraction * 100}%`, top: `${hoverProjection.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>{hover.label}</span>
      <small>{hover.position_m.x.toFixed(2)} · {hover.position_m.y.toFixed(2)} · {hover.position_m.z.toFixed(2)} m</small>
    </div>}
    {pixelTraceEnabled && svoRenderMode === "svo" && voxelRenderMode === "smooth" && <>
      {traceReticle?.visible && <div
        className="pixel-trace-reticle"
        data-testid="pixel-trace-reticle"
        data-pinned={pixelTracePinned ? "true" : "false"}
        style={{ left: `${traceReticle.leftFraction * 100}%`, top: `${traceReticle.topFraction * 100}%` }}
        aria-hidden="true"
      ><i /><em /></div>}
      <PixelTraceHud
        trace={pixelTrace}
        enabledLayers={pixelTraceLayers}
        pinned={pixelTracePinned}
        probeStatus={pixelTraceState.status}
        pointerSeen={pixelTraceState.pointerSeen}
        stale={pixelTraceState.stale}
        onToggleLayer={togglePixelTraceLayer}
        // Pinning asks; unpinning is immediate. The ask is what carries the aim.
        onTogglePinned={() => (pixelTracePinned ? setPixelTracePinned(false) : requestPixelTracePin())}
        onClose={() => setPixelTraceEnabled(false)}
      />
    </>}
    {failure && <div
      className={`viewport-failure-alert tone-${failure.tone}`}
      data-testid="viewport-failure-alert"
      data-failure-id={failure.id}
      role="alert"
      aria-live="assertive"
    >
      <div><strong>{failure.title}</strong><span>{failure.stage}</span></div>
      <p>{failure.detail}</p>
      <button type="button" onClick={() => setDiagnosticsOpen(true)}>INSPECT DIAGNOSTICS</button>
    </div>}
    {failure && failureProjection?.visible && <div
      className={`viewport-failure-marker tone-${failure.tone}`}
      data-testid="viewport-failure-marker"
      style={{ left: `${failureProjection.leftFraction * 100}%`, top: `${failureProjection.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>{failure.locationLabel ?? "first recorded failure"}</span>
    </div>}
    {svoCostOverlay !== "off" && svoRenderMode === "svo" && voxelRenderMode === "smooth" && <div className="svo-cost-legend" data-testid="svo-cost-legend">
      <header><span>SVO · {svoOverlayDefinition.label}</span><span>depth ≤ {svoMaximumTraversalDepth} · visits ≤ {svoMaximumNodeVisits}</span></header>
      <div className="svo-cost-ramp" style={{ background: svoOverlayRamp }} />
      <footer><span>{svoOverlayDefinition.legend[0].label}</span><span>{svoOverlayDefinition.legend.at(-1)?.label}</span></footer>
      <small>{svoOverlayDefinition.description} Adjust its limits and scene blend in Render.</small>
    </div>}
  </>;
}
