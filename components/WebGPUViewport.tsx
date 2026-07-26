"use client";

import { useEffect, useRef, useState } from "react";
import { FluidLabRenderer } from "@/lib/webgpu-renderer";
import { getMethod } from "@/lib/methods";
import { canonicalScene } from "@/lib/model";
import { add, cameraBasis, dot, length, orbit, pan, scale, sub, zoom } from "@/lib/math";
import { boundingRadius, createBodyDescription, type RigidBodyState } from "@/lib/rigid-body";
import { simulation } from "@/lib/simulation/controller";
import { projectToViewport, viewportRayForPointer } from "@/lib/webgpu-camera";
import {
  closestPointOnAxis,
  gizmoAxisDragPosition,
  gizmoHandleAtPointer,
  projectGizmo,
  GIZMO_AXIS_DIRECTIONS,
  type GizmoAxis,
} from "@/lib/editor-gizmo";
import { hoverSceneAt, restOnHover, type EditorHover } from "@/lib/editor-hover";
import { SelectionFlyout } from "./SelectionFlyout";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useMethodStore, resolvedMethodValues } from "@/lib/stores/method-store";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { advancePresentationClock, presentationFrameDue, presentationStateChanged } from "@/lib/frame-pacing";
import { getScenePreset } from "@/lib/scenes";
import { SVO_COST_OVERLAY_LABELS } from "@/lib/svo-render-diagnostics";
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

export function WebGPUViewport() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
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
  const activeTool = useUIStore((state) => state.activeTool);
  const selection = useUIStore((state) => state.selection);
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const [hover, setHover] = useState<EditorHover | null>(null);
  const pointerRef = useRef<
    | { id: number; x: number; y: number; action: "orbit" | "pan" }
    | { id: number; x: number; y: number; action: "pick" }
    | { id: number; action: "body"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3; lastTime: number }
    // Editor gizmo drags preview on runtime state only and commit on release.
    | { id: number; action: "gizmo-axis"; bodyId: string; axis: GizmoAxis; axisOrigin: Vec3; grabOffset: Vec3; lastPosition: Vec3 }
    | { id: number; action: "gizmo-free"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3 }
    | { id: number; action: "slice"; axis: "x" | "y" | "z"; grabY: number; startClientY: number; startSlice: number }
    | null
  >(null);
  const selectedBody = selection?.kind === "body"
    ? bodies.find((body) => body.description.id === selection.id)
    : undefined;
  const gizmo = selectedBody && activeTool === "select"
    ? projectGizmo(selectedBody.position_m, camera, viewportSize.width, viewportSize.height)
    : undefined;
  const hoverProjection = hover ? projectToViewport(hover.position_m, camera, viewportSize.width, viewportSize.height) : undefined;

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
    let lastFrameAt_ms = -Infinity;
    let lastPausedPresentation: readonly unknown[] | undefined;
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
      const render = (now_ms: number) => {
        if (!alive || !running) return;
        frame = requestAnimationFrame(render);
        if (!presentationFrameDue(lastFrameAt_ms, now_ms)) return;
        lastFrameAt_ms = advancePresentationClock(lastFrameAt_ms, now_ms);
        const sceneState = useSceneStore.getState();
        const scene = sceneState.scene;
        const ui = useUIStore.getState();
        const method = useMethodStore.getState();
        const state = useDiagnosticsStore.getState();
        const runtime = useRuntimeStore.getState();
        // A trace needs newly completed submissions, not repeated snapshots
        // of the last on-change frame. Static dry scenes have no fluid solver
        // and are paused by design, so keep presenting while PERF is visible.
        // FluidLabRenderer still admits only one presentation at a time.
        const continuousPerformancePresentation = runtime.runState === "paused"
          && ui.rightPanel === "performance";
        const pausedPresentation = runtime.runState === "paused" && !continuousPerformancePresentation ? [
          sceneState, ui, method, state.bodies, state.fluidRenderState, state.gpuInfo,
          simulation.time(), renderer.presentationRevision,
          canvas.clientWidth, canvas.clientHeight, window.devicePixelRatio
        ] : undefined;
        if (pausedPresentation && !presentationStateChanged(lastPausedPresentation, pausedPresentation)) return;
        if (!pausedPresentation) lastPausedPresentation = undefined;
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
              overlayOpacity: ui.svoOverlayOpacity,
            }
          );
        } catch (error: unknown) {
          void stopGPU(error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped");
          return;
        }
        simulation.recordFrame(metrics, renderer.presentationResolution);
        // Retry a pending paused presentation instead of considering it
        // painted before a command buffer was submitted.
        if (pausedPresentation && metrics.presentationSubmitted) lastPausedPresentation = pausedPresentation;
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
    // pointerRef is a ref, so clearing hover here is what actually re-renders
    // the chip away for the duration of the gesture.
    setHover(null);
    if (event.button === 0 && !event.shiftKey) {
      const ray = pointerRay(event);
      if (beginGizmoDrag(event, ray)) return;
      // Armed tools claim the click before the slice/pick/orbit fallback so a
      // placement never orbits the camera instead.
      if (useUIStore.getState().activeTool === "body-place") { placeBodyAt(ray); return; }
      const grab = sliceGrabHit(ray.origin, ray.direction);
      if (grab) { pointerRef.current = { id: event.pointerId, action: "slice", ...grab, startClientY: event.clientY, startSlice: useUIStore.getState().gridOverlaySlice }; return; }
      if (simulation.backend === "webgpu" && rendererRef.current) {
        const pointerId=event.pointerId,timeStamp=event.timeStamp,x=event.clientX,y=event.clientY;
        pointerRef.current={id:pointerId,x,y,action:"pick"};
        const rect=event.currentTarget.getBoundingClientRect();
        const picked=await rendererRef.current.pickRigidBody(ray.origin,ray.direction,{
          normalizedX:(event.clientX-rect.left)/Math.max(rect.width,1),
          normalizedY:(event.clientY-rect.top)/Math.max(rect.height,1),
        });
        const active=pointerRef.current;
        if(!active||active.id!==pointerId||active.action!=="pick")return;
        const body=picked?useDiagnosticsStore.getState().bodies[picked.bodyIndex]:undefined;
        if(body&&picked){beginBodyDrag(pointerId,timeStamp,ray,body,picked.position_m,picked.orientation,"surfacePosition_m" in picked?picked.surfacePosition_m:picked.position_m);return;}
        pointerRef.current={id:pointerId,x,y,action:"orbit"};
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
    pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, action: event.shiftKey || event.button === 1 ? "pan" : "orbit" };
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const active = pointerRef.current;
    if (!active) {
      // Analytic hover: no GPU readback, so it is safe at pointer-move rate.
      const ray = pointerRay(event);
      setHover(hoverSceneAt(useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray) ?? null);
      return;
    }
    if (active.id !== event.pointerId) return;
    if (active.action === "pick") return;
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
    const active = pointerRef.current;
    if (active?.id !== event.pointerId) return;
    pointerRef.current = null;
    if (active.action === "body") { simulation.dragBody(active.bodyId, active.lastPosition, { x: 0, y: 0, z: 0 }, "end"); return; }
    if (active.action === "gizmo-axis" || active.action === "gizmo-free") {
      // Commit-on-release: one document write and one undo entry per gesture.
      simulation.manipulateBody(active.bodyId, active.lastPosition, "end");
      simulation.updateBody(active.bodyId, { position_m: active.lastPosition, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
      simulation.commitEdit();
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
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={pointerUp}
      onPointerLeave={() => setHover(null)}
      onWheel={(event) => { event.preventDefault(); setCamera((current) => zoom(current, event.deltaY)); }}
      onContextMenu={(event) => event.preventDefault()}
    />
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
      <header><span>SVO · {SVO_COST_OVERLAY_LABELS[svoCostOverlay]}</span><span>depth ≤ {svoMaximumTraversalDepth} · visits ≤ {svoMaximumNodeVisits}</span></header>
      {svoCostOverlay === "exhaustion"
        ? <div className="svo-cost-ramp" style={{ background: "linear-gradient(90deg,#17372f 0 48%,#f5d442 48% 72%,#f04438 72%)" }} />
        : <div className="svo-cost-ramp" />}
      <footer><span>{svoCostOverlay === "exhaustion" ? "within budget" : "lower work"}</span><span>{svoCostOverlay === "exhaustion" ? "exhausted / invalid" : "higher work"}</span></footer>
      <small>Heatmap is blended with the scene radiance; lower the limits in Render to expose expensive rays.</small>
    </div>}
  </>;
}
