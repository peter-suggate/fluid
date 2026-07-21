"use client";

import { useEffect, useRef } from "react";
import { FluidLabRenderer } from "@/lib/webgpu-renderer";
import { getMethod } from "@/lib/methods";
import { canonicalScene } from "@/lib/model";
import { add, cameraBasis, dot, length, normalize, orbit, pan, scale, sub, zoom } from "@/lib/math";
import { boundingRadius, type RigidBodyState } from "@/lib/rigid-body";
import { simulation } from "@/lib/simulation/controller";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useMethodStore, resolvedMethodValues } from "@/lib/stores/method-store";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { advancePresentationClock, presentationFrameDue, presentationStateChanged } from "@/lib/frame-pacing";
import { getScenePreset } from "@/lib/scenes";
import { gpuStageCapture } from "@/lib/gpu-stage-capture";
import { SVO_COST_OVERLAY_LABELS } from "@/lib/svo-render-diagnostics";
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
  const svoCostOverlay = useUIStore((state) => state.svoCostOverlay);
  const svoRenderMode = useUIStore((state) => state.svoRenderMode);
  const voxelRenderMode = useUIStore((state) => state.voxelRenderMode);
  const svoMaximumTraversalDepth = useUIStore((state) => state.svoMaximumTraversalDepth);
  const svoMaximumNodeVisits = useUIStore((state) => state.svoMaximumNodeVisits);
  const pointerRef = useRef<
    | { id: number; x: number; y: number; action: "orbit" | "pan" }
    | { id: number; x: number; y: number; action: "pick" }
    | { id: number; action: "body"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3; lastTime: number }
    | { id: number; action: "slice"; axis: "x" | "y" | "z"; grabY: number; startClientY: number; startSlice: number }
    | null
  >(null);

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
        stageCapturePhase: gpuStageCapture.getSnapshot().phase,
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
      stopPromise = (async () => {
        await shutdownBrowserGPUSession(renderer, pendingLease, releaseGPULease);
        releaseGPULease = undefined;
        stopping = false;
        stopped = true;
        if (publishStatus) diagnostics.set({ gpuStatus: { state: "unavailable", label: releasedLabel } });
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
        const pausedPresentation = runtime.runState === "paused" ? [
          sceneState, ui, method, state.bodies, state.fluidRenderState, state.gpuInfo,
          simulation.time(), renderer.presentationRevision,
          gpuStageCapture.getSnapshot().revision,
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
        // A pending presentation returns zero encode time. Retry that same
        // paused state on the next paced callback instead of considering it
        // painted before any command buffer was submitted.
        if (pausedPresentation && metrics.cpuRenderEncode_ms > 0) lastPausedPresentation = pausedPresentation;
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
    const unsubscribeSafeCapture = gpuStageCapture.subscribe(enforceSafeConfiguration);
    const manualStop = () => { void stopGPU(); };
    window.addEventListener(GPU_MANUAL_STOP_EVENT, manualStop);
    const pageHide = () => { void stopGPU("WebGPU stopped during page close", false); };
    window.addEventListener("pagehide", pageHide, { once: true });
    if (startupMode() === "manual" || startupMode() === "safe") {
      diagnostics.set({ gpuStatus: { state: "manual", label: "WebGPU is waiting for explicit startup" } });
      window.addEventListener(GPU_MANUAL_START_EVENT, beginInitialization);
    } else beginInitialization();
    return () => { alive = false; running = false; window.removeEventListener(GPU_MANUAL_START_EVENT, beginInitialization); window.removeEventListener(GPU_MANUAL_STOP_EVENT, manualStop); window.removeEventListener("pagehide", pageHide); unsubscribeAutomaticStart(); unsubscribeSafeScene(); unsubscribeSafeMethod(); unsubscribeSafeUI(); unsubscribeSafeCapture(); unsubscribeRunState(); cancelAnimationFrame(frame); if(rendererRef.current===renderer)rendererRef.current=null; void stopGPU("WebGPU stopped during component cleanup", false); };
  }, []);

  const pointerRay = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect(), basis = cameraBasis(useUIStore.getState().camera);
    const ndcX = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
    const ndcY = 1 - ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2;
    return { origin: basis.position, direction: normalize(add(basis.forward, add(scale(basis.right, ndcX * rect.width / Math.max(rect.height, 1) * 0.72), scale(basis.up, ndcY * 0.72)))) };
  };
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

  const pointerDown = async (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    if (event.button === 0 && !event.shiftKey) {
      const ray = pointerRay(event);
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
    if (!active || active.id !== event.pointerId) return;
    if (active.action === "pick") return;
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
    if (active?.id === event.pointerId) {
      if (active.action === "body") simulation.dragBody(active.bodyId, active.lastPosition, { x: 0, y: 0, z: 0 }, "end");
      pointerRef.current = null;
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
      onWheel={(event) => { event.preventDefault(); setCamera((current) => zoom(current, event.deltaY)); }}
      onContextMenu={(event) => event.preventDefault()}
    />
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
