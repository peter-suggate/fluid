"use client";

import { useEffect, useRef } from "react";
import { FluidLabRenderer } from "@/lib/webgpu-renderer";
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

type Vec3 = RigidBodyState["position_m"];

export function WebGPUViewport() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FluidLabRenderer | null>(null);
  const camera = useUIStore((state) => state.camera);
  const setCamera = useUIStore((state) => state.setCamera);
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
    let running = true;
    const renderer = new FluidLabRenderer(
      canvas,
      (status) => {
        if (status.state === "lost" || status.state === "unavailable") running = false;
        useDiagnosticsStore.getState().set({ gpuStatus: status });
      },
      (info) => useDiagnosticsStore.getState().set({ gpuInfo: info }),
      undefined,
      (time_s) => simulation.gpuAdvanceCompleted(time_s),
      (effectiveRendererStatus) => useDiagnosticsStore.getState().set({ effectiveRendererStatus })
    );
    const syncRunState = (runState: ReturnType<typeof useRuntimeStore.getState>["runState"]) => {
      const submittedTime_s = renderer.setSimulationRunning(runState === "running");
      if (runState === "paused") simulation.gpuSchedulingPaused(submittedTime_s);
    };
    syncRunState(useRuntimeStore.getState().runState);
    const unsubscribeRunState = useRuntimeStore.subscribe((state, previous) => {
      if (state.runState !== previous.runState) syncRunState(state.runState);
    });
    rendererRef.current = renderer;
    let frame = 0;
    let alive = true;
    let lastFrameAt_ms = -Infinity;
    let lastPausedPresentation: readonly unknown[] | undefined;
    renderer.initialize().then(() => {
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
          simulation.time(), canvas.clientWidth, canvas.clientHeight, window.devicePixelRatio
        ] : undefined;
        if (pausedPresentation && !presentationStateChanged(lastPausedPresentation, pausedPresentation)) return;
        if (!pausedPresentation) lastPausedPresentation = undefined;
        let metrics;
        try {
          metrics = renderer.draw(
            simulation.time(), scene, ui.camera, state.bodies, ui.selectedBodyId,
            state.fluidRenderState ?? undefined, simulation.backend,
            { methodId: method.methodId, quality: method.quality, values: resolvedMethodValues(method) },
            { axis: ui.gridOverlayAxis, position: ui.gridOverlaySlice, mode: ui.gridOverlayMode },
            getScenePreset(sceneState.presetId).background,
            ui.voxelRenderMode,
            ui.svoRenderMode
          );
        } catch (error: unknown) {
          running = false;
          useDiagnosticsStore.getState().set({ gpuStatus: { state: "lost", label: error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped" } });
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
      running = false;
      if (useDiagnosticsStore.getState().gpuStatus.state !== "lost") diagnostics.set({ gpuStatus: { state: "unavailable", label: error instanceof Error ? error.message : "WebGPU initialization failed" } });
    });
    return () => { alive = false; running = false; unsubscribeRunState(); cancelAnimationFrame(frame); if(rendererRef.current===renderer)rendererRef.current=null; renderer.destroy(); };
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
    if (ui.gridOverlayAxis === "off") return undefined;
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

  return (
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
  );
}
