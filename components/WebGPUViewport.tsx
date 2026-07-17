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
import { advancePresentationClock, presentationFrameDue } from "@/lib/frame-pacing";

type Vec3 = RigidBodyState["position_m"];

export function WebGPUViewport() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const camera = useUIStore((state) => state.camera);
  const setCamera = useUIStore((state) => state.setCamera);
  const pointerRef = useRef<
    | { id: number; x: number; y: number; action: "orbit" | "pan" }
    | { id: number; action: "body"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3; lastTime: number }
    | { id: number; action: "slice"; axis: "x" | "z"; grabY: number }
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
      (loads) => simulation.mergeGPULoads(loads),
      (time_s) => simulation.gpuAdvanceCompleted(time_s)
    );
    let frame = 0;
    let alive = true;
    let lastFrameAt_ms = -Infinity;
    renderer.initialize().then(() => {
      const render = (now_ms: number) => {
        if (!alive || !running) return;
        frame = requestAnimationFrame(render);
        const targetFps = useUIStore.getState().targetFps;
        if (!presentationFrameDue(lastFrameAt_ms, now_ms, targetFps)) return;
        lastFrameAt_ms = advancePresentationClock(lastFrameAt_ms, now_ms, targetFps);
        const scene = useSceneStore.getState().scene;
        const ui = useUIStore.getState();
        const method = useMethodStore.getState();
        const state = useDiagnosticsStore.getState();
        let metrics;
        try {
          metrics = renderer.draw(
            simulation.time(), scene, ui.camera, ui.view, state.bodies, ui.selectedBodyId,
            state.fluidRenderState ?? undefined, simulation.backend,
            { methodId: method.methodId, quality: method.quality, values: resolvedMethodValues(method) },
            { axis: ui.view === "scientific" ? ui.gridOverlayAxis : "off", position: ui.gridOverlaySlice, mode: ui.gridOverlayMode },
            ui.waterRenderMode,
            ui.environmentId,
            targetFps
          );
        } catch (error: unknown) {
          running = false;
          useDiagnosticsStore.getState().set({ gpuStatus: { state: "lost", label: error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped" } });
          return;
        }
        simulation.recordFrame(metrics, renderer.presentationResolution);
      };
      frame = requestAnimationFrame(render);
    }).catch((error: unknown) => {
      running = false;
      if (useDiagnosticsStore.getState().gpuStatus.state !== "lost") diagnostics.set({ gpuStatus: { state: "unavailable", label: error instanceof Error ? error.message : "WebGPU initialization failed" } });
    });
    return () => { alive = false; running = false; cancelAnimationFrame(frame); renderer.destroy(); };
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

  // Hit test for the slice gripper: the accent bar along the top edge of the
  // grid-overlay plane. Grabbing it sweeps the slice through the volume.
  const sliceGrabHit = (origin: Vec3, direction: Vec3) => {
    const ui = useUIStore.getState();
    if (ui.view !== "scientific" || ui.gridOverlayAxis === "off") return undefined;
    const axis = ui.gridOverlayAxis;
    const c = useSceneStore.getState().scene.container;
    const planeCoordinate = (axis === "z" ? -c.depth_m / 2 + ui.gridOverlaySlice * c.depth_m : -c.width_m / 2 + ui.gridOverlaySlice * c.width_m);
    const denominator = axis === "z" ? direction.z : direction.x;
    if (Math.abs(denominator) < 1e-5) return undefined;
    const t = (planeCoordinate - (axis === "z" ? origin.z : origin.x)) / denominator;
    if (t <= 0) return undefined;
    const point = add(origin, scale(direction, t));
    const inFootprint = Math.abs(point.x) <= c.width_m / 2 && Math.abs(point.z) <= c.depth_m / 2;
    const nearTop = point.y >= c.height_m * 0.94 && point.y <= c.height_m * 1.02;
    return inFootprint && nearTop ? { axis, grabY: Math.min(point.y, c.height_m) } : undefined;
  };

  const pointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    if (event.button === 0 && !event.shiftKey) {
      const ray = pointerRay(event);
      const grab = sliceGrabHit(ray.origin, ray.direction);
      if (grab) { pointerRef.current = { id: event.pointerId, action: "slice", ...grab }; return; }
      let nearest: { body: RigidBodyState; t: number } | undefined;
      for (const body of useDiagnosticsStore.getState().bodies) {
        const oc = sub(ray.origin, body.position_m), radius = boundingRadius(body), b = dot(oc, ray.direction), c = dot(oc, oc) - radius * radius, discriminant = b * b - c;
        if (discriminant < 0) continue; const t = -b - Math.sqrt(discriminant);
        if (t > 0 && (!nearest || t < nearest.t)) nearest = { body, t };
      }
      if (nearest) {
        const basis = cameraBasis(useUIStore.getState().camera);
        const dragPoint = planeHit(ray.origin, ray.direction, nearest.body.position_m, basis.forward), grabOffset = sub(nearest.body.position_m, dragPoint);
        pointerRef.current = { id: event.pointerId, action: "body", bodyId: nearest.body.description.id, planePoint: nearest.body.position_m, planeNormal: basis.forward, grabOffset, lastPosition: nearest.body.position_m, lastTime: event.timeStamp };
        useUIStore.getState().selectBody(nearest.body.description.id);
        simulation.dragBody(nearest.body.description.id, nearest.body.position_m, { x: 0, y: 0, z: 0 }, "start");
        return;
      }
    }
    pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, action: event.shiftKey || event.button === 1 ? "pan" : "orbit" };
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const active = pointerRef.current;
    if (!active || active.id !== event.pointerId) return;
    if (active.action === "slice") {
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
