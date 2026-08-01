"use client";

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from "react";
import { simulation } from "@/lib/simulation/controller";
import { startQueryStateSync } from "@/lib/url-state";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { WebGPUViewport } from "./WebGPUViewport";
import { EditorToolbar } from "./EditorToolbar";
import { SceneOverlay } from "./SceneOverlay";
import { SceneScaleOverlay } from "./SceneScaleOverlay";
import { SceneConfigPopover } from "./SceneConfigPopover";
import { RigidBodyPanel } from "./RigidBodyTray";
import { VisualPanel } from "./VisualPanel";
import { DiagnosticsPanel } from "./DiagnosticsPanel";
import { PerformancePanel } from "./PerformancePanel";
import { TransportBar } from "./TransportBar";
import { RecordingPlaybackModal } from "./RecordingPlaybackModal";
import type { ResourceActivity, ResourcePluginDefinition } from "@/lib/resource-readiness";
import { resourceActivities } from "@/lib/resource-readiness";
import { requestManualGPUStart } from "@/lib/gpu-startup";
import { useSafeBrowserGPUBringup } from "@/lib/use-safe-browser-gpu-bringup";
import { useEditorShortcuts } from "@/lib/use-editor-shortcuts";
import { MAX_RIGHT_PANEL_WIDTH, MIN_RIGHT_PANEL_WIDTH } from "@/lib/stores/ui-store";

function RightPanelResizer() {
  const rightPanelWidth = useUIStore((state) => state.rightPanelWidth);
  const setRightPanelWidth = useUIStore((state) => state.setRightPanelWidth);
  const dragStart = useRef<{ pointerX: number; width: number } | null>(null);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStart.current = { pointerX: event.clientX, width: rightPanelWidth };
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragStart.current || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
    setRightPanelWidth(dragStart.current.width + dragStart.current.pointerX - event.clientX);
  };
  const finishResize = (event: PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    dragStart.current = null;
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 50 : 10;
    if (event.key === "ArrowLeft") setRightPanelWidth(rightPanelWidth + step);
    else if (event.key === "ArrowRight") setRightPanelWidth(rightPanelWidth - step);
    else if (event.key === "Home") setRightPanelWidth(MIN_RIGHT_PANEL_WIDTH);
    else if (event.key === "End") setRightPanelWidth(MAX_RIGHT_PANEL_WIDTH);
    else return;
    event.preventDefault();
  };

  return <div
    className="right-panel-resizer"
    role="separator"
    aria-label="Resize panel"
    aria-orientation="vertical"
    aria-valuemin={MIN_RIGHT_PANEL_WIDTH}
    aria-valuemax={MAX_RIGHT_PANEL_WIDTH}
    aria-valuenow={rightPanelWidth}
    tabIndex={0}
    onPointerDown={onPointerDown}
    onPointerMove={onPointerMove}
    onPointerUp={finishResize}
    onPointerCancel={finishResize}
    onKeyDown={onKeyDown}
  />;
}

function GPUInitializationPanel({ activity, plugin }: {
  activity: ResourceActivity;
  plugin: ResourcePluginDefinition;
}) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => { const timer = window.setInterval(() => setNow(performance.now()), 100); return () => window.clearInterval(timer); }, []);
  const completed = activity.completed, total = activity.total;
  const elapsed_s = Math.max(0, now - activity.startedAt_ms) / 1000;
  const finalizing = total > 0 && completed >= total;
  const heading = activity.lane === "platform" ? "Starting WebGPU"
    : activity.lane === "fluid" && activity.operation ? "Applying simulation settings"
    : activity.lane === "fluid" ? "Preparing fluid"
    : activity.lane === "svo" ? "Preparing sparse presentation" : "Preparing tool";
  const explanation = plugin.phaseCopy?.[activity.phase]
    ?? "Preparing this resource independently from the rest of the product.";
  return <div className="gpu-build-card gpu-initializing" role="status" aria-live="polite">
    <div className="gpu-build-heading"><i aria-hidden="true" /><strong>{heading}</strong></div>
    {activity.operation && <p className="gpu-build-operation">{activity.operation}</p>}
    <p>{activity.label}</p>
    <progress max={Math.max(1, total)} {...(total > 0 && !finalizing ? { value: Math.min(completed, total) } : {})} aria-label="GPU initialization progress" />
    <div className="gpu-progress-summary"><span>{finalizing ? "Finalizing…" : total > 0 ? `${completed} / ${total} tasks` : "Planning work…"}</span><span>{elapsed_s.toFixed(1)} s</span></div>
    <p className="gpu-stage-explanation">{explanation}</p>
    {elapsed_s >= 10 && <p className="gpu-task-wait">Still working on this task. Elapsed time remains live when the GPU driver exposes no intermediate counters.</p>}
    <small>{activity.retainingPrevious
      ? "The attached generation remains usable. "
      : "The editor, camera, panels, and file actions remain available. "}{plugin.blocks === "transport" ? "Simulation transport unlocks when authoritative fluid is fenced." : "This work does not block product interaction."}</small>
  </div>;
}

export function FluidLab() {
  const safeBringup = useSafeBrowserGPUBringup() === true;
  const runState = useRuntimeStore((state) => state.runState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const resourceReadiness = useDiagnosticsStore((state) => state.resourceReadiness);
  const diagnosticsOpen = useUIStore((state) => state.diagnosticsOpen);
  const setDiagnosticsOpen = useUIStore((state) => state.setDiagnosticsOpen);
  const rightPanel = useUIStore((state) => state.rightPanel);
  const rightPanelWidth = useUIStore((state) => state.rightPanelWidth);
  const setRightPanel = useUIStore((state) => state.setRightPanel);
  const activities = resourceActivities(resourceReadiness);

  useLayoutEffect(() => startQueryStateSync(() => simulation.reset()), []);
  useEditorShortcuts();

  useEffect(() => {
    let frame = 0;
    const tick = (now: number) => { simulation.tick(now); frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);


  return (
    <main className="lab-shell" style={{ "--right-panel-width": `${rightPanelWidth}px` } as CSSProperties} data-run-state={runState} data-solver-mode="eulerian" data-simulation-time={simulationTime.toFixed(6)} data-body-count={bodies.length} data-right-panel-open={Boolean(rightPanel)} data-right-panel={rightPanel ?? "closed"}>
      <section className="viewport-shell" data-resource-active={activities.length > 0} data-gpu-transition={activities.at(-1)?.lane ?? resourceReadiness.platform.state}>
        <WebGPUViewport />
        <EditorToolbar />
        <div className="viewport-topline">
          <div className="topline-left">
            <SceneOverlay />
          </div>
        </div>
        <SceneScaleOverlay />
        <div className="axis-widget"><span className="axis-y">Y</span><span className="axis-x">X</span><span className="axis-z">Z</span></div>
        <nav className="utility-panel-tabs" aria-label="Viewport panels">
          <button className={rightPanel === "visual" ? "active" : ""} onClick={() => setRightPanel(rightPanel === "visual" ? null : "visual")} aria-expanded={rightPanel === "visual"} title="Render and debug controls">RENDER</button>
          <button className={rightPanel === "bodies" ? "active" : ""} onClick={() => setRightPanel(rightPanel === "bodies" ? null : "bodies")} aria-expanded={rightPanel === "bodies"} title="Rigid body controls">BODIES</button>
          <button className={diagnosticsOpen ? "active" : ""} onClick={() => setDiagnosticsOpen(!diagnosticsOpen)} aria-expanded={diagnosticsOpen} title="Live diagnostics">DIAGNOSTICS</button>
          <button className={rightPanel === "performance" ? "active" : ""} onClick={() => setRightPanel(rightPanel === "performance" ? null : "performance")} aria-expanded={rightPanel === "performance"} aria-controls="performance-panel" title="Measured work and paper fields">PERFORMANCE</button>
        </nav>
        {activities.length > 0 && <div className="resource-activity-tray" aria-label="Resource tasks">
          {activities.map((activity) => <GPUInitializationPanel
            key={activity.id}
            activity={activity}
            plugin={resourceReadiness.plugins[activity.pluginId].plugin}
          />)}
        </div>}
        {gpuStatus.state === "manual" && <div className="gpu-fallback gpu-manual-start" role="status">
          <strong>WebGPU startup paused for safety</strong>
          <p>{safeBringup
            ? "Bounded bring-up permits the authored 384-column dam break, one STEP, then an explicit STOP GPU. Close every Dawn process first."
            : "The dam-break GPU workload will not start until you explicitly allow it."}</p>
          <button type="button" onClick={requestManualGPUStart}>START WEBGPU</button>
          <small>{safeBringup
            ? "This browser can exclude other Fluid Lab tabs, but cannot observe Dawn's local filesystem lease."
            : <>Use <code>gpu=off</code> for UI-only inspection or <code>gpu=on</code> to restore automatic startup.</>}</small>
        </div>}
        {gpuStatus.state === "unavailable" && <div className="gpu-fallback"><strong>3D renderer unavailable</strong><p>{gpuStatus.label}</p>
          {gpuStatus.reproduction && <div data-testid="gpu-failure-reproduction">
            <small>Dawn case <strong>{gpuStatus.reproduction.caseId}</strong> · validated · serialized</small>
            <code>{gpuStatus.reproduction.command}</code>
          </div>}
          <small>The scene editor, serialization, and CPU validation remain available.</small>
        </div>}
      </section>

      {rightPanel && <RightPanelResizer />}
      {rightPanel === "visual" && <VisualPanel />}
      {rightPanel === "bodies" && <RigidBodyPanel />}
      {diagnosticsOpen && <DiagnosticsPanel />}
      {rightPanel === "performance" && <PerformancePanel />}
      <TransportBar />

      <RecordingPlaybackModal />
      <SceneConfigPopover />
    </main>
  );
}
