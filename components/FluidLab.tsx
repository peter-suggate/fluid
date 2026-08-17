"use client";

import { useEffect, useLayoutEffect, useState } from "react";
import { simulation } from "../lib/core/simulation/controller";
import { startQueryStateSync } from "../lib/core/url-state";
import { startSceneAutosave } from "../lib/core/scene-autosave";
import { browserSceneLibraryStorage } from "../lib/core/scene-library";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { useRuntimeStore } from "../lib/core/stores/runtime-store";
import { useShellStore } from "../lib/core/stores/shell-store";
import { WebGPUViewport } from "./WebGPUViewport";
import { EditorModeChip } from "./EditorModeChip";
import { PipelineOverlay } from "./PipelineOverlay";
import { RadialMenu } from "./RadialMenu";
import { SceneOverlay } from "./SceneOverlay";
import { SceneScaleOverlay } from "./SceneScaleOverlay";
import { TransportBar } from "./TransportBar";
import { RecordingPlaybackModal } from "./RecordingPlaybackModal";
import type { ResourceActivity, ResourcePluginDefinition } from "../lib/core/resource-readiness";
import { resourceActivities, resourceActivitiesFor } from "../lib/core/resource-readiness";
import { requestManualGPUStart } from "../lib/core/gpu-startup";
import { useSafeBrowserGPUBringup } from "../lib/core/use-safe-browser-gpu-bringup";
import { useEditorShortcuts } from "../lib/core/use-editor-shortcuts";

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
      : plugin.blocks === "viewport"
        ? "The rest of the studio remains available. "
        : "The editor and camera remain available. "}{plugin.blocks === "viewport"
          ? "Scene interaction unlocks when a complete SVO frame is fenced."
          : plugin.blocks === "transport"
            ? "Simulation transport unlocks when authoritative fluid is fenced."
            : "This work does not block product interaction."}</small>
  </div>;
}

/** Work that blocks nothing is reported, but it never takes the tray's width. */
function ResourceActivityPill({ activity }: { activity: ResourceActivity }) {
  return <span className="resource-activity-pill" role="status" aria-live="polite">
    <i aria-hidden="true" />
    <strong>{activity.label}</strong>
    {activity.total > 0 && <small>{Math.min(activity.completed, activity.total)}/{activity.total}</small>}
  </span>;
}

export function FluidLab() {
  const safeBringup = useSafeBrowserGPUBringup() === true;
  const runState = useRuntimeStore((state) => state.runState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const resourceReadiness = useDiagnosticsStore((state) => state.resourceReadiness);
  const shellView = useShellStore((state) => state.view);
  const activities = resourceActivities(resourceReadiness);
  // Transport-blocking work is deliberately absent here: TransportBar states it
  // inline, beside the controls it suspends.
  const trayCards = resourceActivitiesFor(resourceReadiness, "card");
  const trayPills = resourceActivitiesFor(resourceReadiness, "pill");

  useLayoutEffect(() => {
    // A scene has its own route. During client navigation (and Fast Refresh)
    // the opened document is already in the stores, so keep it rather than
    // rebuilding an incomplete URL projection over it. A direct page load has
    // fresh stores and hydrates from the address as before.
    const hydrateFromUrl = !useShellStore.getState().studioEntered;
    useShellStore.getState().enterStudio();
    return startQueryStateSync(() => {
      // /scene is authoritative even for an old link carrying view=library.
      // Enter before the canonical URL write so that legacy flag is removed.
      useShellStore.getState().enterStudio();
      simulation.reset();
    }, { hydrateFromUrl });
  }, []);
  // After the URL has hydrated the document, so the first autosave records the
  // scene the reader actually arrived on rather than the default preset.
  useEffect(() => startSceneAutosave({ storage: browserSceneLibraryStorage() }), []);
  useEditorShortcuts();

  useEffect(() => {
    let frame = 0;
    const tick = (now: number) => { simulation.tick(now); frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);


  return (
    <main className="lab-shell" data-run-state={runState} data-solver-mode="eulerian" data-simulation-time={simulationTime.toFixed(6)} data-body-count={bodies.length} data-shell-view={shellView}>
      <section className="viewport-shell" data-resource-active={activities.length > 0} data-gpu-transition={activities.at(-1)?.lane ?? resourceReadiness.platform.state}>
        <WebGPUViewport />
        <EditorModeChip />
        <RadialMenu />
        <div className="viewport-topline">
          <div className="topline-left">
            <SceneOverlay />
          </div>
        </div>
        <SceneScaleOverlay />
        <PipelineOverlay />
        <TransportBar />
        {(trayCards.length > 0 || trayPills.length > 0) && <div className="resource-activity-tray" aria-label="Resource tasks">
          {trayCards.map((activity) => <GPUInitializationPanel
            key={activity.id}
            activity={activity}
            plugin={resourceReadiness.plugins[activity.pluginId].plugin}
          />)}
          {trayPills.length > 0 && <div className="resource-activity-pills">
            {trayPills.map((activity) => <ResourceActivityPill key={activity.id} activity={activity} />)}
          </div>}
        </div>}
        {gpuStatus.state === "manual" && <div className="gpu-fallback gpu-manual-start" role="status">
          <strong>WebGPU startup paused</strong>
          <p>{safeBringup
            ? "Bounded bring-up permits the authored 384-column dam break, one STEP, then an explicit STOP GPU. Close every Dawn process first."
            : gpuStatus.label}</p>
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

      <RecordingPlaybackModal />
    </main>
  );
}
