"use client";

import { useEffect, useState } from "react";
import { requestManualGPUStart } from "../lib/core/gpu-startup";
import type { ResourceActivity, ResourcePluginDefinition } from "../lib/core/resource-readiness";
import { resourceActivities, resourceActivitiesFor } from "../lib/core/resource-readiness";
import type { PaneId } from "../lib/core/session/session";
import { useSession } from "../lib/core/session/session-context";
import { useSafeBrowserGPUBringup } from "../lib/core/use-safe-browser-gpu-bringup";
import { EditorModeChip } from "./EditorModeChip";
import { PipelineOverlay } from "./PipelineOverlay";
import { RadialMenu } from "./RadialMenu";
import { SceneScaleOverlay } from "./SceneScaleOverlay";
import { SceneSelector } from "./SceneSelector";
import { WebGPUViewport } from "./WebGPUViewport";

/**
 * One pane of scene: a canvas and everything drawn over it.
 *
 * Every child here reads the session this component is mounted under, so a
 * second pane is a second `<SessionProvider>` around a second one of these and
 * nothing below learns a new code path — which is the whole bargain of the
 * session realm. What is deliberately *not* here is the chrome that belongs to
 * the page rather than to a pane: the transport, the scene chip and the
 * recording modal are the host's, because there is one clock, one document
 * being authored and one recorder.
 */

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

export interface ScenePaneProps {
  readonly paneId: PaneId;
  /** Draw the A / B tag. Absent in single-pane mode: there is nothing to tell apart. */
  readonly tagged?: boolean;
  /** The pane the keyboard and the ring belong to. */
  readonly focused?: boolean;
  readonly onFocus?: () => void;
}

export function ScenePane({ paneId, tagged = false, focused = false, onFocus }: ScenePaneProps) {
  const safeBringup = useSafeBrowserGPUBringup() === true;
  const session = useSession();
  const selectorOpen = session.ui((state) => state.sceneSelectorOpen);
  const setSelectorOpen = session.ui((state) => state.setSceneSelectorOpen);
  const gpuStatus = session.diagnostics((state) => state.gpuStatus);
  const resourceReadiness = session.diagnostics((state) => state.resourceReadiness);
  const activities = resourceActivities(resourceReadiness);
  // Transport-blocking work is deliberately absent here: TransportBar states it
  // inline, beside the controls it suspends.
  const trayCards = resourceActivitiesFor(resourceReadiness, "card");
  const trayPills = resourceActivitiesFor(resourceReadiness, "pill");

  return (
    <section
      className="viewport-shell"
      data-pane={paneId}
      data-pane-focused={focused}
      data-resource-active={activities.length > 0}
      data-gpu-transition={activities.at(-1)?.lane ?? resourceReadiness.platform.state}
      // Capture, so focus follows a press that a child stops: a right-click that
      // opens the ring has to focus its own pane before the ring is composed.
      onPointerDownCapture={onFocus}
      onPointerEnter={onFocus}
    >
      <WebGPUViewport paneId={paneId} />
      <EditorModeChip />
      <RadialMenu />
      <SceneScaleOverlay />
      <PipelineOverlay />
      {/* The tag names the pane and is also its scene switch. A compare is only
          worth opening once the two panes can differ, and the coarsest way they
          differ is by running different scenes — so the affordance sits on the
          one mark that already says "this pane", rather than as a second badge
          beside it. */}
      {tagged && <button
        type="button"
        className="pane-tag"
        data-pane={paneId}
        data-pane-focused={focused}
        data-scene-selector-toggle=""
        data-testid={`pane-tag-${paneId}`}
        aria-haspopup="dialog"
        aria-expanded={selectorOpen}
        title={`Pane ${paneId.toUpperCase()} — choose the scene this pane runs`}
        onClick={() => setSelectorOpen(!selectorOpen)}
      >{paneId.toUpperCase()}</button>}
      {selectorOpen && <SceneSelector />}
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
  );
}
