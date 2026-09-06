"use client";

import { WorkProgress } from "./WorkProgress";
import { resourceWorkProgress } from "../lib/core/work-progress";
import { requestManualGPUStart } from "../lib/core/gpu-startup";
import type { ResourceActivity, ResourcePluginDefinition } from "../lib/core/resource-readiness";
import { resourceActivities, resourceActivitiesFor } from "../lib/core/resource-readiness";
import { transportReadiness, transportWorkStatus } from "../lib/core/transport-status";
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
  const heading = activity.lane === "platform" ? "Starting WebGPU"
    : activity.lane === "fluid" && activity.operation ? "Applying simulation settings"
    : activity.lane === "fluid" ? "Preparing fluid"
    : activity.lane === "svo" ? "Preparing sparse presentation" : "Preparing tool";
  return <div className="gpu-build-card gpu-initializing" role="region" aria-label={heading}>
    <div className="gpu-build-heading"><i aria-hidden="true" /><strong>{heading}</strong></div>
    {activity.operation && <p className="gpu-build-operation">{activity.operation}</p>}
    {/* The label is not restated here — WorkProgress leads with it. */}
    <WorkProgress progress={resourceWorkProgress(activity, plugin)} />
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
function ResourceActivityPill({ activity, plugin }: { activity: ResourceActivity; plugin: ResourcePluginDefinition }) {
  return <WorkProgress compact progress={resourceWorkProgress(activity, plugin)} />;
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
  const gpuInfo = session.diagnostics((state) => state.gpuInfo);
  const resourceReadiness = session.diagnostics((state) => state.resourceReadiness);
  const methodId = session.method((state) => state.methodId);
  const activities = resourceActivities(resourceReadiness);
  const trayCards = resourceActivitiesFor(resourceReadiness, "card");
  // Transport-blocking work reports here too: the transport bar only disables
  // its controls (with the reason on each control) and never grows a progress
  // chip, so the story of what is running lives in this tray with the rest.
  const transportInline = resourceActivitiesFor(resourceReadiness, "transport-inline");
  const trayPills = [...transportInline, ...resourceActivitiesFor(resourceReadiness, "pill")];
  // The synthesized gate status earns a pill only when it says something the
  // plugin activities are not already saying: while bring-up cards and pills
  // narrate the same loading with real counts, "Loading sparse world" on top of
  // them is a third telling of one story. Attention states (faults, capacity,
  // deferred resolution) always show — no activity carries those.
  const gateWork = transportWorkStatus(transportReadiness(gpuInfo, methodId), gpuInfo);
  const transportWork = gateWork
    && (gateWork.state !== "active" || (trayCards.length === 0 && transportInline.length === 0))
    ? gateWork : undefined;

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
      {(trayCards.length > 0 || trayPills.length > 0 || transportWork) && <div className="resource-activity-tray" aria-label="Resource tasks">
        {trayCards.map((activity) => <GPUInitializationPanel
          key={activity.id}
          activity={activity}
          plugin={resourceReadiness.plugins[activity.pluginId].plugin}
        />)}
        {(trayPills.length > 0 || transportWork) && <div className="resource-activity-pills">
          {/* Faults keep their explanation; running work is a compact pill. */}
          {transportWork && <WorkProgress compact={transportWork.state !== "error"} progress={transportWork} />}
          {trayPills.map((activity) => <ResourceActivityPill key={activity.id} activity={activity} plugin={resourceReadiness.plugins[activity.pluginId].plugin} />)}
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
