"use client";

import { ActivityIndicator, type ActivityEntry } from "./ActivityIndicator";
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
import { VoxelToolRail } from "./VoxelToolRail";
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

/** What a bring-up is called, by the lane that is doing it. */
function bringupTitle(activity: ResourceActivity): string {
  return activity.lane === "platform" ? "Starting WebGPU"
    : activity.lane === "fluid" && activity.operation ? "Applying simulation settings"
    : activity.lane === "fluid" ? "Preparing fluid"
    : activity.lane === "svo" ? "Preparing sparse presentation" : "Preparing tool";
}

/**
 * One line of the activity mark, from a plugin's in-flight work.
 *
 * The fraction is counted work through *all* of the plugin's declared phases:
 * whole phases behind this one, plus this phase's own count. Phases are taken
 * as equal steps — the plugin declares their order, not their weight — which is
 * what stops the ring emptying each time a phase turns over.
 */
function activityEntry(activity: ResourceActivity, plugin: ResourcePluginDefinition, blocking: boolean): ActivityEntry {
  const progress = resourceWorkProgress(activity, plugin);
  const counted = activity.total > 0 && Number.isFinite(activity.completed);
  const within = counted ? Math.max(0, Math.min(1, activity.completed / activity.total)) : undefined;
  const phases = plugin.progressPhases ?? [];
  const phaseIndex = phases.findIndex((phase) => phase.id === activity.phase);
  const fraction = within === undefined ? undefined
    : phaseIndex >= 0 ? (phaseIndex + within) / phases.length : within;
  return {
    id: activity.id,
    // A blocking bring-up is headed by its lane, so its line says what it is
    // doing right now; everything else is named by its own label.
    label: blocking && activity.operation ? activity.operation : progress.label,
    state: "active",
    count: counted ? `${Math.min(activity.completed, activity.total).toLocaleString()} / ${activity.total.toLocaleString()}` : undefined,
    fraction,
    startedAt_ms: activity.startedAt_ms,
  };
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
  const stopped = ["unavailable", "lost", "stopping", "blocked"].includes(gpuStatus.state);
  const activities = stopped ? [] : resourceActivities(resourceReadiness);
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

  // One mark for all of it. A bring-up that has taken the scene heads the mark
  // with its lane and says, once, what it is holding up; work that blocks less
  // is a line under it, or — alone — the headline itself.
  const lead = trayCards[0];
  const pluginOf = (activity: ResourceActivity) => resourceReadiness.plugins[activity.pluginId].plugin;
  const activityEntries: ActivityEntry[] = [
    ...trayCards.map((activity) => activityEntry(activity, pluginOf(activity), true)),
    ...trayPills.map((activity) => activityEntry(activity, pluginOf(activity), false)),
    ...(transportWork && transportWork.state !== "complete" ? [{
      id: "transport-gate",
      label: transportWork.label,
      state: transportWork.state,
      count: transportWork.total !== undefined && transportWork.completed !== undefined
        ? `${transportWork.completed.toLocaleString()} / ${transportWork.total.toLocaleString()} ${transportWork.unit ?? ""}`.trim()
        : undefined,
      // Running work is its own explanation; a fault or a wait has to say why.
      detail: transportWork.state === "active" ? undefined : transportWork.detail,
    }] : []),
  ];
  const activityTitle = lead ? bringupTitle(lead) : undefined;
  const activityNote = lead?.retainingPrevious ? "The current scene stays in use until this is ready."
    : lead ? "The scene unlocks at its first complete frame."
    : transportInline.length > 0 ? "Playback unlocks when the fluid is ready."
    : undefined;

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
      <VoxelToolRail />
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
      {!stopped && <ActivityIndicator title={activityTitle} entries={activityEntries} note={activityNote} />}
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
    </section>
  );
}
