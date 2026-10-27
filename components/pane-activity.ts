"use client";

import { useEffect } from "react";
import type { ActivityEntry } from "./ActivityIndicator";
import { progressHistory } from "../lib/core/progress-history";
import { resourceWorkProgress } from "../lib/core/work-progress";
import type { ResourceActivity, ResourcePluginDefinition } from "../lib/core/resource-readiness";
import { resourceActivities, resourceActivitiesFor } from "../lib/core/resource-readiness";
import { transportReadiness, transportWorkStatus } from "../lib/core/transport-status";
import type { PaneSession } from "../lib/core/session/session";

/** What a bring-up is called, by the lane that is doing it. */
function bringupTitle(activity: ResourceActivity): string {
  return activity.lane === "platform" ? "Starting WebGPU"
    : activity.lane === "fluid" && activity.operation ? "Applying simulation settings"
    : activity.lane === "fluid" ? "Preparing fluid"
    : activity.lane === "svo" ? "Preparing sparse presentation" : "Preparing tool";
}

/** The work a pane's activity is the same work as, the next time it runs. */
function progressContext(session: PaneSession): string {
  const { presetId } = session.scene.getState();
  const { methodId, quality } = session.method.getState();
  return `${presetId}|${methodId}|${quality}`;
}

/**
 * One line of the activity, from a plugin's in-flight work.
 *
 * The count is what the plugin counted: a task's own sub-count (pipelines
 * settled out of known) where it has one, else the runner's tasks. The
 * fraction is the last finished run's timing where there is one (`estimated`),
 * else counted work through the plugin's declared phases taken as equal steps,
 * which is what stops the bar emptying each time a phase turns over.
 */
function activityEntry(session: PaneSession, activity: ResourceActivity, plugin: ResourcePluginDefinition,
  blocking: boolean, now_ms: number): ActivityEntry {
  const progress = resourceWorkProgress(activity, plugin);
  const work = activity.work && activity.work.total > 0 ? activity.work : undefined;
  const counted = activity.total > 0 && Number.isFinite(activity.completed);
  const within = work ? Math.max(0, Math.min(1, work.completed / work.total))
    : counted ? Math.max(0, Math.min(1, activity.completed / activity.total)) : undefined;
  const phases = plugin.progressPhases ?? [];
  const phaseIndex = phases.findIndex((phase) => phase.id === activity.phase);
  const estimate = progressHistory.estimate(progressContext(session), activity, work ? within : undefined, now_ms);
  const fraction = estimate?.fraction
    ?? (within === undefined ? undefined : phaseIndex >= 0 ? (phaseIndex + within) / phases.length : within);
  return {
    id: activity.id,
    // A blocking bring-up is headed by its lane, so its line says what it is
    // doing right now; everything else is named by its own label.
    label: blocking && activity.operation ? activity.operation : progress.label,
    state: "active",
    count: work ? `${Math.min(work.completed, work.total).toLocaleString()} / ${work.total.toLocaleString()} ${work.unit}`
      : counted ? `${Math.min(activity.completed, activity.total).toLocaleString()} / ${activity.total.toLocaleString()}` : undefined,
    fraction,
    estimated: estimate !== undefined,
    remaining_ms: estimate?.remaining_ms,
    startedAt_ms: activity.startedAt_ms,
  };
}

export interface PaneActivity {
  readonly title?: string;
  readonly entries: readonly ActivityEntry[];
  readonly note?: string;
}

/**
 * Feeds the window's progress history from this pane's readiness, whether or
 * not anything is drawing it, so a run finished while the card was closed is
 * still on record for the next one.
 */
export function useProgressHistory(session: PaneSession): void {
  useEffect(() => session.diagnostics.subscribe((state) =>
    progressHistory.observe(progressContext(session), state.resourceReadiness, performance.now())), [session]);
}

/** Everything this pane is waiting on, as the transport card states it. */
export function usePaneActivity(session: PaneSession): PaneActivity {
  useProgressHistory(session);
  const gpuStatus = session.diagnostics((state) => state.gpuStatus);
  const gpuInfo = session.diagnostics((state) => state.gpuInfo);
  const resourceReadiness = session.diagnostics((state) => state.resourceReadiness);
  const methodId = session.method((state) => state.methodId);
  const stopped = ["unavailable", "lost", "stopping", "blocked"].includes(gpuStatus.state);
  if (stopped || resourceActivities(resourceReadiness).length === 0 && !transportWorkStatus(transportReadiness(gpuInfo, methodId), gpuInfo)) {
    return { entries: [] };
  }
  const now_ms = performance.now();
  const trayCards = resourceActivitiesFor(resourceReadiness, "card");
  // Transport-blocking work reports here too: the transport's controls only
  // disable (with the reason on each control), and this says what is running.
  const transportInline = resourceActivitiesFor(resourceReadiness, "transport-inline");
  const trayPills = [...transportInline, ...resourceActivitiesFor(resourceReadiness, "pill")];
  // The synthesized gate status earns a line only when it says something the
  // plugin activities are not already saying: while bring-up cards and pills
  // narrate the same loading with real counts, "Loading sparse world" on top of
  // them is a third telling of one story. Attention states (faults, capacity,
  // deferred resolution) always show — no activity carries those.
  const gateWork = transportWorkStatus(transportReadiness(gpuInfo, methodId), gpuInfo);
  const transportWork = gateWork
    && (gateWork.state !== "active" || (trayCards.length === 0 && transportInline.length === 0))
    ? gateWork : undefined;
  // A bring-up that has taken the scene heads the activity with its lane and
  // says, once, what it is holding up; work that blocks less is a line under
  // it, or — alone — the headline itself.
  const lead = trayCards[0];
  const pluginOf = (activity: ResourceActivity) => resourceReadiness.plugins[activity.pluginId].plugin;
  const entries: ActivityEntry[] = [
    ...trayCards.map((activity) => activityEntry(session, activity, pluginOf(activity), true, now_ms)),
    ...trayPills.map((activity) => activityEntry(session, activity, pluginOf(activity), false, now_ms)),
    ...(transportWork && transportWork.state !== "complete" ? [{
      id: "transport-gate",
      label: transportWork.label,
      state: transportWork.state,
      count: transportWork.total !== undefined && transportWork.completed !== undefined
        ? `${transportWork.completed.toLocaleString()} / ${transportWork.total.toLocaleString()} ${transportWork.unit ?? ""}`.trim()
        : undefined,
      fraction: transportWork.total && transportWork.completed !== undefined
        ? Math.max(0, Math.min(1, transportWork.completed / transportWork.total)) : undefined,
      // Running work is its own explanation; a fault or a wait has to say why.
      detail: transportWork.state === "active" ? undefined : transportWork.detail,
    } satisfies ActivityEntry] : []),
  ];
  return {
    title: lead ? bringupTitle(lead) : undefined,
    entries,
    note: lead?.retainingPrevious ? "The current scene stays in use until this is ready."
      : lead ? "The scene unlocks at its first complete frame."
      : transportInline.length > 0 ? "Playback unlocks when the fluid is ready."
      : undefined,
  };
}
