/** Serializable progress vocabulary shared by resource plugins and frame stages.
 * Owners provide facts; presentation never estimates completion from elapsed time.
 */
export interface WorkProgressPhase {
  readonly id: string;
  readonly label: string;
  readonly unit?: string;
}
export interface WorkProgress {
  readonly label: string;
  readonly state: "active" | "waiting" | "complete" | "error";
  readonly completed?: number;
  readonly total?: number;
  readonly unit?: string;
  readonly detail?: string;
  readonly phase?: string;
  readonly phases?: readonly WorkProgressPhase[];
  readonly generation?: number;
  /** performance.now() in the consuming UI realm; worker status is rebased at its boundary. */
  readonly startedAt_ms?: number;
}

/** Count completion means this phase is finishing, not that the resource is ready. */
export function resourceWorkProgress(activity: {
  label: string; phase: string; completed: number; total: number; startedAt_ms: number;
}, plugin: { phaseCopy?: Readonly<Record<string, string>>; progressPhases?: readonly WorkProgressPhase[] }): WorkProgress {
  const phase = plugin.progressPhases?.find((item) => item.id === activity.phase);
  const finishing = activity.total > 0 && activity.completed >= activity.total;
  return {
    label: finishing ? `${activity.label} · finalizing` : activity.label,
    state: "active",
    completed: activity.completed,
    total: activity.total,
    unit: phase?.unit ?? "tasks",
    phase: activity.phase,
    phases: plugin.progressPhases,
    startedAt_ms: activity.startedAt_ms,
    detail: plugin.phaseCopy?.[activity.phase],
  };
}
