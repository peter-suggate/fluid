import type { ResourceActivity, ResourceReadinessSnapshot } from "./resource-readiness";

/**
 * How long each phase of a bring-up took the last time it finished, so the
 * next one can be read in time rather than in steps.
 *
 * A runner's count treats its phases as equal steps, and they are not: on a
 * large scene one compile phase is most of the wait, so the count races to
 * 90% and then sits. The phases a plugin moves through are facts, and so is
 * how long each took last time on the same scene, method and quality; their
 * ratio is the only weight here, and it is shown as an estimate ("≈"), never
 * as a count. With no finished run on record there is no estimate at all.
 */

const STORAGE_KEY = "fluid.progress-history.v1";
/** Runs shorter than this finish before anyone reads an estimate of them. */
const MINIMUM_RECORDED_MS = 1000;
/** Contexts kept on record; the oldest is forgotten first. */
const MAXIMUM_RECORDS = 60;

export interface ProgressRecord {
  /** Milliseconds each phase took, by phase id, in the order they ran. */
  readonly phases: readonly (readonly [string, number])[];
  readonly total_ms: number;
}

export interface ProgressEstimate {
  /** 0–1 of the recorded run's time this run has covered. */
  readonly fraction: number;
  readonly remaining_ms: number;
}

interface Run {
  readonly context: string;
  readonly startedAt_ms: number;
  readonly entered: { id: string; at_ms: number }[];
}

type RecordStore = Pick<Storage, "getItem" | "setItem">;

function browserStorage(): RecordStore | undefined {
  try { return typeof localStorage === "undefined" ? undefined : localStorage; } catch { return undefined; }
}

export class ProgressHistory {
  private readonly runs = new Map<string, Run>();
  private records: Record<string, ProgressRecord>;

  constructor(private readonly storage: RecordStore | undefined = browserStorage()) {
    this.records = this.load();
  }

  private load(): Record<string, ProgressRecord> {
    try {
      const parsed: unknown = JSON.parse(this.storage?.getItem(STORAGE_KEY) ?? "{}");
      return parsed && typeof parsed === "object" ? parsed as Record<string, ProgressRecord> : {};
    } catch { return {}; }
  }

  private save(): void {
    try { this.storage?.setItem(STORAGE_KEY, JSON.stringify(this.records)); } catch { /* Storage full or blocked: the estimate is a convenience. */ }
  }

  /**
   * Follows every plugin's activity through its phases. A run is recorded only
   * when its plugin reaches `ready` — a superseded or failed run is not how
   * long the work takes.
   */
  observe(context: string, snapshot: ResourceReadinessSnapshot, now_ms: number): void {
    for (const [pluginId, state] of Object.entries(snapshot.plugins)) {
      const run = this.runs.get(pluginId);
      const activity = state.activity;
      if (state.state === "preparing" && activity) {
        if (!run || run.startedAt_ms !== activity.startedAt_ms || run.context !== context) {
          this.runs.set(pluginId, { context, startedAt_ms: activity.startedAt_ms,
            entered: [{ id: activity.phase, at_ms: now_ms }] });
        } else if (!run.entered.some((phase) => phase.id === activity.phase)) {
          run.entered.push({ id: activity.phase, at_ms: now_ms });
        }
        continue;
      }
      if (!run) continue;
      this.runs.delete(pluginId);
      if (state.state !== "ready") continue;
      const total_ms = now_ms - run.entered[0]!.at_ms;
      if (total_ms < MINIMUM_RECORDED_MS) continue;
      const phases = run.entered.map((phase, index) =>
        [phase.id, (run.entered[index + 1]?.at_ms ?? now_ms) - phase.at_ms] as const);
      const key = `${pluginId}|${run.context}`;
      const { [key]: _replaced, ...rest } = this.records;
      const kept = Object.entries(rest).slice(1 - MAXIMUM_RECORDS);
      this.records = { ...Object.fromEntries(kept), [key]: { phases, total_ms } };
      this.save();
    }
  }

  /**
   * Where `activity` stands against the last finished run of the same work.
   * `within` is the counted share of the current phase where the activity has
   * one; otherwise the phase's own elapsed time stands in, never past its
   * recorded length.
   */
  estimate(context: string, activity: ResourceActivity, within: number | undefined, now_ms: number): ProgressEstimate | undefined {
    const record = this.records[`${activity.pluginId}|${context}`];
    const run = this.runs.get(activity.pluginId);
    if (!record || !run || record.total_ms <= 0) return undefined;
    const recorded = new Map(record.phases);
    const current = run.entered.at(-1)!;
    const currentRecorded_ms = recorded.get(current.id);
    if (currentRecorded_ms === undefined) return undefined;
    const behind_ms = run.entered.slice(0, -1).reduce((sum, phase) => sum + (recorded.get(phase.id) ?? 0), 0);
    const share = within ?? Math.min(0.95, (now_ms - current.at_ms) / Math.max(1, currentRecorded_ms));
    const covered_ms = Math.min(record.total_ms, behind_ms + currentRecorded_ms * Math.max(0, Math.min(1, share)));
    const fraction = Math.min(0.99, covered_ms / record.total_ms);
    // Scaled by how this run is pacing against the record: a machine that is
    // running slow today is slow for the rest of the run too.
    const elapsed_ms = now_ms - run.entered[0]!.at_ms;
    const pace = covered_ms > 0 ? Math.max(0.5, Math.min(2, elapsed_ms / covered_ms)) : 1;
    return { fraction, remaining_ms: Math.max(0, record.total_ms - covered_ms) * pace };
  }
}

/** The window's one history; every pane reads and feeds the same record. */
export const progressHistory = new ProgressHistory();
