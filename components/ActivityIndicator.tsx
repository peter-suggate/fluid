"use client";

import { useEffect, useState } from "react";

/**
 * Everything a pane is waiting on, said under the controls it is holding.
 *
 * This was a mark of its own in the scene's top-right corner, while the Play
 * button it was explaining sat disabled at the bottom centre — the reason and
 * the thing it was the reason for, a screen apart. It is a section of the
 * transport card now: one bar for the whole of the progress, however many
 * pieces of work are running; one line per piece with its count; the time
 * spent; and at most one sentence about what is waiting on them.
 *
 * A count is only ever counted work over total work. Where the last finished
 * run of the same work is on record, the bar follows its timing instead and
 * says so — the percent and the time left wear "≈" — because a count of
 * unequal steps is not progress. With neither, the bar is indeterminate.
 */

export interface ActivityEntry {
  readonly id: string;
  readonly label: string;
  readonly state: "active" | "waiting" | "error";
  /** `completed / total unit`, already phrased; absent when the work has no count. */
  readonly count?: string;
  /** Progress through the whole of this entry, 0–1; absent when unknown. */
  readonly fraction?: number;
  /** The fraction comes from the last finished run's timing, not from a count. */
  readonly estimated?: boolean;
  /** Time left by that same record, paced to this run. */
  readonly remaining_ms?: number;
  /** Only said for an entry that needs attention: why it is not moving. */
  readonly detail?: string;
  readonly startedAt_ms?: number;
}

function useElapsed_s(startedAt_ms: number | undefined): number | undefined {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (startedAt_ms === undefined) return;
    const tick = () => setNow(performance.now());
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [startedAt_ms]);
  return startedAt_ms === undefined || now === 0 ? undefined : Math.max(0, now - startedAt_ms) / 1000;
}

function duration(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, seconds).toFixed(0)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds - minutes * 60).toString().padStart(2, "0")} s`;
}

export function ActivityIndicator({ title, entries, note, tag }: {
  /** What all of this is, in the pane's words. Absent, a lone entry names itself. */
  readonly title?: string;
  readonly entries: readonly ActivityEntry[];
  /** What is waiting on this work — one sentence, or nothing. */
  readonly note?: string;
  /** The pane this is about, where there is more than one. */
  readonly tag?: string;
}) {
  const active = entries.filter((entry) => entry.state === "active");
  const attention = entries.length > active.length;
  const counted = active.flatMap((entry) => entry.fraction === undefined ? [] : [entry.fraction]);
  // Every running entry has to be measured for the bar to be a fraction: one
  // unmeasured piece of work makes the whole of it unknown.
  const fraction = active.length > 0 && counted.length === active.length
    ? Math.max(0, Math.min(1, counted.reduce((sum, value) => sum + value, 0) / counted.length))
    : undefined;
  const estimated = active.some((entry) => entry.estimated);
  const remaining = active.flatMap((entry) => entry.remaining_ms === undefined ? [] : [entry.remaining_ms]);
  const remaining_s = remaining.length > 0 && remaining.length === active.length ? Math.max(...remaining) / 1000 : undefined;
  const starts = active.flatMap((entry) => entry.startedAt_ms === undefined ? [] : [entry.startedAt_ms]);
  const elapsed_s = useElapsed_s(starts.length > 0 ? Math.min(...starts) : undefined);
  if (entries.length === 0) return null;
  // A lone entry under no title is its own headline; its count rides beside it.
  const lone = title === undefined && entries.length === 1 ? entries[0] : undefined;
  const heading = title ?? lone?.label ?? "Working in the background";
  const rows = lone ? [] : entries;
  const percent = fraction === undefined ? undefined : Math.round(fraction * 100);
  return <section className="activity" data-tone={attention ? "attention" : undefined}
    data-testid="activity-indicator" aria-label="Background work">
    <div className="activity-head">
      {tag && <span className="activity-tag">{tag}</span>}
      {active.length === 0 && <i className="activity-flag" aria-hidden />}
      <strong role="status">{heading}</strong>
      {lone?.count && <span className="activity-meta">{lone.count}</span>}
    </div>
    {active.length > 0 && <div className="activity-progress">
      <div className="activity-bar" data-indeterminate={percent === undefined || undefined}
        role="progressbar" aria-label={heading} aria-valuemin={0} aria-valuemax={100}
        {...(percent === undefined ? { "aria-valuetext": "Working; completion unknown" }
          : { "aria-valuenow": percent, "aria-valuetext": `${estimated ? "about " : ""}${percent}%` })}>
        <i style={percent === undefined ? undefined : { width: `${percent}%` }} />
      </div>
      <span className="activity-figures">
        {percent !== undefined && <span title={estimated ? "From how long each phase took the last time this scene, method and quality finished" : "Counted work"}>
          {estimated ? "≈" : ""}{percent}%</span>}
        {elapsed_s !== undefined && <span title="Time since this work started">{duration(elapsed_s)}</span>}
        {remaining_s !== undefined && <span title="Time left by the last finished run, paced to this one">≈ {duration(remaining_s)} left</span>}
      </span>
    </div>}
    {lone?.detail && <p className="activity-note">{lone.detail}</p>}
    {rows.length > 0 && <ul className="activity-rows">
      {rows.map((entry) => <li key={entry.id} data-state={entry.state}>
        <span>{entry.label}</span>
        {entry.count && <span className="activity-meta">{entry.count}</span>}
        {entry.detail && <p>{entry.detail}</p>}
      </li>)}
    </ul>}
    {note && <p className="activity-note">{note}</p>}
  </section>;
}
